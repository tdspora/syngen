import re

import tensorflow as tf
from loguru import logger
import pickle
from pathlib import Path
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    Dense,
    Dropout,
    LeakyReLU,
    concatenate,
    Lambda,
    BatchNormalization,
    Activation,
)
from sklearn.mixture import BayesianGaussianMixture
import numpy as np
import pandas as pd

from syngen.ml.vae.models.custom_layers import FeatureLossLayer


def check_name(name):
    pattern = r"[^A-Za-z0-9_\\.]"
    return re.sub(pattern, "_", name)


class CVAE:
    """
    A class implementing the model architecture.
    """

    def __init__(
        self,
        dataset,
        batch_size=32,
        latent_dim=30,
        intermediate_dim=512,
        latent_components=30,
        cond_columns={},
        is_cond=False,
    ):
        self.dataset = dataset
        self.intermediate_dim = intermediate_dim
        self.batch_size = batch_size
        self.latent_dim = latent_dim
        self.latent_components = min(latent_components, len(self.dataset.df))
        self.model = None
        self.latent_model = None
        self.metrics = {}
        self.cond_features = cond_columns
        self.is_cond = is_cond

    def sample_z(self, args):
        mu, log_sigma = args
        eps = tf.random.normal(shape=(self.latent_dim,), mean=0.0, stddev=1.0)
        return mu + tf.exp(log_sigma / 2) * eps

    def build_model(self):
        self.inputs = list()
        self.encoders = list()
        self.feature_decoders = list()
        self.loss_models = dict()
        self.cond_inputs = list()

        for name, feature in self.dataset.features.items():
            if name in self.cond_features and self.is_cond:
                self.cond_inputs.append(feature.input)

            self.inputs.append(feature.input)
            self.encoders.append(feature.encoder)

        if len(self.encoders) > 1:
            features_encoders = concatenate(self.encoders)
        else:
            features_encoders = self.encoders[0]

        self.mu, self.log_sigma = self.__build_encoder(features_encoders)
        # embed_layer = SampleLayer(gamma=2,
        #                          capacity=30,
        #                          name='sampling_layer')([self.mu, self.log_sigma])
        z = Lambda(self.sample_z)([self.mu, self.log_sigma])

        if self.is_cond:
            if len(self.cond_inputs) > 1:
                self.cond_input = concatenate(self.cond_inputs)
                logger.info(f"Conditioning on {set(self.cond_features)}")
            else:
                self.cond_input = self.cond_inputs[0]
            z_cond = concatenate([z, self.cond_input])

            gen_inp_shape = self.latent_dim + self.cond_input.shape[1]
            decoder_input = z_cond
            encoder_output = z_cond
        else:
            gen_inp_shape = self.latent_dim
            # decoder_input = Lambda(lambda x: x, name='decoder_inp')(embed_layer)
            # encoder_output = embed_layer
            decoder_input = z
            encoder_output = self.mu

        kl_loss = (
            0.2
            * 0.5
            * tf.reduce_sum(
                tf.exp(self.log_sigma) + self.mu ** 2 - 1.0 - self.log_sigma, 1
            )
        )

        generator_input = Input(shape=(gen_inp_shape,))

        self.global_decoder, self.generator = self.__build_decoder(
            decoder_input, generator_input
        )

        generator_outputs = list()
        feature_losses = list()
        for i, (name, feature) in enumerate(self.dataset.features.items()):
            feature_decoder = feature.create_decoder(self.global_decoder)

            FeatureLossLayer(feature, name=check_name(name))
            feature_tensor = feature_decoder
            feature_losses.append(feature.loss)

            self.feature_decoders.append(feature_tensor)

            generator_outputs.append(feature.create_decoder(self.generator))

        self.model = Model(self.inputs, self.feature_decoders)

        self.model.add_loss(feature_losses)
        self.model.add_loss(kl_loss)

        self.encoder_model = Model(self.inputs, encoder_output)

        # generator
        self.generator_model = Model(generator_input, generator_outputs)

        return self.model

    def __append_metric(self, name, tensor):
        self.model.metrics.append(tensor)
        self.model.metrics_names.append(check_name(name))

    def __build_encoder(self, input):
        h0 = Dense(self.intermediate_dim, name="Encoder_0")(input)
        h0 = BatchNormalization()(h0)
        h0 = Activation(tf.nn.leaky_relu)(h0)
        h0 = Dropout(0.2)(h0)

        h1 = Dense(self.intermediate_dim, name="Encoder_1")(h0)
        h1 = BatchNormalization()(h1)
        h1 = Activation(tf.nn.leaky_relu)(h1)
        h1 = Dropout(0.2)(h1)

        h2 = Dense(self.intermediate_dim, name="Encoder_2")(h1)
        h2 = BatchNormalization()(h2)
        h2 = Activation(tf.nn.leaky_relu)(h2)
        h2 = Dropout(0.2)(h2)

        mu = Dense(self.latent_dim, name="mu")(h2)
        log_sigma = Dense(self.latent_dim, name="log_sigma")(h2)
        return mu, log_sigma

    def __build_decoder(self, input_z, generator_input):
        if self.is_cond:
            decoder_h0 = Dense(
                self.intermediate_dim + self.cond_input.shape[1],
                activation=LeakyReLU(),
                name="Decoder_0",
            )
        else:
            decoder_h0 = Dense(
                self.intermediate_dim, activation=LeakyReLU(), name="Decoder_0"
            )
        decoder_h1 = Dense(
            self.intermediate_dim, activation=LeakyReLU(), name="Decoder_1"
        )
        decoder_h2 = Dense(
            self.intermediate_dim, activation=LeakyReLU(), name="Decoder_2"
        )

        h_decoded0 = Dropout(0.2)(decoder_h0(input_z))
        h_decoded1 = Dropout(0.2)(decoder_h1(h_decoded0))
        decoder_output = Dropout(0.2)(decoder_h2(h_decoded1))

        generator0 = decoder_h0(generator_input)
        generator1 = decoder_h1(generator0)
        generator_output = decoder_h2(generator1)

        return decoder_output, generator_output

    def fit(self, data: pd.DataFrame, **kwargs) -> dict:
        transformed_data = self.dataset.transform(data)
        return self.model.fit(transformed_data, batch_size=self.batch_size, **kwargs)

    def fit_sampler(self, data: pd.DataFrame):
        logger.info("Fit sampler")
        transformed_data = self.dataset.transform(data)
        logger.info("Start encoding")
        latent_points = self.encoder_model.predict(transformed_data)

        logger.info("Creating BayesianGaussianMixture")
        self.latent_model = BayesianGaussianMixture(
            n_components=self.latent_components, n_init=10
        )
        logger.info("Fitting BayesianGaussianMixture")
        self.latent_model.fit(latent_points)
        logger.info("Finished fitting BayesianGaussianMixture")

    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        transformed_data = self.dataset.transform(data)
        prediction = self.model.predict(transformed_data, batch_size=self.batch_size)
        return self.dataset.inverse_transform(prediction)

    def sample(self, nb_samples: int) -> pd.DataFrame:
        latent_sample = self.latent_model.sample(nb_samples)[0]
        np.random.shuffle(latent_sample)

        synthetic_prediction = self.generator_model.predict(latent_sample)
        inverse_transformed_df = self.dataset.inverse_transform(synthetic_prediction)

        return inverse_transformed_df

    def less_likely_sample(
        self, nb_samples: int, temp: float = 0.05, variaty: int = 3
    ) -> pd.DataFrame:
        def pop_npoints(latent_vector, log):
            log_probs = {prob: idx for idx, prob in enumerate(log)}
            sorted_log_probs_keys = np.sort(list(log_probs.keys()))
            sorted_samples = {k: log_probs[k] for k in sorted_log_probs_keys}
            idxs = list(sorted_samples.values())[: int(nb_samples * temp / variaty)]
            return latent_vector[idxs]

        sliced_latent_sample = []
        for i in range(variaty):
            latent_sample = self.latent_model.sample(nb_samples)[0]
            log_likelihoods = self.latent_model.score_samples(latent_sample)
            sliced_latent_sample.append(pop_npoints(latent_sample, log_likelihoods))

        synthetic_prediction = self.generator_model.predict(sliced_latent_sample)
        return self.dataset.inverse_transform(synthetic_prediction)

    def show_model(self):
        from IPython.display import SVG
        from keras.utils.vis_utils import model_to_dot
        return SVG(model_to_dot(self.model).create(prog="dot", format="svg"))

    def save_state(self, path: str):
        pth = Path(path)

        if self.model is not None:
            self.model.save_weights(str(pth / "vae.ckpt"))

        if self.generator_model is not None:
            self.generator_model.save_weights(str(pth / "vae_generator.ckpt"))

        if self.latent_model is not None:
            with open(str(pth / "latent_model.pkl"), "wb") as f:
                f.write(pickle.dumps(self.latent_model))

    def load_state(self, path: str):
        pth = Path(path)

        self.model.load_weights(str(pth / "vae.ckpt"))
        self.generator_model.load_weights(str(pth / "vae_generator.ckpt"))

        with open(str(pth / "latent_model.pkl"), "rb") as f:
            self.latent_model = pickle.loads(f.read())
        return self
