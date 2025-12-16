from pathlib import Path
from typing import Optional

import tensorflow as tf
import keras
import keras.ops as ops
import pickle
from keras.models import Model
from keras.layers import (
    Input,
    Dense,
    Dropout,
    LeakyReLU,
    concatenate,
    Lambda,
    BatchNormalization,
)
from sklearn.mixture import BayesianGaussianMixture
import numpy as np
import pandas as pd
from loguru import logger

from syngen.ml.vae.models.custom_layers import FeatureLossLayer, set_seed_generator
from syngen.ml.utils import slugify_parameters, ProgressBarHandler


class CVAE:
    """
    A class implementing the model architecture.
    """

    def __init__(
        self,
        dataset,
        batch_size,
        latent_dim,
        intermediate_dim,
        latent_components,
        random_seed: Optional[int] = None,
        kl_weight: float = 0.0
    ):
        self.dataset = dataset
        self.intermediate_dim = intermediate_dim
        self.batch_size = batch_size
        self.latent_dim = latent_dim
        self.latent_components = min(latent_components, len(self.dataset.order_of_columns))
        self.random_seed = random_seed
        self.kl_weight = kl_weight
        self.model = None
        self.latent_model = None
        self.metrics = {}
        self.cond_features = {}
        self.is_cond = False
        self.inputs = list()
        self.encoders = list()
        self.feature_decoders = list()
        self.feature_losses = dict()
        self.feature_types = dict()
        self.loss_models = dict()
        self.cond_inputs = list()
        self.global_decoder = None
        self.generator = None
        # Set up seed generator for reproducible random operations
        # Store as instance variable for thread-safety with multiple CVAE instances
        self.seed_generator = keras.random.SeedGenerator(random_seed) if random_seed is not None else None
        # Also set module-level for backward compatibility with features
        set_seed_generator(random_seed)

    def sample_z(self, args):
        mu, log_sigma = args
        eps = keras.random.normal(shape=(self.latent_dim,), mean=0.0, stddev=1.0)
        return mu + ops.exp(log_sigma / 2) * eps

    @staticmethod
    @slugify_parameters(exclude_params=("feature",))
    def _create_feature_loss_layer(feature, name):
        FeatureLossLayer(feature, name=name)

    def build_model(self):
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

        generator_input = Input(shape=(gen_inp_shape,))

        self.__build_decoder(decoder_input, generator_input)

        generator_outputs = list()
        for i, (name, feature) in enumerate(self.dataset.features.items()):
            feature_decoder = feature.create_decoder(self.global_decoder)

            # Determine loss type based on feature type
            loss_type = 'categorical'  # default
            if hasattr(feature, 'feature_type'):
                ft = feature.feature_type
                if ft in ('continuous', 'float', 'int'):
                    loss_type = 'continuous'
                elif ft == 'binary':
                    loss_type = 'binary'
                elif ft == 'smart_text':
                    # SmartTextFeature: uses sparse categorical crossentropy
                    # Input is indices [batch, seq_len], output is logits [batch, seq_len, vocab]
                    loss_type = 'sparse_categorical'
                elif ft in ('char_text', 'charbasedtext', 'charbasedtextfeature', 'text'):
                    # CharBasedTextFeature and EmailFeature use one-hot encoding
                    loss_type = 'char_text'
                elif ft in ('datetime', 'datetimefeature'):
                    loss_type = 'datetime'
                # Add more explicit mappings as needed for other feature types

            # Get weight_randomizer from feature if available
            weight_randomizer = getattr(feature, 'weight_randomizer', None)

            # Create a loss layer that's actually wired into the graph
            loss_layer = FeatureLossLayer(
                feature=feature,
                loss_type=loss_type,
                weight=getattr(feature, 'weight', 1.0),
                weight_randomizer=weight_randomizer,
                seed_generator=self.seed_generator,
                name=f"loss_{name}"
            )
            # Wire the loss layer into the graph by passing input and decoder through it
            feature_tensor = loss_layer([feature.input, feature_decoder])

            self.feature_losses[name] = feature.loss  # Keep for compatibility
            self.feature_types[name] = feature.feature_type

            self.feature_decoders.append(feature_tensor)

            generator_outputs.append(feature.create_decoder(self.generator))

        # Use standard Model since losses are added via FeatureLossLayer
        self.model = Model(self.inputs, self.feature_decoders)

        self.encoder_model = Model(self.inputs, encoder_output)

        # generator
        self.generator_model = Model(generator_input, generator_outputs)

        return self.model

    def __build_encoder(self, input):
        h0 = Dense(self.intermediate_dim, name="Encoder_0")(input)
        h0 = BatchNormalization(name="First_encoder_BN")(h0)
        h0 = LeakyReLU()(h0)
        h0 = Dropout(0.2)(h0)

        h1 = Dense(self.intermediate_dim, name="Encoder_1")(h0)
        h1 = BatchNormalization(name="Second_encoder_BN")(h1)
        h1 = LeakyReLU()(h1)
        h1 = Dropout(0.2)(h1)

        h2 = Dense(self.intermediate_dim, name="Encoder_2")(h1)
        h2 = BatchNormalization(name="Third_encoder_BN")(h2)
        h2 = LeakyReLU()(h2)
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
            decoder_h0 = Dense(self.intermediate_dim, activation=LeakyReLU(), name="Decoder_0")
        decoder_h1 = Dense(self.intermediate_dim, activation=LeakyReLU(), name="Decoder_1")
        decoder_h2 = Dense(self.intermediate_dim, activation=LeakyReLU(), name="Decoder_2")

        h_decoded0 = Dropout(0.2)(decoder_h0(input_z))
        h_decoded1 = Dropout(0.2)(decoder_h1(h_decoded0))
        self.global_decoder = Dropout(0.2)(decoder_h2(h_decoded1))

        generator0 = decoder_h0(generator_input)
        generator1 = decoder_h1(generator0)
        self.generator = decoder_h2(generator1)

    def fit(self, data: pd.DataFrame, **kwargs) -> dict:
        transformed_data = self.dataset.transform(data)
        return self.model.fit(transformed_data, batch_size=self.batch_size, **kwargs)

    def fit_sampler(self, data: pd.DataFrame, max_fit_rows: Optional[int] = None):
        log_message = "Fit sampler"
        logger.info(log_message)
        ProgressBarHandler().set_progress(message=log_message)

        transformed_data = None
        if max_fit_rows is not None and len(data) > max_fit_rows:
            logger.info(
                f"Sampler fit row cap is enabled: using {max_fit_rows} of {len(data)} rows"
            )

            # Choose by row position (not index labels) so we can reliably subset arrays later.
            rng = np.random.default_rng(self.random_seed)
            row_indices = rng.choice(len(data), size=max_fit_rows, replace=False)
            row_indices.sort()

            # Fast path: transform only the sampled subset.
            # Some feature transformers may return arrays aligned to the original dataset index,
            # which can cause inconsistent cardinalities. If that happens, we fall back to
            # transforming the full data and slicing per-feature.
            data_for_sampler = data.iloc[row_indices]
            transformed_data = self.dataset.transform(data_for_sampler)

            try:
                sizes = [int(getattr(x, "shape", [len(x)])[0]) for x in transformed_data]
                if len(set(sizes)) != 1:
                    raise ValueError(f"Mismatched transformed feature sizes: {sizes}")
            except Exception as e:
                logger.warning(
                    "Sampler subset transform produced inconsistent feature sizes; "
                    f"falling back to full transform + slicing. Details: {e}"
                )
                transformed_full = self.dataset.transform(data)

                def _subset_rows(arr):
                    # pandas DataFrame/Series use [] as column selection; slice rows via iloc.
                    if isinstance(arr, tf.Tensor):
                        return tf.gather(arr, row_indices, axis=0)
                    if hasattr(arr, "iloc"):
                        arr = arr.iloc[row_indices]
                    else:
                        arr = arr[row_indices]
                    return arr.to_numpy() if hasattr(arr, "to_numpy") else arr

                transformed_data = [_subset_rows(x) for x in transformed_full]
        else:
            transformed_data = self.dataset.transform(data)

        log_message = "Start encoding"
        logger.info(log_message)
        ProgressBarHandler().set_progress(message=log_message)
        latent_points = self.encoder_model.predict(
            transformed_data,
            batch_size=self.batch_size,
            verbose=0,
        )
        # Free potentially large intermediate feature arrays as early as possible.
        del transformed_data

        # Handle NaN values in latent points (can happen if training became unstable)
        if np.isnan(latent_points).any() or np.isinf(latent_points).any():
            nan_count = np.isnan(latent_points).sum()
            inf_count = np.isinf(latent_points).sum()
            logger.warning(
                f"Latent points contain {nan_count} NaN and {inf_count} Inf values. "
                f"Replacing with zeros. This may affect generation quality."
            )
            latent_points = np.nan_to_num(latent_points, nan=0.0, posinf=0.0, neginf=0.0)

        n_samples = int(getattr(latent_points, "shape", [len(latent_points)])[0])
        n_components = max(1, min(int(self.latent_components), n_samples))

        logger.info(
            "Creating BayesianGaussianMixture"
            + (f" (n_components capped to {n_components} due to n_samples={n_samples})" if n_components != int(self.latent_components) else "")
        )
        self.latent_model = BayesianGaussianMixture(n_components=n_components, n_init=10)
        logger.info("Fitting BayesianGaussianMixture")
        self.latent_model.fit(latent_points)
        logger.info("Finished fitting BayesianGaussianMixture")

    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        transformed_data = self.dataset.transform(data)
        prediction = self.model.predict(transformed_data, batch_size=self.batch_size)
        return self.dataset.inverse_transform(prediction)

    def sample(self, nb_samples: int, temperature: float = 0.0) -> pd.DataFrame:
        """
        Sample synthetic data from the learned distribution.
        
        Args:
            nb_samples: Number of samples to generate
            temperature: Sampling temperature for probabilistic features.
                        0 = deterministic (argmax - default, backward compatible)
                        0.5 = conservative (sharper distributions)
                        1.0 = balanced probabilistic sampling
                        2.0 = exploratory (flatter distributions, more variety)
        """
        latent_sample = self.latent_model.sample(nb_samples)[0]
        np.random.shuffle(latent_sample)

        synthetic_prediction = self.generator_model.predict(latent_sample)
        self.inverse_transformed_df = self.dataset.inverse_transform(
            synthetic_prediction, temperature=temperature
        )
        pk_uq_keys_mapping = self.dataset.primary_keys_mapping
        if pk_uq_keys_mapping:
            self.__make_pk_uq_unique(pk_uq_keys_mapping, self.dataset.dropped_columns)
        return self.inverse_transformed_df

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

    def __check_pk_numeric_convertability(self, column, key_type):
        if (
            key_type is str
            and column not in self.dataset.long_text_columns | self.dataset.uuid_columns
        ):
            return self.inverse_transformed_df[column].dropna().str.isnumeric().all()
        else:
            return False

    def __make_pk_uq_unique(self, pk_uq_keys_mapping, empty_columns):
        for key_name, config in pk_uq_keys_mapping.items():
            key_columns = [
                column for column in config.get("columns") if column not in empty_columns
            ]
            for column in key_columns:
                key_type = self.dataset.pk_uq_keys_types[column]
                if key_type is float or self.__check_pk_numeric_convertability(column, key_type):
                    mapped_keys = np.arange(len(self.inverse_transformed_df[column])) + 1
                    self.inverse_transformed_df[column] = mapped_keys

    def save_state(self, path: str):
        pth = Path(path)

        if self.model is not None:
            self.model.save_weights(str(pth / "vae.weights.h5"))

        if self.generator_model is not None:
            self.generator_model.save_weights(str(pth / "vae_generator.weights.h5"))

        if self.latent_model is not None:
            with open(str(pth / "latent_model.pkl"), "wb") as f:
                f.write(pickle.dumps(self.latent_model))

    def load_state(self, path: str):
        pth = Path(path)
        # Load main model weights with fallback to .ckpt
        model_weights_file = pth / "vae.weights.h5"
        if not model_weights_file.exists():
            model_weights_file = pth / "vae.ckpt"
        self.model.load_weights(str(model_weights_file))
        # Load generator model weights with fallback to .ckpt
        generator_weights_file = pth / "vae_generator.weights.h5"
        if not generator_weights_file.exists():
            generator_weights_file = pth / "vae_generator.ckpt"
        self.generator_model.load_weights(str(generator_weights_file))

        with open(str(pth / "latent_model.pkl"), "rb") as f:
            self.latent_model = pickle.loads(f.read())
