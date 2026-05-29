from pathlib import Path
import pickle

import torch
import torch.nn as nn
from sklearn.mixture import BayesianGaussianMixture
import numpy as np
import pandas as pd
from loguru import logger

from syngen.ml.vae.models.custom_layers import reparameterize
from syngen.ml.utils import (
    ProgressBarHandler,
    generate_unique_values_by_regex,
    is_number_regex_pattern
)

# Activation slopes chosen to match the TF graph exactly:
#   encoder used tf.nn.leaky_relu (alpha=0.2);
#   decoder used Keras LeakyReLU() (default alpha=0.3).
ENCODER_LEAKY_SLOPE = 0.2
DECODER_LEAKY_SLOPE = 0.3
# Keras BatchNormalization defaults: momentum=0.99, epsilon=1e-3. PyTorch's
# momentum is the complement, so 0.99 -> 0.01.
BN_MOMENTUM = 0.01
BN_EPS = 1e-3
DROPOUT = 0.2


def _to_tensors(transformed):
    """List of per-feature numpy arrays (in feature order) -> list of float32
    tensors, preserving order (collapse hypothesis #4)."""
    return [torch.as_tensor(np.asarray(arr), dtype=torch.float32) for arr in transformed]


class CVAEModule(nn.Module):
    """The PyTorch CVAE graph (encoder + reparameterization + **shared** decoder).

    The same decoder + per-feature heads serve both the reconstruction path
    (``forward``, fed ``z``) and the generation path (``decode``, fed a BGM
    latent sample). There is exactly one decoder module, so the generator cannot
    diverge from the trained decoder — collapse hypothesis #2 handled by
    construction (replaces the TF layer-reuse at ``model.py:158-176`` and the
    separate ``generator_model``).
    """

    def __init__(self, features: dict, latent_dim: int, intermediate_dim: int):
        super().__init__()
        self.latent_dim = latent_dim
        self.intermediate_dim = intermediate_dim
        feats = list(features.values())

        # Per-feature encoder pieces, in self.features order (identity for tabular
        # features; BiLSTM for text). encoded_dim is the concat contribution.
        self.feature_encoders = nn.ModuleList([f.build_encoder() for f in feats])
        total_encoded = sum(f.encoded_dim for f in feats)

        # Shared encoder: 3 x [Linear -> BatchNorm -> LeakyReLU(0.2) -> Dropout].
        self.enc_block0 = self._encoder_block(total_encoded, intermediate_dim)
        self.enc_block1 = self._encoder_block(intermediate_dim, intermediate_dim)
        self.enc_block2 = self._encoder_block(intermediate_dim, intermediate_dim)
        self.mu_layer = nn.Linear(intermediate_dim, latent_dim)
        self.log_sigma_layer = nn.Linear(intermediate_dim, latent_dim)

        # Shared decoder: 3 x [Linear -> LeakyReLU(0.3) -> Dropout].
        self.dec_block0 = self._decoder_block(latent_dim, intermediate_dim)
        self.dec_block1 = self._decoder_block(intermediate_dim, intermediate_dim)
        self.dec_block2 = self._decoder_block(intermediate_dim, intermediate_dim)

        # Per-feature decoder heads, in self.features order.
        self.feature_heads = nn.ModuleList(
            [f.build_decoder_head(intermediate_dim) for f in feats]
        )

        # Match Keras Dense defaults (glorot_uniform kernel, zero bias). With only
        # a handful of epochs the result is init-dominated, so matching TF's init
        # keeps the generated spread aligned with the baseline.
        self.apply(self._init_linear)

    @staticmethod
    def _init_linear(module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    @staticmethod
    def _encoder_block(in_features: int, out_features: int) -> nn.Module:
        return nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.BatchNorm1d(out_features, eps=BN_EPS, momentum=BN_MOMENTUM),
            nn.LeakyReLU(ENCODER_LEAKY_SLOPE),
            nn.Dropout(DROPOUT),
        )

    @staticmethod
    def _decoder_block(in_features: int, out_features: int) -> nn.Module:
        return nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.LeakyReLU(DECODER_LEAKY_SLOPE),
            nn.Dropout(DROPOUT),
        )

    def _encode_features(self, inputs):
        encoded = [enc(x) for enc, x in zip(self.feature_encoders, inputs)]
        return torch.cat(encoded, dim=-1) if len(encoded) > 1 else encoded[0]

    def encode(self, inputs):
        h = self._encode_features(inputs)
        h = self.enc_block2(self.enc_block1(self.enc_block0(h)))
        return self.mu_layer(h), self.log_sigma_layer(h)

    def decode(self, latent):
        d = self.dec_block2(self.dec_block1(self.dec_block0(latent)))
        return [head(d) for head in self.feature_heads]

    def forward(self, inputs):
        mu, log_sigma = self.encode(inputs)
        z = reparameterize(mu, log_sigma)
        return self.decode(z), mu, log_sigma


def kl_divergence(mu: torch.Tensor, log_sigma: torch.Tensor) -> torch.Tensor:
    """0.5 * sum(exp(log_sigma) + mu^2 - 1 - log_sigma), mean over the batch.

    Same form as ``model.py:103-107``. Reported under ``kl_loss`` but added to the
    total with **weight 0** in the training loop (collapse hypothesis #1)."""
    per_sample = 0.5 * torch.sum(
        torch.exp(log_sigma) + mu ** 2 - 1.0 - log_sigma, dim=1
    )
    return per_sample.mean()


class CVAE:
    """
    A class implementing the model architecture (PyTorch backend).
    """

    BACKEND = "pytorch"
    ARTIFACT_VERSION = 1

    def __init__(self, dataset, batch_size, latent_dim, intermediate_dim, latent_components):
        self.dataset = dataset
        self.intermediate_dim = intermediate_dim
        self.batch_size = batch_size
        self.latent_dim = latent_dim
        self.latent_components = min(latent_components, len(self.dataset.order_of_columns))
        self.model = None              # CVAEModule (nn.Module)
        self.latent_model = None       # BayesianGaussianMixture
        self.feature_types = dict()
        self.feature_order = list()

    def build_model(self):
        features = self.dataset.features
        self.feature_order = list(features.keys())
        self.feature_types = {name: feature.feature_type for name, feature in features.items()}
        self.model = CVAEModule(features, self.latent_dim, self.intermediate_dim)
        return self.model

    def fit_sampler(self, data: pd.DataFrame):
        log_message = "Fit sampler"
        logger.info(log_message)
        ProgressBarHandler().set_progress(message=log_message)
        transformed_data = self.dataset.transform(data)
        log_message = "Start encoding"
        logger.info(log_message)
        ProgressBarHandler().set_progress(message=log_message)
        # eval() so BatchNorm uses running stats and Dropout is off during the
        # deterministic encode (collapse hypothesis #3); mu only, like the TF
        # encoder_model that outputs `mu` (model.py:101,190).
        self.model.eval()
        with torch.no_grad():
            mu, _ = self.model.encode(_to_tensors(transformed_data))
        latent_points = mu.cpu().numpy()

        logger.info("Creating BayesianGaussianMixture")
        self.latent_model = BayesianGaussianMixture(n_components=self.latent_components, n_init=10)
        logger.info("Fitting BayesianGaussianMixture")
        self.latent_model.fit(latent_points)
        logger.info("Finished fitting BayesianGaussianMixture")

    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        self.model.eval()
        with torch.no_grad():
            recons, _, _ = self.model(_to_tensors(self.dataset.transform(data)))
        prediction = [r.cpu().numpy() for r in recons]
        return self.dataset.inverse_transform(prediction)

    def sample(self, nb_samples: int) -> pd.DataFrame:
        latent_sample = self.latent_model.sample(nb_samples)[0]
        np.random.shuffle(latent_sample)

        # eval() + no_grad so the shared decoder runs with Dropout off
        # (collapse hypothesis #3); decode the BGM draw, not N(0,1) (hypothesis #6).
        self.model.eval()
        with torch.no_grad():
            latent = torch.as_tensor(latent_sample, dtype=torch.float32)
            outputs = self.model.decode(latent)
        synthetic_prediction = [o.cpu().numpy() for o in outputs]
        self.inverse_transformed_df = self.dataset.inverse_transform(synthetic_prediction)
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

        latent = torch.as_tensor(np.concatenate(sliced_latent_sample), dtype=torch.float32)
        self.model.eval()
        with torch.no_grad():
            outputs = self.model.decode(latent)
        synthetic_prediction = [o.cpu().numpy() for o in outputs]
        return self.dataset.inverse_transform(synthetic_prediction)

    def __check_pk_numeric_convertability(self, column, key_type):
        if (
            key_type is str
            and column not in self.dataset.long_text_columns | self.dataset.uuid_columns
        ):
            return self.inverse_transformed_df[column].dropna().str.isnumeric().all()
        else:
            return False

    def __apply_sequential_keys(self, column):
        self.inverse_transformed_df[column] = (
            np.arange(len(self.inverse_transformed_df[column])) + 1
        )

    def __apply_regex_keys(self, column, key_name, regex_pattern):
        logger.info(
            f"The key column `{column}` (key `{key_name}`) "
            "has been generated using the provided regular expression."
        )
        self.inverse_transformed_df[column] = generate_unique_values_by_regex(
            regex_pattern=regex_pattern,
            size=self.inverse_transformed_df.shape[0]
        )

    def __make_pk_uq_unique(self, pk_uq_keys_mapping, empty_columns):
        for key_name, config in pk_uq_keys_mapping.items():
            key_columns = [
                column for column in config.get("columns") if column not in empty_columns
            ]
            for column in key_columns:
                key_type = self.dataset.pk_uq_keys_types[column]
                is_number_key_type = (
                    key_type is float or self.__check_pk_numeric_convertability(column, key_type)
                )
                regex_pattern = pk_uq_keys_mapping[key_name].get(
                    "regex_patterns", {}
                ).get(f"{column}")
                if is_number_key_type and regex_pattern is None:
                    self.__apply_sequential_keys(column)
                elif (
                    is_number_key_type
                    and regex_pattern is not None
                    and not is_number_regex_pattern(regex_pattern)
                ):
                    logger.warning(
                        f"The provided regex for generating key column `{column}` "
                        f"(key `{key_name}`) may produce non-numeric string values "
                        "while the column is expected to be numeric. "
                        "The regular expression will be ignored."
                    )
                    self.__apply_sequential_keys(column)

                elif regex_pattern is not None:
                    if (
                        not is_number_key_type
                        or (is_number_key_type and is_number_regex_pattern(regex_pattern))
                    ):
                        self.__apply_regex_keys(column, key_name, regex_pattern)

    def save_state(self, path: str):
        pth = Path(path)
        pth.mkdir(parents=True, exist_ok=True)

        if self.model is not None:
            torch.save(
                {
                    "backend": self.BACKEND,
                    "version": self.ARTIFACT_VERSION,
                    "state_dict": self.model.state_dict(),
                },
                str(pth / "vae_state.pt"),
            )

        if self.latent_model is not None:
            with open(str(pth / "latent_model.pkl"), "wb") as f:
                f.write(pickle.dumps(self.latent_model))

    def load_state(self, path: str):
        pth = Path(path)
        state_file = pth / "vae_state.pt"
        if not state_file.exists():
            legacy = any((pth / name).exists() for name in
                         ("vae.ckpt", "vae.ckpt.index", "vae_generator.ckpt.index"))
            hint = (
                " A TensorFlow-era checkpoint (vae.ckpt) is present; the PyTorch "
                "backend cannot load it. Retrain with the PyTorch backend."
            ) if legacy else ""
            raise FileNotFoundError(f"Missing PyTorch VAE state at '{state_file}'.{hint}")

        checkpoint = torch.load(str(state_file), map_location="cpu", weights_only=False)
        backend = checkpoint.get("backend")
        if backend != self.BACKEND:
            raise ValueError(
                f"Incompatible model artifact: backend {backend!r}, "
                f"expected {self.BACKEND!r}. Retrain with the current backend."
            )
        self.model.load_state_dict(checkpoint["state_dict"])

        with open(str(pth / "latent_model.pkl"), "rb") as f:
            self.latent_model = pickle.loads(f.read())
