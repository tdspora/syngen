from pathlib import Path

from typing import Dict, Tuple, List
import pickle

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.mixture import BayesianGaussianMixture
from loguru import logger
from syngen.ml.utils import ProgressBarHandler
import math


class CVAE(nn.Module):
    """
    PyTorch implementation of Conditional Variational Autoencoder used by Syngen.

    Notes:
    - Reconstruction losses are computed explicitly outside the module (in the wrapper)
      via `compute_losses` to preserve per-feature accounting.
    - KL term can be combined with a weight (currently set to 0.0 in wrapper to match
      previous behavior).
    """

    def __init__(self, dataset, batch_size, latent_dim, intermediate_dim, latent_components):
        super().__init__()
        self.dataset = dataset
        self.intermediate_dim = intermediate_dim
        self.batch_size = batch_size
        self.latent_dim = latent_dim
        self.latent_components = min(latent_components, len(self.dataset.order_of_columns))

        # Collect feature metadata and shapes
        self.feature_types: Dict[str, str] = {}
        self.feature_shapes: Dict[str, Tuple[int, ...]] = {}
        self.flatten_dims: Dict[str, int] = {}
        for name, feature in self.dataset.features.items():
            ftype = getattr(feature, "feature_type", "numeric")
            self.feature_types[name] = ftype
            if ftype == "text":
                L = getattr(feature, "text_max_len")
                V = getattr(feature, "vocab_size")
                shape = (L, V)
            else:
                dim = getattr(feature, "input_dimension", 1)
                shape = (dim,)
            self.feature_shapes[name] = shape
            self.flatten_dims[name] = int(np.prod(shape))

        self.feature_names: List[str] = list(self.dataset.features.keys())
        self.total_in_dim = int(sum(self.flatten_dims.values()))

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(self.total_in_dim, self.intermediate_dim),
            nn.BatchNorm1d(self.intermediate_dim),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(self.intermediate_dim, self.intermediate_dim),
            nn.BatchNorm1d(self.intermediate_dim),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(self.intermediate_dim, self.intermediate_dim),
            nn.BatchNorm1d(self.intermediate_dim),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
        )
        self.mu_head = nn.Linear(self.intermediate_dim, self.latent_dim)
        self.logvar_head = nn.Linear(self.intermediate_dim, self.latent_dim)

        # Decoder (shared)
        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, self.intermediate_dim),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(self.intermediate_dim, self.intermediate_dim),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(self.intermediate_dim, self.intermediate_dim),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
        )

        # Per-feature heads
        self.heads = nn.ModuleDict({
            name: nn.Linear(self.intermediate_dim, out_dim)
            for name, out_dim in self.flatten_dims.items()
        })

        # Sampler for latent space
        self.latent_model: BayesianGaussianMixture | None = None

    @staticmethod
    def _flatten_inputs(inputs: List[torch.Tensor]) -> torch.Tensor:
        flats = []
        for x in inputs:
            if x.dim() > 2:
                flats.append(x.view(x.size(0), -1))
            else:
                flats.append(x)
        return torch.cat(flats, dim=1)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, inputs: List[torch.Tensor]):
        x = self._flatten_inputs(inputs)
        h = self.encoder(x)
        mu = self.mu_head(h)
        logvar = self.logvar_head(h)
        z = self.reparameterize(mu, logvar)
        h_dec = self.decoder(z)
        outputs: Dict[str, torch.Tensor] = {}
        for name, head in self.heads.items():
            out = head(h_dec)
            shape = self.feature_shapes[name]
            if len(shape) == 1:
                outputs[name] = out
            else:
                outputs[name] = out.view(-1, *shape)
        return outputs, mu, logvar

    def compute_losses(
        self,
        inputs_by_name: Dict[str, torch.Tensor],
        outputs_by_name: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        total = torch.tensor(0.0, device=next(self.parameters()).device)
        per_feature: Dict[str, torch.Tensor] = {}
        for name, y_pred in outputs_by_name.items():
            ftype = self.feature_types.get(name, "numeric")
            y_true = inputs_by_name[name]
            if ftype == "categorical":
                # y_true (B, C) one-hot -> indices
                target = torch.argmax(y_true, dim=-1)
                loss = F.cross_entropy(y_pred, target)
            elif ftype == "text":
                # y_true (B, L, V) one-hot
                B, L, V = y_pred.shape
                target = torch.argmax(y_true, dim=-1).view(-1)
                logits = y_pred.view(B * L, V)
                loss = F.cross_entropy(logits, target)
            elif ftype == "binary":
                loss = F.mse_loss(y_pred, y_true)
            else:
                loss = F.mse_loss(y_pred, y_true)
            per_feature[name] = loss.detach()
            total = total + loss
        return total, per_feature

    @staticmethod
    def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        return 0.5 * torch.sum(torch.exp(logvar) + mu ** 2 - 1.0 - logvar, dim=1).mean()

    # ----------------- Inference/sampling helpers -----------------
    @torch.no_grad()
    def fit_sampler(self, data: pd.DataFrame):
        inputs_np = self.dataset.transform(data)
        normalized: List[np.ndarray] = []
        for inp in inputs_np:
            if isinstance(inp, pd.DataFrame):
                arr = inp.to_numpy(dtype=np.float32)
            else:
                arr = np.asarray(inp, dtype=np.float32)
                if arr.ndim == 1:
                    arr = arr.reshape(-1, 1)
            normalized.append(arr)
        tensors = [torch.from_numpy(arr).float().to(next(self.parameters()).device) for arr in normalized]
        mu = self.forward(tensors)[1].cpu().numpy()
        logger.info("Creating BayesianGaussianMixture")
        self.latent_model = BayesianGaussianMixture(n_components=self.latent_components, n_init=10)
        logger.info("Fitting BayesianGaussianMixture")
        self.latent_model.fit(mu)
        logger.info("Finished fitting BayesianGaussianMixture")

    @torch.no_grad()
    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        inputs_np = self.dataset.transform(data)
        normalized: List[np.ndarray] = []
        for inp in inputs_np:
            if isinstance(inp, pd.DataFrame):
                arr = inp.to_numpy(dtype=np.float32)
            else:
                arr = np.asarray(inp, dtype=np.float32)
                if arr.ndim == 1:
                    arr = arr.reshape(-1, 1)
            normalized.append(arr)
        tensors = [torch.from_numpy(arr).float().to(next(self.parameters()).device) for arr in normalized]
        outputs, _, _ = self.forward(tensors)
        preds_list = [outputs[name].detach().cpu().numpy() for name in self.feature_names]
        return self.dataset.inverse_transform(preds_list)

    @torch.no_grad()
    def sample(self, nb_samples: int) -> pd.DataFrame:
        if self.latent_model is None:
            raise RuntimeError("Latent sampler is not fitted. Call fit_sampler first.")
        latent_sample = self.latent_model.sample(nb_samples)[0]
        np.random.shuffle(latent_sample)
        z = torch.from_numpy(latent_sample).float().to(next(self.parameters()).device)
        h_dec = self.decoder(z)
        outputs = []
        for name, head in self.heads.items():
            out = head(h_dec)
            shape = self.feature_shapes[name]
            if len(shape) > 1:
                out = out.view(-1, *shape)
            outputs.append(out.detach().cpu().numpy())
        self.inverse_transformed_df = self.dataset.inverse_transform(outputs)
        pk_uq_keys_mapping = self.dataset.primary_keys_mapping
        if pk_uq_keys_mapping:
            self.__make_pk_uq_unique(pk_uq_keys_mapping, self.dataset.dropped_columns)
        return self.inverse_transformed_df

    @torch.no_grad()
    def less_likely_sample(self, nb_samples: int, temp: float = 0.05, variaty: int = 3) -> pd.DataFrame:
        if self.latent_model is None:
            raise RuntimeError("Latent sampler is not fitted. Call fit_sampler first.")

        def pop_npoints(latent_vector, log):
            log_probs = {prob: idx for idx, prob in enumerate(log)}
            sorted_log_probs_keys = np.sort(list(log_probs.keys()))
            sorted_samples = {k: log_probs[k] for k in sorted_log_probs_keys}
            idxs = list(sorted_samples.values())[: int(nb_samples * temp / variaty)]
            return latent_vector[idxs]

        sliced_latent_sample = []
        for _ in range(variaty):
            latent_sample = self.latent_model.sample(nb_samples)[0]
            log_likelihoods = self.latent_model.score_samples(latent_sample)
            sliced_latent_sample.append(pop_npoints(latent_sample, log_likelihoods))

        z = torch.from_numpy(np.concatenate(sliced_latent_sample, axis=0)).float().to(next(self.parameters()).device)
        h_dec = self.decoder(z)
        outputs = []
        for name, head in self.heads.items():
            out = head(h_dec)
            shape = self.feature_shapes[name]
            if len(shape) > 1:
                out = out.view(-1, *shape)
            outputs.append(out.detach().cpu().numpy())
        return self.dataset.inverse_transform(outputs)

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
        pth.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), str(pth / "vae.pt"))
        if self.latent_model is not None:
            with open(str(pth / "latent_model.pkl"), "wb") as f:
                f.write(pickle.dumps(self.latent_model))

    def load_state(self, path: str):
        pth = Path(path)
        self.load_state_dict(torch.load(str(pth / "vae.pt"), map_location="cpu"))
        with open(str(pth / "latent_model.pkl"), "rb") as f:
            self.latent_model = pickle.loads(f.read())
