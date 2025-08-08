# flake8: noqa
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


class _TabularDataset(Dataset):
    def __init__(self, arrays: List[np.ndarray]):
        self.arrs = [torch.tensor(a, dtype=torch.float32) for a in arrays]
        self.N = self.arrs[0].shape[0]

    def __len__(self) -> int:
        return self.N

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        xs = [a[idx] for a in self.arrs]
        x_cat = torch.cat(xs, dim=-1)
        return x_cat, xs


def build_dataloader(transformed: List[np.ndarray], batch_size: int) -> DataLoader:
    ds = _TabularDataset(transformed)
    return DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)


class TorchCVAE(nn.Module):
    def __init__(
        self,
        feature_dims: Dict[str, int],
        feature_types: Dict[str, str],
        latent_dim: int = 10,
        intermediate_dim: int = 128,
        cond_dims: int = 0,
    ):
        super().__init__()
        self.feature_dims = feature_dims
        self.feature_types = feature_types
        self.latent_dim = latent_dim
        self.cond_dims = cond_dims
        in_dim = sum(feature_dims.values())
        # Encoder trunk
        self.enc = nn.Sequential(
            nn.Linear(in_dim, intermediate_dim),
            nn.BatchNorm1d(intermediate_dim),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(intermediate_dim, intermediate_dim),
            nn.BatchNorm1d(intermediate_dim),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(intermediate_dim, intermediate_dim),
            nn.BatchNorm1d(intermediate_dim),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
        )
        self.mu = nn.Linear(intermediate_dim, latent_dim)
        self.log_sigma = nn.Linear(intermediate_dim, latent_dim)
        # Decoder trunk
        dec_in = latent_dim + cond_dims
        self.dec0 = nn.Linear(dec_in, intermediate_dim + cond_dims)
        self.dec1 = nn.Linear(intermediate_dim, intermediate_dim)
        self.dec2 = nn.Linear(intermediate_dim, intermediate_dim)
        self.drop = nn.Dropout(0.2)
        # per-feature heads
        self.heads = nn.ModuleDict(
            {
                name: nn.Linear(intermediate_dim, dim)
                for name, dim in feature_dims.items()
            }
        )

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.enc(x)
        mu = self.mu(h)
        log_sigma = self.log_sigma(h)
        return mu, log_sigma

    def reparameterize(self, mu: torch.Tensor, log_sigma: torch.Tensor) -> torch.Tensor:
        eps = torch.randn_like(mu)
        return mu + torch.exp(log_sigma * 0.5) * eps

    def decode_shared(self, z: torch.Tensor, cond: torch.Tensor = None) -> torch.Tensor:
        if cond is not None and self.cond_dims > 0:
            z = torch.cat([z, cond], dim=1)
        h = self.drop(F.leaky_relu(self.dec0(z)))
        h = self.drop(F.leaky_relu(self.dec1(h)))
        h = self.drop(F.leaky_relu(self.dec2(h)))
        return h

    def decode_heads(self, h: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {name: head(h) for name, head in self.heads.items()}

    def forward(self, x: torch.Tensor, cond: torch.Tensor = None):
        mu, log_sigma = self.encode(x)
        z = self.reparameterize(mu, log_sigma)
        h = self.decode_shared(z, cond)
        outs = self.decode_heads(h)
        return mu, log_sigma, outs


def kl_loss(mu: torch.Tensor, log_sigma: torch.Tensor) -> torch.Tensor:
    return (
        0.5
        * torch.sum(torch.exp(log_sigma) + mu.pow(2) - 1.0 - log_sigma, dim=1).mean()
    )


def recon_loss(
    preds: Dict[str, torch.Tensor],
    targets: List[torch.Tensor],
    feature_dims: Dict[str, int],
    feature_types: Dict[str, str],
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    total = torch.tensor(0.0, device=next(iter(preds.values())).device)
    by_feature: Dict[str, torch.Tensor] = {}
    for (name, dim), t in zip(feature_dims.items(), targets):
        p = preds[name]
        ftype = feature_types.get(name, "numeric")
        if ftype == "categorical" and dim > 1:
            idx = torch.argmax(t, dim=1)
            loss = F.cross_entropy(p, idx)
        elif ftype == "categorical" and dim == 1:
            loss = F.binary_cross_entropy_with_logits(p.view(-1), t.view(-1))
        else:
            loss = F.mse_loss(p, t)
        by_feature[name] = loss.detach()
        total = total + loss
    return total, by_feature


def train_epoch(
    model: TorchCVAE,
    loader: DataLoader,
    optim: torch.optim.Optimizer,
    feature_dims: Dict[str, int],
    feature_types: Dict[str, str],
) -> Tuple[float, float, Dict[str, float]]:
    model.train()
    total, total_kl = 0.0, 0.0
    sums: Dict[str, float] = {k: 0.0 for k in feature_dims}
    n = 0
    for x_cat, xs in loader:
        x_cat = x_cat.to(next(model.parameters()).device)
        xs_t = [a.to(x_cat.device) for a in xs]
        optim.zero_grad()
        mu, log_sigma, outs = model(x_cat)
        rec, by_f = recon_loss(outs, xs_t, feature_dims, feature_types)
        kl = kl_loss(mu, log_sigma)
        loss = rec + kl
        loss.backward()
        optim.step()
        total += loss.item()
        total_kl += kl.item()
        for name, v in by_f.items():
            sums[name] += float(v.item())
        n += 1
    means = {k: v / max(1, n) for k, v in sums.items()}
    return total / max(1, n), total_kl / max(1, n), means


@torch.no_grad()
def encode_dataset(model: TorchCVAE, loader: DataLoader) -> np.ndarray:
    model.eval()
    zs: List[np.ndarray] = []
    for x_cat, _ in loader:
        x_cat = x_cat.to(next(model.parameters()).device)
        mu, log_sigma = model.encode(x_cat)
        zs.append(mu.cpu().numpy())
    return np.concatenate(zs, axis=0)


@torch.no_grad()
def generate_from_latent(model: TorchCVAE, latent: np.ndarray) -> Dict[str, np.ndarray]:
    model.eval()
    z = torch.tensor(
        latent, dtype=torch.float32, device=next(model.parameters()).device
    )
    h = model.decode_shared(z)
    outs = model.decode_heads(h)
    res: Dict[str, np.ndarray] = {}
    for name, logits in outs.items():
        ftype = model.feature_types.get(name, "numeric")
        if ftype == "categorical" and model.feature_dims[name] > 1:
            res[name] = F.softmax(logits, dim=1).cpu().numpy()
        elif ftype == "categorical" and model.feature_dims[name] == 1:
            res[name] = torch.sigmoid(logits).cpu().numpy()
        else:
            res[name] = logits.cpu().numpy()
    return res
