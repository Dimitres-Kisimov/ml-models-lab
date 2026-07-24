"""Global cross-series MLP with a negative-binomial head (DeepAR-lite, no RNN)."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


class DemandNet(nn.Module):
    def __init__(self, n_features: int, n_skus: int, embed_dim: int = 8):
        super().__init__()
        self.embed = nn.Embedding(n_skus, embed_dim)
        self.body = nn.Sequential(
            nn.Linear(n_features + embed_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )
        self.head_mu = nn.Linear(64, 1)
        self.head_r = nn.Linear(64, 1)

    def forward(self, x, sku_idx):
        h = torch.cat([x, self.embed(sku_idx)], dim=1)
        h = self.body(h)
        mu = torch.nn.functional.softplus(self.head_mu(h)).squeeze(1) + 1e-4
        r = torch.nn.functional.softplus(self.head_r(h)).squeeze(1) + 1e-4
        return mu, r


def nb_nll(
    y: torch.Tensor, mu: torch.Tensor, r: torch.Tensor, weight: torch.Tensor | None = None
) -> torch.Tensor:
    """Weighted negative-binomial NLL (mean ``mu``, dispersion ``r``).

    Per-sample ``weight`` lets us align training with the scale-free MASE/RMSSE
    metric by up-weighting low-volume series (whose relative errors dominate).
    """
    eps = 1e-8
    log_r_mu = torch.log(r + mu + eps)
    ll = (
        torch.lgamma(y + r)
        - torch.lgamma(r)
        - torch.lgamma(y + 1.0)
        + r * (torch.log(r + eps) - log_r_mu)
        + y * (torch.log(mu + eps) - log_r_mu)
    )
    if weight is None:
        return -ll.mean()
    return -(ll * weight).sum() / weight.sum()


def train_net(
    X: np.ndarray,
    sku_idx: np.ndarray,
    y: np.ndarray,
    n_skus: int,
    mean: np.ndarray,
    std: np.ndarray,
    epochs: int = 250,
    lr: float = 0.01,
    seed: int = 0,
    sample_weight: np.ndarray | None = None,
) -> DemandNet:
    torch.manual_seed(seed)
    Xs = (X - mean) / std
    model = DemandNet(X.shape[1], n_skus)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    xt = torch.tensor(Xs, dtype=torch.float32)
    st = torch.tensor(sku_idx, dtype=torch.long)
    yt = torch.tensor(y, dtype=torch.float32)
    wt = None if sample_weight is None else torch.tensor(sample_weight, dtype=torch.float32)
    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        mu, r = model(xt, st)
        loss = nb_nll(yt, mu, r, wt)
        loss.backward()
        opt.step()
    model.eval()
    return model


def predict_mean(model: DemandNet, X: np.ndarray, sku_idx: np.ndarray, mean, std) -> np.ndarray:
    Xs = (X - mean) / std
    with torch.no_grad():
        mu, _ = model(
            torch.tensor(Xs, dtype=torch.float32),
            torch.tensor(sku_idx, dtype=torch.long),
        )
    return mu.numpy()
