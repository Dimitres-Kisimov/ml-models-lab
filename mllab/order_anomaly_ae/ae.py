"""Undercomplete autoencoder d-16-4-16-d (torch), trained normal-only."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


class TorchAutoencoder(nn.Module):
    def __init__(self, d: int, bottleneck: int = 4, denoise_std: float = 0.0):
        super().__init__()
        self.denoise_std = denoise_std
        self.encoder = nn.Sequential(
            nn.Linear(d, 16), nn.ReLU(), nn.Linear(16, bottleneck), nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck, 16), nn.ReLU(), nn.Linear(16, d)
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


def train_ae(
    X_norm: np.ndarray,
    seed: int = 0,
    epochs: int = 300,
    lr: float = 1e-2,
    bottleneck: int = 4,
    denoise_std: float = 0.1,
) -> TorchAutoencoder:
    torch.manual_seed(seed)
    np.random.seed(seed)
    d = X_norm.shape[1]
    model = TorchAutoencoder(d, bottleneck=bottleneck, denoise_std=denoise_std)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    X = torch.tensor(X_norm, dtype=torch.float32)
    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        inp = X
        if denoise_std > 0:
            inp = X + denoise_std * torch.randn_like(X)
        recon = model(inp)
        loss = loss_fn(recon, X)
        loss.backward()
        opt.step()
    model.eval()
    return model


def ae_score(model: TorchAutoencoder, X: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        xt = torch.tensor(X, dtype=torch.float32)
        recon = model(xt)
        err = ((xt - recon) ** 2).sum(dim=1).numpy()
    return err
