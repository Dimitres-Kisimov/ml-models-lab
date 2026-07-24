"""Anomaly scoring by reconstruction error.

Two reconstructors:
- ``PCAReconstructor`` - numpy PCA-SVD baseline (mandatory challenger).
- ``TorchAutoencoder`` - undercomplete AE d-16-4-16-d, trained normal-only.

Both are fit on normal data only; the anomaly score is the per-row squared
reconstruction error in standardised feature space.
"""

from __future__ import annotations

import numpy as np


class Standardizer:
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, X: np.ndarray) -> Standardizer:
        self.mean = X.mean(axis=0)
        self.std = X.std(axis=0) + 1e-9
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean) / self.std


class PCAReconstructor:
    """PCA via SVD; reconstruct from the top-``k`` components (numpy only)."""

    def __init__(self, k: int = 4):
        self.k = k
        self.mean = None
        self.components = None  # (k, d)

    def fit(self, X: np.ndarray) -> PCAReconstructor:
        self.mean = X.mean(axis=0)
        Xc = X - self.mean
        _, _, vt = np.linalg.svd(Xc, full_matrices=False)
        self.components = vt[: self.k]
        return self

    def reconstruct(self, X: np.ndarray) -> np.ndarray:
        Xc = X - self.mean
        proj = Xc @ self.components.T
        return proj @ self.components + self.mean

    def score(self, X: np.ndarray) -> np.ndarray:
        recon = self.reconstruct(X)
        return np.sum((X - recon) ** 2, axis=1)

    def per_feature_error(self, X: np.ndarray) -> np.ndarray:
        recon = self.reconstruct(X)
        return (X - recon) ** 2


def pca_scores(X_train_norm: np.ndarray, X_eval: np.ndarray, k: int = 4):
    """Convenience: standardise on normal data, fit PCA, score eval set."""
    sc = Standardizer().fit(X_train_norm)
    pca = PCAReconstructor(k=k).fit(sc.transform(X_train_norm))
    return pca.score(sc.transform(X_eval)), sc, pca
