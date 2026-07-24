"""L2-penalised logistic regression from scratch (numpy) + Platt calibration.

Core method: RFM (Bult & Wansbeek 1995) + L2 logistic regression. Optional
class-weighted / focal loss. Analytic gradient, full-batch gradient descent.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

FEATURE_COLS = [
    "recency_days",
    "frequency",
    "monetary",
    "tenure_days",
    "freq_slope",
    "avg_basket",
]


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -50.0, 50.0)))


def build_features(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(X, y, mean_std)`` with standardised RFM + engineered features."""
    X = df[FEATURE_COLS].to_numpy(dtype=float)
    y = df["churn"].to_numpy(dtype=float)
    mean = X.mean(axis=0)
    std = X.std(axis=0) + 1e-9
    Xs = (X - mean) / std
    return Xs, y, np.vstack([mean, std])


def standardize(df: pd.DataFrame, mean_std: np.ndarray) -> np.ndarray:
    X = df[FEATURE_COLS].to_numpy(dtype=float)
    return (X - mean_std[0]) / mean_std[1]


class LogisticRegressionScratch:
    """Full-batch gradient-descent logistic regression.

    Loss: L2-penalised BCE. With ``focal_gamma > 0`` the focal modulation
    (Lin et al. 2017) is applied; ``class_weight`` up-weights the positives.
    """

    def __init__(
        self,
        l2: float = 1.0,
        lr: float = 0.5,
        n_iter: int = 2000,
        class_weight: float = 1.0,
        focal_gamma: float = 0.0,
    ):
        self.l2 = l2
        self.lr = lr
        self.n_iter = n_iter
        self.class_weight = class_weight
        self.focal_gamma = focal_gamma
        self.w: np.ndarray | None = None
        self.b: float = 0.0

    def _weights(self, y: np.ndarray) -> np.ndarray:
        wt = np.ones_like(y)
        wt[y == 1] = self.class_weight
        return wt

    def loss(self, X: np.ndarray, y: np.ndarray) -> float:
        p = _sigmoid(X @ self.w + self.b)
        eps = 1e-9
        wt = self._weights(y)
        if self.focal_gamma > 0:
            pt = np.where(y == 1, p, 1.0 - p)
            ce = -(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))
            per = wt * ((1 - pt) ** self.focal_gamma) * ce
        else:
            per = -wt * (y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))
        return float(per.mean() + self.l2 * np.sum(self.w**2))

    def gradient(self, X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
        """Analytic gradient of the (weighted, plain-BCE) L2 objective.

        For the plain weighted BCE this is exactly
        ``X^T (wt*(sigma - y)) / n + 2*l2*w``.
        """
        n = X.shape[0]
        p = _sigmoid(X @ self.w + self.b)
        wt = self._weights(y)
        resid = wt * (p - y)
        gw = X.T @ resid / n + 2.0 * self.l2 * self.w
        gb = float(resid.mean())
        return gw, gb

    def fit(self, X: np.ndarray, y: np.ndarray) -> LogisticRegressionScratch:
        n, d = X.shape
        self.w = np.zeros(d)
        self.b = 0.0
        for _ in range(self.n_iter):
            if self.focal_gamma > 0:
                gw, gb = self._focal_gradient(X, y)
            else:
                gw, gb = self.gradient(X, y)
            self.w -= self.lr * gw
            self.b -= self.lr * gb
        return self

    def _focal_gradient(self, X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
        # numerically-stable focal gradient via finite structure is messy;
        # use autograd-free chain rule on the focal BCE.
        n = X.shape[0]
        p = _sigmoid(X @ self.w + self.b)
        wt = self._weights(y)
        g = self.focal_gamma
        eps = 1e-9
        pt = np.where(y == 1, p, 1.0 - p)
        # d(focal)/d(logit); sign folded so positives push logit up
        mod = (1 - pt) ** g
        dce = p - y  # d BCE / d logit
        dmod = -g * (1 - pt) ** (g - 1)
        dpt = np.where(y == 1, p * (1 - p), -(p * (1 - p)))
        ce = -(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))
        dlogit = wt * (mod * dce + dmod * dpt * ce)
        gw = X.T @ dlogit / n + 2.0 * self.l2 * self.w
        gb = float(dlogit.mean())
        return gw, gb

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        return X @ self.w + self.b

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return _sigmoid(self.decision_function(X))


class PlattCalibrator:
    """1-D logistic calibration of raw logits (Platt 1999), fit by GD."""

    def __init__(self, lr: float = 0.1, n_iter: int = 2000):
        self.a = 1.0
        self.b = 0.0
        self.lr = lr
        self.n_iter = n_iter

    def fit(self, logits: np.ndarray, y: np.ndarray) -> PlattCalibrator:
        logits = np.asarray(logits, dtype=float)
        y = np.asarray(y, dtype=float)
        for _ in range(self.n_iter):
            p = _sigmoid(self.a * logits + self.b)
            resid = p - y
            ga = float((resid * logits).mean())
            gb = float(resid.mean())
            self.a -= self.lr * ga
            self.b -= self.lr * gb
        return self

    def transform(self, logits: np.ndarray) -> np.ndarray:
        return _sigmoid(self.a * np.asarray(logits, dtype=float) + self.b)
