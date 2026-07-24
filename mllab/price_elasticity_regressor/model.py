"""Ridge and Lasso regression from scratch for log-log price elasticity.

- Ridge: closed form via ``scipy.linalg.solve`` (oracle) + gradient descent.
- Lasso: coordinate descent with soft-thresholding.
- Optimal price: Lerner rule ``P* = MC*|e|/(|e|-1)`` guarded for |e|<=1.
- Hierarchical shrinkage of thin-segment elasticities toward the pooled mean.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import solve


def _add_intercept(X: np.ndarray) -> np.ndarray:
    return np.hstack([np.ones((X.shape[0], 1)), X])


def ridge_closed_form(X: np.ndarray, y: np.ndarray, l2: float = 1.0) -> np.ndarray:
    """Closed-form ridge (intercept not penalised). Returns ``[b0, b1, ...]``."""
    Xi = _add_intercept(X)
    d = Xi.shape[1]
    A = Xi.T @ Xi + l2 * np.eye(d)
    A[0, 0] -= l2  # do not penalise intercept
    b = Xi.T @ y
    return solve(A, b, assume_a="sym")


def ridge_gd(
    X: np.ndarray,
    y: np.ndarray,
    l2: float = 1.0,
    lr: float = 0.05,
    n_iter: int = 5000,
) -> np.ndarray:
    """Ridge via full-batch gradient descent (intercept not penalised)."""
    Xi = _add_intercept(X)
    n, d = Xi.shape
    w = np.zeros(d)
    for _ in range(n_iter):
        grad = Xi.T @ (Xi @ w - y) / n
        pen = 2.0 * l2 * w / n
        pen[0] = 0.0
        w -= lr * (grad + pen)
    return w


def _soft_threshold(rho: float, lam: float) -> float:
    if rho > lam:
        return rho - lam
    if rho < -lam:
        return rho + lam
    return 0.0


def lasso_coordinate_descent(
    X: np.ndarray,
    y: np.ndarray,
    l1: float = 0.1,
    n_iter: int = 500,
    tol: float = 1e-7,
) -> np.ndarray:
    """Lasso via coordinate descent with soft-thresholding.

    Intercept (column 0 of the augmented design) is not penalised. Returns
    ``[b0, b1, ...]``.
    """
    Xi = _add_intercept(X)
    n, d = Xi.shape
    w = np.zeros(d)
    col_sq = (Xi**2).sum(axis=0)
    for _ in range(n_iter):
        w_old = w.copy()
        for j in range(d):
            r_j = y - Xi @ w + w[j] * Xi[:, j]
            rho = Xi[:, j] @ r_j
            if j == 0:
                w[j] = rho / col_sq[j]  # unpenalised intercept
            else:
                w[j] = _soft_threshold(rho, l1 * n) / col_sq[j]
        if np.max(np.abs(w - w_old)) < tol:
            break
    return w


def optimal_price(mc: float, elasticity: float, price_cap: float = 1e6) -> float:
    """Lerner optimal price ``P* = MC*|e|/(|e|-1)``, guarded for |e|<=1.

    For inelastic demand (|e|<=1) profit is monotically increasing in price;
    we return the cap (no interior optimum), which is the honest answer.
    """
    e = abs(elasticity)
    if e <= 1.0 + 1e-9:
        return float(price_cap)
    return float(mc * e / (e - 1.0))


def hierarchical_shrinkage(
    seg_estimates: np.ndarray,
    seg_counts: np.ndarray,
    seg_var: np.ndarray | None = None,
) -> np.ndarray:
    """Partial pooling: shrink each segment estimate toward the pooled mean.

    Shrinkage weight grows for thin segments (small counts). Uses a simple
    James-Stein-style precision weighting.
    """
    pooled = np.average(seg_estimates, weights=seg_counts)
    if seg_var is None:
        seg_var = np.ones_like(seg_estimates)
    # more data => tighter => less shrinkage
    tau2 = np.var(seg_estimates) + 1e-9
    shrunk = np.empty_like(seg_estimates)
    for i in range(len(seg_estimates)):
        se2 = seg_var[i] / max(seg_counts[i], 1)
        w = tau2 / (tau2 + se2)
        shrunk[i] = w * seg_estimates[i] + (1 - w) * pooled
    return shrunk
