"""Feature engineering + classical baselines for demand forecasting (numpy).

Features per (SKU, month t): lags {1,2,3,12,13}, rolling means {3,6,12},
6 Fourier month features, and causal Croston smoothed size/interval states.
All features use only information available strictly before t.
"""

from __future__ import annotations

import numpy as np

LAGS = (1, 2, 3, 12, 13)
ROLL = (3, 6, 12)
MIN_T = 13  # need lag 13


def croston_states(series: np.ndarray, alpha: float = 0.1) -> tuple[np.ndarray, np.ndarray]:
    """Causal Croston smoothed demand size ``z`` and interval ``p`` at each t.

    Returns arrays aligned to ``series`` giving, for each t, the state estimated
    from data strictly before t (so they are safe as features for target t).
    """
    n = len(series)
    z = np.zeros(n)  # smoothed nonzero size
    p = np.zeros(n)  # smoothed interval between demands
    z_cur, p_cur = 0.0, 1.0
    q = 1  # periods since last demand
    seen = False
    for t in range(n):
        z[t] = z_cur
        p[t] = p_cur
        yt = series[t]
        if yt > 0:
            if not seen:
                z_cur = float(yt)
                p_cur = float(q)
                seen = True
            else:
                z_cur = alpha * yt + (1 - alpha) * z_cur
                p_cur = alpha * q + (1 - alpha) * p_cur
            q = 1
        else:
            q += 1
    return z, p


def _fourier(month: int) -> list[float]:
    feats = []
    for k in (1, 2, 3):
        ang = 2.0 * np.pi * k * month / 12.0
        feats.extend([np.sin(ang), np.cos(ang)])
    return feats


FEATURE_NAMES = (
    [f"lag{lag}" for lag in LAGS]
    + [f"roll{w}" for w in ROLL]
    + [f"fourier{i}" for i in range(6)]
    + ["croston_z", "croston_p"]
)


def build_dataset(y: np.ndarray, max_t: int | None = None):
    """Build the global training table across all SKUs.

    Returns ``(X, sku_idx, target, time)`` for every (SKU, t) with t >= MIN_T
    and (if given) t <= max_t.
    """
    n_skus, n_months = y.shape
    if max_t is None:
        max_t = n_months - 1
    cros = [croston_states(y[s]) for s in range(n_skus)]
    X, sku_idx, target, time = [], [], [], []
    for s in range(n_skus):
        z, p = cros[s]
        for t in range(MIN_T, max_t + 1):
            feats = [float(y[s, t - lag]) for lag in LAGS]
            feats += [float(y[s, t - w : t].mean()) for w in ROLL]
            feats += _fourier(t % 12)
            feats += [float(z[t]), float(p[t])]
            X.append(feats)
            sku_idx.append(s)
            target.append(float(y[s, t]))
            time.append(t)
    return (
        np.asarray(X, dtype=np.float64),
        np.asarray(sku_idx, dtype=np.int64),
        np.asarray(target, dtype=np.float64),
        np.asarray(time, dtype=np.int64),
    )


def seasonal_naive(y: np.ndarray, t: int, m: int = 12) -> np.ndarray:
    """Seasonal-naive forecast for month t: value from t-m."""
    return y[:, t - m].astype(float)


def holt_winters_forecast(
    series: np.ndarray,
    t: int,
    m: int = 12,
    alpha: float = 0.3,
    beta: float = 0.05,
    gamma: float = 0.3,
) -> float:
    """One-step additive Holt-Winters forecast for month t from history < t."""
    hist = series[:t].astype(float)
    if len(hist) < 2 * m:
        # not enough for seasonality: fall back to last value
        return float(hist[-1]) if len(hist) else 0.0
    level = hist[:m].mean()
    trend = (hist[m : 2 * m].mean() - hist[:m].mean()) / m
    seasonal = hist[:m] - level
    for i in range(m, len(hist)):
        val = hist[i]
        s_idx = i % m
        prev_level = level
        level = alpha * (val - seasonal[s_idx]) + (1 - alpha) * (level + trend)
        trend = beta * (level - prev_level) + (1 - beta) * trend
        seasonal[s_idx] = gamma * (val - level) + (1 - gamma) * seasonal[s_idx]
    return float(level + trend + seasonal[t % m])
