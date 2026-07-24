"""Seeded synthetic data generators shared by all five models.

Every generator takes an explicit ``seed`` and returns deterministic output:
the same seed produces byte-identical arrays. All data here is synthetic - no
real customer, order, or catalogue data is used anywhere in this lab.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _nb_draw(rng: np.random.Generator, mu: np.ndarray, r: float) -> np.ndarray:
    """Negative-binomial draw parameterised by mean ``mu`` and dispersion ``r``.

    Variance = mu + mu^2 / r (overdispersed relative to Poisson).
    """
    mu = np.asarray(mu, dtype=float)
    mu = np.clip(mu, 1e-6, None)
    p = r / (r + mu)
    return rng.negative_binomial(r, p)


# ---------------------------------------------------------------------------
# 1. demand panel
# ---------------------------------------------------------------------------


def make_demand_panel(
    seed: int = 0,
    n_skus: int = 200,
    n_months: int = 42,
    intermittent_frac: float = 0.25,
) -> tuple[np.ndarray, pd.DataFrame]:
    """~200 SKUs x 42 months of monthly demand, mix of seasonal + intermittent.

    Returns
    -------
    y : int array (n_skus, n_months)
        Monthly unit demand.
    meta : DataFrame
        One row per SKU with ``sku_id``, ``is_intermittent``, ``seasonality``.
    """
    rng = np.random.default_rng(seed)
    months = np.arange(n_months)
    y = np.zeros((n_skus, n_months), dtype=np.int64)
    is_interm = rng.random(n_skus) < intermittent_frac

    levels = np.zeros(n_skus)
    amps = np.zeros(n_skus)
    for i in range(n_skus):
        if is_interm[i]:
            # sparse, low-volume intermittent series
            base = rng.uniform(2.0, 12.0)
            p_occur = rng.uniform(0.15, 0.45)
            occur = rng.random(n_months) < p_occur
            sizes = _nb_draw(rng, np.full(n_months, base), r=3.0)
            y[i] = np.where(occur, sizes, 0)
            levels[i] = base
            amps[i] = 0.0
        else:
            level = rng.uniform(20.0, 200.0)
            amp = rng.uniform(0.15, 0.6)
            phase = rng.uniform(0.0, 12.0)
            trend = rng.uniform(-0.4, 0.9) / n_months
            season = 1.0 + amp * np.sin(2.0 * np.pi * (months + phase) / 12.0)
            mu = level * season * (1.0 + trend * months)
            y[i] = _nb_draw(rng, mu, r=6.0)
            levels[i] = level
            amps[i] = amp

    meta = pd.DataFrame(
        {
            "sku_id": np.arange(n_skus),
            "is_intermittent": is_interm,
            "level": levels,
            "amp": amps,
        }
    )
    return y, meta


# ---------------------------------------------------------------------------
# 2. SKU text
# ---------------------------------------------------------------------------

_CATEGORIES = [
    "hex_bolt",
    "hex_nut",
    "flat_washer",
    "wood_screw",
    "steel_pipe",
    "ball_valve",
    "power_cable",
    "ball_bearing",
    "rubber_gasket",
    "steel_bracket",
    "epoxy_adhesive",
    "spray_paint",
]

_CAT_CORE = {
    "hex_bolt": ["bolt", "hex bolt", "cap screw", "hex head bolt"],
    "hex_nut": ["nut", "hex nut", "lock nut", "hex lock nut"],
    "flat_washer": ["washer", "flat washer", "plain washer", "spring washer"],
    "wood_screw": ["wood screw", "screw", "chipboard screw", "deck screw"],
    "steel_pipe": ["pipe", "steel pipe", "tube", "seamless pipe"],
    "ball_valve": ["valve", "ball valve", "shutoff valve", "gate valve"],
    "power_cable": ["cable", "power cable", "wire", "cord"],
    "ball_bearing": ["bearing", "ball bearing", "roller bearing", "deep groove bearing"],
    "rubber_gasket": ["gasket", "rubber gasket", "seal", "o-ring"],
    "steel_bracket": ["bracket", "angle bracket", "mounting bracket", "l bracket"],
    "epoxy_adhesive": ["adhesive", "epoxy", "glue", "epoxy adhesive"],
    "spray_paint": ["paint", "spray paint", "primer", "aerosol paint"],
}

# deliberately shared across categories to force overlap
_MATERIALS = ["steel", "stainless", "stainless steel", "galvanized", "brass", "zinc", "carbon"]
_BRANDS = ["ACME", "BoltPro", "TitanFix", "EuroFast", "MRO", "IndustCo"]


def make_sku_text(
    seed: int = 0,
    n_per_class: int = 300,
    classes: list[str] | None = None,
) -> pd.DataFrame:
    """~12 categories x ~300 templated part descriptions with vocab overlap.

    Returns a DataFrame with ``text``, ``label`` (int), ``label_name``.
    """
    rng = np.random.default_rng(seed)
    classes = classes or _CATEGORIES
    rows: list[tuple[str, int, str]] = []
    for ci, cat in enumerate(classes):
        cores = _CAT_CORE[cat]
        for _ in range(n_per_class):
            core = cores[rng.integers(len(cores))]
            mat = _MATERIALS[rng.integers(len(_MATERIALS))]
            brand = _BRANDS[rng.integers(len(_BRANDS))]
            metric = rng.integers(3, 24)
            length = rng.integers(6, 120)
            parts = []
            # word order and optional tokens vary for robustness
            if rng.random() < 0.6:
                parts.append(brand)
            if rng.random() < 0.7:
                parts.append(mat)
            parts.append(core)
            if rng.random() < 0.65:
                parts.append(f"M{metric}x{length}")
            if rng.random() < 0.35:
                parts.append(f"{metric}mm")
            if rng.random() < 0.25:
                parts.append(f"PN{rng.integers(1000, 9999)}")
            text = " ".join(parts)
            rows.append((text, ci, cat))
    df = pd.DataFrame(rows, columns=["text", "label", "label_name"])
    # shuffle deterministically
    perm = rng.permutation(len(df))
    return df.iloc[perm].reset_index(drop=True)


# ---------------------------------------------------------------------------
# 3. orders / anomalies
# ---------------------------------------------------------------------------

_ORDER_FEATURES = [
    "order_value",
    "quantity",
    "discount_pct",
    "lead_time_days",
    "unit_price",
    "num_line_items",
    "customer_tenure",
]


def make_orders(
    seed: int = 0,
    n: int = 5000,
    anomaly_frac: float = 0.03,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """~5000 orders x 7 correlated features with ~3% injected anomalies.

    Three anomaly kinds: extreme univariate, broken correlation, impossible combo.

    Returns ``(X, y, feature_names)`` where ``y`` is 1 for anomalies.
    """
    rng = np.random.default_rng(seed)
    d = len(_ORDER_FEATURES)

    # correlated normal core via a random low-rank factor structure
    k = 3
    load = rng.normal(size=(d, k)) * 0.8
    factors = rng.normal(size=(n, k))
    noise = rng.normal(size=(n, d)) * 0.4
    z = factors @ load.T + noise  # standardized latent space
    # map to interpretable positive scales
    means = np.array([500.0, 20.0, 0.10, 7.0, 25.0, 4.0, 36.0])
    scales = np.array([180.0, 8.0, 0.05, 3.0, 9.0, 2.0, 20.0])
    X = means + scales * z
    X[:, 2] = np.clip(X[:, 2], 0.0, 0.6)  # discount_pct in [0, .6]
    X[:, 1] = np.clip(X[:, 1], 1.0, None)
    X[:, 3] = np.clip(X[:, 3], 1.0, None)
    X[:, 4] = np.clip(X[:, 4], 1.0, None)
    X[:, 5] = np.clip(np.round(X[:, 5]), 1.0, None)
    X[:, 6] = np.clip(X[:, 6], 0.0, None)

    y = np.zeros(n, dtype=np.int64)
    n_anom = int(round(n * anomaly_frac))
    idx = rng.choice(n, size=n_anom, replace=False)
    kinds = rng.integers(0, 3, size=n_anom)
    for j, i in enumerate(idx):
        y[i] = 1
        if kinds[j] == 0:
            # extreme univariate: blow up one feature
            f = rng.integers(d)
            X[i, f] = means[f] + scales[f] * rng.uniform(6.0, 10.0) * rng.choice([-1.0, 1.0])
            X[i, 2] = np.clip(X[i, 2], 0.0, 0.6)
        elif kinds[j] == 1:
            # broken correlation: redraw two coupled features independently
            X[i, 0] = means[0] + scales[0] * rng.uniform(2.5, 4.0)  # big value
            X[i, 1] = np.clip(means[1] - scales[1] * rng.uniform(1.5, 2.0), 1.0, None)  # low qty
        else:
            # impossible combo: huge discount on a small, cheap order
            X[i, 2] = rng.uniform(0.55, 0.6)  # near-max discount
            X[i, 0] = np.clip(means[0] - scales[0] * 2.2, 1.0, None)  # tiny value
            X[i, 4] = np.clip(means[4] + scales[4] * 3.5, 1.0, None)  # high unit price
    return X, y, list(_ORDER_FEATURES)


# ---------------------------------------------------------------------------
# 4. churn customers
# ---------------------------------------------------------------------------

_CHURN_FEATURES = [
    "recency_days",
    "frequency",
    "monetary",
    "tenure_days",
    "freq_slope",  # declining-order-frequency slope (engineered)
    "avg_basket",
]


def make_customers(
    seed: int = 0,
    n: int = 4000,
    churn_rate: float = 0.15,
) -> pd.DataFrame:
    """~4000 customers with RFM + engineered slope features and a churn label.

    A ``cohort_month`` column supports a time-based (leakage-free) split.
    Returns a DataFrame of features + ``churn`` + ``cohort_month``.
    """
    rng = np.random.default_rng(seed)
    recency = rng.gamma(2.0, 30.0, size=n)  # days since last order
    frequency = rng.poisson(6.0, size=n) + 1
    monetary = rng.gamma(2.0, 400.0, size=n)
    tenure = rng.uniform(60.0, 1200.0, size=n)
    # declining-order-frequency slope: negative => declining engagement
    freq_slope = rng.normal(-0.02, 0.05, size=n)
    avg_basket = monetary / frequency
    cohort_month = rng.integers(0, 24, size=n)  # signup month, for time split

    # standardize for the latent logistic
    def z(a):
        return (a - a.mean()) / (a.std() + 1e-9)

    latent = (
        0.9 * z(recency)
        - 0.5 * z(frequency)
        - 0.3 * z(monetary)
        - 0.6 * z(tenure)
        - 1.1 * z(freq_slope)  # steeper decline => more churn
        + 0.8 * z(recency) * z(-freq_slope)  # R x slope interaction
    )
    # calibrate intercept so mean churn ~ churn_rate
    from scipy.optimize import brentq

    def mean_p(b0):
        return float(np.mean(1.0 / (1.0 + np.exp(-(latent + b0)))))

    b0 = brentq(lambda b: mean_p(b) - churn_rate, -20.0, 20.0)
    p = 1.0 / (1.0 + np.exp(-(latent + b0)))
    churn = (rng.random(n) < p).astype(np.int64)

    return pd.DataFrame(
        {
            "recency_days": recency,
            "frequency": frequency.astype(float),
            "monetary": monetary,
            "tenure_days": tenure,
            "freq_slope": freq_slope,
            "avg_basket": avg_basket,
            "cohort_month": cohort_month,
            "churn": churn,
        }
    )


# ---------------------------------------------------------------------------
# 5. price / elasticity transactions
# ---------------------------------------------------------------------------


def make_transactions(
    seed: int = 0,
    n_products: int = 60,
    min_tx: int = 50,
    max_tx: int = 300,
    n_segments: int = 6,
) -> pd.DataFrame:
    """Per-product transactions with a known true elasticity and a confounder.

    ``ln Q = a - |elasticity| * ln P + demand_coef * Z + noise`` with price set
    partly from the latent demand shock ``Z`` (endogeneity). ``Z`` is observable
    and can be used as a control to recover the true elasticity.

    Returns a DataFrame with product/segment ids, log_price, log_qty, the
    control ``z``, ``true_elasticity`` and ``mc``.
    """
    rng = np.random.default_rng(seed)
    # segment-level pooled elasticity (all elastic, |e|>1)
    seg_elast = rng.uniform(-2.6, -1.3, size=n_segments)
    rows = []
    for pid in range(n_products):
        seg = pid % n_segments
        # thin segments: some products get very few transactions
        n_tx = int(rng.integers(min_tx, max_tx + 1))
        if rng.random() < 0.25:
            n_tx = int(rng.integers(min_tx, min_tx + 20))
        # product elasticity drawn around its segment mean (partial pooling target)
        true_e = seg_elast[seg] + rng.normal(0.0, 0.15)
        a = rng.uniform(8.0, 11.0)  # demand intercept
        mc = rng.uniform(5.0, 40.0)  # marginal cost
        demand_coef = rng.uniform(0.6, 1.2)
        gamma = rng.uniform(0.4, 0.8)  # how strongly price tracks the shock
        z_shock = rng.normal(0.0, 1.0, size=n_tx)  # latent demand confounder
        base_lnp = np.log(mc) + rng.uniform(0.4, 0.9)
        ln_p = base_lnp + gamma * z_shock + rng.normal(0.0, 0.12, size=n_tx)
        ln_q = a + true_e * ln_p + demand_coef * z_shock + rng.normal(0.0, 0.15, size=n_tx)
        for t in range(n_tx):
            rows.append((pid, seg, ln_p[t], ln_q[t], z_shock[t], true_e, mc, a, demand_coef))
    return pd.DataFrame(
        rows,
        columns=[
            "product_id",
            "segment",
            "log_price",
            "log_qty",
            "z",
            "true_elasticity",
            "mc",
            "true_a",
            "true_demand_coef",
        ],
    )
