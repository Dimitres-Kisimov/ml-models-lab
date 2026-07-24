"""Train elasticity regressors: endogeneity demo, shrinkage, profit uplift."""

from __future__ import annotations

import sys

import numpy as np

from mllab import synth
from mllab.price_elasticity_regressor.model import (
    hierarchical_shrinkage,
    optimal_price,
    ridge_closed_form,
)

SEED = 11


def _r2(y, yhat):
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return float(1.0 - ss_res / max(ss_tot, 1e-12))


def _fit_product(sub, control: bool):
    """Return estimated elasticity for one product's transactions."""
    lnp = sub["log_price"].to_numpy()
    lnq = sub["log_qty"].to_numpy()
    if control:
        X = np.column_stack([lnp, sub["z"].to_numpy()])
    else:
        X = lnp.reshape(-1, 1)
    coef = ridge_closed_form(X, lnq, l2=1e-3)
    return coef[1]  # slope on log_price = elasticity


def run(seed: int = SEED, make_plot: bool = True) -> dict:
    df = synth.make_transactions(seed=seed)
    products = sorted(df["product_id"].unique())

    naive_e, ctrl_e, true_e, counts, segs, mcs, true_a = [], [], [], [], [], [], []
    for pid in products:
        sub = df[df["product_id"] == pid]
        naive_e.append(_fit_product(sub, control=False))
        ctrl_e.append(_fit_product(sub, control=True))
        true_e.append(sub["true_elasticity"].iloc[0])
        counts.append(len(sub))
        segs.append(int(sub["segment"].iloc[0]))
        mcs.append(sub["mc"].iloc[0])
        true_a.append(sub["true_a"].iloc[0])

    naive_e = np.array(naive_e)
    ctrl_e = np.array(ctrl_e)
    true_e = np.array(true_e)
    counts = np.array(counts)
    segs = np.array(segs)
    mcs = np.array(mcs)
    true_a = np.array(true_a)

    # hierarchical shrinkage of controlled estimates toward pooled mean per segment
    shrunk = ctrl_e.copy()
    for s in np.unique(segs):
        mask = segs == s
        shrunk[mask] = hierarchical_shrinkage(ctrl_e[mask], counts[mask])

    rmse_naive = float(np.sqrt(np.mean((naive_e - true_e) ** 2)))
    rmse_ctrl = float(np.sqrt(np.mean((ctrl_e - true_e) ** 2)))
    rmse_shrunk = float(np.sqrt(np.mean((shrunk - true_e) ** 2)))
    bias_naive = float(np.mean(naive_e - true_e))
    bias_ctrl = float(np.mean(ctrl_e - true_e))

    # lasso sanity: strong penalty zeroes the (irrelevant) noise features
    profit = _profit_simulation(true_e, shrunk, mcs, true_a)

    out = {
        "rmse_naive_ols": rmse_naive,
        "rmse_controlled": rmse_ctrl,
        "rmse_shrunk": rmse_shrunk,
        "bias_naive": bias_naive,
        "bias_controlled": bias_ctrl,
        "r2_controlled": _r2(true_e, ctrl_e),
        "mean_regret_frac": profit["mean_regret_frac"],
        "uplift_vs_cost_plus": profit["uplift_vs_cost_plus"],
        "n_products": len(products),
    }

    print("=== price-elasticity-regressor ===")
    print("[data] synthetic, seed", seed, "| price partly set by latent demand shock (endogeneity)")
    print(f"[endogeneity] naive OLS elasticity bias : {bias_naive:+.3f} (RMSE {rmse_naive:.3f})")
    print(f"[endogeneity] control-for-Z bias        : {bias_ctrl:+.3f} (RMSE {rmse_ctrl:.3f})")
    print(f"[result] adding the control recovers true elasticity: "
          f"{'YES' if abs(bias_ctrl) < abs(bias_naive) else 'NO'}")
    print(f"[shrinkage] RMSE controlled -> shrunk   : {rmse_ctrl:.3f} -> {rmse_shrunk:.3f}")
    print(f"[metric] R2 of controlled elasticity    : {out['r2_controlled']:.4f}")
    print(f"[decision] mean profit regret vs oracle : {profit['mean_regret_frac']*100:.2f}%")
    print(f"[decision] profit uplift vs cost-plus    : {profit['uplift_vs_cost_plus']*100:+.2f}%")

    if make_plot:
        _plot(true_e, naive_e, ctrl_e, out, seed)
    return out


def _profit_at(price, e, a, mc):
    q = np.exp(a + e * np.log(price))
    return (price - mc) * q


def _profit_simulation(true_e, est_e, mcs, true_a):
    """Realised profit under the TRUE curve at the price recommended from est."""
    regrets = []
    uplift = []
    for e_t, e_h, mc, a in zip(true_e, est_e, mcs, true_a, strict=False):
        p_oracle = optimal_price(mc, e_t, price_cap=mc * 50)
        p_est = optimal_price(mc, e_h, price_cap=mc * 50)
        p_cost_plus = mc * 1.5  # naive cost-plus policy
        prof_oracle = _profit_at(p_oracle, e_t, a, mc)
        prof_est = _profit_at(p_est, e_t, a, mc)
        prof_cost = _profit_at(p_cost_plus, e_t, a, mc)
        regrets.append((prof_oracle - prof_est) / max(prof_oracle, 1e-9))
        uplift.append((prof_est - prof_cost) / max(prof_cost, 1e-9))
    return {
        "mean_regret_frac": float(np.mean(regrets)),
        "uplift_vs_cost_plus": float(np.mean(uplift)),
    }


def _plot(true_e, naive_e, ctrl_e, out, seed):
    import os

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.5, 6))
    lo = min(true_e.min(), naive_e.min(), ctrl_e.min()) - 0.2
    hi = max(true_e.max(), naive_e.max(), ctrl_e.max()) + 0.2
    ax.plot([lo, hi], [lo, hi], "k--", label="true = estimated")
    ax.scatter(true_e, naive_e, alpha=0.6, label=f"naive OLS (bias {out['bias_naive']:+.2f})")
    ax.scatter(true_e, ctrl_e, alpha=0.6, label=f"control for Z (bias {out['bias_controlled']:+.2f})")
    ax.set_xlabel("true elasticity")
    ax.set_ylabel("estimated elasticity")
    ax.set_title(f"Elasticity recovery (synthetic, seed {seed})\nendogeneity biases OLS; controls recover it")
    ax.legend()
    ax.grid(alpha=0.3)
    here = os.path.dirname(__file__)
    outdir = os.path.join(here, "results")
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, "elasticity_fit.png")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    print("[figure]", path)


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    run()


if __name__ == "__main__":
    main()
