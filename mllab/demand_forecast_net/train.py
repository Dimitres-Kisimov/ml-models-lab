"""Rolling-origin CV for the demand net vs seasonal-naive and Holt-Winters."""

from __future__ import annotations

import sys

import numpy as np

from mllab import synth
from mllab.demand_forecast_net.features import (
    build_dataset,
    holt_winters_forecast,
    seasonal_naive,
)

SEED = 13
TEST_ORIGINS = (39, 40, 41)  # last three months, one-step-ahead rolling origin


def _scaled_errors(y_panel, preds_by_t, denom_abs, denom_sq):
    """Accumulate absolute/squared errors scaled per series (MASE / RMSSE)."""
    abs_scaled, sq_scaled = [], []
    for t, pred in preds_by_t.items():
        actual = y_panel[:, t].astype(float)
        ae = np.abs(actual - pred) / denom_abs
        se = (actual - pred) ** 2 / denom_sq
        abs_scaled.append(ae)
        sq_scaled.append(se)
    mase = float(np.mean(np.concatenate(abs_scaled)))
    rmsse = float(np.sqrt(np.mean(np.concatenate(sq_scaled))))
    return mase, rmsse


def run(seed: int = SEED, make_plot: bool = True) -> dict:
    from mllab.demand_forecast_net.model import predict_mean, train_net

    y, meta = synth.make_demand_panel(seed=seed)
    n_skus, n_months = y.shape
    m = 12

    # per-series scaling denominators from history before the first test origin
    first = TEST_ORIGINS[0]
    denom_abs = np.array(
        [max(np.mean(np.abs(y[s, m:first] - y[s, : first - m])), 1e-6) for s in range(n_skus)]
    )
    denom_sq = np.array(
        [max(np.mean((y[s, m:first] - y[s, : first - m]) ** 2), 1e-6) for s in range(n_skus)]
    )

    net_preds, snaive_preds, hw_preds = {}, {}, {}
    for origin in TEST_ORIGINS:
        # train on all samples with target time < origin (leakage-free)
        Xtr, sku_tr, ytr, ttr = build_dataset(y, max_t=origin - 1)
        mean = Xtr.mean(axis=0)
        std = Xtr.std(axis=0) + 1e-9
        # weight each sample by inverse series scale so training targets the
        # scale-free MASE/RMSSE metric (low-volume series otherwise dominate it)
        series_scale = np.array(
            [max(float(y[s, :origin].mean()), 1.0) for s in range(n_skus)]
        )
        w = 1.0 / series_scale[sku_tr]
        w = w / w.mean()
        model = train_net(
            Xtr, sku_tr, ytr, n_skus, mean, std, epochs=400, lr=0.02, seed=seed, sample_weight=w
        )

        # build test features for exactly month = origin, all SKUs
        Xte, sku_te, _, tte = build_dataset(y, max_t=origin)
        sel = tte == origin
        pred = predict_mean(model, Xte[sel], sku_te[sel], mean, std)
        order = np.argsort(sku_te[sel])
        net_preds[origin] = pred[order]
        snaive_preds[origin] = seasonal_naive(y, origin, m)
        hw_preds[origin] = np.array(
            [holt_winters_forecast(y[s], origin, m) for s in range(n_skus)]
        )

    net_mase, net_rmsse = _scaled_errors(y, net_preds, denom_abs, denom_sq)
    sn_mase, sn_rmsse = _scaled_errors(y, snaive_preds, denom_abs, denom_sq)
    hw_mase, hw_rmsse = _scaled_errors(y, hw_preds, denom_abs, denom_sq)

    out = {
        "net_mase": net_mase,
        "net_rmsse": net_rmsse,
        "seasonal_naive_mase": sn_mase,
        "seasonal_naive_rmsse": sn_rmsse,
        "holt_winters_mase": hw_mase,
        "holt_winters_rmsse": hw_rmsse,
        "n_skus": n_skus,
    }
    beat = net_mase < sn_mase and net_mase < hw_mase
    out["beats_both_baselines"] = bool(beat)

    print("=== demand-forecast-net ===")
    print("[data] synthetic, seed", seed, f"| {n_skus} SKUs x {n_months} months, rolling-origin CV")
    print("[metric]              MASE     RMSSE")
    print(f"[global NB MLP]       {net_mase:.4f}   {net_rmsse:.4f}")
    print(f"[seasonal-naive]      {sn_mase:.4f}   {sn_rmsse:.4f}")
    print(f"[Holt-Winters]        {hw_mase:.4f}   {hw_rmsse:.4f}")
    print(f"[result] net beats seasonal-naive AND Holt-Winters (MASE): {'YES' if beat else 'NO'}")

    if make_plot:
        _plot(y, net_preds, snaive_preds, meta, out, seed)
    return out


def _plot(y, net_preds, snaive_preds, meta, out, seed):
    import os

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # pick a representative seasonal SKU
    seasonal = np.where(~meta["is_intermittent"].to_numpy())[0]
    s = int(seasonal[0])
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(y.shape[1]), y[s], "o-", color="0.5", label="actual demand", ms=3)
    origins = sorted(net_preds.keys())
    ax.plot(origins, [net_preds[o][s] for o in origins], "D", color="C0", label="global NB MLP")
    ax.plot(origins, [snaive_preds[o][s] for o in origins], "s", color="C1", label="seasonal-naive")
    ax.set_xlabel("month")
    ax.set_ylabel("units")
    ax.set_title(
        f"Demand forecast, SKU {s} (synthetic, seed {seed})\n"
        f"MASE net={out['net_mase']:.3f} vs snaive={out['seasonal_naive_mase']:.3f} "
        f"vs HW={out['holt_winters_mase']:.3f}"
    )
    ax.legend()
    ax.grid(alpha=0.3)
    here = os.path.dirname(__file__)
    outdir = os.path.join(here, "results")
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, "forecast_fit.png")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    print("[figure]", path)


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    run()


if __name__ == "__main__":
    main()
