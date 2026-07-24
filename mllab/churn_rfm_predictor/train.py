"""Train the churn predictor: time-based split, calibrate, report PR-AUC first."""

from __future__ import annotations

import sys

import numpy as np

from mllab import metrics, synth
from mllab.churn_rfm_predictor.model import (
    LogisticRegressionScratch,
    PlattCalibrator,
    build_features,
    standardize,
)

SEED = 7


def time_based_split(df, cutoff: int = 16):
    """Train on earlier cohorts, test on later ones (leakage-free)."""
    train = df[df["cohort_month"] < cutoff].reset_index(drop=True)
    test = df[df["cohort_month"] >= cutoff].reset_index(drop=True)
    return train, test


def run(seed: int = SEED, make_plot: bool = True) -> dict:
    df = synth.make_customers(seed=seed)
    train_df, test_df = time_based_split(df)

    Xtr, ytr, mean_std = build_features(train_df)
    Xte = standardize(test_df, mean_std)
    yte = test_df["churn"].to_numpy(dtype=float)

    # inner split of TRAIN to fit Platt calibration on held-out logits
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(Xtr))
    cut = int(0.8 * len(idx))
    fit_idx, cal_idx = idx[:cut], idx[cut:]

    pos_rate = ytr[fit_idx].mean()
    class_weight = (1.0 - pos_rate) / max(pos_rate, 1e-6)

    model = LogisticRegressionScratch(
        l2=0.01, lr=0.5, n_iter=3000, class_weight=class_weight
    )
    model.fit(Xtr[fit_idx], ytr[fit_idx])

    cal = PlattCalibrator().fit(model.decision_function(Xtr[cal_idx]), ytr[cal_idx])

    logits_te = model.decision_function(Xte)
    p_uncal = model.predict_proba(Xte)
    p_cal = cal.transform(logits_te)

    # baselines: recency-only score (higher recency => more churn) and prevalence
    recency_score = test_df["recency_days"].to_numpy(dtype=float)
    base_rate = ytr.mean()

    out = {
        "pr_auc": metrics.pr_auc(yte, p_cal),
        "roc_auc": metrics.roc_auc(yte, p_cal),
        "brier_uncal": metrics.brier_score(yte, p_uncal),
        "brier_cal": metrics.brier_score(yte, p_cal),
        "ece_uncal": metrics.expected_calibration_error(yte, p_uncal),
        "ece_cal": metrics.expected_calibration_error(yte, p_cal),
        "pr_auc_recency": metrics.pr_auc(yte, recency_score),
        "pr_auc_prevalence": base_rate,
        "test_base_rate": float(yte.mean()),
        "n_train": int(len(Xtr)),
        "n_test": int(len(Xte)),
    }

    print("=== churn-rfm-predictor ===")
    print("[data] synthetic, seed", seed, "| time-based cohort split")
    print(f"[train] n={out['n_train']}  [test] n={out['n_test']}  churn rate(test)={out['test_base_rate']:.3f}")
    print(f"[metric] PR-AUC (primary)   : {out['pr_auc']:.4f}")
    print(f"[baseline] PR-AUC recency   : {out['pr_auc_recency']:.4f}")
    print(f"[baseline] PR-AUC prevalence: {out['pr_auc_prevalence']:.4f}")
    print(f"[metric] ROC-AUC            : {out['roc_auc']:.4f}")
    print(f"[metric] Brier  uncal/cal   : {out['brier_uncal']:.4f} / {out['brier_cal']:.4f}")
    print(f"[metric] ECE    uncal/cal   : {out['ece_uncal']:.4f} / {out['ece_cal']:.4f}")
    beat = out["pr_auc"] > max(out["pr_auc_recency"], out["pr_auc_prevalence"])
    print(f"[result] beats recency + prevalence baselines: {'YES' if beat else 'NO'}")

    if make_plot:
        _plot(yte, p_uncal, p_cal, out, seed)
    return out


def _plot(yte, p_uncal, p_cal, out, seed):
    import os

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    conf_u, acc_u, _ = metrics.reliability_curve(yte, p_uncal)
    conf_c, acc_c, _ = metrics.reliability_curve(yte, p_cal)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k--", label="perfectly calibrated")
    m = ~np.isnan(conf_u)
    ax.plot(conf_u[m], acc_u[m], "o-", label="uncalibrated")
    m = ~np.isnan(conf_c)
    ax.plot(conf_c[m], acc_c[m], "s-", label="Platt-calibrated")
    ax.set_xlabel("predicted probability")
    ax.set_ylabel("observed churn frequency")
    ax.set_title(
        f"Churn reliability (synthetic, seed {seed})\n"
        f"PR-AUC={out['pr_auc']:.3f}  ECE {out['ece_uncal']:.3f}->{out['ece_cal']:.3f}"
    )
    ax.legend()
    ax.grid(alpha=0.3)
    here = os.path.dirname(__file__)
    outdir = os.path.join(here, "results")
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, "reliability_curve.png")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    print("[figure]", path)


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    run()


if __name__ == "__main__":
    main()
