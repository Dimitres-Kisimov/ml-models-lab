"""Train AE + PCA anomaly detectors and report the honest comparison."""

from __future__ import annotations

import sys

import numpy as np

from mllab import metrics, synth
from mllab.order_anomaly_ae.model import PCAReconstructor, Standardizer

SEED = 3


def run(seed: int = SEED, make_plot: bool = True) -> dict:
    X, y, feat_names = synth.make_orders(seed=seed)
    rng = np.random.default_rng(seed)

    # normal-only training split; evaluate on a held-out mix
    normal_idx = np.where(y == 0)[0]
    rng.shuffle(normal_idx)
    n_train = int(0.6 * len(normal_idx))
    train_norm = normal_idx[:n_train]
    val_norm = normal_idx[n_train : n_train + int(0.15 * len(normal_idx))]
    used = set(train_norm.tolist()) | set(val_norm.tolist())
    eval_idx = np.array([i for i in range(len(y)) if i not in used])

    sc = Standardizer().fit(X[train_norm])
    Xtr = sc.transform(X[train_norm])
    Xval = sc.transform(X[val_norm])
    Xev = sc.transform(X[eval_idx])
    yev = y[eval_idx]

    # PCA baseline (numpy)
    pca = PCAReconstructor(k=4).fit(Xtr)
    pca_ev = pca.score(Xev)

    # AE (torch); import lazily so numpy-only environments still import this file
    from mllab.order_anomaly_ae.ae import ae_score, train_ae

    ae = train_ae(Xtr, seed=seed, epochs=300, denoise_std=0.1)
    ae_ev = ae_score(ae, Xev)

    contamination = float(y.mean())
    # threshold from a clean validation split (FPR budget = contamination)
    thr = np.quantile(pca.score(Xval), 1.0 - contamination)

    k = int(yev.sum())
    out = {
        "contamination": contamination,
        "pca_roc_auc": metrics.roc_auc(yev, pca_ev),
        "pca_pr_auc": metrics.pr_auc(yev, pca_ev),
        "pca_p_at_k": metrics.precision_at_k(yev, pca_ev, k),
        "ae_roc_auc": metrics.roc_auc(yev, ae_ev),
        "ae_pr_auc": metrics.pr_auc(yev, ae_ev),
        "ae_p_at_k": metrics.precision_at_k(yev, ae_ev, k),
        "threshold": float(thr),
        "n_eval": int(len(yev)),
        "n_anom_eval": k,
    }
    winner = "AE" if out["ae_pr_auc"] > out["pca_pr_auc"] else "PCA"
    out["winner_by_pr_auc"] = winner

    print("=== order-anomaly-ae ===")
    print("[data] synthetic, seed", seed, f"| assumed contamination = {contamination:.3f}")
    print(f"[eval] n={out['n_eval']}  anomalies={out['n_anom_eval']}")
    print("[metric]           ROC-AUC   PR-AUC   P@k")
    print(f"[PCA baseline]     {out['pca_roc_auc']:.3f}     {out['pca_pr_auc']:.3f}    {out['pca_p_at_k']:.3f}")
    print(f"[Autoencoder]      {out['ae_roc_auc']:.3f}     {out['ae_pr_auc']:.3f}    {out['ae_p_at_k']:.3f}")
    print(f"[HONEST RESULT] higher PR-AUC: {winner}")
    if winner == "PCA":
        print("[recommendation] AE does NOT beat PCA here -> prefer the simpler PCA baseline.")
    else:
        print("[recommendation] AE beats PCA on PR-AUC on this seed; PCA remains the honest baseline.")

    # per-feature explanation for the top flagged anomaly
    per_feat = pca.per_feature_error(Xev)
    top = int(np.argsort(-pca_ev)[0])
    contrib = per_feat[top] / per_feat[top].sum()
    ranked = sorted(zip(feat_names, contrib, strict=False), key=lambda t: -t[1])[:2]
    expl = ", ".join(f"{n} ({w*100:.0f}%)" for n, w in ranked)
    print(f"[explanation] top-flagged order driven by: {expl}")

    if make_plot:
        _plot(yev, pca_ev, ae_ev, out, seed)
    return out


def _plot(yev, pca_ev, ae_ev, out, seed):
    import os

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def pr_points(y, score):
        order = np.argsort(-score)
        ys = y[order]
        tp = np.cumsum(ys == 1)
        fp = np.cumsum(ys == 0)
        prec = tp / np.maximum(tp + fp, 1)
        rec = tp / max((y == 1).sum(), 1)
        return rec, prec

    fig, ax = plt.subplots(figsize=(6.5, 6))
    for name, score, key in [("PCA", pca_ev, "pca_pr_auc"), ("Autoencoder", ae_ev, "ae_pr_auc")]:
        rec, prec = pr_points(yev, score)
        ax.plot(rec, prec, label=f"{name} (PR-AUC={out[key]:.3f})")
    ax.axhline(out["contamination"], color="k", ls="--", label=f"prevalence={out['contamination']:.3f}")
    ax.set_xlabel("recall")
    ax.set_ylabel("precision")
    ax.set_title(f"Anomaly PR curves (synthetic, seed {seed})\nhonest winner: {out['winner_by_pr_auc']}")
    ax.legend()
    ax.grid(alpha=0.3)
    here = os.path.dirname(__file__)
    outdir = os.path.join(here, "results")
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, "pr_curve.png")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    print("[figure]", path)


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    run()


if __name__ == "__main__":
    main()
