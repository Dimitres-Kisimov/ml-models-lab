"""Train the text classifier; report macro-F1 (not accuracy) + confusion matrix."""

from __future__ import annotations

import sys

import numpy as np

from mllab import metrics, synth

SEED = 5
N_BUCKETS = 2**17


def stratified_split(labels: np.ndarray, frac: float, seed: int):
    rng = np.random.default_rng(seed)
    train_idx, test_idx = [], []
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        rng.shuffle(idx)
        cut = int(frac * len(idx))
        train_idx.extend(idx[:cut])
        test_idx.extend(idx[cut:])
    return np.array(train_idx), np.array(test_idx)


def run(seed: int = SEED, make_plot: bool = True) -> dict:
    from mllab.sku_text_classifier.model import predict, train_classifier

    df = synth.make_sku_text(seed=seed)
    texts = df["text"].tolist()
    labels = df["label"].to_numpy()
    class_names = [c for c in df.sort_values("label")["label_name"].unique()]
    n_classes = len(class_names)

    tr, te = stratified_split(labels, 0.75, seed)
    tr_texts = [texts[i] for i in tr]
    te_texts = [texts[i] for i in te]

    model = train_classifier(
        tr_texts, labels[tr], n_classes, n_buckets=N_BUCKETS, epochs=60, seed=seed
    )
    pred = predict(model, te_texts, N_BUCKETS)
    y_true = labels[te]

    macro_f1 = metrics.macro_f1(y_true, pred, n_classes)
    acc = float((pred == y_true).mean())
    # baseline: majority class predictor
    majority = np.bincount(labels[tr]).argmax()
    base_pred = np.full_like(y_true, majority)
    base_f1 = metrics.macro_f1(y_true, base_pred, n_classes)

    out = {
        "macro_f1": macro_f1,
        "accuracy": acc,
        "macro_f1_majority": base_f1,
        "n_classes": n_classes,
        "n_test": int(len(te)),
    }
    print("=== sku-text-classifier ===")
    print("[data] synthetic, seed", seed, f"| {n_classes} categories, stratified split")
    print(f"[metric] macro-F1 (primary)   : {macro_f1:.4f}")
    print(f"[baseline] macro-F1 majority  : {base_f1:.4f}")
    print(f"[reference] accuracy          : {acc:.4f}")
    print(f"[result] beats majority baseline: {'YES' if macro_f1 > base_f1 else 'NO'}")

    if make_plot:
        _plot(y_true, pred, class_names, macro_f1, seed)
    return out


def _plot(y_true, pred, class_names, macro_f1, seed):
    import os

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(class_names)
    cm = metrics.confusion_matrix(y_true, pred, n)
    cm_norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(class_names, rotation=90, fontsize=7)
    ax.set_yticklabels(class_names, fontsize=7)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(f"Confusion matrix (row-normalised)\nmacro-F1={macro_f1:.3f} (synthetic, seed {seed})")
    fig.colorbar(im, ax=ax, fraction=0.046)
    here = os.path.dirname(__file__)
    outdir = os.path.join(here, "results")
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, "confusion_matrix.png")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    print("[figure]", path)


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    run()


if __name__ == "__main__":
    main()
