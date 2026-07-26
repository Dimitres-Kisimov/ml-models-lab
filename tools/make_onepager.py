"""Build deliverables/ml_models_lab_onepager.pdf - a one-page summary of the lab.

Every number below is copied verbatim from the repo README's measured results
(synthetic, seeded data; see docs/BUSINESS_CASE.md for the honesty framing).
Run:  python tools/make_onepager.py
"""

import os
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "deliverables")
OUT_PDF = os.path.join(OUT_DIR, "ml_models_lab_onepager.pdf")
IMG_DIR = os.path.join(ROOT, "docs", "img")

INK = "#1a1a1a"
INK2 = "#4a4a4a"
INK3 = "#6e6e6e"
RULE = "#c8c8c8"
BOX_BG = "#f2f2f2"

# (name, method -> improvement, measured result) - verbatim from README.md
MODELS = [
    (
        "demand-forecast-net",
        "Global NB MLP (DeepAR-lite) -> intermittent routing + Croston states, inverse-scale loss",
        "MASE 0.987 / RMSSE 0.948 vs seasonal-naive 1.080/1.062 and Holt-Winters 1.101/1.019",
    ),
    (
        "sku-text-classifier",
        "fastText hashed char n-grams -> regex part-number split + inverse-frequency class weights",
        "macro-F1 0.963 vs majority-class baseline 0.013",
    ),
    (
        "order-anomaly-ae",
        "Reconstruction-error AE -> per-feature explanations + clean-split threshold; PCA baseline",
        "AE PR-AUC 0.963 vs PCA 0.951 - narrow win; PCA recommended as the first reach",
    ),
    (
        "churn-rfm-predictor",
        "From-scratch numpy logistic regression -> frequency-slope feature + Platt calibration",
        "PR-AUC 0.653 vs recency 0.361 / prevalence 0.150; ECE 0.197 -> 0.021 after Platt",
    ),
    (
        "price-elasticity-regressor",
        "From-scratch ridge/lasso, log-log elasticity -> hierarchical shrinkage + endogeneity control",
        "naive-OLS bias +1.52 -> +0.03 controlled; profit regret 0.89% vs analytic optimum",
    ),
]

HONESTY = [
    "PCA over the autoencoder: the AE wins only narrowly (PR-AUC 0.963 vs 0.951), so the"
    " simpler PCA-SVD baseline stays in the report and is the recommended first reach.",
    "Naive-baseline transparency: seasonal-naive and Holt-Winters sit in the forecasting"
    " headline table; recency and prevalence beside churn; majority-class beside macro-F1.",
]

FIGURES = [
    ("pr_curve.png", "AE vs PCA precision-recall curves"),
    ("forecast_fit.png", "Forecasts vs the naive baselines"),
    ("reliability_curve.png", "Platt calibration: ECE 0.197 -> 0.021"),
]


def build() -> str:
    os.makedirs(OUT_DIR, exist_ok=True)
    fig = plt.figure(figsize=(8.27, 11.69))  # A4 portrait
    lm, rm = 0.06, 0.94

    fig.text(lm, 0.962, "ml-models-lab", fontsize=19, fontweight="bold", color=INK)
    fig.text(
        lm,
        0.9435,
        "Five ML models built from scratch for a B2B distributor setting - "
        "method, one improvement, honest measured result.",
        fontsize=9.5,
        color=INK2,
    )
    fig.text(
        lm,
        0.9305,
        "All data synthetic and seeded; numbers are exactly what the code prints at the "
        "default seed. No euro claims.",
        fontsize=8.5,
        color=INK3,
        style="italic",
    )
    fig.add_artist(plt.Line2D([lm, rm], [0.921, 0.921], color=RULE, lw=0.8))

    y = 0.905
    for name, method, result in MODELS:
        fig.text(lm, y, name, fontsize=10.5, fontweight="bold", color=INK, family="monospace")
        fig.text(lm + 0.005, y - 0.0165, method, fontsize=8.8, color=INK2)
        fig.text(lm + 0.005, y - 0.031, "Measured:  " + result, fontsize=8.8, color=INK)
        y -= 0.0525
        if name != MODELS[-1][0]:
            fig.add_artist(plt.Line2D([lm, rm], [y + 0.0075, y + 0.0075], color=RULE, lw=0.5))

    box_top, box_h = y - 0.002, 0.098
    fig.add_artist(
        FancyBboxPatch(
            (lm, box_top - box_h),
            rm - lm,
            box_h,
            boxstyle="round,pad=0.006",
            facecolor=BOX_BG,
            edgecolor=RULE,
            linewidth=0.8,
            transform=fig.transFigure,
        )
    )
    fig.text(
        lm + 0.012, box_top - 0.019, "Honesty highlights", fontsize=10, fontweight="bold", color=INK
    )
    by = box_top - 0.038
    for bullet in HONESTY:
        for j, line in enumerate(textwrap.wrap(bullet, width=112)):
            prefix = "- " if j == 0 else "  "
            fig.text(lm + 0.012, by, prefix + line, fontsize=8.4, color=INK2)
            by -= 0.0135
        by -= 0.004

    fig_y, fig_h = 0.315, 0.185
    fig_w = (rm - lm - 0.04) / 3
    for i, (fname, caption) in enumerate(FIGURES):
        x = lm + i * (fig_w + 0.02)
        ax = fig.add_axes([x, fig_y, fig_w, fig_h], anchor="N")
        ax.imshow(mpimg.imread(os.path.join(IMG_DIR, fname)))
        ax.axis("off")
        fig.text(x + fig_w / 2, fig_y - 0.010, caption, fontsize=7.6, color=INK3, ha="center")

    fig.text(lm, 0.245, "Reproduce it", fontsize=10, fontweight="bold", color=INK)
    commands = [
        ("pip install -r requirements.txt", "numpy / scipy / pandas / matplotlib / torch"),
        ("python -m mllab.<model>", "train + print metrics + save the labelled figure"),
        ("python -m ruff check .", "lint gate (clean)"),
        ("python -m pytest -q", "20 deterministic / baseline / invariant tests"),
        ("python tools/make_onepager.py", "rebuild this PDF from the repo figures"),
    ]
    cy = 0.226
    for cmd, note in commands:
        fig.text(lm + 0.005, cy, cmd, fontsize=8.6, color=INK, family="monospace")
        fig.text(lm + 0.36, cy, "# " + note, fontsize=8.2, color=INK3)
        cy -= 0.0165

    fig.add_artist(plt.Line2D([lm, rm], [0.055, 0.055], color=RULE, lw=0.8))
    fig.text(
        lm,
        0.040,
        "Deliverable of the ml-models-lab repo - full specs in docs/METHODOLOGY.md, framing in "
        "docs/BUSINESS_CASE.md.",
        fontsize=8,
        color=INK3,
    )
    fig.text(lm, 0.027, "Author: Dimitres Kisimov, 2026.", fontsize=8, color=INK3)

    fig.savefig(OUT_PDF, format="pdf")
    plt.close(fig)
    return OUT_PDF


def main() -> None:
    path = build()
    size = os.path.getsize(path)
    print("ml-models-lab one-pager")
    print("=" * 60)
    for name, _method, result in MODELS:
        print(f"[model] {name}")
        print(f"        {result}")
    print("=" * 60)
    print(f"[ok] wrote {os.path.relpath(path, ROOT)} ({size / 1024:.1f} KB)")
    if size <= 10 * 1024:
        raise SystemExit(f"[fail] PDF is {size} bytes; expected > 10 KB")
    print("[ok] size check passed (> 10 KB)")


if __name__ == "__main__":
    main()
