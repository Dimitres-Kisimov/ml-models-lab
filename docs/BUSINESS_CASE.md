# Why from-scratch models matter — a business case

*This repo is a teaching-grade lab on synthetic data, not a product, so framing
it in euros would be dishonest. Instead: here is the problem it addresses, what
the five models actually prove, who would care, and what I am explicitly
**not** claiming.*

## The situation

Most applied ML today is library-driven: `fit()`, `predict()`, ship. That works
until it doesn't — and when it doesn't, the person who assembled the pipeline
often cannot say *why*, because the math lives inside someone else's black box.
A B2B distributor's decisions (how much stock to order, which customer to call,
what price to set, which order to hold for review) are exactly the kind where
the failure modes matter more than the demo: class imbalance, short and
intermittent series, confounded prices, drifting base rates. A model whose
builder understands its failure modes can be trusted with a decision; a model
whose builder only understands its API cannot.

## What the lab proves

Five published methods, re-implemented at the math level in numpy or plain
PyTorch (no scikit-learn, no xgboost), each with one principled improvement and
an evaluation metric chosen to survive the thing that actually breaks the
problem — per the portfolio's KPI methodology in
[`docs/METHODOLOGY.md`](METHODOLOGY.md):

- **Demand forecasting** (global NB MLP, DeepAR-lite): MASE **0.987** / RMSSE
  **0.948** under rolling-origin CV, reported *beside* seasonal-naive
  (1.080 / 1.062) and Holt-Winters (1.101 / 1.019) — the naive baselines stay
  in the headline table so the margin is visible, not implied.
- **SKU text classification** (fastText-style hashed n-grams): macro-F1
  **0.963** vs a majority-class baseline of 0.013 — macro-F1, not accuracy,
  because rare categories are the point.
- **Order anomaly detection** (autoencoder + mandatory PCA baseline): AE PR-AUC
  **0.963** vs PCA **0.951** — and the recommendation is still **PCA first**
  (see below).
- **Churn prediction** (from-scratch logistic regression + Platt): PR-AUC
  **0.653** vs recency-only 0.361 / prevalence 0.150, with calibration error
  ECE **0.197 → 0.021** — because a retention budget needs probabilities, not
  just a ranking.
- **Price elasticity** (from-scratch ridge/lasso + Lerner pricing): naive-OLS
  elasticity bias **+1.52** collapses to **+0.03** with the confounder
  controlled, and pricing from the estimates costs **0.89%** profit regret vs
  the analytic optimum — decision quality measured, not just RMSE.

Two results are worth singling out, because in both the **simpler method won
and was recommended** — which is the whole point of building from scratch:

1. **PCA over the autoencoder.** The AE beats the numpy PCA-SVD baseline only
   narrowly on the default seed (PR-AUC 0.963 vs 0.951). The repo's stated
   recommendation is to reach for PCA first, and on seeds where PCA wins, to
   prefer PCA. A library-driven workflow would have shipped the deep model and
   never run the baseline.
2. **The pooled estimate over per-segment fits.** For thin pricing segments,
   hierarchically shrinking each segment's elasticity toward the simple pooled
   estimate tightens RMSE from 0.16 to 0.13 — the boring partial-pooling
   estimator beats trusting every segment's own noisy fit.

## Who benefits

- **Whoever consumes the numbers** — a distributor's planner, pricing analyst,
  or retention owner gets metrics chosen for their decision (MASE not MAPE,
  PR-AUC not accuracy, calibration not just ranking, profit regret not just
  R²), with the baselines left in view.
- **A reviewer assessing the builder** — the from-scratch implementations plus
  the two simpler-method-won calls are a direct demonstration of judgment, not
  just library fluency.
- **Anyone learning these methods** — each model is the smallest credible
  version that still demonstrates the method, trains in seconds on a laptop
  CPU, and is pinned by 20 deterministic tests.

## What I am NOT claiming

- **All data is synthetic and seeded.** No real customer, order, or catalogue
  data exists anywhere in the repo, and no euro impact is claimed or implied.
- The headline numbers are single-seed measurements on toy-scale data; they
  demonstrate the *methods and the evaluation discipline*, not production
  performance. Real deployment would need the drift, leakage, and label-quality
  checks each model's README spells out.
- Not SOTA, not benchmarked against tuned library implementations — these are
  deliberately tiny models held to a teaching bar.

## Deliverable

The repo itself (five documented models under `mllab/`, the from-scratch
metrics in `mllab/metrics.py`, 20 tests, and the cited specs in
[`docs/METHODOLOGY.md`](METHODOLOGY.md)), plus a one-page PDF summary —
[`deliverables/ml_models_lab_onepager.pdf`](../deliverables/ml_models_lab_onepager.pdf),
regenerable with `python tools/make_onepager.py` — showing each model's method,
improvement, and measured result, with the honesty highlights above.
