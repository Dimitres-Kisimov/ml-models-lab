# ml-models-lab

This is a small lab of five machine-learning models I built from scratch for a
B2B distributor setting (demand, catalogue text, orders, customers, pricing).
Each one takes a published, well-regarded method, re-implements the core in
numpy or plain PyTorch, adds **one** principled improvement for the use case,
and is evaluated with a metric that survives the thing that actually breaks
these problems - class imbalance, short series, or lots of zeros - **not
accuracy**.

Everything trains on a laptop CPU in seconds to a few seconds. I kept each model
the smallest credible version that still demonstrates the method.

## Honesty and data note

**All data in this repo is synthetic and seeded.** There is no real customer,
order, or catalogue data anywhere. Each model ships its own generator in
`mllab/synth.py` with an explicit seed, so the same seed reproduces the same
numbers (the tests enforce this). The results below are exactly what the code
prints on my machine at the default seed - nothing is rounded up or cherry
picked. Where a from-scratch model does **not** clearly beat its simpler
baseline (the anomaly autoencoder vs PCA), I say so and keep the baseline in
front of you. No SOTA / "beats everyone" claims - these are deliberately tiny
models on toy data, meant to show the method and an honest evaluation.

## The five models

| Model | Base method | The one improvement | Metric | Measured result (seeded) |
|---|---|---|---|---|
| `demand-forecast-net` | DeepAR / N-BEATS global model | intermittent routing + Croston states as features; inverse-scale loss weighting | MASE / RMSSE, rolling-origin CV | **0.987 / 0.948**, beats seasonal-naive (1.080 / 1.062) and Holt-Winters (1.101 / 1.019) |
| `sku-text-classifier` | fastText hashed n-grams | regex sub-token split of part numbers + inverse-frequency class weights | macro-F1 | **0.963** vs majority-class 0.013 |
| `order-anomaly-ae` | reconstruction-error autoencoder | per-feature error explanation + clean-split percentile threshold; **mandatory PCA baseline** | PR-AUC / ROC-AUC / P@k | AE **0.963** PR-AUC vs PCA **0.951** (AE wins narrowly on this seed; PCA is the honest baseline) |
| `churn-rfm-predictor` | RFM + L2 logistic regression | declining-order-frequency slope + Platt calibration after class-weighting; time-based split | PR-AUC (+ Brier, ECE) | **0.653** PR-AUC vs recency 0.361 / prevalence 0.150; ECE 0.197 -> **0.021** after calibration |
| `price-elasticity-regressor` | Ridge + Lasso, log-log elasticity | hierarchical shrinkage of thin segments + endogeneity control | elasticity RMSE + profit regret | naive-OLS bias **+1.52** -> controlled **+0.03**; profit regret **0.89%** vs the analytic optimum |

## How to run

Install the stack (numpy / scipy / pandas / matplotlib / torch):

```
pip install -r requirements.txt
```

Each model has a CLI entry point that trains, prints its metrics with plain
ASCII markers, and saves a labelled figure to `mllab/<model>/results/`:

```
python -m mllab.demand_forecast_net        # forecast fit, MASE/RMSSE table
python -m mllab.sku_text_classifier        # confusion matrix, macro-F1
python -m mllab.order_anomaly_ae           # PR curves, AE vs PCA
python -m mllab.churn_rfm_predictor        # reliability curve, PR-AUC
python -m mllab.price_elasticity_regressor # elasticity fit, endogeneity demo
```

Run the gates:

```
python -m ruff check .
python -m pytest -q
```

## What each model does (and its sources)

Short version below; the deeper spec with citations is in
[`docs/METHODOLOGY.md`](docs/METHODOLOGY.md), and each model has its own
`README.md` with the exact source URLs.

- **`demand-forecast-net`** - a global cross-series MLP (a "DeepAR-lite without
  the RNN"): lags {1,2,3,12,13} + rolling means {3,6,12} + Fourier month
  features + an 8-dim SKU embedding, with a negative-binomial head. Intermittent
  SKUs are routed and fed Croston smoothed size/interval states. I weight the NB
  loss by inverse series scale so training targets the scale-free MASE/RMSSE
  metric instead of over-fitting the few high-volume SKUs. See
  [`mllab/demand_forecast_net/README.md`](mllab/demand_forecast_net/README.md).

- **`sku-text-classifier`** - hashed char n-grams (3-5) into an
  `nn.EmbeddingBag(mean)` and a linear softmax (fastText core). Before hashing I
  split alphanumeric part numbers with a regex (`M8x40` -> `m8`, `x40`, `8`,
  `40`) so unseen SKUs are represented by their *shape*, and I use
  inverse-frequency class weights so rare categories are not drowned. Reported
  as **macro-F1**, not accuracy. See
  [`mllab/sku_text_classifier/README.md`](mllab/sku_text_classifier/README.md).

- **`order-anomaly-ae`** - an undercomplete autoencoder (d-16-4-16-d) trained on
  normal orders only; the anomaly score is reconstruction error. The
  **mandatory numpy PCA-SVD baseline** runs alongside it, and I report the
  honest side-by-side. The AE also explains each flag by its top per-feature
  errors. See [`mllab/order_anomaly_ae/README.md`](mllab/order_anomaly_ae/README.md).

- **`churn-rfm-predictor`** - logistic regression written from scratch in numpy
  (sigmoid, L2-penalised BCE, analytic gradient, full-batch GD), with
  class-weighting and Platt calibration fit on held-out logits. Features include
  an engineered declining-order-frequency slope, and the split is time-based to
  avoid leakage. See
  [`mllab/churn_rfm_predictor/README.md`](mllab/churn_rfm_predictor/README.md).

- **`price-elasticity-regressor`** - ridge (closed form via `scipy.linalg.solve`
  plus gradient descent from scratch) and lasso (coordinate descent with
  soft-thresholding) on per-segment log-log elasticity, with hierarchical
  shrinkage of thin segments toward the pooled estimate. The synthetic data
  bakes in a demand confounder, so I can show naive OLS is biased and adding the
  control recovers the true elasticity - then translate that into simulated
  profit regret vs the Lerner optimum. See
  [`mllab/price_elasticity_regressor/README.md`](mllab/price_elasticity_regressor/README.md).

## Repo layout

```
mllab/
  synth.py            seeded synthetic data generators (all 5)
  metrics.py          from-scratch metrics (ROC/PR-AUC, MASE, ECE, macro-F1, ...)
  demand_forecast_net/
  sku_text_classifier/
  order_anomaly_ae/
  churn_rfm_predictor/
  price_elasticity_regressor/
tests/                deterministic + baseline-beating + invariant checks
docs/METHODOLOGY.md   the deeper spec, with citations
```

## License

MIT - see [`LICENSE`](LICENSE). Author: Dimitres Kisimov, 2026.
