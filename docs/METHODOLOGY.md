# ML model-training batch — methodology specs

Five small models that **train locally** on this machine (PyTorch + numpy only; no
scikit-learn/xgboost; seconds-to-minutes on CPU / RTX 4050). Each takes a
published, well-regarded method and adapts/improves it for a specific B2B
distributor use case, with an honest evaluation. Sources verified.

The recurring, defensible thread across all five: **implement the core from
scratch, add one principled improvement for the use case, and evaluate with the
metric that survives class imbalance / short series / zeros — not accuracy.**

---

## 1. Demand-forecasting net  (`demand-forecast-net`)
- **Base method:** DeepAR (Salinas et al., IJF 2020, arXiv:1704.04110) and N-BEATS
  (Oreshkin et al., ICLR 2020, arXiv:1905.10437); M4/M5 evidence that a **global**
  model on lag+calendar features wins on short series.
- **Build:** a global cross-series MLP (input ≈ lags {1,2,3,12,13} + rolling means
  {3,6,12} + 6 Fourier month features + SKU embedding dim 8; body 64→64 ReLU) with a
  **negative-binomial head** (or pinball/quantile loss) — a "DeepAR-lite without the
  RNN". Trim N-BEATS (1 trend + 1 seasonality block) as the challenger.
- **Improvement:** route intermittent SKUs (zero-fraction > 30% / ADI ≥ 1.32) to the
  NB head + feed Croston's smoothed size/interval states as extra features.
- **Data:** 200 SKUs × 42 months (~75% seasonal, ~25% intermittent), numpy NB draws.
- **Metric:** MASE + RMSSE under rolling-origin CV; beat seasonal-naive & Holt-Winters.

## 2. Product-category text classifier  (`sku-text-classifier`)
- **Base method:** fastText (Joulin et al., arXiv:1607.01759) + TextCNN (Kim, EMNLP
  2014, arXiv:1408.5882); character-level (Zhang/Zhao/LeCun, NIPS 2015).
- **Build:** hashed **char n-gram (3–5) → `nn.EmbeddingBag(mean)` → linear softmax**
  (fastText core, trains in seconds), with an optional char-TextCNN branch.
- **Improvement:** regex sub-token split of alphanumerics (`M8x40` → `M8`,`x40`,`8`,`40`)
  before hashing + word-boundary markers + inverse-frequency class weights — so unseen
  part numbers are represented by their *shape*, and rare categories aren't drowned.
- **Data:** ~12 categories × ~300 templated strings with deliberate cross-category
  vocab overlap.
- **Metric:** **macro-F1** (not accuracy) on a stratified split + confusion matrix.

## 3. Order/delivery anomaly detector  (`order-anomaly-ae`)
- **Base method:** autoencoder reconstruction-error anomaly detection (Sakurada &
  Yairi, MLSDA 2014; Hawkins 2002); PCA/Mahalanobis baselines; Deep SVDD as the named
  alternative.
- **Build:** undercomplete AE `d-16-4-16-d` (ReLU hidden, linear output), trained
  **normal-only**, score = squared reconstruction error. Mandatory **numpy PCA-SVD
  baseline** — if the AE can't beat it, prefer PCA (and say so).
- **Improvement:** percentile threshold on a clean validation split (FPR budget) +
  **per-feature error as explanation** ("flagged by discount_pct & order_value") +
  denoising AE for contamination robustness.
- **Data:** ~5000 orders, 7 correlated features; inject ~3% anomalies of 3 kinds
  (extreme univariate, broken correlation, impossible combo).
- **Metric:** ROC-AUC + **PR-AUC** + precision@k; report the assumed contamination rate.

## 4. Churn / at-risk predictor  (`churn-rfm-predictor`)
- **Base method:** RFM (Bult & Wansbeek, Marketing Science 1995) + L2 logistic
  regression; focal loss (Lin et al., ICCV 2017); Platt/temperature calibration
  (Platt 1999; Guo et al., ICML 2017).
- **Build:** **logistic regression from scratch** (numpy: sigmoid, L2-penalized BCE,
  analytic gradient `Xᵀ(σ(Xw+b)−y)/n + 2λw`, full-batch GD) + class-weighted/focal loss
  + **Platt calibration** on held-out logits. Shallow MLP as the nonlinear challenger.
- **Improvement:** engineered **declining-order-frequency slope** + recency/tenure
  features; calibrate *after* weighting; **time-based split** (features from before a
  cutoff, label from the future window) to avoid leakage.
- **Data:** ~4000 customers, ~15% churn via a logistic latent with an R×slope
  interaction + irreducible noise.
- **Metric:** **PR-AUC** (primary) + ROC-AUC + Brier + ECE + reliability curve.

## 5. Price / margin elasticity regressor  (`price-elasticity-regressor`)
- **Base method:** Ridge (Hoerl & Kennard 1970) + Lasso (Tibshirani 1996); log-log
  constant-elasticity demand (`ln Q = a + b ln P`); Lerner optimal markup
  `(P−MC)/P = 1/|ε|`.
- **Build:** ridge (closed form via `scipy.linalg.solve` as oracle + GD from scratch)
  and lasso (**coordinate descent with soft-thresholding**) on a per-segment log-log
  elasticity regression; price `P* = MC·|ε|/(|ε|−1)`, clipped, guarded for |ε|≤1.
- **Improvement:** **hierarchical shrinkage** of each segment's elasticity toward the
  pooled estimate (partial pooling in numpy) for thin segments; explicit **endogeneity
  caveat** demonstrated — a confounder in the synthetic data biases naive OLS, and
  adding controls recovers the true elasticity.
- **Data:** 30–100 products × 50–300 transactions with a known true elasticity and a
  demand confounder baked in (price set partly from the latent demand shock).
- **Metric:** RMSE/R² **and** decision-quality: held-out **simulated profit uplift /
  regret** vs the analytic optimum (accuracy ≠ good pricing).

---

*Author: Dimitres Kisimov. Specs grounded in the cited literature; every model is
deliberately the smallest credible version that still demonstrates the method, so it
trains on a laptop in seconds. Full source URLs live in each model's own README.*
