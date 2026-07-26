# order-anomaly-ae

Reconstruction-error anomaly detection for orders/deliveries: an undercomplete
autoencoder trained on normal orders only, scored by squared reconstruction
error - with a **mandatory PCA-SVD baseline** so the comparison stays honest.

- **Build:** AE `d-16-4-16-d` (ReLU hidden, linear output), trained normal-only
  with a small denoising perturbation; numpy PCA-SVD reconstructor as the
  challenger (`model.py`, torch-free and unit-tested).
- **Improvement:** the anomaly threshold is set from a percentile on a *clean*
  validation split (an explicit FPR/contamination budget), and every flag is
  explained by its top per-feature reconstruction errors, attributed by the
  same scorer that produced the flag ("driven by customer_tenure,
  lead_time_days").
- **Data:** 5000 orders, 7 correlated features via a low-rank factor model, with
  ~3% injected anomalies of three kinds: extreme-univariate, broken-correlation,
  and impossible-combo (`mllab/synth.make_orders`).
- **Metric:** ROC-AUC + PR-AUC + precision@k, with the assumed contamination
  rate reported.

Measured (seed 3): AE PR-AUC 0.963 (ROC 0.982) vs PCA PR-AUC 0.951 (ROC 0.976).
The AE wins narrowly on this seed, but PCA is the simpler model and remains the
baseline you should reach for first; the CLI prints the honest side-by-side and
recommends whichever wins. On seeds where PCA wins, prefer PCA.

## Sources

- Anomaly detection using autoencoders - Sakurada & Yairi, MLSDA 2014 - https://doi.org/10.1145/2689746.2689747
- Outlier detection via replicator neural networks - Hawkins et al., 2002 - https://doi.org/10.1007/3-540-46145-0_17
- Deep SVDD (named deep alternative) - Ruff et al., ICML 2018 - https://proceedings.mlr.press/v80/ruff18a.html
- PCA / Mahalanobis reconstruction baselines - Jolliffe, *Principal Component Analysis*, 2002
