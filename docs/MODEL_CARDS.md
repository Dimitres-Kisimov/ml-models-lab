# Model cards roll-up

One row per model. Each links to its full
[Mitchell et al.](https://arxiv.org/abs/1810.03993)-structured card under
[`model_cards/`](model_cards/). **Every number below is exactly what the code
prints at its default seed on synthetic data** (see each card's *Evaluation*
section) - nothing is invented, rounded up, or cherry-picked. All five models
are trained and evaluated exclusively on the seeded synthetic generators in
[`../mllab/synth.py`](../mllab/synth.py); none is validated on real data, and
each card's *Intended use* section says so explicitly.

| Model | What it is | Metric measured | Result vs fair baseline (seeded) | Do not use for |
|---|---|---|---|---|
| [demand-forecast-net](model_cards/demand-forecast-net.md) | Global NB-head MLP demand forecaster | MASE / RMSSE, one-step rolling-origin CV | **0.987 / 0.948** vs seasonal-naive 1.080 / 1.062 and Holt-Winters 1.101 / 1.019 | real purchasing, inventory, or staffing decisions |
| [sku-text-classifier](model_cards/sku-text-classifier.md) | Hashed char n-gram SKU category classifier | macro-F1 | **0.963** vs majority-class 0.013 | open-set classification (it cannot say "none of these") |
| [order-anomaly-ae](model_cards/order-anomaly-ae.md) | Order anomaly autoencoder + PCA baseline | PR-AUC / ROC-AUC / P@k | AE **0.963** PR-AUC vs PCA **0.951** - a narrow win on this seed, so the PCA baseline stays in the report | automated blocking or reversal of orders with no human in the loop |
| [churn-rfm-predictor](model_cards/churn-rfm-predictor.md) | Calibrated RFM churn predictor | PR-AUC (+ Brier, ECE) | **0.653** PR-AUC vs recency 0.361 / prevalence 0.150; ECE **0.197 -> 0.021** after Platt calibration | decisions about individual real customers |
| [price-elasticity-regressor](model_cards/price-elasticity-regressor.md) | Price elasticity + Lerner-optimal pricing | elasticity bias + profit regret | naive-OLS bias **+1.52 -> +0.03** once the demand control is added; profit regret **0.89%** vs the analytic optimum | setting real prices from observational data without an instrument |

Every model package under `mllab/` that exposes a `python -m mllab.<model>`
entry point must have a card here - [`tests/test_model_cards.py`](../tests/test_model_cards.py)
fails if one is missing, if a card is orphaned, or if a card drops a required
section, so this table cannot silently drift out of sync with the code.

*Author: Dimitres Kisimov, 2026. Card structure follows Mitchell et al., "Model
Cards for Model Reporting", FAT\* 2019.*
