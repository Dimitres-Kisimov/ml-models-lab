# Model cards

One card per model, following the structure of Mitchell et al., *Model Cards
for Model Reporting*, FAT\* 2019 - https://arxiv.org/abs/1810.03993 - adapted
to this lab: every card states the method, the exact training configuration,
the synthetic data generator and its assumptions, why the reported metric was
chosen, the measured results against named baselines, and concrete limitations
and failure modes. No number appears in a card that the code does not print at
its default seed.

A blanket note that applies to all five cards: **every model in this lab is
trained and evaluated exclusively on seeded synthetic data** from
[`mllab/synth.py`](../../mllab/synth.py). The cards therefore describe what the
implementation demonstrably does on that data - they are not evidence of
real-world performance, and each card's *Intended use* section says so
explicitly.

| Card | Model |
|---|---|
| [demand-forecast-net.md](demand-forecast-net.md) | Global NB-head MLP demand forecaster |
| [sku-text-classifier.md](sku-text-classifier.md) | Hashed char n-gram SKU category classifier |
| [order-anomaly-ae.md](order-anomaly-ae.md) | Order anomaly autoencoder + PCA baseline |
| [churn-rfm-predictor.md](churn-rfm-predictor.md) | Calibrated RFM churn predictor |
| [price-elasticity-regressor.md](price-elasticity-regressor.md) | Price elasticity + Lerner pricing |

Author: Dimitres Kisimov, 2026.
