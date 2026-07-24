# demand-forecast-net

Global cross-series MLP with a negative-binomial head - a "DeepAR-lite without
the RNN" - trained on lag + calendar features across all SKUs at once, which is
the setup that wins on short retail series (M4/M5 evidence).

- **Build:** input = lags {1,2,3,12,13} + rolling means {3,6,12} + 6 Fourier
  month features + 8-dim SKU embedding; body 64->64 ReLU; NB head (mean +
  dispersion), trained by NB negative log-likelihood.
- **Improvement:** intermittent SKUs (zero-fraction > 30% / ADI >= 1.32) are fed
  Croston smoothed size/interval states as extra causal features, and the NB
  loss is weighted by inverse series scale so training optimises the scale-free
  MASE/RMSSE metric rather than the handful of high-volume SKUs.
- **Data:** 200 SKUs x 42 months, ~75% seasonal + ~25% intermittent, numpy
  negative-binomial draws (`mllab/synth.make_demand_panel`).
- **Metric:** MASE + RMSSE under one-step rolling-origin CV (last 3 months as
  successive origins). Baselines: seasonal-naive and additive Holt-Winters.

Measured (seed 13): net MASE 0.987 / RMSSE 0.948, beating seasonal-naive
(1.080 / 1.062) and Holt-Winters (1.101 / 1.019).

## Sources

- DeepAR - Salinas et al., *Int. J. Forecasting* 2020 - https://arxiv.org/abs/1704.04110
- N-BEATS - Oreshkin et al., ICLR 2020 - https://arxiv.org/abs/1905.10437
- Croston's method for intermittent demand - Croston, *Oper. Res. Q.* 1972;
  syntetos-Boylan ADI/CV2 classification - https://doi.org/10.1016/j.ijforecast.2004.10.001
- M5 accuracy competition (MASE/RMSSE) - Makridakis et al., 2022 - https://doi.org/10.1016/j.ijforecast.2021.11.013
