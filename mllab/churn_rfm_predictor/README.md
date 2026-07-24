# churn-rfm-predictor

At-risk / churn prediction on RFM features with **logistic regression written
from scratch** in numpy, plus probability calibration - because for a retention
budget you need well-calibrated *probabilities*, not just a ranking.

- **Build:** numpy logistic regression - sigmoid, L2-penalised BCE, analytic
  gradient `X^T(sigma(Xw+b) - y)/n + 2*lambda*w`, full-batch gradient descent
  (a finite-difference test pins the gradient). Optional class-weighted / focal
  loss. Platt calibration is fit on held-out logits.
- **Improvement:** an engineered declining-order-frequency slope plus
  recency/tenure features; calibration happens *after* class-weighting; and the
  train/test split is **time-based** (train on earlier signup cohorts, test on
  later ones) to avoid leakage.
- **Data:** 4000 customers, ~15% churn from a logistic latent with an
  R x slope interaction and irreducible noise (`mllab/synth.make_customers`).
- **Metric:** PR-AUC (primary) + ROC-AUC + Brier + ECE + a reliability curve.

Measured (seed 7): PR-AUC 0.653 (vs recency-only 0.361 and prevalence 0.150),
ROC-AUC 0.871, Brier 0.135 -> 0.078 and ECE 0.197 -> 0.021 after Platt
calibration.

## Sources

- RFM optimal-selection model - Bult & Wansbeek, *Marketing Science* 1995 - https://doi.org/10.1287/mksc.14.4.362
- Focal loss - Lin et al., ICCV 2017 - https://arxiv.org/abs/1708.02002
- Platt scaling - Platt, 1999 - https://www.researchgate.net/publication/2594015
- On calibration of modern neural networks (ECE / temperature) - Guo et al., ICML 2017 - https://arxiv.org/abs/1706.04599
