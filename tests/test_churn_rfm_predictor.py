import numpy as np

from mllab.churn_rfm_predictor.model import LogisticRegressionScratch
from mllab.churn_rfm_predictor.train import run


def test_logistic_gradient_matches_finite_difference():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 4))
    y = (rng.random(40) < 0.5).astype(float)
    clf = LogisticRegressionScratch(l2=0.1, class_weight=2.0, focal_gamma=0.0)
    clf.w = rng.normal(size=4)
    clf.b = 0.3
    gw, gb = clf.gradient(X, y)

    eps = 1e-6
    num_gw = np.zeros_like(gw)
    for j in range(len(gw)):
        wp = clf.w.copy()
        wp[j] += eps
        wm = clf.w.copy()
        wm[j] -= eps
        base_w = clf.w
        clf.w = wp
        lp = clf.loss(X, y)
        clf.w = wm
        lm = clf.loss(X, y)
        clf.w = base_w
        num_gw[j] = (lp - lm) / (2 * eps)
    assert np.allclose(gw, num_gw, atol=1e-4)

    bp, bm = clf.b + eps, clf.b - eps
    base_b = clf.b
    clf.b = bp
    lp = clf.loss(X, y)
    clf.b = bm
    lm = clf.loss(X, y)
    clf.b = base_b
    num_gb = (lp - lm) / (2 * eps)
    assert abs(gb - num_gb) < 1e-4


def test_probabilities_are_valid():
    out = run(make_plot=False)
    assert 0.0 <= out["pr_auc"] <= 1.0
    assert 0.0 <= out["roc_auc"] <= 1.0


def test_beats_baselines_and_calibration_helps():
    out = run(make_plot=False)
    assert out["pr_auc"] > out["pr_auc_recency"]
    assert out["pr_auc"] > out["pr_auc_prevalence"]
    # Platt calibration should not worsen ECE
    assert out["ece_cal"] <= out["ece_uncal"]
