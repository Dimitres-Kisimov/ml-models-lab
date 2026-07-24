import numpy as np

from mllab import metrics, synth
from mllab.order_anomaly_ae.model import PCAReconstructor, Standardizer


def test_pca_reconstruct_shapes_and_scores():
    X, y, _ = synth.make_orders(0)
    sc = Standardizer().fit(X[y == 0])
    Xs = sc.transform(X)
    pca = PCAReconstructor(k=4).fit(Xs[y == 0])
    recon = pca.reconstruct(Xs)
    assert recon.shape == Xs.shape
    scores = pca.score(Xs)
    assert scores.shape == (len(Xs),)
    assert np.all(scores >= 0)


def test_pca_baseline_separates_anomalies():
    X, y, _ = synth.make_orders(0)
    sc = Standardizer().fit(X[y == 0])
    pca = PCAReconstructor(k=4).fit(sc.transform(X[y == 0]))
    scores = pca.score(sc.transform(X))
    assert metrics.roc_auc(y, scores) > 0.7  # clearly better than chance


def test_honest_comparison_is_reported():
    # importorskip so numpy-only CI can still collect this module
    import pytest

    pytest.importorskip("torch")
    from mllab.order_anomaly_ae.train import run

    out = run(make_plot=False)
    for key in ("pca_pr_auc", "ae_pr_auc", "winner_by_pr_auc"):
        assert key in out
    assert out["winner_by_pr_auc"] in ("AE", "PCA")
