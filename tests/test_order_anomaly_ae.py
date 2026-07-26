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
    # the explanation must be attributed by the scorer that produced the flag
    assert out["explanation_scorer"] == out["winner_by_pr_auc"]
    assert len(out["explanation_features"]) == 2


def test_ae_per_feature_error_shape_and_range():
    import pytest

    pytest.importorskip("torch")
    from mllab.order_anomaly_ae.ae import ae_per_feature_error, ae_score, train_ae

    X, y, _ = synth.make_orders(0)
    sc = Standardizer().fit(X[y == 0])
    Xs = sc.transform(X[y == 0])[:300]
    ae = train_ae(Xs, seed=0, epochs=60)
    per_feat = ae_per_feature_error(ae, Xs)
    assert per_feat.shape == Xs.shape  # one error per input feature
    assert np.all(np.isfinite(per_feat))
    assert np.all(per_feat >= 0)
    # rows sum to the AE anomaly score for the same input
    np.testing.assert_allclose(per_feat.sum(axis=1), ae_score(ae, Xs), rtol=1e-4)


def test_ae_explanation_localises_corrupted_feature():
    import pytest

    pytest.importorskip("torch")
    from mllab.order_anomaly_ae.ae import ae_per_feature_error, train_ae

    X, y, feat_names = synth.make_orders(3)
    normal = X[y == 0]
    sc = Standardizer().fit(normal)
    ae = train_ae(sc.transform(normal[:1500]), seed=3, epochs=300, denoise_std=0.1)
    # corrupt exactly ONE known feature of a held-out normal row (+8 sigma)
    row = sc.transform(normal[2000:2001]).copy()
    j = feat_names.index("discount_pct")
    row[0, j] += 8.0
    per_feat = ae_per_feature_error(ae, row)[0]
    top2 = np.argsort(-per_feat)[:2]
    assert j in top2  # loose (top-2) to avoid seed brittleness


def test_explain_top_flag_pairs_explanation_with_scorer():
    from mllab.order_anomaly_ae.train import explain_top_flag

    feat_names = ["a", "b", "c"]
    # PCA-style scorer flags row 0 (driven by feature "a") ...
    per_feat_pca = np.array([[9.0, 0.5, 0.5], [0.2, 0.2, 0.2]])
    # ... while an AE-style scorer flags row 1 (driven by feature "b")
    per_feat_ae = np.array([[0.2, 0.2, 0.2], [0.3, 8.0, 0.3]])
    pca_ranked = explain_top_flag(per_feat_pca.sum(axis=1), per_feat_pca, feat_names)
    ae_ranked = explain_top_flag(per_feat_ae.sum(axis=1), per_feat_ae, feat_names)
    assert pca_ranked[0][0] == "a"  # PCA flag -> PCA attribution
    assert ae_ranked[0][0] == "b"  # AE flag -> AE attribution
