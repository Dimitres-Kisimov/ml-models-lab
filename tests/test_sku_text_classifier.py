import pytest

from mllab.sku_text_classifier.hashing import (
    char_ngrams,
    hash_document,
    subtoken_split,
)


def test_subtoken_split_expands_alphanumerics():
    toks = subtoken_split("M8x40")
    assert "m8" in toks and "x40" in toks and "8" in toks and "40" in toks


def test_hashing_is_deterministic_and_bounded():
    a = hash_document("ACME stainless hex bolt M8x40", n_buckets=1024)
    b = hash_document("ACME stainless hex bolt M8x40", n_buckets=1024)
    assert a == b
    assert all(0 <= h < 1024 for h in a)
    assert len(char_ngrams("m8")) > 0


def test_macro_f1_beats_majority_baseline():
    pytest.importorskip("torch")
    from mllab.sku_text_classifier.train import run

    out = run(make_plot=False)
    assert out["macro_f1"] > out["macro_f1_majority"]
    assert out["macro_f1"] > 0.5  # non-trivial despite cross-category overlap
