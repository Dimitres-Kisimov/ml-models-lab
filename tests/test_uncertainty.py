"""Tests for the bootstrap skill-CI harness (mllab/uncertainty.py).

Three layers, mirroring the benchmark tests:

* Pure, torch-free unit tests of the metric evaluator and the bootstrap on
  hand-built data (run anywhere, offline, instantly).
* A numpy-model integration check: the two numpy models are deterministic on
  every platform, so their point skills and bootstrap intervals are pinned.
* A drift guard: recompute every interval and assert the committed
  ``docs/BENCHMARK_CI.md`` still matches, within the same tight tolerance the
  benchmark uses for numpy models.

Nothing here touches torch: the harness is numpy-only by construction (only the
two numpy models are in scope), so the whole file runs in a numpy-only env.
"""

from pathlib import Path

import numpy as np
import pytest

from mllab.benchmark import skill_score
from mllab.uncertainty import (
    BENCHMARK_CI_MD,
    CIRow,
    _percentile_ci,
    bootstrap_row,
    collect_ci_rows,
    metric_pair,
    render_markdown,
)

# --------------------------------------------------------------------------- #
# pure unit tests: metric evaluator + percentile helper (no training)
# --------------------------------------------------------------------------- #


def test_metric_pair_binary_matches_hand_computed_pr_auc():
    # y=[1,1,0,0]; model ranks the positives first (perfect AP=1.0); the baseline
    # ranks them last -> AP = 0.5*(1/3) + 0.5*(0.5) = 0.41666... (hand-checked).
    units = {
        "kind": "binary_scores",
        "y_true": np.array([1.0, 1.0, 0.0, 0.0]),
        "model_score": np.array([0.9, 0.8, 0.2, 0.1]),
        "baseline_score": np.array([0.1, 0.2, 0.8, 0.9]),
    }
    model, base = metric_pair(units, np.arange(4))
    assert model == pytest.approx(1.0)
    assert base == pytest.approx(0.41666667, abs=1e-6)


def test_metric_pair_regression_matches_hand_computed_rmse():
    # model error vector [0.1,-0.1,0.0] -> rmse = sqrt((0.01+0.01+0)/3)=0.08165;
    # baseline error [1,1,1] -> rmse = 1.0 (hand-checked).
    units = {
        "kind": "regression_estimates",
        "target": np.array([0.0, 0.0, 0.0]),
        "model_estimate": np.array([0.1, -0.1, 0.0]),
        "baseline_estimate": np.array([1.0, 1.0, 1.0]),
    }
    model, base = metric_pair(units, np.arange(3))
    assert model == pytest.approx(np.sqrt(0.02 / 3), abs=1e-9)
    assert base == pytest.approx(1.0)


def test_metric_pair_rejects_unknown_kind():
    with pytest.raises(ValueError, match="unknown eval_units kind"):
        metric_pair({"kind": "nonsense"}, np.arange(1))


def test_percentile_ci_is_the_plain_percentile():
    vals = np.arange(0.0, 101.0)  # 0..100
    lo, hi = _percentile_ci(vals, ci_level=0.95)
    assert lo == pytest.approx(2.5)
    assert hi == pytest.approx(97.5)


# --------------------------------------------------------------------------- #
# bootstrap behaviour on synthetic, hand-reasoned data
# --------------------------------------------------------------------------- #


def _binary_units(n=400, sep=0.6, seed=0):
    """A model score that separates the classes better than a noise baseline."""
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.4).astype(float)
    model_score = sep * y + rng.normal(0.0, 0.4, size=n)  # informative
    baseline_score = rng.normal(0.0, 1.0, size=n)  # pure noise
    return {
        "kind": "binary_scores",
        "y_true": y,
        "model_score": model_score,
        "baseline_score": baseline_score,
    }


def test_bootstrap_point_equals_full_sample_skill():
    units = _binary_units()
    spec = {"model": "m", "metric": "PR-AUC", "higher_is_better": True, "baseline_name": "noise"}
    row = bootstrap_row(spec, units, n_boot=500, seed=0)
    m0, b0 = metric_pair(units, np.arange(len(units["y_true"])))
    assert row.model_value == pytest.approx(m0)
    assert row.baseline_value == pytest.approx(b0)
    assert row.skill == pytest.approx(skill_score(m0, b0, True))


def test_bootstrap_ci_is_ordered_and_brackets_reasonably():
    units = _binary_units()
    spec = {"model": "m", "metric": "PR-AUC", "higher_is_better": True, "baseline_name": "noise"}
    row = bootstrap_row(spec, units, n_boot=800, seed=0)
    assert row.skill_lo <= row.skill_hi
    assert row.model_lo <= row.model_hi
    assert row.baseline_lo <= row.baseline_hi
    # an informative model over noise should win, and clearly
    assert row.skill > 0.0
    assert row.skill_lo > 0.0 and row.ci_excludes_zero
    assert 0.0 <= row.p_beats_baseline <= 1.0
    assert row.p_beats_baseline > 0.9


def test_bootstrap_is_deterministic_in_seed():
    units = _binary_units()
    spec = {"model": "m", "metric": "PR-AUC", "higher_is_better": True, "baseline_name": "noise"}
    a = bootstrap_row(spec, units, n_boot=300, seed=123)
    b = bootstrap_row(spec, units, n_boot=300, seed=123)
    assert a == b  # frozen dataclass equality -> every field identical


def test_bootstrap_ci_brackets_zero_when_model_equals_baseline():
    # identical model and baseline scores -> skill 0 everywhere; the CI must
    # straddle 0 and the "clears zero" flag must be False.
    rng = np.random.default_rng(1)
    y = (rng.random(300) < 0.5).astype(float)
    s = rng.normal(size=300)
    units = {"kind": "binary_scores", "y_true": y, "model_score": s, "baseline_score": s.copy()}
    spec = {"model": "m", "metric": "PR-AUC", "higher_is_better": True, "baseline_name": "same"}
    row = bootstrap_row(spec, units, n_boot=400, seed=0)
    assert row.skill == pytest.approx(0.0, abs=1e-12)
    assert row.skill_lo <= 0.0 <= row.skill_hi
    assert not row.ci_excludes_zero


def test_bootstrap_regression_lower_is_better_clears_zero():
    rng = np.random.default_rng(2)
    target = rng.normal(size=80)
    units = {
        "kind": "regression_estimates",
        "target": target,
        "model_estimate": target + rng.normal(0.0, 0.1, size=80),  # close
        "baseline_estimate": target + rng.normal(0.0, 1.0, size=80),  # far
    }
    spec = {
        "model": "m",
        "metric": "elasticity RMSE",
        "higher_is_better": False,
        "baseline_name": "far",
    }
    row = bootstrap_row(spec, units, n_boot=600, seed=0)
    assert row.skill > 0.0 and row.skill_lo > 0.0
    assert row.p_beats_baseline == pytest.approx(1.0)


def test_bootstrap_drops_degenerate_resamples_but_stays_finite():
    # only one positive in five rows: many size-5 resamples contain no positive,
    # so PR-AUC is undefined there; those resamples must be dropped, and every
    # reported number must still be finite.
    units = {
        "kind": "binary_scores",
        "y_true": np.array([1.0, 0.0, 0.0, 0.0, 0.0]),
        "model_score": np.array([0.9, 0.4, 0.3, 0.2, 0.1]),
        "baseline_score": np.array([0.5, 0.5, 0.5, 0.5, 0.5]),
    }
    spec = {"model": "m", "metric": "PR-AUC", "higher_is_better": True, "baseline_name": "flat"}
    row = bootstrap_row(spec, units, n_boot=500, seed=0)
    assert 0 < row.n_valid < row.n_boot  # some, but not all, resamples survived
    for v in (row.skill_lo, row.skill_hi, row.model_lo, row.model_hi,
              row.baseline_lo, row.baseline_hi, row.p_beats_baseline):
        assert np.isfinite(v)


def test_render_is_deterministic_and_ascii():
    rows = collect_ci_rows(n_boot=200, seed=0)
    first = render_markdown(rows)
    second = render_markdown(rows)
    assert first == second
    assert first.isascii()  # no unicode -> byte-stable on any OS


# --------------------------------------------------------------------------- #
# numpy-model integration + committed-artifact drift guard (CI-safe: no torch)
# --------------------------------------------------------------------------- #

# The two numpy models are deterministic on every platform, so - exactly like
# the benchmark's numpy-model pins - their values are safe to assert tightly.
_TOL_POINT = 5e-3  # same tolerance the benchmark drift guard uses for numpy models
_TOL_CI = 1e-2  # percentile bounds: a touch looser, still catches real drift


@pytest.fixture(scope="module")
def ci_rows():
    return collect_ci_rows()


def test_collects_exactly_the_two_numpy_models(ci_rows):
    names = [r.model for r in ci_rows]
    assert names == ["churn-rfm-predictor", "price-elasticity-regressor"]
    for r in ci_rows:
        assert isinstance(r, CIRow)
        assert r.n_valid == r.n_boot == 2000  # both fully non-degenerate at default seed


def test_default_run_is_deterministic(ci_rows):
    again = collect_ci_rows()
    assert again == ci_rows  # frozen dataclasses, fixed seeds -> identical


def test_both_numpy_models_clear_zero_and_point_matches_leaderboard(ci_rows):
    from mllab.benchmark import BENCHMARK_MD

    # parse the committed leaderboard's skill cell for the two numpy models
    leaderboard = Path(BENCHMARK_MD).read_text(encoding="utf-8")
    committed_skill = {}
    for line in leaderboard.splitlines():
        s = line.strip()
        if s.startswith("| `"):
            cells = [c.strip() for c in s.strip("|").split("|")]
            committed_skill[cells[0].strip("`")] = float(cells[8])

    for r in ci_rows:
        assert r.ci_excludes_zero, f"{r.model}: 95% skill CI should clear zero"
        assert r.skill_lo <= r.skill <= r.skill_hi
        # the point skill must equal the leaderboard's value (same rule, same data)
        assert r.skill == pytest.approx(committed_skill[r.model], abs=_TOL_POINT)


def _parse_ci_cell(cell: str) -> tuple[float, float]:
    lo, hi = cell.strip().strip("[]").split(",")
    return float(lo), float(hi)


def _parse_committed_ci_table(text: str) -> dict[str, dict]:
    parsed: dict[str, dict] = {}
    for line in text.splitlines():
        s = line.strip()
        if not s.startswith("| `"):
            continue
        cells = [c.strip() for c in s.strip("|").split("|")]
        parsed[cells[0].strip("`")] = {
            "metric": cells[1],
            "better": cells[2],
            "units": int(cells[3]),
            "skill": float(cells[4]),
            "skill_ci": _parse_ci_cell(cells[5]),
            "model_ci": _parse_ci_cell(cells[6]),
            "baseline_ci": _parse_ci_cell(cells[7]),
            "wins": cells[8],
            "clears": cells[9],
        }
    return parsed


def test_committed_benchmark_ci_matches_code(ci_rows):
    """docs/BENCHMARK_CI.md must not drift from what the harness produces.

    Numpy models are deterministic everywhere, so values are pinned (labels
    exactly; numbers within the same tight tolerance the benchmark uses).
    """
    assert BENCHMARK_CI_MD.is_file(), (
        "docs/BENCHMARK_CI.md is missing - run: python -m mllab.uncertainty"
    )
    committed = _parse_committed_ci_table(Path(BENCHMARK_CI_MD).read_text(encoding="utf-8"))
    fresh = {r.model: r for r in ci_rows}
    assert set(committed) == set(fresh), "committed CI table and harness disagree on models"

    for name, r in fresh.items():
        c = committed[name]
        assert c["metric"] == r.metric, f"{name}: metric label drifted"
        assert c["better"] == ("higher" if r.higher_is_better else "lower"), (
            f"{name}: direction drifted"
        )
        assert c["units"] == r.n_units, f"{name}: unit count drifted"
        assert c["clears"] == ("yes" if r.ci_excludes_zero else "no"), (
            f"{name}: clears-zero flag drifted"
        )
        assert c["skill"] == pytest.approx(r.skill, abs=_TOL_POINT), f"{name}: point skill drifted"
        assert c["skill_ci"][0] == pytest.approx(r.skill_lo, abs=_TOL_CI), f"{name}: skill CI lo"
        assert c["skill_ci"][1] == pytest.approx(r.skill_hi, abs=_TOL_CI), f"{name}: skill CI hi"
        assert c["model_ci"][0] == pytest.approx(r.model_lo, abs=_TOL_CI), f"{name}: model CI lo"
        assert c["model_ci"][1] == pytest.approx(r.model_hi, abs=_TOL_CI), f"{name}: model CI hi"
        assert c["baseline_ci"][0] == pytest.approx(r.baseline_lo, abs=_TOL_CI), f"{name}: base CI lo"
        assert c["baseline_ci"][1] == pytest.approx(r.baseline_hi, abs=_TOL_CI), f"{name}: base CI hi"

    n_clear = sum(r.ci_excludes_zero for r in ci_rows)
    text = Path(BENCHMARK_CI_MD).read_text(encoding="utf-8")
    assert f"**{n_clear} of {len(ci_rows)} numpy models have a 95% skill CI" in text
