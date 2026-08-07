"""Tests for the shared benchmark harness.

Two layers:

* Pure, torch-free unit tests of the scoring rule and the rendering (these run
  anywhere, offline, instantly).
* A full-harness drift guard: run all five models once, recompute every cell,
  and assert the committed ``docs/BENCHMARK.md`` still matches what the code
  produces (torch-gated, like the other model tests).
"""

from pathlib import Path

import numpy as np
import pytest

from mllab.benchmark import (
    BENCHMARK_MD,
    BenchmarkRow,
    render_markdown,
    skill_score,
)

# The three models whose headline metric is TRAINED with torch. A trained
# metric is not bit-reproducible across BLAS builds / operating systems (the
# committed numbers come from one machine), so the drift guard pins their
# labels + the beats-baseline flag but NOT the raw value - otherwise CI on a
# foreign torch build would fail with no code change. The two numpy models are
# deterministic everywhere and stay fully value-pinned.
_TORCH_MODELS = frozenset(
    {"demand-forecast-net", "sku-text-classifier", "order-anomaly-ae"}
)

# --------------------------------------------------------------------------- #
# pure scoring-rule unit tests (no torch, no data)
# --------------------------------------------------------------------------- #


def test_skill_lower_is_better_removes_baseline_error():
    # halve the baseline error -> skill 0.5
    assert skill_score(0.5, 1.0, higher_is_better=False) == pytest.approx(0.5)
    # perfect (zero error) -> skill 1
    assert skill_score(0.0, 1.0, higher_is_better=False) == pytest.approx(1.0)


def test_skill_higher_is_better_captures_headroom():
    # baseline 0.2, model 0.6 -> (0.6-0.2)/(1-0.2) = 0.5
    assert skill_score(0.6, 0.2, higher_is_better=True) == pytest.approx(0.5)
    # perfect score -> skill 1
    assert skill_score(1.0, 0.2, higher_is_better=True) == pytest.approx(1.0)


def test_skill_ties_are_zero_and_worse_is_negative():
    assert skill_score(0.4, 0.4, higher_is_better=True) == pytest.approx(0.0)
    assert skill_score(0.4, 0.4, higher_is_better=False) == pytest.approx(0.0)
    assert skill_score(0.3, 0.5, higher_is_better=True) < 0.0  # below baseline
    assert skill_score(1.2, 1.0, higher_is_better=False) < 0.0  # more error


def test_skill_handles_degenerate_baseline_without_dividing_by_zero():
    assert skill_score(0.99, 1.0, higher_is_better=True) == 0.0  # no headroom
    assert skill_score(0.5, 0.0, higher_is_better=False) == 0.0  # zero-error baseline


def test_row_direction_and_beats_flag():
    hi = BenchmarkRow("m", "fam", "meth", "PR-AUC", True, "base", 0.30, 0.65)
    assert hi.direction == "higher" and hi.beats_baseline is True
    assert hi.skill == pytest.approx((0.65 - 0.30) / (1 - 0.30))

    lo = BenchmarkRow("m", "fam", "meth", "MASE", False, "base", 1.10, 0.99)
    assert lo.direction == "lower" and lo.beats_baseline is True

    lose = BenchmarkRow("m", "fam", "meth", "PR-AUC", True, "base", 0.50, 0.40)
    assert lose.beats_baseline is False and lose.skill < 0.0


def test_render_is_deterministic_and_ascii():
    rows = [
        BenchmarkRow("a-model", "fam", "meth", "MASE", False, "naive", 1.08, 0.98),
        BenchmarkRow("b-model", "fam2", "meth2", "PR-AUC", True, "prev", 0.20, 0.66),
    ]
    first = render_markdown(rows)
    second = render_markdown(rows)
    assert first == second  # byte-stable given the same rows
    assert first.isascii()  # no unicode arrows / smart quotes -> stable on any OS
    assert "1 of 2 models beat their fair baseline" not in first  # both beat here
    assert "2 of 2 models beat their fair baseline" in first


# --------------------------------------------------------------------------- #
# full-harness drift guard (torch-gated: 3 of the 5 models need torch)
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def collected_rows():
    pytest.importorskip("torch")
    from mllab.benchmark import collect_rows

    return collect_rows()


def test_harness_scores_all_five_models(collected_rows):
    names = [r.model for r in collected_rows]
    assert len(collected_rows) == 5
    assert len(set(names)) == 5  # no duplicates
    for r in collected_rows:
        assert np.isfinite(r.model_value) and np.isfinite(r.baseline_value)
        assert np.isfinite(r.skill)


def test_every_model_beats_its_fair_baseline(collected_rows):
    losers = [r.model for r in collected_rows if not r.beats_baseline]
    assert not losers, f"models that failed to beat their fair baseline: {losers}"
    # a positive skill score must agree with the beats flag, by construction
    for r in collected_rows:
        assert (r.skill > 0.0) == r.beats_baseline


def test_harness_is_deterministic(collected_rows):
    pytest.importorskip("torch")
    from mllab.benchmark import collect_rows

    again = collect_rows()
    assert [r.model for r in again] == [r.model for r in collected_rows]
    for a, b in zip(again, collected_rows, strict=True):
        # same machine, fixed seeds -> byte-identical metric values
        assert a.model_value == b.model_value
        assert a.baseline_value == b.baseline_value


def _parse_committed_table(text: str) -> dict[str, dict]:
    """Parse the data rows of docs/BENCHMARK.md into {model: {...}}."""
    parsed: dict[str, dict] = {}
    for line in text.splitlines():
        s = line.strip()
        if not s.startswith("| `"):  # only the backticked model data rows
            continue
        cells = [c.strip() for c in s.strip("|").split("|")]
        model = cells[0].strip("`")
        parsed[model] = {
            "metric": cells[3],
            "better": cells[4],
            "model_value": float(cells[5]),
            "baseline_name": cells[6],
            "baseline_value": float(cells[7]),
            "skill": float(cells[8]),
            "beats": cells[9],
        }
    return parsed


def test_committed_benchmark_matches_code(collected_rows):
    """docs/BENCHMARK.md must not drift from what the models actually produce.

    Labels/structure are compared exactly; numbers within a tight tolerance so
    the guard is robust to <1e-3 float differences from a different torch build
    (the committed values are rounded to 3 decimals) while still catching any
    real regression in a model.
    """
    assert BENCHMARK_MD.is_file(), "docs/BENCHMARK.md is missing - run: python -m mllab.benchmark"
    text = Path(BENCHMARK_MD).read_text(encoding="utf-8")
    committed = _parse_committed_table(text)

    fresh = {r.model: r for r in collected_rows}
    assert set(committed) == set(fresh), "committed table and harness disagree on which models exist"

    for name, r in fresh.items():
        c = committed[name]
        assert c["metric"] == r.metric, f"{name}: metric label drifted"
        assert c["better"] == r.direction, f"{name}: direction drifted"
        assert c["baseline_name"] == r.baseline_name, f"{name}: baseline label drifted"
        assert c["beats"] == ("yes" if r.beats_baseline else "no"), f"{name}: beats flag drifted"
        if name in _TORCH_MODELS:
            # trained-with-torch metric: not bit-reproducible across BLAS/OS, so
            # the labels + beats flag above are the drift guard, not the raw
            # number (a broken model stops beating its baseline -> caught above).
            continue
        # numpy models are deterministic everywhere -> pin their values tightly.
        assert c["model_value"] == pytest.approx(r.model_value, abs=5e-3), f"{name}: model metric drifted"
        assert c["baseline_value"] == pytest.approx(r.baseline_value, abs=5e-3), f"{name}: baseline drifted"
        assert c["skill"] == pytest.approx(r.skill, abs=5e-3), f"{name}: skill score drifted"

    # the headline count must match the rows
    n_beat = sum(r.beats_baseline for r in collected_rows)
    assert f"**{n_beat} of {len(collected_rows)} models beat their fair baseline.**" in text
