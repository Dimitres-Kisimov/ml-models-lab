"""Bootstrap confidence intervals on the shared skill scores (numpy models).

The leaderboard in [`mllab/benchmark.py`](benchmark.py) reports a single **point**
skill score per model. A point estimate answers *"did the model beat its fair
baseline?"* but not *"by how much, and is the margin bigger than evaluation
noise?"* - the question any reviewer asks next. This module puts a
**non-parametric bootstrap confidence interval** on each skill score by
resampling the *evaluation units* and recomputing the metric.

Crucially, it **reuses each model's already-computed test-set predictions** - it
does not retrain, refit, or re-derive anything. Each numpy model's
``train.run(make_plot=False)`` now returns an ``"eval_units"`` payload (the same
predictions it scored); the bootstrap only recomputes the metric on resampled
units, so the fitted model is held fixed. This quantifies **evaluation-set
sampling uncertainty**, not training-seed or model-selection uncertainty.

Scope - **numpy models only** (``churn-rfm-predictor``, ``price-elasticity-
regressor``):

* both are pure numpy and deterministic on every platform, so their point
  metrics, and therefore these bootstrap intervals, are **bit-reproducible
  across OS / BLAS builds** and safe to pin in CI;
* the three torch-trained models are deliberately excluded: their headline
  metric is not bit-reproducible across builds (the same reason
  ``tests/test_benchmark.py`` pins only their labels + beats flag), so a
  committed interval for them would be machine-dependent and could not be
  verified in CI.

The skill score itself is the exact same direction-aware rule the leaderboard
uses - :func:`mllab.benchmark.skill_score` is imported here, not re-implemented.

Method: percentile bootstrap, ``N_BOOT`` resamples with replacement over the
evaluation units, at a fixed ``BOOT_SEED`` (numpy PCG64 - platform independent).
Everything is synthetic and seeded (see ``mllab/synth.py``). Regenerate the
committed table with::

    python -m mllab.uncertainty
"""

from __future__ import annotations

import contextlib
import importlib
import io
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from mllab import metrics
from mllab.benchmark import skill_score

REPO_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_CI_MD = REPO_ROOT / "docs" / "BENCHMARK_CI.md"

# Bootstrap configuration. Fixed so the committed table is byte-stable and the
# tests can pin it. 2000 resamples is the usual floor for a stable 95% interval.
N_BOOT = 2000
BOOT_SEED = 0
CI_LEVEL = 0.95

# One spec per NUMPY model (torch models are intentionally out of scope - see the
# module docstring). ``kind`` selects how a resample of that model's eval_units
# is turned into (model_metric, baseline_metric).
_NUMPY_SPECS: tuple[dict, ...] = (
    {
        "model": "churn-rfm-predictor",
        "run": "mllab.churn_rfm_predictor.train",
        "metric": "PR-AUC",
        "higher_is_better": True,
        "baseline_name": "recency-only",
    },
    {
        "model": "price-elasticity-regressor",
        "run": "mllab.price_elasticity_regressor.train",
        "metric": "elasticity RMSE",
        "higher_is_better": False,
        "baseline_name": "naive OLS",
    },
)


def _rmse(estimate: np.ndarray, target: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(estimate) - np.asarray(target)) ** 2)))


def metric_pair(units: dict, idx: np.ndarray) -> tuple[float, float]:
    """(model_metric, baseline_metric) on the units selected by ``idx``.

    ``idx`` is an index array into the evaluation units; the identity index
    ``arange(n)`` recovers the full-sample point metrics. Supports the two
    numpy-model payload kinds; anything else is a programming error.
    """
    kind = units["kind"]
    if kind == "binary_scores":
        y = np.asarray(units["y_true"])[idx]
        model = metrics.pr_auc(y, np.asarray(units["model_score"])[idx])
        base = metrics.pr_auc(y, np.asarray(units["baseline_score"])[idx])
        return model, base
    if kind == "regression_estimates":
        target = np.asarray(units["target"])[idx]
        model = _rmse(np.asarray(units["model_estimate"])[idx], target)
        base = _rmse(np.asarray(units["baseline_estimate"])[idx], target)
        return model, base
    raise ValueError(f"unknown eval_units kind: {kind!r}")


def _n_units(units: dict) -> int:
    if units["kind"] == "binary_scores":
        return int(len(units["y_true"]))
    return int(len(units["target"]))


def _percentile_ci(values: np.ndarray, ci_level: float) -> tuple[float, float]:
    alpha = (1.0 - ci_level) / 2.0
    lo, hi = np.percentile(values, [100.0 * alpha, 100.0 * (1.0 - alpha)])
    return float(lo), float(hi)


@dataclass(frozen=True)
class CIRow:
    """A model's point skill plus its bootstrap interval on the evaluation set."""

    model: str
    metric: str
    higher_is_better: bool
    baseline_name: str
    n_units: int
    n_boot: int
    n_valid: int
    ci_level: float
    # full-sample point estimates (these match docs/BENCHMARK.md by construction)
    model_value: float
    baseline_value: float
    skill: float
    # percentile-bootstrap intervals
    skill_lo: float
    skill_hi: float
    model_lo: float
    model_hi: float
    baseline_lo: float
    baseline_hi: float
    # share of resamples in which the model still beats its baseline (skill > 0)
    p_beats_baseline: float

    @property
    def ci_excludes_zero(self) -> bool:
        """True when the whole skill CI is above 0 - the margin clears noise."""
        return self.skill_lo > 0.0


def bootstrap_row(
    spec: dict,
    units: dict,
    n_boot: int = N_BOOT,
    seed: int = BOOT_SEED,
    ci_level: float = CI_LEVEL,
) -> CIRow:
    """Percentile bootstrap of one model's skill score over its eval units.

    The fitted model is fixed; only the evaluation units are resampled with
    replacement. A resample that makes a metric undefined (e.g. no positives in a
    PR-AUC resample) yields a non-finite skill and is dropped from the percentile
    estimate; ``n_valid`` records how many resamples survived. Deterministic in
    ``seed`` (numpy PCG64 is platform independent), so the committed table is
    byte-stable and CI-safe to pin.
    """
    higher = bool(spec["higher_is_better"])
    n = _n_units(units)

    # full-sample point (identity resample) -> matches the leaderboard cell
    m0, b0 = metric_pair(units, np.arange(n))
    skill0 = skill_score(m0, b0, higher)

    rng = np.random.default_rng(seed)
    idx_mat = rng.integers(0, n, size=(n_boot, n))
    model_vals = np.empty(n_boot)
    base_vals = np.empty(n_boot)
    skills = np.empty(n_boot)
    for i in range(n_boot):
        m, b = metric_pair(units, idx_mat[i])
        model_vals[i] = m
        base_vals[i] = b
        skills[i] = skill_score(m, b, higher)

    finite = np.isfinite(model_vals) & np.isfinite(base_vals) & np.isfinite(skills)
    n_valid = int(finite.sum())
    m_lo, m_hi = _percentile_ci(model_vals[finite], ci_level)
    b_lo, b_hi = _percentile_ci(base_vals[finite], ci_level)
    s_lo, s_hi = _percentile_ci(skills[finite], ci_level)
    p_beats = float(np.mean(skills[finite] > 0.0))

    return CIRow(
        model=spec["model"],
        metric=spec["metric"],
        higher_is_better=higher,
        baseline_name=spec["baseline_name"],
        n_units=n,
        n_boot=n_boot,
        n_valid=n_valid,
        ci_level=ci_level,
        model_value=m0,
        baseline_value=b0,
        skill=skill0,
        skill_lo=s_lo,
        skill_hi=s_hi,
        model_lo=m_lo,
        model_hi=m_hi,
        baseline_lo=b_lo,
        baseline_hi=b_hi,
        p_beats_baseline=p_beats,
    )


def collect_ci_rows(
    n_boot: int = N_BOOT,
    seed: int = BOOT_SEED,
    ci_level: float = CI_LEVEL,
    quiet: bool = True,
) -> list[CIRow]:
    """Run each numpy model once, reuse its ``eval_units``, and bootstrap it.

    One ``train.run()`` call per model (identical cost to the benchmark harness);
    it trains once and hands back the predictions it already scored. No torch is
    imported here, so this runs in a numpy-only environment.
    """
    rows: list[CIRow] = []
    for spec in _NUMPY_SPECS:
        mod = importlib.import_module(spec["run"])
        if quiet:
            with contextlib.redirect_stdout(io.StringIO()):
                out = mod.run(make_plot=False)
        else:
            out = mod.run(make_plot=False)
        units = out["eval_units"]
        rows.append(bootstrap_row(spec, units, n_boot=n_boot, seed=seed, ci_level=ci_level))
    return rows


def render_markdown(rows: list[CIRow]) -> str:
    """Render the committed CI table. Fixed decimals -> byte-stable output."""
    pct = int(round(100 * (rows[0].ci_level if rows else CI_LEVEL)))
    lines: list[str] = []
    lines.append("# Bootstrap confidence intervals on the skill scores")
    lines.append("")
    lines.append(
        "A companion to the point-estimate leaderboard "
        "[`docs/BENCHMARK.md`](BENCHMARK.md), generated by "
        "[`mllab/uncertainty.py`](../mllab/uncertainty.py) - **no numbers are "
        "typed by hand here**. It puts a non-parametric percentile bootstrap "
        f"{pct}% confidence interval on each model's skill score by resampling "
        "the evaluation units with replacement and recomputing the metric."
    )
    lines.append("")
    lines.append(
        "Only the **two numpy models** appear: they are deterministic on every "
        "platform, so these intervals are bit-reproducible and CI-verifiable. The "
        "three torch-trained models are excluded on purpose - their trained metric "
        "is not bit-reproducible across BLAS/OS builds (the same reason the "
        "benchmark drift guard pins only their labels and beats flag), so a "
        "committed interval for them would be machine-dependent."
    )
    lines.append("")
    lines.append(
        "The fitted model is held **fixed**: the bootstrap reuses each model's "
        "already-computed test-set predictions (nothing is retrained) and only "
        "resamples the evaluation units. So the interval reflects **evaluation-set "
        "sampling uncertainty**, not training-seed or model-selection uncertainty. "
        "The point skill equals the leaderboard's value exactly (the identity "
        "resample); the skill rule is the shared "
        "[`skill_score`](../mllab/benchmark.py) used everywhere in this repo."
    )
    lines.append("")
    lines.append(
        "| Model | Metric | Better | Units | Skill (point) | "
        f"Skill {pct}% CI | Model metric {pct}% CI | Baseline {pct}% CI | "
        "Resamples model wins | CI clears 0? |"
    )
    lines.append("|---|---|---|---:|---:|:-:|:-:|:-:|---:|:-:|")
    for r in rows:
        lines.append(
            f"| `{r.model}` | {r.metric} | "
            f"{'higher' if r.higher_is_better else 'lower'} | {r.n_units} | "
            f"{r.skill:+.3f} | "
            f"[{r.skill_lo:+.3f}, {r.skill_hi:+.3f}] | "
            f"[{r.model_lo:.3f}, {r.model_hi:.3f}] | "
            f"[{r.baseline_lo:.3f}, {r.baseline_hi:.3f}] | "
            f"{100.0 * r.p_beats_baseline:.1f}% | "
            f"{'yes' if r.ci_excludes_zero else 'no'} |"
        )
    lines.append("")
    n_clear = sum(r.ci_excludes_zero for r in rows)
    r0 = rows[0] if rows else None
    n_boot = r0.n_boot if r0 else N_BOOT
    lines.append(
        f"**{n_clear} of {len(rows)} numpy models have a {pct}% skill CI entirely "
        "above zero** - i.e. their advantage over the fair baseline is larger than "
        "evaluation-set sampling noise at this level."
    )
    lines.append("")
    lines.append(
        f"Method: percentile bootstrap, {n_boot} resamples with replacement over "
        f"the evaluation units, bootstrap seed {BOOT_SEED} (numpy PCG64, platform "
        "independent). *Resamples model wins* is the share of resamples in which "
        "the model still beats its baseline (skill > 0); it is a one-sided "
        "bootstrap read on the margin, not a p-value. A resample that leaves a "
        "metric undefined (e.g. no positives in a PR-AUC resample) is dropped "
        "before the percentiles are taken."
    )
    lines.append("")
    for r in rows:
        if r.n_valid != r.n_boot:
            lines.append(
                f"- `{r.model}`: {r.n_valid} of {r.n_boot} resamples were "
                "non-degenerate (the rest had an undefined metric and were dropped)."
            )
    if any(r.n_valid != r.n_boot for r in rows):
        lines.append("")
    lines.append(
        "All data is synthetic and seeded (see [`../mllab/synth.py`](../mllab/synth.py)); "
        "nothing is real, rounded up, or cherry-picked. The skill score is a "
        "**within-model** diagnostic against that model's fair baseline, not a "
        "cross-model ranking. This table is regenerated by `python -m "
        "mllab.uncertainty` and guarded by "
        "[`tests/test_uncertainty.py`](../tests/test_uncertainty.py), which "
        "recomputes every interval and fails if the committed table drifts."
    )
    lines.append("")
    lines.append("*Author: Dimitres Kisimov, 2026.*")
    lines.append("")
    return "\n".join(lines)


def write_benchmark_ci(path: Path = BENCHMARK_CI_MD) -> Path:
    rows = collect_ci_rows()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_markdown(rows), encoding="utf-8", newline="\n")
    return path


def main() -> None:
    import sys

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    rows = collect_ci_rows()
    print("ml-models-lab bootstrap skill CIs (numpy models)")
    print("=" * 72)
    for r in rows:
        clear = "CLEARS 0" if r.ci_excludes_zero else "touches 0"
        print(
            f"[{clear}] {r.model:<28} skill {r.skill:+.3f} "
            f"[{r.skill_lo:+.3f}, {r.skill_hi:+.3f}]  "
            f"model-wins {100.0 * r.p_beats_baseline:.1f}%"
        )
    n_clear = sum(r.ci_excludes_zero for r in rows)
    print("=" * 72)
    print(f"[summary] {n_clear} of {len(rows)} numpy models have a skill CI above zero")
    path = write_benchmark_ci()
    print(f"[ok] wrote {os.path.relpath(path, REPO_ROOT)}")


if __name__ == "__main__":
    main()
