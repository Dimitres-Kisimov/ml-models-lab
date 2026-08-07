"""Shared benchmark harness: score all five models under one consistent rule.

The five models solve different tasks and are reported in different natural
metrics (MASE, macro-F1, PR-AUC, elasticity RMSE). This harness reuses each
model's existing ``train.run(make_plot=False)`` - it does **not** retrain
anything new - and folds every model into a single, self-verifying leaderboard:
each model's primary metric, its fair baseline, and a **unified skill score**.

Skill score (per model, direction-aware):

* lower-is-better error metrics (MASE, elasticity RMSE):
  ``skill = 1 - model / baseline`` - the fraction of the fair baseline's error
  the model removes.
* bounded [0, 1] higher-is-better score metrics (macro-F1, PR-AUC):
  ``skill = (model - baseline) / (1 - baseline)`` - the fraction of the
  headroom between the baseline and a perfect score the model captures.

Both conventions read the same way: ``> 0`` beats the baseline, ``0`` ties it,
``< 0`` is worse, ``1`` is perfect. The skill score is a **within-model**
diagnostic against that model's fair baseline; it is deliberately *not* a
cross-model ranking, because the five models answer different questions against
different baselines.

Everything is synthetic and seeded (see ``mllab/synth.py``); each model runs at
its own default seed, so the table reproduces the numbers the individual model
CLIs print. Regenerate the committed table with::

    python -m mllab.benchmark
"""

from __future__ import annotations

import contextlib
import importlib
import io
import os
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_MD = REPO_ROOT / "docs" / "BENCHMARK.md"


def skill_score(model_value: float, baseline_value: float, higher_is_better: bool) -> float:
    """Direction-aware skill of ``model_value`` against ``baseline_value``.

    See the module docstring for the two conventions. Returns ``0.0`` for a
    degenerate baseline (a perfect score baseline, or a zero-error baseline)
    so the harness never divides by zero.
    """
    if higher_is_better:
        headroom = 1.0 - baseline_value
        if headroom <= 0.0:
            return 0.0
        return (model_value - baseline_value) / headroom
    if baseline_value <= 0.0:
        return 0.0
    return 1.0 - model_value / baseline_value


@dataclass(frozen=True)
class BenchmarkRow:
    """One model's benchmark line: its metric vs its fair baseline."""

    model: str
    family: str
    method: str
    metric: str
    higher_is_better: bool
    baseline_name: str
    baseline_value: float
    model_value: float

    @property
    def skill(self) -> float:
        return skill_score(self.model_value, self.baseline_value, self.higher_is_better)

    @property
    def beats_baseline(self) -> bool:
        if self.higher_is_better:
            return self.model_value > self.baseline_value
        return self.model_value < self.baseline_value

    @property
    def direction(self) -> str:
        return "higher" if self.higher_is_better else "lower"


# One spec per model: where its run() lives, and which keys of its result dict
# are the headline model metric and its fair baseline. The fair baseline is the
# *stronger* simple reference in each case (seasonal-naive over Holt-Winters;
# recency over prevalence; the PCA-SVD reconstructor over the AE's own prior),
# so no skill score is inflated by a straw-man baseline.
_SPECS: tuple[dict, ...] = (
    {
        "model": "demand-forecast-net",
        "family": "demand forecasting",
        "method": "global NB-head MLP (DeepAR-lite)",
        "metric": "MASE",
        "higher_is_better": False,
        "baseline_name": "seasonal-naive",
        "run": "mllab.demand_forecast_net.train",
        "model_key": "net_mase",
        "baseline_key": "seasonal_naive_mase",
    },
    {
        "model": "sku-text-classifier",
        "family": "catalogue text",
        "method": "hashed char n-gram fastText",
        "metric": "macro-F1",
        "higher_is_better": True,
        "baseline_name": "majority-class",
        "run": "mllab.sku_text_classifier.train",
        "model_key": "macro_f1",
        "baseline_key": "macro_f1_majority",
    },
    {
        "model": "order-anomaly-ae",
        "family": "anomaly detection",
        "method": "undercomplete autoencoder",
        "metric": "PR-AUC",
        "higher_is_better": True,
        "baseline_name": "PCA-SVD",
        "run": "mllab.order_anomaly_ae.train",
        "model_key": "ae_pr_auc",
        "baseline_key": "pca_pr_auc",
    },
    {
        "model": "churn-rfm-predictor",
        "family": "churn / calibration",
        "method": "calibrated logistic RFM",
        "metric": "PR-AUC",
        "higher_is_better": True,
        "baseline_name": "recency-only",
        "run": "mllab.churn_rfm_predictor.train",
        "model_key": "pr_auc",
        "baseline_key": "pr_auc_recency",
    },
    {
        "model": "price-elasticity-regressor",
        "family": "causal pricing",
        "method": "controlled + shrunk ridge",
        "metric": "elasticity RMSE",
        "higher_is_better": False,
        "baseline_name": "naive OLS",
        "run": "mllab.price_elasticity_regressor.train",
        "model_key": "rmse_shrunk",
        "baseline_key": "rmse_naive_ols",
    },
)


def collect_rows(seed: int | None = None, quiet: bool = True) -> list[BenchmarkRow]:
    """Run every model's ``run(make_plot=False)`` and fold it into a row.

    Reuses the existing training entry points (no new training code). ``seed``
    of ``None`` lets each model use its own default seed. torch is imported
    lazily by three of the five models; a numpy-only environment cannot build
    the full table (the tests skip it there).
    """
    rows: list[BenchmarkRow] = []
    for spec in _SPECS:
        mod = importlib.import_module(spec["run"])
        kwargs: dict = {"make_plot": False}
        if seed is not None:
            kwargs["seed"] = seed
        if quiet:
            with contextlib.redirect_stdout(io.StringIO()):
                out = mod.run(**kwargs)
        else:
            out = mod.run(**kwargs)
        rows.append(
            BenchmarkRow(
                model=spec["model"],
                family=spec["family"],
                method=spec["method"],
                metric=spec["metric"],
                higher_is_better=bool(spec["higher_is_better"]),
                baseline_name=spec["baseline_name"],
                baseline_value=float(out[spec["baseline_key"]]),
                model_value=float(out[spec["model_key"]]),
            )
        )
    return rows


def render_markdown(rows: list[BenchmarkRow]) -> str:
    """Render the committed leaderboard. Fixed decimals -> byte-stable output."""
    n_beat = sum(r.beats_baseline for r in rows)
    lines: list[str] = []
    lines.append("# Shared benchmark leaderboard")
    lines.append("")
    lines.append(
        "One consistent scoring rule across all five models, generated by "
        "[`mllab/benchmark.py`](../mllab/benchmark.py) from each model's own "
        "`train.run()` - **no numbers are typed by hand here**. Every value is "
        "exactly what the code prints at each model's default seed on the seeded "
        "synthetic data in [`../mllab/synth.py`](../mllab/synth.py); nothing is "
        "real, rounded up, or cherry-picked."
    )
    lines.append("")
    lines.append("**Skill score** (direction-aware, against each model's *fair* baseline):")
    lines.append("")
    lines.append(
        "- lower-is-better error metrics (MASE, elasticity RMSE): "
        "`skill = 1 - model / baseline` - the fraction of the baseline's error removed."
    )
    lines.append(
        "- bounded [0, 1] score metrics (macro-F1, PR-AUC): "
        "`skill = (model - baseline) / (1 - baseline)` - the fraction of the headroom "
        "to a perfect score captured."
    )
    lines.append("")
    lines.append(
        "`> 0` beats the baseline, `0` ties, `< 0` is worse, `1` is perfect. The "
        "skill score is a **within-model** diagnostic against that model's fair "
        "baseline - it is *not* a cross-model ranking, because the five models "
        "solve different tasks against different baselines."
    )
    lines.append("")
    lines.append(
        "| Model | Task | Method | Metric | Better | Model | Fair baseline | Baseline | Skill | Beats? |"
    )
    lines.append(
        "|---|---|---|---|---|---:|---|---:|---:|:-:|"
    )
    for r in rows:
        lines.append(
            f"| `{r.model}` | {r.family} | {r.method} | {r.metric} | {r.direction} | "
            f"{r.model_value:.3f} | {r.baseline_name} | {r.baseline_value:.3f} | "
            f"{r.skill:+.3f} | {'yes' if r.beats_baseline else 'no'} |"
        )
    lines.append("")
    lines.append(f"**{n_beat} of {len(rows)} models beat their fair baseline.**")
    lines.append("")
    lines.append(
        "The fair baseline is the *stronger* simple reference in each case "
        "(seasonal-naive over Holt-Winters for the forecaster; recency over "
        "prevalence for churn; the PCA-SVD reconstructor for the anomaly AE), so "
        "no skill score is propped up by a straw-man. Where a model wins only "
        "narrowly - the anomaly autoencoder over PCA - the small skill score "
        "shows it, and the PCA baseline stays in the report."
    )
    lines.append("")
    lines.append(
        "This table is regenerated by `python -m mllab.benchmark` and guarded by "
        "[`tests/test_benchmark.py`](../tests/test_benchmark.py), which recomputes "
        "every cell from the code and fails if the committed table drifts from what "
        "the models actually produce."
    )
    lines.append("")
    lines.append("*Author: Dimitres Kisimov, 2026.*")
    lines.append("")
    return "\n".join(lines)


def write_benchmark(path: Path = BENCHMARK_MD) -> Path:
    rows = collect_rows()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_markdown(rows), encoding="utf-8", newline="\n")
    return path


def main() -> None:
    import sys

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    rows = collect_rows(quiet=True)
    print("ml-models-lab shared benchmark")
    print("=" * 68)
    for r in rows:
        mark = "beats" if r.beats_baseline else "LOSES"
        print(
            f"[{mark}] {r.model:<26} {r.metric:<16} "
            f"model={r.model_value:.3f} vs {r.baseline_name} {r.baseline_value:.3f} "
            f"| skill {r.skill:+.3f}"
        )
    n_beat = sum(r.beats_baseline for r in rows)
    print("=" * 68)
    print(f"[summary] {n_beat} of {len(rows)} models beat their fair baseline")
    path = write_benchmark()
    print(f"[ok] wrote {os.path.relpath(path, REPO_ROOT)}")


if __name__ == "__main__":
    main()
