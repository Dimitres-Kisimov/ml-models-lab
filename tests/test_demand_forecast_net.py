import numpy as np
import pytest

from mllab import synth
from mllab.demand_forecast_net.features import (
    FEATURE_NAMES,
    build_dataset,
    croston_states,
)


def test_croston_states_are_causal():
    series = np.array([0, 0, 5, 0, 3, 0, 0, 4])
    z, p = croston_states(series)
    # state at t uses only data strictly before t: first two are the initial state
    assert z[0] == 0.0 and z[1] == 0.0
    # after the first demand at index 2, state at index 3 is updated
    assert z[3] > 0.0
    assert len(z) == len(series) and len(p) == len(series)


def test_build_dataset_shapes():
    y, _ = synth.make_demand_panel(0)
    X, sku, target, time = build_dataset(y, max_t=41)
    assert X.shape[1] == len(FEATURE_NAMES)
    assert len(sku) == len(target) == len(time) == X.shape[0]
    assert time.min() >= 13  # needs lag 13


def test_net_beats_seasonal_naive_and_holt_winters():
    pytest.importorskip("torch")
    from mllab.demand_forecast_net.train import run

    out = run(make_plot=False)
    assert out["net_mase"] < out["seasonal_naive_mase"]
    assert out["net_mase"] < out["holt_winters_mase"]
    assert out["beats_both_baselines"] is True
