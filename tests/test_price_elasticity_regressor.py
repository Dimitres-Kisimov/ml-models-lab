import numpy as np

from mllab.price_elasticity_regressor.model import (
    lasso_coordinate_descent,
    optimal_price,
    ridge_closed_form,
    ridge_gd,
)
from mllab.price_elasticity_regressor.train import run


def _toy(seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(200, 3))
    beta = np.array([1.5, -2.0, 0.0])
    y = 0.5 + X @ beta + rng.normal(scale=0.1, size=200)
    return X, y


def test_ridge_gd_approaches_closed_form():
    X, y = _toy()
    cf = ridge_closed_form(X, y, l2=1.0)
    gd = ridge_gd(X, y, l2=1.0, lr=0.05, n_iter=20000)
    assert np.allclose(cf, gd, atol=1e-2)


def test_lasso_zeros_coeffs_under_strong_penalty():
    X, y = _toy()
    weak = lasso_coordinate_descent(X, y, l1=1e-4)
    strong = lasso_coordinate_descent(X, y, l1=100.0)
    assert np.count_nonzero(np.abs(strong[1:]) > 1e-6) == 0
    assert np.count_nonzero(np.abs(weak[1:]) > 1e-6) >= 2


def test_optimal_price_guard_for_inelastic():
    # |e| <= 1 has no interior optimum -> returns the cap
    assert optimal_price(10.0, -0.8, price_cap=999.0) == 999.0
    # elastic case matches the Lerner formula
    assert abs(optimal_price(10.0, -2.0) - 20.0) < 1e-6


def test_controls_recover_true_elasticity():
    out = run(make_plot=False)
    assert abs(out["bias_controlled"]) < abs(out["bias_naive"])
    assert out["rmse_controlled"] < out["rmse_naive_ols"]
