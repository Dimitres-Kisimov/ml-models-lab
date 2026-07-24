import numpy as np

from mllab import synth


def test_generators_are_deterministic():
    y1, m1 = synth.make_demand_panel(0)
    y2, m2 = synth.make_demand_panel(0)
    assert np.array_equal(y1, y2)
    assert m1.equals(m2)

    assert synth.make_sku_text(0).equals(synth.make_sku_text(0))

    x1, l1, _ = synth.make_orders(0)
    x2, l2, _ = synth.make_orders(0)
    assert np.array_equal(x1, x2) and np.array_equal(l1, l2)

    assert synth.make_customers(0).equals(synth.make_customers(0))
    assert synth.make_transactions(0).equals(synth.make_transactions(0))


def test_different_seeds_differ():
    y1, _ = synth.make_demand_panel(0)
    y2, _ = synth.make_demand_panel(1)
    assert not np.array_equal(y1, y2)


def test_shapes_and_prevalences():
    y, meta = synth.make_demand_panel(0)
    assert y.shape == (200, 42)
    assert 0.10 < meta["is_intermittent"].mean() < 0.40

    X, lab, names = synth.make_orders(0)
    assert X.shape == (5000, 7) and len(names) == 7
    assert 0.02 < lab.mean() < 0.05  # ~3% anomalies

    cust = synth.make_customers(0)
    assert len(cust) == 4000
    assert 0.10 < cust["churn"].mean() < 0.20  # ~15% churn


def test_transactions_have_confounder_and_truth():
    df = synth.make_transactions(0)
    for col in ("log_price", "log_qty", "z", "true_elasticity", "mc", "true_a"):
        assert col in df.columns
    assert (df["true_elasticity"] < -1.0).all()  # elastic by construction
