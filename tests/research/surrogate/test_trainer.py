import numpy as np
from hydrus_research.surrogate import train_gp, evaluate
from hydrus_research.batch import BatchResult


def _toy_batch(N=40, seed=42):
    rng = np.random.default_rng(seed)
    thetas = rng.uniform(-np.pi, np.pi, size=(N, 2))
    ys = np.column_stack([np.sin(thetas[:, 0]) + np.cos(thetas[:, 1])])
    return BatchResult(thetas=thetas, ys=ys, wall_s=np.zeros(N),
                       converged=np.ones(N, dtype=bool),
                       param_names=["x", "y"], obs_names=["f"], meta={})


def test_train_gp_then_evaluate():
    train = _toy_batch(40, seed=1)
    test = _toy_batch(20, seed=2)
    surr = train_gp(train)
    m = evaluate(surr, test)
    assert "NSE" in m
    assert m["NSE"][0] > 0.5
