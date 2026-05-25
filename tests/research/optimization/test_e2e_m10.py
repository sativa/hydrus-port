"""M10 acceptance: NSGA-II + Optuna on synthetic problems."""
import numpy as np
import pytest

pymoo = pytest.importorskip("pymoo")
from hydrus_research.optimization import nsga_optimize, optuna_optimize


def test_nsga_finds_pareto_front_on_zdt1():
    """ZDT1 — classic 2-objective benchmark; Pareto front is the curve
    f2 = 1 - sqrt(f1) for theta[0] in [0, 1] (other thetas = 0)."""
    def zdt1(theta):
        f1 = theta[0]
        g = 1 + 9 * np.sum(theta[1:]) / (len(theta) - 1)
        f2 = g * (1 - np.sqrt(f1 / g))
        return np.array([f1, f2])
    bounds = np.array([[0.0, 1.0]] * 3)
    r = nsga_optimize(forward=zdt1, bounds=bounds,
                      param_names=["x0", "x1", "x2"],
                      objective_names=["f1", "f2"],
                      pop_size=30, n_gen=20, seed=42)
    objs = np.array(r.pareto_objectives)
    # f1 axis should span most of [0, 1]
    assert objs[:, 0].min() < 0.3
    assert objs[:, 0].max() > 0.7


def test_optuna_finds_min_of_quadratic():
    optuna = pytest.importorskip("optuna")
    def f(theta): return float((theta[0] - 2.5) ** 2 + (theta[1] + 1.0) ** 2)
    r = optuna_optimize(forward_scalar=f,
                       bounds=np.array([[-5.0, 5.0], [-5.0, 5.0]]),
                       param_names=["a", "b"], objective_name="q",
                       n_trials=50, seed=11)
    assert abs(r.pareto_thetas[0][0] - 2.5) < 0.5
    assert abs(r.pareto_thetas[0][1] + 1.0) < 0.5
