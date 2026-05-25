import numpy as np
import pytest

optuna = pytest.importorskip("optuna", reason="optuna not installed; in [research-opt]")
from hydrus_research.optimization import optuna_optimize, OptimizationResult


def test_optuna_finds_min_of_parabola():
    def f(theta): return float((theta[0] - 2.5) ** 2)
    bounds = np.array([[-5.0, 5.0]])
    r = optuna_optimize(forward_scalar=f, bounds=bounds,
                        param_names=["a"], objective_name="quadratic",
                        n_trials=30, seed=42)
    assert isinstance(r, OptimizationResult)
    assert r.method.startswith("optuna_")
    # Best should be near 2.5
    assert abs(r.pareto_thetas[0][0] - 2.5) < 0.5
