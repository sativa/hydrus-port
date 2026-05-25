import numpy as np
import pytest

pymoo = pytest.importorskip("pymoo", reason="pymoo not installed; in [research-opt]")
from hydrus_research.optimization import nsga_optimize, OptimizationResult


def test_nsga_on_toy_problem():
    """Minimise two conflicting objectives:
       f1 = (theta[0] - 1)^2 + (theta[1] - 2)^2
       f2 = (theta[0] + 1)^2 + (theta[1] + 2)^2
    Pareto front is a non-trivial curve in objective space."""
    def fwd(theta):
        f1 = (theta[0] - 1) ** 2 + (theta[1] - 2) ** 2
        f2 = (theta[0] + 1) ** 2 + (theta[1] + 2) ** 2
        return np.array([f1, f2])
    bounds = np.array([[-3.0, 3.0], [-3.0, 3.0]])
    r = nsga_optimize(forward=fwd, bounds=bounds,
                      param_names=["x", "y"], objective_names=["f1", "f2"],
                      pop_size=20, n_gen=10, seed=42)
    assert isinstance(r, OptimizationResult)
    assert r.method in ("nsga2", "nsga3")
    assert len(r.pareto_thetas) >= 5
    # Pareto front: non-dominated points
    objs = np.array(r.pareto_objectives)
    assert objs.shape[1] == 2
