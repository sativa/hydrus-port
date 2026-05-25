import pytest
from hydrus_research.optimization import OptimizationResult


def test_result_construction():
    r = OptimizationResult(method="nsga2",
                           param_names=["a", "b"], objective_names=["o1", "o2"],
                           pareto_thetas=[[0.1, 0.2], [0.3, 0.4]],
                           pareto_objectives=[[1.0, 2.0], [1.5, 1.0]],
                           n_evaluations=200, wall_s=5.0)
    assert r.method == "nsga2"
    assert len(r.pareto_thetas) == 2
