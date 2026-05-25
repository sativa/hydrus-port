"""pymoo NSGA-II / NSGA-III multi-objective optimization."""
from __future__ import annotations
import time
from typing import Callable
import numpy as np

from .result import OptimizationResult


def nsga_optimize(forward: Callable[[np.ndarray], np.ndarray],
                  bounds: np.ndarray,
                  param_names: list[str],
                  objective_names: list[str],
                  pop_size: int = 50,
                  n_gen: int = 20,
                  seed: int | None = None,
                  variant: str = "nsga2") -> OptimizationResult:
    try:
        from pymoo.core.problem import Problem
        from pymoo.algorithms.moo.nsga2 import NSGA2
        from pymoo.algorithms.moo.nsga3 import NSGA3
        from pymoo.optimize import minimize
        from pymoo.util.ref_dirs import get_reference_directions
    except ImportError as e:
        raise ImportError(
            "nsga_optimize requires pymoo. Install with:\n"
            "    pip install 'hydrus-port[research,research-opt]'"
        ) from e

    bounds = np.asarray(bounds, dtype=float)
    D = bounds.shape[0]
    M = len(objective_names)

    class _UserProblem(Problem):
        def __init__(self):
            super().__init__(n_var=D, n_obj=M, xl=bounds[:, 0], xu=bounds[:, 1])

        def _evaluate(self, X, out, *args, **kwargs):
            F = np.array([forward(x) for x in X])
            out["F"] = F

    if variant == "nsga3":
        ref_dirs = get_reference_directions("das-dennis", M, n_partitions=12)
        algo = NSGA3(pop_size=pop_size, ref_dirs=ref_dirs)
    else:
        algo = NSGA2(pop_size=pop_size)

    t0 = time.time()
    res = minimize(_UserProblem(), algo,
                   termination=("n_gen", n_gen),
                   seed=seed, verbose=False)
    wall = time.time() - t0

    pareto_thetas = res.X.tolist() if res.X is not None else []
    pareto_objs = res.F.tolist() if res.F is not None else []
    return OptimizationResult(
        method=variant,                      # type: ignore[arg-type]
        param_names=param_names,
        objective_names=objective_names,
        pareto_thetas=pareto_thetas,
        pareto_objectives=pareto_objs,
        n_evaluations=int(pop_size * n_gen),
        wall_s=float(wall),
        diagnostics={"pop_size": pop_size, "n_gen": n_gen},
    )
