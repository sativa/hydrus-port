"""Morris elementary effects — screening method.

Wraps SALib's morris.sample + morris.analyze. Useful for screening dozens
of parameters cheaply (~10·(D+1) forward calls)."""
from __future__ import annotations
from typing import Callable
import numpy as np

from .result import SensitivityResult
from ._runner import evaluate_samples


def morris_screen(forward: Callable[[np.ndarray], np.ndarray],
                  param_map,
                  obs_names: list[str],
                  n_trajectories: int = 20,
                  num_levels: int = 4,
                  seed: int | None = None,
                  n_workers: int = 1) -> SensitivityResult:
    """Morris elementary effects via SALib.

    `param_map` only needs `.names` and `.bounds_array()` (the duck-typed
    minimum). Pass an internal-coords ParameterMap if your forward expects
    internal coords; pass a user-coords ParameterMap if it expects user."""
    from SALib.sample import morris as morris_sample
    from SALib.analyze import morris as morris_analyze

    bounds = np.asarray(param_map.bounds_array(), dtype=float)
    problem = {
        "num_vars": len(param_map.names),
        "names": list(param_map.names),
        "bounds": bounds.tolist(),
    }
    samples = morris_sample.sample(problem, N=n_trajectories,
                                   num_levels=num_levels, seed=seed)
    ys, wall = evaluate_samples(forward, samples,
                                param_names=list(param_map.names),
                                obs_names=obs_names,
                                n_workers=n_workers)
    # SALib analyses one y vector at a time
    indices: dict[str, list[list[float]]] = {k: [] for k in
                                              ("mu", "mu_star", "sigma", "mu_star_conf")}
    for j in range(ys.shape[1]):
        Si = morris_analyze.analyze(problem, samples, ys[:, j],
                                    num_levels=num_levels, seed=seed)
        for key in indices:
            indices[key].append([float(v) for v in Si[key]])

    return SensitivityResult(
        method="morris",
        param_names=list(param_map.names),
        obs_names=obs_names,
        indices=indices,
        sample_size=samples.shape[0],
        forward_cost_s=wall,
        diagnostics={"n_trajectories": n_trajectories, "num_levels": num_levels},
    )
