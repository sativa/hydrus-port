"""Sobol variance-based decomposition.

Wraps SALib's saltelli.sample + sobol.analyze. Cost: (2D+2)·N forward
calls. Use Morris first to screen down to ≤ 10 params, THEN Sobol for
quantitative decomposition."""
from __future__ import annotations
from typing import Callable
import numpy as np

from .result import SensitivityResult
from ._runner import evaluate_samples


def sobol_decompose(forward: Callable[[np.ndarray], np.ndarray],
                    param_map,
                    obs_names: list[str],
                    n_base: int = 1024,
                    calc_second_order: bool = False,
                    seed: int | None = None,
                    n_workers: int = 1) -> SensitivityResult:
    """Sobol decomposition via SALib.

    `param_map` only needs `.names` and `.bounds_array()` (the duck-typed
    minimum). Pass an internal-coords ParameterMap if your forward expects
    internal coords; pass a user-coords ParameterMap if it expects user."""
    from SALib.analyze import sobol

    # Import sobol.sample; handle API transition: saltelli → sobol (SALib 1.5+)
    try:
        from SALib.sample import sobol as sobol_sample
    except ImportError:
        from SALib.sample import saltelli as sobol_sample

    if seed is not None:
        np.random.seed(seed)

    bounds = np.asarray(param_map.bounds_array(), dtype=float)
    problem = {
        "num_vars": len(param_map.names),
        "names": list(param_map.names),
        "bounds": bounds.tolist(),
    }
    samples = sobol_sample.sample(problem, n_base,
                                  calc_second_order=calc_second_order)
    ys, wall = evaluate_samples(forward, samples,
                                param_names=list(param_map.names),
                                obs_names=obs_names,
                                n_workers=n_workers)
    base_keys = ["S1", "S1_conf", "ST", "ST_conf"]
    if calc_second_order:
        base_keys += ["S2", "S2_conf"]
    indices: dict[str, list] = {k: [] for k in base_keys}
    for j in range(ys.shape[1]):
        Si = sobol.analyze(problem, ys[:, j],
                           calc_second_order=calc_second_order,
                           seed=seed)
        for key in base_keys:
            v = Si[key]
            # S2 is a (D, D) matrix; flatten to list-of-lists
            indices[key].append(
                v.tolist() if hasattr(v, "tolist") else list(v))

    return SensitivityResult(
        method="sobol",
        param_names=list(param_map.names),
        obs_names=obs_names,
        indices=indices,
        sample_size=samples.shape[0],
        forward_cost_s=wall,
        diagnostics={"n_base": n_base, "calc_second_order": calc_second_order},
    )
