"""FAST / eFAST Fourier-based sensitivity indices."""
from __future__ import annotations
from typing import Callable
import numpy as np

from .result import SensitivityResult
from ._runner import evaluate_samples


def fast_indices(forward: Callable[[np.ndarray], np.ndarray],
                 param_map,
                 obs_names: list[str],
                 n: int = 1024,
                 m: int = 4,
                 seed: int | None = None,
                 n_workers: int = 1) -> SensitivityResult:
    from SALib.sample import fast_sampler
    from SALib.analyze import fast

    bounds = np.asarray(param_map.bounds_array(), dtype=float)
    problem = {
        "num_vars": len(param_map.names),
        "names": list(param_map.names),
        "bounds": bounds.tolist(),
    }
    samples = fast_sampler.sample(problem, N=n, M=m, seed=seed)
    ys, wall = evaluate_samples(forward, samples,
                                param_names=list(param_map.names),
                                obs_names=obs_names,
                                n_workers=n_workers)
    indices: dict[str, list[list[float]]] = {"S1": [], "ST": []}
    for j in range(ys.shape[1]):
        Si = fast.analyze(problem, ys[:, j], M=m)
        indices["S1"].append([float(v) for v in Si["S1"]])
        indices["ST"].append([float(v) for v in Si["ST"]])

    return SensitivityResult(
        method="fast",
        param_names=list(param_map.names),
        obs_names=obs_names,
        indices=indices,
        sample_size=samples.shape[0],
        forward_cost_s=wall,
        diagnostics={"n": n, "m": m},
    )
