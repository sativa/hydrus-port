"""PAWN distribution-based sensitivity — non-parametric, robust to
non-linear / non-monotonic responses."""
from __future__ import annotations
from typing import Callable
import numpy as np

from .result import SensitivityResult
from ._runner import evaluate_samples


def pawn_kde(forward: Callable[[np.ndarray], np.ndarray],
             param_map,
             obs_names: list[str],
             n: int = 2000,
             s: int = 10,
             seed: int | None = None,
             n_workers: int = 1) -> SensitivityResult:
    from SALib.sample import latin
    from SALib.analyze import pawn

    bounds = np.asarray(param_map.bounds_array(), dtype=float)
    problem = {
        "num_vars": len(param_map.names),
        "names": list(param_map.names),
        "bounds": bounds.tolist(),
    }
    samples = latin.sample(problem, n, seed=seed)
    ys, wall = evaluate_samples(forward, samples,
                                param_names=list(param_map.names),
                                obs_names=obs_names,
                                n_workers=n_workers)
    keys = ["minimum", "mean", "median", "maximum", "CV"]
    indices: dict[str, list[list[float]]] = {k: [] for k in keys}
    for j in range(ys.shape[1]):
        Si = pawn.analyze(problem, samples, ys[:, j], S=s)
        for key in keys:
            indices[key].append([float(v) for v in Si[key]])

    return SensitivityResult(
        method="pawn",
        param_names=list(param_map.names),
        obs_names=obs_names,
        indices=indices,
        sample_size=samples.shape[0],
        forward_cost_s=wall,
        diagnostics={"n": n, "S": s},
    )
