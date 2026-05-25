"""Internal — shared bookkeeping for the four sensitivity methods.

Given a SALib sample matrix `samples (N, D)` and a forward callable,
runs them through BatchRunner and returns `ys (N, M)` ready for SALib
analyzer functions."""
from __future__ import annotations
import time
from typing import Callable
import numpy as np

from ..batch import BatchRunner


def evaluate_samples(forward: Callable[[np.ndarray], np.ndarray],
                     samples: np.ndarray,
                     param_names: list[str],
                     obs_names: list[str],
                     n_workers: int = 1) -> tuple[np.ndarray, float]:
    """Run forward over `samples` and return (ys, wall_s_total).

    Failed runs propagate as NaN rows — SALib analyzers handle these
    by raising; callers should filter the input first or accept the
    failure as a flagged result. This helper does NOT silently drop
    rows."""
    runner = BatchRunner(forward=forward,
                         param_names=param_names,
                         obs_names=obs_names,
                         n_workers=n_workers,
                         show_progress=False)
    t0 = time.time()
    br = runner.run(samples)
    wall = time.time() - t0
    return br.ys, wall
