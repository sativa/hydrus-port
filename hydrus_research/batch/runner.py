"""BatchRunner — orchestrates N forward-model evaluations.

Two backends:
- joblib (default): single-host parallelism via threads/processes
- pyemu_tcp (M3.5): runs as a PEST++ TCP worker that receives thetas from
  a remote master process

This file defines the joblib backend. pyemu_tcp lives in pyemu_worker.py."""
from __future__ import annotations
import os
import time
from typing import Callable, Literal

import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

from .result import BatchResult


def _detect_workers() -> int:
    try:
        return max(1, os.cpu_count() or 1)
    except Exception:
        return 1


class BatchRunner:
    """Run `forward(theta) → y_sim` over many thetas in parallel.

    Parameters
    ----------
    forward : callable (theta_vector: np.ndarray) -> y_vector: np.ndarray
        The narrow-waist callable — typically `make_forward(simulator, ...)`.
    param_names : list[str]
        Names for the columns of `thetas` (one per parameter).
    obs_names : list[str]
        Names for the columns of `ys` (one per observation).
    n_workers : int | "auto"
        Number of parallel workers. "auto" = os.cpu_count(). 1 = serial.
    backend : "joblib" | "pyemu_tcp"
        Parallelism backend. "joblib" is default; "pyemu_tcp" is M3.5.
    show_progress : bool
        Show a tqdm bar (only meaningful when n_workers == 1; joblib backends
        report progress via a different mechanism — see runner internals).
    """

    def __init__(self,
                 forward: Callable[[np.ndarray], np.ndarray],
                 param_names: list[str],
                 obs_names: list[str],
                 n_workers: int | Literal["auto"] = "auto",
                 backend: Literal["joblib", "pyemu_tcp"] = "joblib",
                 show_progress: bool = True):
        self.forward = forward
        self.param_names = list(param_names)
        self.obs_names = list(obs_names)
        self.n_workers = _detect_workers() if n_workers == "auto" else int(n_workers)
        self.backend = backend
        self.show_progress = show_progress

    def _run_one(self, theta: np.ndarray) -> tuple[np.ndarray, float, bool]:
        """Returns (y_vec, wall_s, converged). On failure: NaN ys + converged=False."""
        t0 = time.time()
        try:
            y = np.asarray(self.forward(theta), dtype=float)
            return y, time.time() - t0, True
        except Exception:
            return (np.full(len(self.obs_names), np.nan), time.time() - t0, False)

    def run(self, thetas: np.ndarray) -> BatchResult:
        thetas = np.asarray(thetas, dtype=float)
        if thetas.ndim != 2 or thetas.shape[1] != len(self.param_names):
            raise ValueError(
                f"thetas shape {thetas.shape} incompatible with "
                f"{len(self.param_names)} param_names"
            )
        N = thetas.shape[0]

        if self.backend == "joblib":
            results = self._run_joblib(thetas)
        elif self.backend == "pyemu_tcp":
            raise NotImplementedError(
                "backend='pyemu_tcp' is for worker-mode (see pyemu_worker.py); "
                "use the `hydrus research worker` CLI instead of BatchRunner.run"
            )
        else:
            raise ValueError(f"unknown backend {self.backend!r}")

        ys = np.stack([r[0] for r in results])
        wall_s = np.array([r[1] for r in results])
        converged = np.array([r[2] for r in results], dtype=bool)

        return BatchResult(
            thetas=thetas, ys=ys, wall_s=wall_s, converged=converged,
            param_names=self.param_names, obs_names=self.obs_names,
            meta={"backend": self.backend, "n_workers": self.n_workers,
                  "n_total": N, "n_failed": int((~converged).sum())},
        )

    def _run_joblib(self, thetas: np.ndarray):
        N = thetas.shape[0]
        if self.n_workers <= 1:
            it = tqdm(thetas, total=N, disable=not self.show_progress)
            return [self._run_one(t) for t in it]
        # Parallel path — joblib loky backend works well for CPU- and IO-bound tasks
        with Parallel(n_jobs=self.n_workers, backend="loky") as parallel:
            return parallel(delayed(self._run_one)(t) for t in thetas)
