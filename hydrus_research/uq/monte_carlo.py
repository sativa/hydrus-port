"""Monte Carlo propagation of PTF parameter uncertainty.

Samples N parameter vectors from N(ptf.mean, ptf.covariance), runs forward
once per sample, returns the ensemble + 95% quantile bands."""
from __future__ import annotations
from typing import Callable
import numpy as np

from .result import UQResult


def propagate_ptf_uncertainty(forward: Callable[[np.ndarray], np.ndarray],
                              ptf,
                              param_names: list[str],
                              obs_names: list[str],
                              n: int = 500,
                              seed: int | None = None) -> UQResult:
    """Sample N parameter vectors from the PTF covariance, run forward each
    time, return the ensemble + 95% quantile bands per observable."""
    rng = np.random.default_rng(seed)
    mean = np.array([ptf.theta_r, ptf.theta_s, ptf.alpha, ptf.n, ptf.Ks],
                    dtype=float)
    if ptf.covariance is None:
        raise ValueError("ptf.covariance is None; cannot propagate")
    cov = np.asarray(ptf.covariance, dtype=float)

    samples = rng.multivariate_normal(mean, cov, size=n)

    ys: list[list[float]] = []
    for theta in samples:
        try:
            y = np.asarray(forward(theta), dtype=float)
            ys.append([float(v) for v in y])
        except Exception:
            ys.append([float("nan")] * len(obs_names))

    arr = np.array(ys)
    quantiles = {
        "p2.5":  [float(v) for v in np.nanpercentile(arr,  2.5, axis=0)],
        "p50":   [float(v) for v in np.nanpercentile(arr, 50.0, axis=0)],
        "p97.5": [float(v) for v in np.nanpercentile(arr, 97.5, axis=0)],
    }
    return UQResult(method="ptf_mc", param_names=param_names,
                    obs_names=obs_names, ys=ys, quantiles=quantiles,
                    n_samples=n,
                    diagnostics={"mean": mean.tolist(), "seed": seed})
