"""Posterior predictive — reuse F3 posterior ensemble (no new sampling).

For each member of inv_result.posterior_ensemble, run forward once and
collect the predicted y."""
from __future__ import annotations
from typing import Callable
import numpy as np

from .result import UQResult


def predict_with_posterior(forward: Callable[[np.ndarray], np.ndarray],
                           inv_result,
                           obs_names: list[str]) -> UQResult:
    if inv_result.posterior_ensemble is None:
        backend = getattr(inv_result, "backend", "unknown")
        raise ValueError(
            f"inv_result has no posterior_ensemble (backend={backend!r}); "
            "LM doesn't produce one — use IES or PyMC."
        )
    posterior = np.asarray(inv_result.posterior_ensemble, dtype=float)
    ys: list[list[float]] = []
    for theta in posterior:
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
    return UQResult(method="posterior_predict",
                    param_names=list(inv_result.posterior_param_names),
                    obs_names=obs_names, ys=ys, quantiles=quantiles,
                    n_samples=len(posterior),
                    diagnostics={"source_backend": inv_result.backend})
