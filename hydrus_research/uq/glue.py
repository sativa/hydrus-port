"""GLUE — Generalised Likelihood Uncertainty Estimation (Beven & Binley 1992).

Filter an existing BatchResult: compute Gaussian likelihood L_i = exp(-0.5
* Σ((y_sim_i - obs)/sigma)²) for each row; keep rows with L_i ≥ cutoff *
L_max; renormalise weights to sum to 1; return UQResult with weights +
weighted quantile bands."""
from __future__ import annotations
import numpy as np

from .result import UQResult


def glue_filter(batch_result,
                obs_values: np.ndarray,
                obs_sigmas: np.ndarray,
                likelihood_cutoff: float = 0.5) -> UQResult:
    obs_values = np.asarray(obs_values, dtype=float)
    obs_sigmas = np.asarray(obs_sigmas, dtype=float)
    ys = batch_result.ys
    if ys.shape[1] != obs_values.shape[0]:
        raise ValueError(f"obs_values has shape {obs_values.shape}; "
                         f"batch_result.ys has {ys.shape[1]} observables")

    # Per-row negative log-likelihood (Gaussian)
    z = (ys - obs_values) / obs_sigmas
    nll = 0.5 * np.nansum(z * z, axis=1)
    L = np.exp(-(nll - np.nanmin(nll)))      # normalise to max=1 for stability

    # Behavioral filter
    keep = L >= likelihood_cutoff * np.nanmax(L)
    keep_idx = np.flatnonzero(keep)
    if keep_idx.size == 0:
        raise ValueError(f"no rows pass cutoff={likelihood_cutoff}; "
                         "lower it or relax obs_sigmas")
    ys_kept = ys[keep_idx]
    weights = L[keep_idx]
    weights = weights / weights.sum()        # renormalise to sum=1

    # Weighted quantiles
    def _wq(col, q):
        order = np.argsort(col)
        cw = np.cumsum(weights[order])
        return float(np.interp(q, cw / cw[-1], col[order]))

    quantiles = {"p2.5": [], "p50": [], "p97.5": []}
    for j in range(ys_kept.shape[1]):
        col = ys_kept[:, j]
        quantiles["p2.5"].append(_wq(col, 0.025))
        quantiles["p50"].append(_wq(col, 0.50))
        quantiles["p97.5"].append(_wq(col, 0.975))

    return UQResult(method="glue",
                    param_names=list(batch_result.param_names),
                    obs_names=list(batch_result.obs_names),
                    ys=[[float(v) for v in row] for row in ys_kept],
                    weights=[float(w) for w in weights],
                    quantiles=quantiles,
                    n_samples=int(keep.sum()),
                    diagnostics={"cutoff": likelihood_cutoff,
                                 "n_total": int(ys.shape[0]),
                                 "n_kept": int(keep.sum())})
