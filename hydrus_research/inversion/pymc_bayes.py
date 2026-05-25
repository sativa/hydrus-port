"""PyMC Bayesian inversion via SMC (Sequential Monte Carlo) — P1 backend.

Wraps a generic black-box forward(theta: ndarray) -> ndarray callable as a
pm.Potential log-likelihood using a pytensor black-box Op (pytensor.wrap_py).
Runs SMC via pm.sample_smc — no autodiff/gradients required, compatible with
any numpy-based forward.

PyMC version: 6.x (tested 6.0.1)
Sampler used: pm.sample_smc with IMH kernel (default). NUTS is NOT used
because the forward is a black-box numpy callable without autodiff support.
The backend name is kept "pymc_nuts" for API consistency with the plan.

Adaptation from plan:
  - Plan suggests NUTS; we use SMC (no gradient required).
  - Plan shows pm.math.stack; we use pt.stack from pytensor.tensor.
  - Plan shows as_op; we use pytensor.wrap_py (newer non-deprecated API).
  - PyMC 6 returns a DataTree from sample_smc; we access .posterior group.
  - arviz 1.1 summary uses 'r_hat' and 'ess_bulk' column names (confirmed).
"""
from __future__ import annotations
import time
import warnings
from typing import Callable
import numpy as np


def fit_pymc(forward: Callable[[np.ndarray], np.ndarray],
             param_map,
             obs,
             draws: int = 1000,
             tune: int = 1000,
             chains: int = 2,
             seed: int | None = None) -> "InversionResult":
    """Bayesian inversion via SMC (black-box, no gradients needed).

    Uses uniform priors on each parameter's bounds by default; if a
    ParameterSpec has prior_mean + prior_std, switches to a truncated
    normal prior on that parameter (truncated to bounds).

    Parameters
    ----------
    forward : callable
        Black-box forward model: theta (1-D ndarray, length D) -> y (1-D ndarray).
    param_map : ParameterMap
        Describes parameter names, bounds, and optional prior moments.
    obs : ObservationSet
        Observed values and per-observation standard deviations.
    draws : int
        Number of posterior draws per chain (= SMC particle count).
    tune : int
        Ignored for SMC (kept for API compatibility with NUTS signature).
    chains : int
        Number of independent SMC chains.
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    InversionResult
        backend="pymc_nuts", posterior_ensemble filled, diagnostics include
        r_hat and ess_bulk per parameter.
    """
    try:
        import pymc as pm
        import arviz as az
        import pytensor
        import pytensor.tensor as pt
    except ImportError as e:
        raise ImportError(
            "fit_pymc requires pymc + arviz. Install with:\n"
            "    pip install pymc arviz"
        ) from e

    from .base import InversionResult

    par_names = list(param_map.names)
    bounds = param_map.bounds_array()          # shape (D, 2)
    obs_values = np.asarray(obs.values, dtype=float)
    obs_sigmas = np.asarray(obs.sigmas, dtype=float)

    # Build a pytensor black-box Op wrapping the forward callable.
    # wrap_py is the non-deprecated replacement for as_op in pytensor >= 2.x.
    @pytensor.wrap_py(itypes=[pt.dvector], otypes=[pt.dvector])
    def _forward_op(theta_val):
        return np.asarray(forward(theta_val), dtype=float)

    t0 = time.time()

    # Suppress numba object-mode warning (expected for black-box Ops).
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Numba will use object mode",
            category=UserWarning,
        )

        with pm.Model() as _model:
            # Build priors per ParameterSpec.
            theta_rvs = []
            for spec, (lo, hi) in zip(param_map.specs, bounds):
                lo, hi = float(lo), float(hi)
                if spec.prior_mean is not None and spec.prior_std is not None:
                    rv = pm.TruncatedNormal(
                        spec.name,
                        mu=float(spec.prior_mean),
                        sigma=float(spec.prior_std),
                        lower=lo,
                        upper=hi,
                    )
                else:
                    rv = pm.Uniform(spec.name, lower=lo, upper=hi)
                theta_rvs.append(rv)

            # Stack parameters into a single dvector for the forward Op.
            theta_vec = pt.stack(theta_rvs)

            # Compute forward model output via black-box Op.
            sim = _forward_op(theta_vec)

            # Gaussian log-likelihood as a Potential term.
            loglik = -0.5 * pt.sum(((sim - obs_values) / obs_sigmas) ** 2)
            pm.Potential("obs_loglik", loglik)

            # SMC sampling — no gradients required.
            # cores=1 forces sequential chain execution to avoid multiprocessing
            # serialisation limitations with local wrap_py Ops.
            idata = pm.sample_smc(
                draws=draws,
                chains=chains,
                cores=1,
                random_seed=seed,
                progressbar=False,
            )

    wall = time.time() - t0

    # Extract posterior samples.
    # PyMC 6 returns a DataTree; .posterior is an xarray Dataset.
    posterior_ds = idata.posterior  # dims: (chain, draw)
    # Shape: (D, chains*draws) -> transpose to (chains*draws, D)
    arr = np.stack(
        [posterior_ds[name].values.flatten() for name in par_names],
        axis=1,
    )
    posterior_list = [[float(v) for v in row] for row in arr]

    # Compute diagnostics via arviz.
    # Suppress low-ESS warnings for small draw counts.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        summary = az.summary(idata, var_names=par_names)

    r_hat = {name: float(summary.loc[name, "r_hat"]) for name in par_names}
    ess_bulk = {name: float(summary.loc[name, "ess_bulk"]) for name in par_names}

    best = {name: float(arr[:, j].mean()) for j, name in enumerate(par_names)}
    ci_lo = {
        name: float(np.percentile(arr[:, j], 2.5))
        for j, name in enumerate(par_names)
    }
    ci_hi = {
        name: float(np.percentile(arr[:, j], 97.5))
        for j, name in enumerate(par_names)
    }

    return InversionResult(
        backend="pymc_nuts",
        best_params=best,
        parameter_ci_lo=ci_lo,
        parameter_ci_hi=ci_hi,
        posterior_ensemble=posterior_list,
        posterior_param_names=par_names,
        n_forward_calls=int(draws * chains),
        wall_s=float(wall),
        diagnostics={
            "r_hat": r_hat,
            "ess_bulk": ess_bulk,
            "draws": draws,
            "tune": tune,
            "chains": chains,
            "sampler": "smc_imh",
        },
    )
