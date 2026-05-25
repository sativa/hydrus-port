"""Scipy Levenberg-Marquardt — the in-process fast inversion path.

Uses scipy.optimize.least_squares with the 'trf' (trust-region) variant,
which handles bounded problems more robustly than pure LM."""
from __future__ import annotations
import time
from typing import Callable
import numpy as np

from .base import InversionResult


def fit_lm(forward: Callable[[np.ndarray], np.ndarray],
           param_map,
           obs,
           x0: np.ndarray | None = None,
           max_nfev: int = 200,
           diff_step: float | None = None) -> InversionResult:
    """LM-fit `forward(theta) → y_sim` to `obs.values` via least_squares.

    Returns InversionResult with best_params + jacobian-derived CIs.

    `diff_step` controls the finite-difference step size for the jacobian.
    When the forward simulator outputs rounded values (e.g. HYDRUS-1D text
    output precision = 4 decimal places), the default scipy step of ~1.5e-8
    may be below the resolution floor — set diff_step=0.05 (5% in parameter
    space) to ensure the gradient is detectable."""
    from scipy.optimize import least_squares

    if x0 is None:
        x0 = param_map.midpoints()

    bounds = param_map.bounds_array()
    history: list[float] = []
    nfev = {"count": 0}

    def residuals(theta: np.ndarray) -> np.ndarray:
        nfev["count"] += 1
        sim = forward(theta)
        r = obs.residuals(sim)
        history.append(float(np.sum(r * r)))
        return r

    t0 = time.time()
    res = least_squares(
        residuals, x0=np.asarray(x0, dtype=float),
        bounds=(bounds[:, 0], bounds[:, 1]),
        method="trf", jac="2-point", x_scale="jac",
        max_nfev=max_nfev,
        **({"diff_step": diff_step} if diff_step is not None else {}),
    )
    wall = time.time() - t0

    best_named = param_map.from_vector(res.x)

    # Jacobian-based ±1σ CIs via the pseudo-inverse of J^T J
    ci_lo: dict[str, float] = {}
    ci_hi: dict[str, float] = {}
    try:
        J = res.jac
        if J is not None and J.size > 0:
            JTJ = J.T @ J
            cov = np.linalg.pinv(JTJ)
            sigmas = np.sqrt(np.maximum(np.diag(cov), 0.0))
            for spec, sigma in zip(param_map.specs, sigmas):
                mean_user = best_named[spec.name]
                if spec.transform == "log":
                    su = mean_user * sigma
                elif spec.transform == "logit":
                    lo_b, hi_b = spec.bounds
                    range_w = hi_b - lo_b
                    u = (mean_user - lo_b) / range_w
                    su = u * (1 - u) * range_w * sigma
                else:
                    su = sigma
                ci_lo[spec.name] = float(mean_user - su)
                ci_hi[spec.name] = float(mean_user + su)
    except (np.linalg.LinAlgError, ValueError):
        pass

    return InversionResult(
        backend="lm_scipy",
        best_params={k: float(v) for k, v in best_named.items()},
        parameter_ci_lo=ci_lo,
        parameter_ci_hi=ci_hi,
        posterior_ensemble=None,
        objective_history=history,
        n_forward_calls=int(nfev["count"]),
        wall_s=float(wall),
        diagnostics={"scipy_status": int(res.status),
                     "scipy_message": str(res.message),
                     "scipy_optimality": float(res.optimality)},
    )
