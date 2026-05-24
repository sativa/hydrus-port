"""Convert a PTFResult into M0 ParameterSpec priors for use as a starting
point in F3 inversion / F5 UQ."""
from __future__ import annotations
import math
import numpy as np

from .result import PTFResult
from ..parameters import ParameterSpec


def vg_to_prior(ptf: PTFResult, material_index: int = 0) -> list[ParameterSpec]:
    """Build 5 ParameterSpec entries (theta_r/theta_s/alpha/n/Ks) from a PTF.

    `alpha` and `Ks` are log-transformed (always positive); `theta_r`, `theta_s`
    and `n` are linear. Bounds span ±4σ when covariance is available; otherwise
    fall back to ±50% multiplicative for log params and ±20% additive for theta_r
    and theta_s, ±0.5 for n."""
    # Diagonal stddevs from covariance, or None
    if ptf.covariance is not None:
        diag = [math.sqrt(max(ptf.covariance[i][i], 0.0)) for i in range(5)]
    else:
        diag = [None] * 5

    means = (ptf.theta_r, ptf.theta_s, ptf.alpha, ptf.n, ptf.Ks)
    names = ("theta_r", "theta_s", "alpha", "n", "Ks")
    transforms = ("linear", "linear", "log", "linear", "log")
    fallback_bounds = (
        lambda m: (max(m - 0.04, 0.0), m + 0.04),                   # theta_r
        lambda m: (max(m - 0.04, 0.0), min(m + 0.04, 1.0)),         # theta_s
        lambda m: (m * 0.5, m * 2.0),                                # alpha
        lambda m: (max(m - 0.3, 1.05), m + 0.5),                    # n
        lambda m: (m * 0.2, m * 5.0),                                # Ks
    )

    specs: list[ParameterSpec] = []
    for i, (name, mean, trans) in enumerate(zip(names, means, transforms)):
        sigma = diag[i]
        if sigma is not None and sigma > 0:
            if trans == "log":
                # 4-sigma window in user units (multiplicative-ish)
                lo = max(mean - 4 * sigma, mean * 0.05)
                hi = mean + 4 * sigma
            else:
                lo, hi = mean - 4 * sigma, mean + 4 * sigma
        else:
            lo, hi = fallback_bounds[i](mean)

        if trans == "log":
            lo = max(lo, mean * 0.01)              # never let lo touch zero
        if name == "theta_r":
            lo = max(lo, 0.0)
        if name == "theta_s":
            hi = min(hi, 1.0)
        if name == "n":
            lo = max(lo, 1.05)

        specs.append(ParameterSpec(
            name=name,
            target=f"materials[{material_index}].{name}",
            bounds=(float(lo), float(hi)),
            transform=trans,
            prior_mean=float(mean),
            prior_std=float(sigma) if sigma is not None else None,
            group=f"mat{material_index}_vg",
        ))
    return specs
