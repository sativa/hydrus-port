"""Wrapper around the USDA-ARS rosetta-soil package.

The public API picks the right hierarchical model from the inputs:
  - 3 inputs (sand/silt/clay)       → H1 (ROSETTA code 2)
  - + bulk density                  → H2 (ROSETTA code 3)
  - + theta at 33 kPa               → H3 (ROSETTA code 4)
  - + theta at 1500 kPa             → H4 (ROSETTA code 5)

API NOTE (deviation from plan): the rosetta-soil 0.3.x API is:
  rosetta.rosetta(rosetta_version, soildata, estimate_type='arith')
where rosetta_version is 1|2|3 (the ROSETTA model generation, not
the hierarchy level). We use version=3 (latest).

Return columns (7 total): theta_r, theta_s, alpha, npar, Ksat, K0, Lpar.
With estimate_type='log': columns 2-4 (alpha, npar, Ksat) are log10 values;
theta_r, theta_s, K0, Lpar are always arithmetic means.
We back-transform alpha, npar, Ksat from log10 and build the 5×5 covariance
using delta-method on the log-scale stdevs.
"""
from __future__ import annotations
import numpy as np
import rosetta as _rosetta

from .result import PTFResult


def _validate_texture(sand_pct, silt_pct, clay_pct):
    total = sand_pct + silt_pct + clay_pct
    if not 99.0 <= total <= 101.0:
        raise ValueError(f"sand+silt+clay must sum to 100; got {total}")
    for v, name in [(sand_pct, "sand"), (silt_pct, "silt"), (clay_pct, "clay")]:
        if not 0.0 <= v <= 100.0:
            raise ValueError(f"{name}_pct out of range [0, 100]: {v}")


def rosetta3_predict(sand_pct: float, silt_pct: float, clay_pct: float,
                     bulk_density_g_cm3: float | None = None,
                     theta_33: float | None = None,
                     theta_1500: float | None = None) -> PTFResult:
    _validate_texture(sand_pct, silt_pct, clay_pct)

    # Build data row: [sand%, silt%, clay%, (bd), (th33), (th1500)]
    cols = [sand_pct, silt_pct, clay_pct]
    method = "rosetta3_h1"
    if bulk_density_g_cm3 is not None:
        cols.append(bulk_density_g_cm3)
        method = "rosetta3_h2"
    if theta_33 is not None:
        cols.append(theta_33)
        method = "rosetta3_h3"
    if theta_1500 is not None:
        cols.append(theta_1500)
        method = "rosetta3_h4"

    arr = [cols]  # rosetta expects list of rows

    # Use estimate_type='log' so alpha/npar/Ksat come as log10 values;
    # stdevs are then on the log scale, enabling delta-method covariance.
    mean, std, _code = _rosetta.rosetta(3, arr, estimate_type='log')
    mean = np.asarray(mean).reshape(-1, 7)[0]
    std = np.asarray(std).reshape(-1, 7)[0]

    # Columns: theta_r, theta_s, log10(alpha), log10(npar), log10(Ksat), K0, Lpar
    theta_r = float(mean[0])
    theta_s = float(mean[1])
    alpha = float(10.0 ** mean[2])
    n = float(10.0 ** mean[3])
    Ks = float(10.0 ** mean[4])

    # Build a 5×5 diagonal covariance from per-param stddevs.
    # theta_r, theta_s: linear stddev → variance directly.
    # alpha, n, Ks: log10 stddev → delta-method: σ_x ≈ x * ln(10) * σ_log10x
    diag = np.zeros(5, dtype=float)
    diag[0] = std[0] ** 2                                  # theta_r
    diag[1] = std[1] ** 2                                  # theta_s
    diag[2] = (alpha * np.log(10) * std[2]) ** 2           # alpha from log10
    diag[3] = (n * np.log(10) * std[3]) ** 2               # n from log10
    diag[4] = (Ks * np.log(10) * std[4]) ** 2              # Ks from log10
    cov = np.diag(diag).tolist()

    return PTFResult(theta_r=theta_r, theta_s=theta_s,
                     alpha=alpha, n=n, Ks=Ks,
                     method=method, covariance=cov)
