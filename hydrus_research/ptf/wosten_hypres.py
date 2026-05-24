"""Wösten et al. (1999) HYPRES continuous PTF.

Reference: Wösten, J. H. M., Lilly, A., Nemes, A., & Le Bas, C. (1999).
Development and use of a database of hydraulic properties of European
soils. Geoderma, 90(3-4), 169-185.

Inputs: sand%, silt%, clay%, bulk density (g/cm³), organic matter %,
topsoil/subsoil flag (boolean). Returns the 5 VG params via closed-form
multivariate polynomial regressions."""
from __future__ import annotations
import math
from .result import PTFResult


def _validate(sand_pct, silt_pct, clay_pct):
    total = sand_pct + silt_pct + clay_pct
    if not 99.0 <= total <= 101.0:
        raise ValueError(f"sand+silt+clay must sum to 100; got {total}")
    for v, name in [(sand_pct, "sand"), (silt_pct, "silt"), (clay_pct, "clay")]:
        if not 0.0 <= v <= 100.0:
            raise ValueError(f"{name}_pct out of range [0, 100]: {v}")


def wosten_predict(sand_pct: float, silt_pct: float, clay_pct: float,
                   bulk_density_g_cm3: float,
                   organic_matter_pct: float,
                   topsoil: bool) -> PTFResult:
    _validate(sand_pct, silt_pct, clay_pct)
    C = clay_pct
    S = silt_pct                                # silt + clay = fine fraction
    D = bulk_density_g_cm3
    OM = max(organic_matter_pct, 0.01)
    topsoil_i = 1 if topsoil else 0

    # Saturated water content (theta_s) — Eq. 1
    theta_s = (0.7919
               + 0.001691 * C
               - 0.29619 * D
               - 0.000001491 * S * S
               + 0.0000821 * OM * OM
               + 0.02427 / C
               + 0.01113 / S
               + 0.01472 * math.log(S)
               - 0.0000733 * OM * C
               - 0.000619 * D * C
               - 0.001183 * D * OM
               - 0.0001664 * topsoil_i * S)

    # ln(alpha*) — Eq. 2  (alpha in 1/cm)
    ln_alpha_star = (-14.96
                     + 0.03135 * C
                     + 0.0351 * S
                     + 0.646 * OM
                     + 15.29 * D
                     - 0.192 * topsoil_i
                     - 4.671 * D * D
                     - 0.000781 * C * C
                     - 0.00687 * OM * OM
                     + 0.0449 / OM
                     + 0.0663 * math.log(S)
                     + 0.1482 * math.log(OM)
                     - 0.04546 * D * S
                     - 0.4852 * D * OM
                     + 0.00673 * topsoil_i * C)
    alpha = math.exp(ln_alpha_star)

    # ln(n*-1) — Eq. 3  (n > 1 always)
    ln_n_minus_1 = (-25.23
                    - 0.02195 * C
                    + 0.0074 * S
                    - 0.1940 * OM
                    + 45.5 * D
                    - 7.24 * D * D
                    + 0.0003658 * C * C
                    + 0.002885 * OM * OM
                    - 12.81 / D
                    - 0.1524 / S
                    - 0.01958 / OM
                    - 0.2876 * math.log(S)
                    - 0.0709 * math.log(OM)
                    - 44.6 * math.log(D)
                    - 0.02264 * D * C
                    + 0.0896 * D * OM
                    + 0.00718 * topsoil_i * C)
    n = math.exp(ln_n_minus_1) + 1.0

    # ln(Ks) — Eq. 4 (cm/day)
    ln_Ks = (7.755
             + 0.0352 * S
             + 0.93 * topsoil_i
             - 0.967 * D * D
             - 0.000484 * C * C
             - 0.000322 * S * S
             + 0.001 / S
             - 0.0748 / OM
             - 0.643 * math.log(S)
             - 0.01398 * D * C
             - 0.1673 * D * OM
             + 0.02986 * topsoil_i * C
             - 0.03305 * topsoil_i * S)
    Ks = math.exp(ln_Ks)

    # theta_r is not predicted by Wösten 1999; HYDRUS convention default
    theta_r = 0.01

    return PTFResult(theta_r=theta_r, theta_s=float(theta_s),
                     alpha=float(alpha), n=float(n), Ks=float(Ks),
                     method="wosten")
