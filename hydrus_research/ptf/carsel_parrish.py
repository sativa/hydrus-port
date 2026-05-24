"""Carsel & Parrish (1988) WRR 24(5) Table 2 — mean VG parameters per
USDA texture class. Ks in cm/day; alpha in 1/cm; n dimensionless.

Reference: Carsel, R. F., & Parrish, R. S. (1988). Developing joint
probability distributions of soil water retention characteristics.
Water Resources Research, 24(5), 755-769."""
from __future__ import annotations
from .result import PTFResult


# (theta_r, theta_s, alpha [1/cm], n, Ks [cm/day])  — published means
USDA_CLASSES: dict[str, tuple[float, float, float, float, float]] = {
    "sand":             (0.045, 0.43, 0.145, 2.68, 712.8),
    "loamy_sand":       (0.057, 0.41, 0.124, 2.28, 350.2),
    "sandy_loam":       (0.065, 0.41, 0.075, 1.89, 106.1),
    "loam":             (0.078, 0.43, 0.036, 1.56, 24.96),
    "silt":             (0.034, 0.46, 0.016, 1.37, 6.0),
    "silt_loam":        (0.067, 0.45, 0.020, 1.41, 10.8),
    "sandy_clay_loam":  (0.100, 0.39, 0.059, 1.48, 31.44),
    "clay_loam":        (0.095, 0.41, 0.019, 1.31, 6.24),
    "silty_clay_loam":  (0.089, 0.43, 0.010, 1.23, 1.68),
    "sandy_clay":       (0.100, 0.38, 0.027, 1.23, 2.88),
    "silty_clay":       (0.070, 0.36, 0.005, 1.09, 0.48),
    "clay":             (0.068, 0.38, 0.008, 1.09, 4.8),
}


def carsel_parrish_lookup(class_name: str) -> PTFResult:
    name = class_name.lower().strip().replace(" ", "_").replace("-", "_")
    if name not in USDA_CLASSES:
        raise KeyError(f"unknown USDA class {class_name!r}; available: {sorted(USDA_CLASSES)}")
    tr, ts, a, n, ks = USDA_CLASSES[name]
    return PTFResult(theta_r=tr, theta_s=ts, alpha=a, n=n, Ks=ks,
                     method="carsel_parrish")
