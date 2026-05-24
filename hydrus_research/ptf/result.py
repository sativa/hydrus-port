"""PTFResult — output of every pedotransfer function call.

`theta_r, theta_s, alpha, n, Ks, L` are the van Genuchten-Mualem 6 parameters
in HYDRUS conventions (cm⁻¹ for alpha, dimensionless n, cm/day for Ks).
`covariance` is the 5×5 covariance of (theta_r, theta_s, alpha, n, Ks) when
the backend provides one (ROSETTA-3 returns a per-prediction stddev; we
store it as the diagonal of `covariance`)."""
from __future__ import annotations
from typing import Literal
from pydantic import BaseModel


PTFMethod = Literal[
    "rosetta3_h1", "rosetta3_h2", "rosetta3_h3", "rosetta3_h4",
    "carsel_parrish", "wosten",
]


class PTFResult(BaseModel):
    theta_r: float
    theta_s: float
    alpha: float                                # 1/cm
    n: float                                    # > 1, dimensionless
    Ks: float                                   # cm/day
    L: float = 0.5                              # Mualem pore-connectivity tortuosity
    method: PTFMethod
    covariance: list[list[float]] | None = None
