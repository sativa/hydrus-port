"""Pydantic 11-item DNDC ↔ HYDRUS data contract.

Filled by GUI manual form, CSV import, or (future B2) live DNDC. Converted to
a `Forcing` for the engine by `to_forcing.py`.

Decoupling discipline: this file imports ONLY pydantic + stdlib. The numeric
conversion to `Forcing` lives in `to_forcing.py`, which is the single seam
that touches `hydrus_research.simulator`.
"""
from __future__ import annotations
from datetime import date
from pathlib import Path
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator


class AtmDaily(BaseModel):                            # #1 atmospheric forcing
    dates: list[date]
    precip_cm: list[float]
    pet_cm: list[float] | None = None
    t_min_c: list[float] | None = None
    t_max_c: list[float] | None = None
    rh_pct: list[float] | None = None
    wind_m_s: list[float] | None = None
    solar_mj_m2: list[float] | None = None

    @model_validator(mode="after")
    def _lengths_match(self):
        n = len(self.dates)
        for fname in ("precip_cm", "pet_cm", "t_min_c", "t_max_c",
                      "rh_pct", "wind_m_s", "solar_mj_m2"):
            v = getattr(self, fname)
            if v is not None and len(v) != n:
                raise ValueError(f"{fname} length {len(v)} != dates length {n}")
        return self


class EtPartition(BaseModel):                         # #2 E/T split
    mode: Literal["lai_beer", "explicit_split", "kc_dual"]
    lai: list[float] | None = None
    extinction_k: float = 0.6
    explicit_e_frac: list[float] | None = None
    kcb: list[float] | None = None

    @model_validator(mode="after")
    def _mode_requires_fields(self):
        if self.mode == "explicit_split" and self.explicit_e_frac is None:
            raise ValueError("mode='explicit_split' requires explicit_e_frac")
        if self.mode == "kc_dual" and self.kcb is None:
            raise ValueError("mode='kc_dual' requires kcb")
        return self


class RootGrowth(BaseModel):                          # #3 root dynamics
    z_max_cm: float
    growth_curve: Literal["linear", "logistic", "table"]
    days_to_zmax: float | None = None
    table: list[tuple[date, float]] | None = None
    density_profile: Literal["uniform", "linear_decline",
                             "exponential", "raats"] = "linear_decline"
    density_param: float | None = None

    @model_validator(mode="after")
    def _curve_requires_fields(self):
        if self.growth_curve in ("linear", "logistic") and self.days_to_zmax is None:
            raise ValueError(f"growth_curve='{self.growth_curve}' requires days_to_zmax")
        if self.growth_curve == "table" and not self.table:
            raise ValueError("growth_curve='table' requires non-empty table")
        return self


class FeddesParams(BaseModel):                        # #4 water stress
    h1: float                 # cm pressure head; anaerobiosis onset
    h2: float
    h3_high: float
    h3_low: float
    h4: float                 # wilting
    pet_high_cm_d: float = 0.5
    pet_low_cm_d: float = 0.1

    @model_validator(mode="after")
    def _ordering(self):
        # In HYDRUS convention all h are pressures (negative for unsaturated);
        # required ordering: h1 >= h2 >= h3_high >= h3_low >= h4
        seq = [self.h1, self.h2, self.h3_high, self.h3_low, self.h4]
        for a, b in zip(seq[:-1], seq[1:]):
            if a < b:
                raise ValueError(
                    "Feddes h's must satisfy h1 >= h2 >= h3_high >= h3_low >= h4; "
                    f"got {seq}"
                )
        return self
