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


class FertEvent(BaseModel):                           # #5 fertilizer event
    date: date
    depth_cm: float                                   # 0 = surface; >0 = injected/banded
    mass_kg_n_ha: float
    form: Literal["NH4", "NO3", "urea", "NH4NO3", "compound"]
    composition: dict[str, float] | None = None

    @model_validator(mode="after")
    def _compound_needs_composition(self):
        if self.form == "compound" and not self.composition:
            raise ValueError("form='compound' requires composition dict (e.g. {'NH4':0.5,'NO3':0.5})")
        if self.composition is not None:
            total = sum(self.composition.values())
            if not (0.99 < total < 1.01):
                raise ValueError(f"composition fractions must sum to ~1.0; got {total}")
        return self


class IrrigEvent(BaseModel):                          # #6 irrigation event
    date: date
    method: Literal["flood", "sprinkler", "drip", "subsurface"]
    amount_cm: float
    duration_h: float
    solute_concs_mg_l: dict[str, float] = Field(default_factory=dict)
    drip_emitter_xyz: tuple[float, float, float] | None = None


class NTransformation(BaseModel):                     # #7 N transformation (B2 hook)
    mode: Literal["constant_rates", "external_callable", "lookup_table"]
    # constant_rates mode
    k_mineralization_d: float | None = None
    k_nitrification_d: float | None = None
    k_denitrification_d: float | None = None
    k_volatilization_d: float | None = None
    # external_callable mode (B2)
    callable_ref: str | None = None                   # e.g. "dndc.n_module:compute_rates"
    # lookup_table mode (interim)
    table_path: Path | None = None

    @model_validator(mode="after")
    def _mode_required_fields(self):
        if self.mode == "constant_rates":
            if all(getattr(self, k) is None for k in
                   ("k_mineralization_d", "k_nitrification_d",
                    "k_denitrification_d", "k_volatilization_d")):
                raise ValueError("mode='constant_rates' requires at least one k_*_d rate")
        elif self.mode == "external_callable":
            if not self.callable_ref:
                raise ValueError("mode='external_callable' requires callable_ref "
                                 "(e.g. 'dndc.n_module:compute_rates')")
        elif self.mode == "lookup_table":
            if not self.table_path:
                raise ValueError("mode='lookup_table' requires table_path")
        return self


class PlantNUptake(BaseModel):                        # #8 plant N uptake
    mode: Literal["passive_with_water", "michaelis_menten",
                  "demand_driven", "external"]
    km_mg_l: float | None = None
    vmax_mg_per_day_per_root_cm: float | None = None
    daily_demand_kg_n_ha: list[float] | None = None
    callable_ref: str | None = None                   # B2

    @model_validator(mode="after")
    def _mode_required_fields(self):
        if self.mode == "michaelis_menten":
            if self.km_mg_l is None or self.vmax_mg_per_day_per_root_cm is None:
                raise ValueError("mode='michaelis_menten' requires km_mg_l and vmax_mg_per_day_per_root_cm")
        if self.mode == "demand_driven" and not self.daily_demand_kg_n_ha:
            raise ValueError("mode='demand_driven' requires daily_demand_kg_n_ha")
        if self.mode == "external" and not self.callable_ref:
            raise ValueError("mode='external' requires callable_ref")
        return self


class StateExchange(BaseModel):                       # #9 bidirectional state
    z_grid_cm: list[float]
    initial_theta: list[float] | None = None
    initial_h: list[float] | None = None
    initial_c: dict[str, list[float]] | None = None
    initial_t: list[float] | None = None
    writeback_daily: bool = False                     # True only in B2
    writeback_path: Path | None = None

    @model_validator(mode="after")
    def _profiles_match_grid(self):
        nz = len(self.z_grid_cm)
        for fname in ("initial_theta", "initial_h", "initial_t"):
            v = getattr(self, fname)
            if v is not None and len(v) != nz:
                raise ValueError(f"{fname} length {len(v)} != z_grid_cm length {nz}")
        if self.initial_c is not None:
            for sp, prof in self.initial_c.items():
                if len(prof) != nz:
                    raise ValueError(f"initial_c[{sp!r}] length {len(prof)} != z_grid_cm length {nz}")
        return self


class SoilTemp(BaseModel):                            # #10 soil temperature
    enabled: bool = False
    surface_t_daily_c: list[float] | None = None


class Residue(BaseModel):                             # #11 residue / mulch
    mulch_fraction: float = 0.0
    residue_kg_ha: float = 0.0
    e_reduction_factor: float = 1.0


class DndcSeamInputs(BaseModel):
    """Complete data contract DNDC → HYDRUS. Used by:
       - manual GUI form (current)
       - DndcLiveAdapter (future B2)
       - CSV / YAML / JSON file ingestion (testing, batch)
    """
    model_config = ConfigDict(extra="allow")       # forward-compat buffer per spec §5.5

    atm: AtmDaily
    et: EtPartition
    root: RootGrowth
    feddes: FeddesParams
    fert_events: list[FertEvent] = Field(default_factory=list)
    irrig_events: list[IrrigEvent] = Field(default_factory=list)
    n_transform: NTransformation
    plant_n_uptake: PlantNUptake
    state: StateExchange
    soil_temp: SoilTemp = Field(default_factory=SoilTemp)
    residue: Residue = Field(default_factory=Residue)
    extras: dict = Field(default_factory=dict)     # extras-aware container (B2 surprises)
