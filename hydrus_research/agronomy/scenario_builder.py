"""Map (crop, soil, weather_series, AgronomyRequest) → canonical Scenario.

Design notes
------------
The canonical ``Scenario`` dataclass (hydrus_port.schema) uses its own field
layout which does not match the agronomy-oriented dict shape the workflow layer
expects (keys like "dimension", "sim", "profile", "sink").  Rather than mutating
the canonical schema we return an ``AgronomyScenario`` thin wrapper that:

  1. Holds a fully-validated canonical ``Scenario`` accessible as ``.scenario``.
  2. Overrides ``to_dict()`` to produce the agronomy-shaped dict that test
     assertions (and future Task-5 runner code) expect:
       * d["dimension"]      — "1d" | "2d" | "3d"
       * d["sim"]["t_max"]   — simulation horizon in days
       * d["profile"]["depth_cm"]  — total soil column depth
       * d["sink"]["feddes"] — Feddes params keyed by the crop's field names
                               (uppercase P0/P2H/P2L/P3, r2H/r2L)

Feddes field-name note
----------------------
``Crop.feddes`` uses uppercase (P0, P2H, P2L, P3, r2H, r2L — from the JSON
library).  ``FeddesRootUptake`` in the canonical schema uses lowercase (p0,
p2H, p2L, p3 …).  The agronomy dict keeps the crop's original uppercase keys
in d["sink"]["feddes"] so Task-5 can round-trip them back into the crop model
without casing conversion.
"""
from __future__ import annotations
import copy
from datetime import date, timedelta
from dataclasses import dataclass, field as dc_field
from typing import Any

from hydrus_port.schema import (
    Scenario,
    _scenario_from_dict,
    HydraulicMaterial,
    Geometry1D,
    AtmosphericBC,
    AtmosphericRow,
    FeddesRootUptake,
    TimeControl,
)
from hydrus_research.library.crops import Crop
from hydrus_research.library.soils import Soil
from .types import AgronomyRequest


# ---------------------------------------------------------------- public util

def event_to_t_day(event_date: date, sow_date: date) -> int:
    """Return the integer offset (days) from *sow_date* to *event_date*.

    Day 0 = sow date itself.  Can be negative if event precedes sowing.
    """
    return (event_date - sow_date).days


# ---------------------------------------------------------------- wrapper

@dataclass
class AgronomyScenario:
    """Agronomy-shaped wrapper around a canonical Scenario.

    Attributes
    ----------
    scenario : Scenario
        The underlying canonical HYDRUS-port scenario ready for adapters.
    _agro_dict : dict
        Agronomy-shaped metadata dict (dimension / sim / profile / sink keys).
        Produced once at build time and returned verbatim by ``to_dict()``.
    """
    scenario: Scenario
    _agro_dict: dict = dc_field(repr=False)

    def to_dict(self) -> dict:
        """Return the agronomy-shaped dict (dimension / sim / profile / sink)."""
        return copy.deepcopy(self._agro_dict)


# ---------------------------------------------------------------- private helpers

def _sow_date(req: AgronomyRequest, crop: Crop) -> date:
    return date(req.start_year, 1, 1) + timedelta(days=crop.season.sow_doy - 1)


def _build_atm_rows(
    sow: date,
    horizon: int,
    weather: dict[str, list[float]],
    req: AgronomyRequest,
) -> tuple[list[AtmosphericRow], dict[int, float]]:
    """Build one AtmosphericRow per simulation day (precip + PET).

    Irrigation events on the matching calendar date are added to precipitation.
    Units: precip/evap in cm/day (HYDRUS-1D convention; weather data in mm).

    Returns
    -------
    rows : list[AtmosphericRow]
    ctop_by_day : dict[int, float]
        Map from 1-based simulation-day index to cTop value (mg/L equivalent
        concentration for solute 1 = N-NO₃) for that day.  Only days that have
        a fertilizer event are present; all other days default to 0.0 in the
        ATMOSPH.IN writer.
    """
    # Build a lookup: calendar date → fertilizer events
    fert_by_date: dict[date, list] = {}
    for fe in req.fertilizer:
        fert_by_date.setdefault(fe.date, []).append(fe)

    rows = []
    ctop_by_day: dict[int, float] = {}

    for i in range(horizon):
        cur = sow + timedelta(days=i)
        doy = cur.timetuple().tm_yday
        idx = (doy - 1) % 365
        prec_mm = weather["P_mm"][idx]
        pet_mm  = weather["PET_mm"][idx]

        # Add scheduled irrigation events falling on this calendar day
        irrig_mm = 0.0
        for ie in req.irrigation:
            if ie.date == cur:
                prec_mm += ie.depth_mm
                irrig_mm += ie.depth_mm

        rows.append(AtmosphericRow(
            t=float(i + 1),
            precip=prec_mm / 10.0,   # mm → cm
            evap=pet_mm  / 10.0,
        ))

        # Fertilizer events on this day → cTop concentration for ATMOSPH.IN
        if cur in fert_by_date:
            total_ctop = 0.0
            for fe in fert_by_date[cur]:
                if fe.conc_mg_l is not None:
                    total_ctop += fe.conc_mg_l
                else:
                    # Dissolve kg_n_ha in the available water (precip + irrig) for this day
                    # kg/ha × 100 → g/m² = mg/cm² ; / water_cm = mg/cm³ = mg/mL = mg/L
                    # (factor: 1 kg/ha = 100 g/1000 m² × 10^6 mg/kg = 0.1 g/m²)
                    water_cm = prec_mm / 10.0  # already includes irrig above
                    water_cm = max(water_cm, 0.1)  # guard against zero
                    # cTop in mg/L: (kg_n_ha * 0.1 g/m²) / (water_cm * 0.01 m) / 1000 mg/g
                    # Simplified: cTop_mg_L = kg_n_ha * 10 / water_cm
                    total_ctop += fe.kg_n_ha * 10.0 / water_cm
            ctop_by_day[i + 1] = total_ctop   # 1-based day index

    return rows, ctop_by_day


def _build_materials(soil: Soil) -> list[HydraulicMaterial]:
    return [
        HydraulicMaterial(
            theta_r=L.vg.theta_r,
            theta_s=L.vg.theta_s,
            alpha=L.vg.alpha,
            n=L.vg.n,
            Ks=L.vg.Ks,
            l=L.vg.L,
        )
        for L in soil.layers
    ]


def _build_geometry1d(soil: Soil) -> Geometry1D:
    """Uniform 1-cm node spacing from 0 (surface) down to total depth.

    ``mat_num`` assigns each node to the correct soil layer material (1-based).
    """
    total_depth = sum(L.depth_cm for L in soil.layers)
    n_nodes = max(101, int(total_depth) + 1)
    dz = total_depth / (n_nodes - 1)

    # Build layer-depth cumulative boundaries (top-down positive)
    cum_depths = []
    acc = 0.0
    for L in soil.layers:
        acc += L.depth_cm
        cum_depths.append(acc)

    z_vals = []
    mat_nums = []
    layer_nums = []
    initial_h = []

    for i in range(n_nodes):
        depth_from_surface = i * dz
        # z is negative-downward (surface=0, bottom=-total_depth) per HYDRUS-1D Profile.dat convention
        z_vals.append(-depth_from_surface)

        # Determine which layer this node belongs to (1-based)
        mat = 1
        for j, cum in enumerate(cum_depths):
            if depth_from_surface <= cum + 1e-9:
                mat = j + 1
                break
        mat_nums.append(mat)
        layer_nums.append(mat)
        initial_h.append(-100.0)    # pressure head initial condition (cm, unsaturated)

    return Geometry1D(
        z=z_vals,
        initial_h=initial_h,
        mat_num=mat_nums,
        layer=layer_nums,
        beta=[0.0] * n_nodes,   # no explicit beta distribution (FeddesRootUptake handles uptake)
        axz=[1.0] * n_nodes,
        bxz=[1.0] * n_nodes,
        dxz=[1.0] * n_nodes,
    )


def _build_feddes(crop: Crop) -> FeddesRootUptake:
    """Convert Crop.feddes (uppercase Pydantic) → canonical FeddesRootUptake."""
    f = crop.feddes
    return FeddesRootUptake(
        p0=f.P0,
        p2H=f.P2H,
        p2L=f.P2L,
        p3=f.P3,
        r2H=f.r2H,
        r2L=f.r2L,
        p_optm=[],
    )


def _build_agro_dict(
    crop: Crop,
    soil: Soil,
    req: AgronomyRequest,
    solute_events: list[dict],
) -> dict[str, Any]:
    """Build the agronomy-shaped dict the test assertions expect.

    Keys:
      dimension           "1d"
      sim.t_max           horizon in days
      profile.depth_cm    total soil column depth
      sink.feddes         crop Feddes params (uppercase, matching Crop.feddes)
      sink.root_*         root depth parameters
      solute.events       fertilizer injection records
      name                scenario identifier
    """
    total_depth = sum(L.depth_cm for L in soil.layers)
    return {
        "dimension": "1d",
        "name": f"agronomy_{crop.id}_{soil.id}_{req.weather_id}",
        "sim": {
            "t_init": 0.0,
            "t_max": float(req.horizon_days),
            "dt": 0.01,
        },
        "profile": {
            "depth_cm": total_depth,
            "n_nodes": max(101, int(total_depth) + 1),
            "layer_depths": [L.depth_cm for L in soil.layers],
        },
        "sink": {
            # Keep crop's original uppercase field names so Task-5 can
            # round-trip back into the Crop.feddes Pydantic model directly.
            "feddes": {
                "P0":  crop.feddes.P0,
                "P0pt": crop.feddes.P0pt,
                "P2H": crop.feddes.P2H,
                "P2L": crop.feddes.P2L,
                "P3":  crop.feddes.P3,
                "r2H": crop.feddes.r2H,
                "r2L": crop.feddes.r2L,
            },
            "root_depth_cm": crop.root.max_depth_cm,
            "root_z50_cm":   crop.root.z50_cm,
            "root_z95_cm":   crop.root.z95_cm,
        },
        "solute": {
            "enabled": bool(solute_events),
            "events": solute_events,
        },
    }


# ---------------------------------------------------------------- public API

def build_scenario(
    crop: Crop,
    soil: Soil,
    weather: dict[str, list[float]],
    req: AgronomyRequest,
) -> AgronomyScenario:
    """Build a canonical 1D HYDRUS scenario for the given agronomy inputs.

    Returns an ``AgronomyScenario`` wrapper that exposes both:
    * ``.scenario``   — fully-validated canonical ``Scenario`` for adapters
    * ``.to_dict()``  — agronomy-shaped dict for workflow/test assertions

    Parameters
    ----------
    crop : Crop
        Loaded from ``hydrus_research.library.crops.get_crop()``.
    soil : Soil
        Loaded from ``hydrus_research.library.soils.get_soil()``.
    weather : dict[str, list[float]]
        Typical-year series from ``load_weather_series()``; keys P_mm, PET_mm.
    req : AgronomyRequest
        Validated request object (horizon, events, IDs, start year).
    """
    sow = _sow_date(req, crop)
    horizon = req.horizon_days

    # 1. Atmospheric BC (precipitation + ET per day; irrigation added to precip)
    atm_rows, ctop_by_day = _build_atm_rows(sow, horizon, weather, req)

    # 2. Materials (one per soil layer; VG params)
    materials = _build_materials(soil)

    # 3. 1D geometry (uniform node spacing)
    geometry = _build_geometry1d(soil)

    # 4. Root uptake (Feddes 1978)
    feddes = _build_feddes(crop)

    # 5. Fertilizer events → solute injection records
    solute_events = []
    for fe in req.fertilizer:
        t_day = event_to_t_day(fe.date, sow) + 1   # 1-based simulation day
        if 1 <= t_day <= horizon:
            solute_events.append({
                "t_day": t_day,
                "kg_n_ha": fe.kg_n_ha,
                "conc_mg_l": fe.conc_mg_l,
            })

    # 6. Determine whether to enable solute transport
    has_fert = bool(req.fertilizer)
    n_mat = len(soil.layers)
    n_nodes = max(101, int(sum(L.depth_cm for L in soil.layers)) + 1)

    # 7. Build canonical Scenario via _scenario_from_dict.
    solver_dict: dict[str, Any] = {
        "water_flow": True,
        "atmospheric_bc": True,
        "root_uptake": True,
        "solute_transport": has_fert,
    }

    scenario_dict: dict[str, Any] = {
        "meta": {
            "name": f"agronomy_{crop.id}_{soil.id}_{req.weather_id}",
            "description": (
                f"crop={crop.id}, soil={soil.id}, weather={req.weather_id}, "
                f"horizon={horizon}d"
            ),
        },
        "solver": solver_dict,
        "materials": [
            {
                "theta_r": m.theta_r,
                "theta_s": m.theta_s,
                "alpha": m.alpha,
                "n": m.n,
                "Ks": m.Ks,
                "l": m.l,
            }
            for m in materials
        ],
        "time": {
            "t_init": 0.0,
            "t_max": float(horizon),
            "dt": 0.01,
            "print_times": [float(d) for d in range(1, horizon + 1)],
        },
        "root_uptake": {
            "p0":   feddes.p0,
            "p2H":  feddes.p2H,
            "p2L":  feddes.p2L,
            "p3":   feddes.p3,
            "r2H":  feddes.r2H,
            "r2L":  feddes.r2L,
            "p_optm": [],
        },
        "geometry": {
            "kind": "1d",
            "z": geometry.z,
            "initial_h": geometry.initial_h,
            "mat_num": geometry.mat_num,
            "layer": geometry.layer,
            "beta": geometry.beta,
            "axz": geometry.axz,
            "bxz": geometry.bxz,
            "dxz": geometry.dxz,
        },
        "atmospheric": {
            "sink_flag": False,
            "qgwlf": False,
            "rows": [
                {
                    "t": r.t,
                    "precip": r.precip,
                    "evap": r.evap,
                    "h_critA": r.h_critA,
                    "rRoot": r.rRoot,
                    "rB": r.rB,
                    "hB": r.hB,
                    "ht": r.ht,
                    "tTop": r.tTop,
                    "tBot": r.tBot,
                    "Ampl": r.Ampl,
                }
                for r in atm_rows
            ],
        },
        "legacy_extras": {
            "agronomy": {
                "crop_id":      crop.id,
                "soil_id":      soil.id,
                "weather_id":   req.weather_id,
                "sow_date":     sow.isoformat(),
                "solute_events": solute_events,
                "root_z50_cm":  crop.root.z50_cm,
                "root_z95_cm":  crop.root.z95_cm,
            },
            # Per-day cTop concentrations for ATMOSPH.IN (only populated when
            # fert events exist; keyed by 1-based simulation day).
            "agronomy_cTop": ctop_by_day,
            # Per-node initial concentration for Profile.dat (all zero at t=0).
            "agronomy_initial_conc_per_node": [0.0] * n_nodes,
        },
    }

    # 8. SoluteTransport block (BLOCK F) — only when fertilizer events exist
    if has_fert:
        scenario_dict["solute"] = {
            "epsi": 0.5,
            # Per-material: Bulk.d, Disp.L, Frac, ImmobWC
            "chem_params": [[1.35, 5.0, 1.0, 0.0]] * n_mat,
            # kTopCh=-1 (Cauchy/3rd-type, uses ATMOSPH.IN cTop per timestep)
            # kBotCh=0  (zero-gradient bottom)
            "kod_cb": [-1, 0],
            "c_bound": [0.0, 0.0],
            "t_pulse": 1.0e30,
        }

    canonical = _scenario_from_dict(scenario_dict)

    # 7. Agronomy-shaped dict (what test assertions and the workflow layer use)
    agro_dict = _build_agro_dict(crop, soil, req, solute_events)

    return AgronomyScenario(scenario=canonical, _agro_dict=agro_dict)
