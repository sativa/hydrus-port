"""End-to-end agronomy runner: AgronomyRequest → AgronomyResult.

Calls the existing Hydrus1DSimulator adapter. Pure Python entry; no
GUI / REST dependency.

Adapter integration note
------------------------
Hydrus1DSimulator.run() returns a SimResult whose meta dict has key
``"out_dir"`` pointing to the directory where NOD_INF.OUT / BALANCE.OUT
were written (see hydrus_research/simulator/hydrus1d_adapter.py line ~83).
We read that key in _resolve_run_dir() rather than re-deriving the path.
"""
from __future__ import annotations
from pathlib import Path
import tempfile

from hydrus_research.library.crops import get_crop
from hydrus_research.library.soils import get_soil
from hydrus_research.library.weather import load_weather_series
from hydrus_research.simulator.hydrus1d_adapter import Hydrus1DSimulator

from .scenario_builder import build_scenario, event_to_t_day, _sow_date
from .result_parser import parse_nod_inf, parse_balance, parse_conc_from_nod_inf
from .types import (
    AgronomyRequest, AgronomyResult,
    WaterBalance, NBudget, EventTick,
)


def run_agronomy(req: AgronomyRequest, work_dir: Path | str | None = None) -> AgronomyResult:
    """Run a full HYDRUS-1D agronomy simulation for *req*.

    Parameters
    ----------
    req:
        Validated agronomy request (crop/soil/weather IDs, events, horizon).
    work_dir:
        Root directory under which the adapter will create a temp sub-directory
        for HYDRUS-1D I/O files. Defaults to a system temp folder. Passing a
        ``tmp_path`` fixture here lets tests inspect the run artefacts.

    Returns
    -------
    AgronomyResult
        Parsed soil-water state + water balance for the full simulation horizon.
    """
    crop = get_crop(req.crop_id)
    soil = get_soil(req.soil_id)
    weather = load_weather_series(req.weather_id)

    sc = build_scenario(crop, soil, weather, req)

    work_root = Path(work_dir) if work_dir is not None else Path(tempfile.gettempdir()) / "agronomy"
    sim = Hydrus1DSimulator(work_root=work_root)

    # sc.scenario is the canonical Scenario; sc.to_dict() is agronomy-shaped.
    sim_result = sim.run(sc.scenario.to_dict(), forcing=None, ic=None)

    # Locate the HYDRUS-1D output directory the adapter wrote into.
    out_dir = _resolve_run_dir(sim_result)

    # parse_nod_inf returns (z_ascending_positive, t_days, theta[nT, nZ])
    z, t, theta = parse_nod_inf(out_dir / "NOD_INF.OUT")
    balance = parse_balance(out_dir / "BALANCE.OUT")

    # Parse N-NO₃ concentration field when solute transport was enabled.
    try:
        n_field = parse_conc_from_nod_inf(out_dir / "NOD_INF.OUT")
        n_zt = n_field.tolist()
    except Exception:
        # Non-solute run, or file format without Conc column → zeros.
        n_zt = [[0.0] * len(z) for _ in t]

    irrig_mm = sum(e.depth_mm for e in req.irrigation)
    sow = _sow_date(req, crop)

    events: list[EventTick] = [
        EventTick(t_day=float(event_to_t_day(e.date, sow)), amount=e.depth_mm, label="irrig")
        for e in req.irrigation
    ] + [
        EventTick(t_day=float(event_to_t_day(e.date, sow)), amount=e.kg_n_ha, label="fert")
        for e in req.fertilizer
    ]

    return AgronomyResult(
        z_cm=z.tolist(),
        t_days=t.tolist(),
        theta_zt=theta.tolist(),
        n_zt=n_zt,
        water_balance=WaterBalance(
            rain_mm=balance.get("rain_mm", 0.0),
            irrig_mm=irrig_mm,
            et_mm=balance.get("et_mm", 0.0),
            percolation_mm=balance.get("percolation_mm", 0.0),
            storage_change_mm=balance.get("storage_change_mm", 0.0),
        ),
        n_budget=NBudget(
            applied_kg_ha=sum(e.kg_n_ha for e in req.fertilizer),
            uptake_kg_ha=0.0,
            leached_kg_ha=0.0,           # TODO M3: integrate solute flux at bottom node
            residual_kg_ha=sum(e.kg_n_ha for e in req.fertilizer),   # applied - leached - uptake
        ),
        events=events,
    )


def _resolve_run_dir(sim_result) -> Path:
    """Locate the HYDRUS-1D output directory from a SimResult.

    The Hydrus1DSimulator stores the path under meta["out_dir"]
    (hydrus_research/simulator/hydrus1d_adapter.py, _load_outputs).
    """
    meta = getattr(sim_result, "meta", {}) or {}
    # Primary key used by hydrus1d_adapter._load_outputs
    for key in ("out_dir", "run_dir", "work_dir", "output_dir"):
        v = meta.get(key)
        if v:
            return Path(v)
    raise RuntimeError(
        f"SimResult.meta has no recognisable run-dir key (keys: {list(meta.keys())}); "
        "inspect hydrus_research/simulator/hydrus1d_adapter.py to find the correct one."
    )
