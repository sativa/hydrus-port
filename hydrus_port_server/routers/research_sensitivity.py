"""/research/sensitivity/{method} — F2 REST surface.  # M4:

Synchronous endpoint: runs the sweep + analysis in-process and returns
the SensitivityResult. For long sweeps the client should use the M3
/research/batch/* async pattern instead.
"""
from __future__ import annotations
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel


router = APIRouter()  # M4:


class ParamSpecPayload(BaseModel):  # M4:
    name: str
    target: str
    bounds: tuple[float, float]
    transform: Literal["linear", "log", "logit"] = "linear"


class ObsSpecPayload(BaseModel):  # M4:
    name: str
    kind: Literal["theta", "h", "c", "flux", "cumulative_flux", "concentration_flux"]
    location: dict
    time_day: float


class SensitivityRequest(BaseModel):  # M4:
    scenario_dir: str
    params: list[ParamSpecPayload]
    obs: list[ObsSpecPayload]
    n: int = 100
    workers: int = 1
    seed: int | None = None
    # Method-specific knobs (optional)
    num_levels: int = 4                    # morris
    calc_second_order: bool = False        # sobol
    m: int = 4                             # fast
    s: int = 10                            # pawn


_VALID_METHODS = {"morris", "sobol", "fast", "pawn"}  # M4:


@router.post("/{method}")  # M4:
def run(method: str, req: SensitivityRequest):
    if method not in _VALID_METHODS:
        raise HTTPException(status_code=404,
                            detail=f"unknown method {method!r}; "
                                   f"available: {sorted(_VALID_METHODS)}")
    from hydrus_research.parameters import ParameterSpec, ParameterMap
    from hydrus_research.observations import ObservationSpec
    from hydrus_research.simulator import make_forward
    from hydrus_research.simulator.hydrus1d_adapter import Hydrus1DSimulator
    from hydrus_port.adapters.hydrus1d import load as _load_h1d

    specs = [ParameterSpec(name=p.name, target=p.target,
                           bounds=p.bounds, transform=p.transform)
             for p in req.params]
    pm = ParameterMap(specs)
    obs_specs = [ObservationSpec(name=o.name, kind=o.kind,
                                 location=o.location, time_day=o.time_day)
                 for o in req.obs]
    template = _load_h1d(Path(req.scenario_dir)).to_dict()
    sim = Hydrus1DSimulator()
    forward = make_forward(sim, pm, template_scenario=template,
                           forcing=None, ic=None, obs_specs=obs_specs)
    obs_names = [o.name for o in req.obs]

    if method == "morris":
        from hydrus_research.sensitivity import morris_screen
        r = morris_screen(forward, pm, obs_names,
                          n_trajectories=req.n, num_levels=req.num_levels,
                          seed=req.seed, n_workers=req.workers)
    elif method == "sobol":
        from hydrus_research.sensitivity import sobol_decompose
        r = sobol_decompose(forward, pm, obs_names,
                            n_base=req.n,
                            calc_second_order=req.calc_second_order,
                            seed=req.seed, n_workers=req.workers)
    elif method == "fast":
        from hydrus_research.sensitivity import fast_indices
        r = fast_indices(forward, pm, obs_names,
                         n=req.n, m=req.m,
                         seed=req.seed, n_workers=req.workers)
    else:                                    # pawn
        from hydrus_research.sensitivity import pawn_kde
        r = pawn_kde(forward, pm, obs_names,
                     n=req.n, s=req.s,
                     seed=req.seed, n_workers=req.workers)

    return r.model_dump()
