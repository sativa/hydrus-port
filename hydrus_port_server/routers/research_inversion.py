"""/research/inversion/{backend} — F3 REST surface."""
from __future__ import annotations
from pathlib import Path
from typing import Any, Literal

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel


router = APIRouter()


class ParamSpecPayload(BaseModel):
    name: str
    target: str
    bounds: tuple[float, float]
    transform: Literal["linear", "log", "logit"] = "linear"


class ObsSpecPayload(BaseModel):
    name: str
    kind: Literal["theta", "h", "c", "flux", "cumulative_flux", "concentration_flux"]
    location: dict
    time_day: float


class ObsInline(BaseModel):
    specs: list[ObsSpecPayload]
    values: list[float]
    sigmas: list[float]


class InversionRequest(BaseModel):
    scenario_dir: str
    params: list[ParamSpecPayload]
    obs_inline: ObsInline | None = None
    obs_csv: str | None = None
    max_nfev: int = 200
    n_real: int = 200
    n_iter: int = 4


_VALID = {"lm", "lm_scipy", "ies", "pyemu_ies", "glm", "pyemu_glm",
          "nuts", "pymc_nuts", "auto"}


@router.post("/{backend}")
def run(backend: str, req: InversionRequest):
    if backend not in _VALID:
        raise HTTPException(status_code=404,
                            detail=f"unknown backend {backend!r}; "
                                   f"available: {sorted(_VALID)}")
    from hydrus_research.parameters import ParameterSpec, ParameterMap
    from hydrus_research.observations import ObservationSpec, ObservationSet
    from hydrus_research.simulator import make_forward
    from hydrus_research.simulator.hydrus1d_adapter import Hydrus1DSimulator
    from hydrus_research.inversion import fit
    from hydrus_port.adapters.hydrus1d import load as _load_h1d

    specs = [ParameterSpec(name=p.name, target=p.target,
                           bounds=p.bounds, transform=p.transform)
             for p in req.params]
    pm = ParameterMap(specs)

    if req.obs_inline:
        obs_specs = [ObservationSpec(name=s.name, kind=s.kind,
                                     location=s.location, time_day=s.time_day)
                     for s in req.obs_inline.specs]
        obs = ObservationSet(specs=obs_specs,
                             values=np.array(req.obs_inline.values),
                             sigmas=np.array(req.obs_inline.sigmas))
    elif req.obs_csv:
        obs = ObservationSet.from_csv(Path(req.obs_csv))
    else:
        raise HTTPException(status_code=400,
                            detail="must provide obs_inline or obs_csv")

    template = _load_h1d(Path(req.scenario_dir)).to_dict()
    sim = Hydrus1DSimulator()
    forward = make_forward(sim, pm, template_scenario=template,
                           forcing=None, ic=None, obs_specs=obs.specs)

    try:
        result = fit(forward=forward, param_map=pm, obs=obs,
                     scenario_dir=req.scenario_dir,
                     backend=backend, simulator_dimension=1,
                     max_nfev=req.max_nfev,
                     n_real=req.n_real, n_iter=req.n_iter)
    except RuntimeError as e:                    # pestpp-ies missing, etc.
        raise HTTPException(status_code=503, detail=str(e))

    return result.model_dump()
