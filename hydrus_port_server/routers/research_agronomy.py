"""/research/agronomy/* — crop/soil/weather libraries + run endpoint.

Decoupling: this file imports only fastapi + hydrus_research. No direct
hydrus1d / desktop imports, and hydrus_research must not import from
hydrus_port_server (no reverse dep).
"""
from __future__ import annotations
import shutil
import tempfile
from pathlib import Path

from fastapi import APIRouter, HTTPException

from hydrus_research.library.crops import load_crops
from hydrus_research.library.soils import load_soils
from hydrus_research.library.weather import (
    load_weather_meta, load_weather_series,
)
from hydrus_research.agronomy import AgronomyRequest, AgronomyResult, run_agronomy


router = APIRouter()


@router.get("/lib/crops")
def lib_crops():
    return {"crops": [c.model_dump() for c in load_crops()]}


@router.get("/lib/soils")
def lib_soils():
    return {"soils": [s.model_dump() for s in load_soils()]}


@router.get("/lib/weather")
def lib_weather():
    return {"weather": load_weather_meta()}


@router.get("/lib/weather/{weather_id}")
def lib_weather_series(weather_id: str):
    try:
        return load_weather_series(weather_id)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.post("/run", response_model=AgronomyResult)
def run(req: AgronomyRequest):
    work = Path(tempfile.mkdtemp(prefix="agronomy_"))
    try:
        return run_agronomy(req, work_dir=work)
    except KeyError as e:
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        shutil.rmtree(work, ignore_errors=True)
