"""/research/ptf/* — pedotransfer function REST endpoints."""  # M2:
from __future__ import annotations
from typing import Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from hydrus_research.ptf import texture_to_vg
from hydrus_research.ptf.presets import USDA_TEXTURE_CENTERS


router = APIRouter()


class PredictRequest(BaseModel):
    sand_pct: float
    silt_pct: float
    clay_pct: float
    bulk_density_g_cm3: float | None = None
    theta_33: float | None = None
    theta_1500: float | None = None
    organic_matter_pct: float | None = None
    organic_carbon_pct: float | None = None
    topsoil: bool = True
    method: Literal["rosetta3_auto", "carsel_parrish", "wosten",
                    "rosetta3_h1", "rosetta3_h2", "rosetta3_h3", "rosetta3_h4"
                    ] = "rosetta3_auto"


def _validate_texture_sum(sand_pct: float, silt_pct: float, clay_pct: float):
    total = sand_pct + silt_pct + clay_pct
    if not 99.0 <= total <= 101.0:
        raise HTTPException(status_code=422,
                            detail=f"sand+silt+clay must sum to 100; got {total}")


@router.post("/predict")
def predict(req: PredictRequest):
    _validate_texture_sum(req.sand_pct, req.silt_pct, req.clay_pct)
    try:
        r = texture_to_vg(
            sand_pct=req.sand_pct, silt_pct=req.silt_pct, clay_pct=req.clay_pct,
            bulk_density_g_cm3=req.bulk_density_g_cm3,
            theta_33=req.theta_33, theta_1500=req.theta_1500,
            organic_matter_pct=req.organic_matter_pct,
            organic_carbon_pct=req.organic_carbon_pct,
            topsoil=req.topsoil, method=req.method,
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    return r.model_dump()


@router.get("/usda-classes")
def usda_classes():
    return USDA_TEXTURE_CENTERS
