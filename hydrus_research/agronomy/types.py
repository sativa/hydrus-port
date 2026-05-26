"""Pydantic request/response types for the agronomy decision workflow."""
from __future__ import annotations
from datetime import date
from pydantic import BaseModel, Field


class IrrigEvent(BaseModel):
    date: date
    depth_mm: float = Field(gt=0)


class FertEvent(BaseModel):
    date: date
    kg_n_ha: float = Field(gt=0)
    conc_mg_l: float | None = None  # if provided, applied with the next irrigation


class AgronomyRequest(BaseModel):
    crop_id: str
    soil_id: str
    weather_id: str
    horizon_days: int = Field(gt=0, le=400)
    irrigation: list[IrrigEvent] = []
    fertilizer: list[FertEvent] = []
    start_year: int = 2026


class WaterBalance(BaseModel):
    rain_mm: float
    irrig_mm: float
    et_mm: float
    percolation_mm: float
    storage_change_mm: float


class NBudget(BaseModel):
    applied_kg_ha: float
    uptake_kg_ha: float
    leached_kg_ha: float
    residual_kg_ha: float


class EventTick(BaseModel):
    t_day: float
    amount: float            # mm for irrig, kg N/ha for fert
    label: str               # "irrig" | "fert"


class AgronomyResult(BaseModel):
    z_cm: list[float]                # ascending (surface=0 → deep positive)
    t_days: list[float]
    theta_zt: list[list[float]]      # shape (nT, nZ)
    n_zt: list[list[float]]
    water_balance: WaterBalance
    n_budget: NBudget
    events: list[EventTick]
