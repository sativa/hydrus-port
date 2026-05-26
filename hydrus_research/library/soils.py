"""Soil library: Pydantic models + JSON loader."""
from __future__ import annotations
import json
from functools import lru_cache
from pathlib import Path
from pydantic import BaseModel


_DATA = Path(__file__).parent / "data" / "soils.json"


class VanGenuchten(BaseModel):
    theta_r: float
    theta_s: float
    alpha: float      # 1/cm
    n: float
    Ks: float         # cm/day
    L: float = 0.5


class SoilLayer(BaseModel):
    depth_cm: float
    vg: VanGenuchten


class Soil(BaseModel):
    id: str
    name_zh: str
    name_en: str
    layers: list[SoilLayer]


@lru_cache(maxsize=1)
def load_soils() -> list[Soil]:
    raw = json.loads(_DATA.read_text(encoding="utf-8"))
    return [Soil.model_validate(s) for s in raw["soils"]]


def get_soil(soil_id: str) -> Soil:
    for s in load_soils():
        if s.id == soil_id:
            return s
    raise KeyError(f"unknown soil id: {soil_id}")
