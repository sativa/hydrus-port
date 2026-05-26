"""Crop library: Pydantic models + JSON loader."""
from __future__ import annotations
import json
from functools import lru_cache
from pathlib import Path
from pydantic import BaseModel


_DATA = Path(__file__).parent / "data" / "crops.json"


class Feddes(BaseModel):
    P0: float
    P0pt: float
    P2H: float
    P2L: float
    P3: float
    r2H: float
    r2L: float


class Root(BaseModel):
    max_depth_cm: float
    z50_cm: float
    z95_cm: float


class Season(BaseModel):
    sow_doy: int
    harvest_doy: int


class KcPoint(BaseModel):
    doy: int
    kc: float


class Crop(BaseModel):
    id: str
    name_zh: str
    name_en: str
    feddes: Feddes
    root: Root
    season: Season
    kc_curve: list[KcPoint]


@lru_cache(maxsize=1)
def _load_crops_cached() -> tuple[Crop, ...]:
    raw = json.loads(_DATA.read_text(encoding="utf-8"))
    return tuple(Crop.model_validate(c) for c in raw["crops"])


def load_crops() -> list[Crop]:
    return list(_load_crops_cached())


def get_crop(crop_id: str) -> Crop:
    for c in _load_crops_cached():
        if c.id == crop_id:
            return c
    raise KeyError(f"unknown crop id: {crop_id}")
