"""Crop-preset loader — reads crop_presets.yaml and builds typed sub-models."""
from __future__ import annotations
from functools import cache
from pathlib import Path
import yaml

from .schema import FeddesParams, RootGrowth


_YAML_PATH = Path(__file__).parent / "crop_presets.yaml"


@cache
def _load_yaml() -> dict:
    with _YAML_PATH.open() as f:
        return yaml.safe_load(f)


def list_crop_presets() -> list[str]:
    return list(_load_yaml().keys())


def load_crop_preset(name: str) -> tuple[FeddesParams, RootGrowth, str]:
    data = _load_yaml()
    if name not in data:
        raise KeyError(f"unknown crop preset {name!r}; available: {sorted(data)}")
    entry = data[name]
    return (
        FeddesParams(**entry["feddes"]),
        RootGrowth(**entry["root"]),
        entry.get("description", ""),
    )
