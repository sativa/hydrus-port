"""USDA texture-class centers + thin `usda_class_to_vg` wrapper around
Carsel-Parrish lookup. The centers are useful to seed the texture triangle
in the GUI when the user picks a class instead of clicking the triangle."""
from __future__ import annotations
from .carsel_parrish import carsel_parrish_lookup
from .result import PTFResult


# Approximate centroids of each USDA texture class in (sand%, silt%, clay%).
# Source: USDA Soil Survey Manual texture triangle vertices.
USDA_TEXTURE_CENTERS: dict[str, dict[str, float]] = {
    "sand":             {"sand_pct": 92.0, "silt_pct": 5.0,  "clay_pct": 3.0},
    "loamy_sand":       {"sand_pct": 82.0, "silt_pct": 12.0, "clay_pct": 6.0},
    "sandy_loam":       {"sand_pct": 65.0, "silt_pct": 25.0, "clay_pct": 10.0},
    "loam":             {"sand_pct": 40.0, "silt_pct": 40.0, "clay_pct": 20.0},
    "silt":             {"sand_pct": 5.0,  "silt_pct": 88.0, "clay_pct": 7.0},
    "silt_loam":        {"sand_pct": 20.0, "silt_pct": 65.0, "clay_pct": 15.0},
    "sandy_clay_loam":  {"sand_pct": 60.0, "silt_pct": 13.0, "clay_pct": 27.0},
    "clay_loam":        {"sand_pct": 32.0, "silt_pct": 34.0, "clay_pct": 34.0},
    "silty_clay_loam":  {"sand_pct": 10.0, "silt_pct": 56.0, "clay_pct": 34.0},
    "sandy_clay":       {"sand_pct": 52.0, "silt_pct": 6.0,  "clay_pct": 42.0},
    "silty_clay":       {"sand_pct": 6.0,  "silt_pct": 47.0, "clay_pct": 47.0},
    "clay":             {"sand_pct": 22.0, "silt_pct": 20.0, "clay_pct": 58.0},
}


def usda_class_to_vg(class_name: str) -> PTFResult:
    """Look up VG parameters for a USDA texture class (Carsel-Parrish 1988 means)."""
    return carsel_parrish_lookup(class_name)
