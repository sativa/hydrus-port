"""Public texture_to_vg() entry point — dispatches to ROSETTA-3, Carsel-Parrish,
or Wösten HYPRES based on `method`. `method='rosetta3_auto'` picks the deepest
ROSETTA hierarchy supported by the provided inputs."""
from __future__ import annotations
from typing import Literal

from .result import PTFResult


PTFMethodArg = Literal["rosetta3_auto", "carsel_parrish", "wosten",
                       "rosetta3_h1", "rosetta3_h2", "rosetta3_h3", "rosetta3_h4"]


def texture_to_vg(sand_pct: float, silt_pct: float, clay_pct: float,
                  bulk_density_g_cm3: float | None = None,
                  theta_33: float | None = None,
                  theta_1500: float | None = None,
                  organic_matter_pct: float | None = None,
                  organic_carbon_pct: float | None = None,
                  topsoil: bool = True,
                  method: PTFMethodArg = "rosetta3_auto") -> PTFResult:
    """Pedotransfer dispatcher. See package docstring for backend semantics."""
    if method.startswith("rosetta3"):
        from .rosetta import rosetta3_predict
        # `rosetta3_auto` lets rosetta3_predict pick model by which inputs are non-None
        return rosetta3_predict(
            sand_pct=sand_pct, silt_pct=silt_pct, clay_pct=clay_pct,
            bulk_density_g_cm3=bulk_density_g_cm3,
            theta_33=theta_33, theta_1500=theta_1500,
        )
    if method == "carsel_parrish":
        from .carsel_parrish import carsel_parrish_lookup
        from .presets import USDA_TEXTURE_CENTERS
        # Snap to the nearest USDA class center
        nearest, best_d2 = "loam", float("inf")
        for cname, c in USDA_TEXTURE_CENTERS.items():
            d2 = ((c["sand_pct"] - sand_pct) ** 2
                  + (c["silt_pct"] - silt_pct) ** 2
                  + (c["clay_pct"] - clay_pct) ** 2)
            if d2 < best_d2:
                nearest, best_d2 = cname, d2
        return carsel_parrish_lookup(nearest)
    if method == "wosten":
        from .wosten_hypres import wosten_predict
        # Wösten needs OM; OC is often the available measurement (OM ≈ 1.724 * OC)
        om = organic_matter_pct
        if om is None:
            if organic_carbon_pct is not None:
                om = 1.724 * organic_carbon_pct
            else:
                raise ValueError("wosten requires organic_matter_pct or organic_carbon_pct")
        if bulk_density_g_cm3 is None:
            raise ValueError("wosten requires bulk_density_g_cm3")
        return wosten_predict(sand_pct=sand_pct, silt_pct=silt_pct, clay_pct=clay_pct,
                              bulk_density_g_cm3=bulk_density_g_cm3,
                              organic_matter_pct=om, topsoil=topsoil)
    raise ValueError(f"unknown method {method!r}")
