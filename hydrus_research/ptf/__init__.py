"""Pedotransfer functions (F1) — texture → van Genuchten hydraulic params.

Three backends share a single PTFResult schema:
  - rosetta3_*  — neural-net PTFs (rosetta-soil package, USDA-ARS)
  - carsel_parrish — 12 USDA-class lookup (1988)
  - wosten — HYPRES continuous closed-form (Wösten 1999)

Entry point: texture_to_vg(...). For the 12-class shortcut use usda_class_to_vg(name).
"""
from .result import PTFResult
from .api import texture_to_vg
from .presets import usda_class_to_vg
from .uncertainty import vg_to_prior

__all__ = ["PTFResult", "texture_to_vg", "usda_class_to_vg", "vg_to_prior"]
