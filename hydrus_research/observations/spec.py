"""ObservationSpec — one scalar observation point.

A spec carries enough information for any Simulator.observable_at()
implementation to sample the right scalar from a SimResult."""
from __future__ import annotations
from typing import Literal
from pydantic import BaseModel, Field


ObsKind = Literal[
    "theta", "h", "c",
    "flux", "cumulative_flux", "concentration_flux",
]


class ObservationSpec(BaseModel):
    name: str
    kind: ObsKind
    location: dict                       # 1D: {"z_cm": float}; 2D/3D: {"node": int} or {"xyz": [x,y,z]}
    time_day: float
    weight: float = 1.0
    species: str | None = None           # solute species name, only for kind == "c" or "concentration_flux"
