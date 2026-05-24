"""ObservationSpec — one scalar observation point.

A spec carries enough information for any Simulator.observable_at()
implementation to sample the right scalar from a SimResult."""
from __future__ import annotations
from typing import Literal
from pydantic import BaseModel, Field, model_validator


ObsKind = Literal[
    "theta", "h", "c",
    "flux", "cumulative_flux", "concentration_flux",
]

_ALLOWED_LOCATION_KEYS = {"z_cm", "node", "xyz"}


class ObservationSpec(BaseModel):
    name: str
    kind: ObsKind
    location: dict                       # 1D: {"z_cm": float}; 2D/3D: {"node": int} or {"xyz": [x,y,z]}
    time_day: float
    weight: float = 1.0
    species: str | None = None           # solute species name, only for kind == "c" or "concentration_flux"

    @model_validator(mode="after")
    def _location_well_formed(self):
        if not self.location:
            raise ValueError("location dict must not be empty")
        keys = set(self.location)
        unknown = keys - _ALLOWED_LOCATION_KEYS
        if unknown:
            raise ValueError(f"location has unknown keys {sorted(unknown)}; "
                             f"allowed: {sorted(_ALLOWED_LOCATION_KEYS)}")
        return self
