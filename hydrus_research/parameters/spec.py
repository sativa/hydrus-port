"""ParameterSpec — one calibrated/sampled parameter, in user units.

Optimizers see an *internal* coordinate (linear / log / logit) that is
unbounded or trivially bounded, making bound-handling robust. The
back-transform happens here, not in the optimizer."""
from __future__ import annotations
from typing import Literal
import numpy as np
from pydantic import BaseModel, Field, field_validator


Transform = Literal["linear", "log", "logit"]


class ParameterSpec(BaseModel):
    name: str
    target: str                              # path into canonical scenario, e.g. "materials[0].alpha"
    bounds: tuple[float, float]
    transform: Transform = "linear"
    prior_mean: float | None = None          # in user units
    prior_std: float | None = None
    group: str = "default"

    @field_validator("bounds")
    @classmethod
    def _bounds_ordered(cls, v):
        lo, hi = v
        if lo >= hi:
            raise ValueError(f"bounds must satisfy lo < hi; got {v}")
        return v

    @field_validator("transform")
    @classmethod
    def _transform_known(cls, v):
        if v not in ("linear", "log", "logit"):
            raise ValueError(f"unknown transform: {v!r}")
        return v

    def to_internal(self, user: float) -> float:
        if self.transform == "linear":
            return user
        if self.transform == "log":
            return float(np.log(user))
        # logit
        lo, hi = self.bounds
        u = (user - lo) / (hi - lo)
        return float(np.log(u / (1.0 - u)))

    def from_internal(self, internal: float) -> float:
        if self.transform == "linear":
            return internal
        if self.transform == "log":
            return float(np.exp(internal))
        # logit
        lo, hi = self.bounds
        u = 1.0 / (1.0 + np.exp(-internal))
        return float(lo + u * (hi - lo))

    def internal_bounds(self) -> tuple[float, float]:
        lo, hi = self.bounds
        if self.transform == "linear":
            return lo, hi
        if self.transform == "log":
            return float(np.log(lo)), float(np.log(hi))
        # logit: maps (lo, hi) -> (-inf, inf); choose wide finite bounds for optimizers
        return -1e6, 1e6
