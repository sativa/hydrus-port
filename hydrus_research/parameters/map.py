"""ParameterMap — bijection between an optimizer's theta vector and a
named-parameter dict that the Simulator (and scenario JSON) understands."""
from __future__ import annotations
from typing import Any
import numpy as np

from .spec import ParameterSpec


class ParameterMap:
    def __init__(self, specs: list[ParameterSpec]):
        names = [s.name for s in specs]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate ParameterSpec names: {names}")
        self.specs = list(specs)
        self._index = {s.name: i for i, s in enumerate(self.specs)}

    @property
    def names(self) -> list[str]:
        return [s.name for s in self.specs]

    @property
    def D(self) -> int:
        return len(self.specs)

    def to_vector(self, named: dict[str, float]) -> np.ndarray:
        theta = np.empty(self.D, dtype=float)
        for s in self.specs:
            if s.name not in named:
                raise KeyError(f"missing parameter {s.name!r} in {named!r}")
            theta[self._index[s.name]] = s.to_internal(named[s.name])
        return theta

    def from_vector(self, theta: np.ndarray) -> dict[str, float]:
        theta = np.asarray(theta, dtype=float)
        if theta.shape != (self.D,):
            raise ValueError(f"theta shape {theta.shape}, expected ({self.D},)")
        return {s.name: s.from_internal(theta[self._index[s.name]]) for s in self.specs}

    def bounds_array(self) -> np.ndarray:
        """(D, 2) array of internal-coord bounds, ready for scipy / pymoo."""
        return np.array([s.internal_bounds() for s in self.specs], dtype=float)

    def midpoints(self) -> np.ndarray:
        """In *user* coords, midpoint of each spec's bounds, expressed back as
        an internal theta vector. Useful as x0 for optimizers when no prior."""
        mids_user = {s.name: 0.5 * (s.bounds[0] + s.bounds[1]) for s in self.specs}
        return self.to_vector(mids_user)
