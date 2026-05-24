"""ParameterMap — bijection between an optimizer's theta vector and a
named-parameter dict that the Simulator (and scenario JSON) understands."""
from __future__ import annotations
import copy
import re
from typing import Any
import numpy as np

from .spec import ParameterSpec


_INDEX_RE = re.compile(r"^([^\[]+)\[(\d+)\]$")


def _walk(d: Any, parts: list[str]) -> tuple[Any, str | int]:
    """Walk `parts` over a nested dict / list; return (container, last_key).
    `parts` are dotted keys, optionally suffixed with `[N]` for list indexing."""
    cur = d
    for i, part in enumerate(parts[:-1]):
        m = _INDEX_RE.match(part)
        if m:
            key, idx = m.group(1), int(m.group(2))
            cur = cur[key]
            cur = cur[idx]
        else:
            cur = cur[part]
    last = parts[-1]
    m = _INDEX_RE.match(last)
    if m:
        key, idx = m.group(1), int(m.group(2))
        return cur[key], idx
    return cur, last


class _ApplyMixin:
    def apply_to_scenario(self, scenario: dict, named: dict[str, float]) -> dict:
        """Return a deep copy of `scenario` with each named value patched into
        its ParameterSpec.target path. `target` syntax: dotted keys with
        optional `[N]` indexing, e.g. `materials[0].alpha`, `solver.tol_theta`,
        `geometry.nodes[12].h_init`."""
        out = copy.deepcopy(scenario)
        for s in self.specs:
            if s.name not in named:
                continue
            value = named[s.name]
            parts = s.target.split(".")
            try:
                container, last_key = _walk(out, parts)
            except (KeyError, IndexError, TypeError) as e:
                raise KeyError(f"target {s.target!r} not found in scenario: {e}") from e
            try:
                container[last_key] = value
            except (KeyError, IndexError, TypeError) as e:
                raise KeyError(f"cannot set {s.target!r} in scenario: {e}") from e
        return out


class ParameterMap(_ApplyMixin):
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
