"""Core dataclasses + Simulator ABC for the research platform.

Forcing  — time-varying drivers (populated by dndc_seam.to_forcing()).
InitialState — z-profiles of theta / h / c / T at t=0 (also used for the
               returned final state, enabling DNDC day-step restart).
SimResult — what every Simulator.run() returns.
Event — irrigation or fertilizer event.
Simulator — ABC; subclasses wrap the real solvers (1D / 2D / 3D / surrogate).
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable
import numpy as np


@dataclass(frozen=True)
class Event:
    """Irrigation or fertilizer event. `method` distinguishes the two."""
    time_day: float
    depth_cm: float                              # 0 = surface
    amount: float                                # cm for irrig; kg N / ha for fert
    method: str                                  # "drip" | "sprinkler" | "flood" | "subsurface" | "fert"
    solute_concs_mg_l: dict[str, float] = field(default_factory=dict)
    form: str | None = None                      # "NH4" | "NO3" | "urea" | ... (fert only)


@dataclass(frozen=True, eq=False)
class Forcing:
    """All time-varying drivers for one Simulator.run().

    Treated as immutable by convention; ndarray fields prevent hash-based collections.
    """
    times_days: np.ndarray
    precip_cm_per_day: np.ndarray
    pet_cm_per_day: np.ndarray
    lai: np.ndarray
    root_depth_cm: np.ndarray
    root_density_fn: Callable[[np.ndarray, float], np.ndarray]   # (z, t) -> normalized beta(z)
    irrigation_events: list[Event]
    fert_events: list[Event]
    n_source_terms: Callable[..., tuple[float, float]]           # (z, t, theta, c) -> (gamma_w, gamma_s); B2 hook
    air_temp_c: np.ndarray | None


@dataclass(frozen=True, eq=False)
class InitialState:
    """Treated as immutable by convention; ndarray fields prevent hash-based collections."""
    z_cm: np.ndarray
    theta: np.ndarray | None
    h_cm: np.ndarray | None
    c_mg_per_L: np.ndarray | None
    t_celsius: np.ndarray | None


@dataclass(frozen=True, eq=False)
class SimResult:
    """Raw simulator output. 1D adapters store theta/h with shape (NT, NZ);
    2D/3D adapters store mesh-node arrays of shape (NT, Nnode) and use
    `z` slot for mesh metadata or a dummy axis.

    Treated as immutable by convention; ndarray fields prevent hash-based collections.
    """
    times: np.ndarray
    z: np.ndarray
    theta: np.ndarray
    h: np.ndarray
    c: np.ndarray | None
    fluxes: dict[str, np.ndarray]
    mass_balance: dict[str, float]
    final_state: InitialState
    meta: dict[str, Any]


class Simulator(ABC):
    """Pure-function interface. No hidden state between runs.

    Subclasses must set class attributes `name` and `dimension`.

    `run` takes a **fully patched canonical scenario dict** (parameter
    application happens upstream in `make_forward` via
    `ParameterMap.apply_to_scenario`). `forcing=None` means "use whatever
    atmospheric / sink data is already inside the scenario"; non-None
    forcing overrides them (DNDC seam consumes this from M1 on).
    """

    name: str = ""
    dimension: int = 0

    @abstractmethod
    def run(self, scenario: dict, forcing: Forcing | None,
            ic: InitialState | None) -> SimResult: ...

    @abstractmethod
    def observable_at(self, result: SimResult,
                      spec: ObservationSpec) -> float: ...

    def batch_observables(self, result: SimResult,
                          specs: list[ObservationSpec]) -> np.ndarray:
        return np.array([self.observable_at(result, s) for s in specs])
