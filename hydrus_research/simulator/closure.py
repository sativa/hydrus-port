"""make_forward — produces the single (theta -> y) callable that every
research module consumes. This is the narrow waist of the architecture."""
from __future__ import annotations
from typing import Callable
import numpy as np

from .base import Simulator, Forcing, InitialState
from ..parameters import ParameterMap
from ..observations import ObservationSpec


def make_forward(
    simulator: Simulator,
    param_map: ParameterMap,
    template_scenario: dict,
    forcing: Forcing | None,
    ic: InitialState | None,
    obs_specs: list[ObservationSpec],
) -> Callable[[np.ndarray], np.ndarray]:
    """Build a pure function `forward(theta) -> y_sim` of length M = len(obs_specs).

    `theta` is in *internal* coords (see ParameterSpec transforms). On each
    call: from_vector → apply_to_scenario(template_scenario, named) → run
    → batch_observables. The adapter never sees parameter specs — its only
    contract is "take patched scenario dict → return SimResult"."""
    def forward(theta: np.ndarray) -> np.ndarray:
        named = param_map.from_vector(theta)
        scenario = param_map.apply_to_scenario(template_scenario, named)
        result = simulator.run(scenario, forcing, ic)
        return simulator.batch_observables(result, obs_specs)
    return forward
