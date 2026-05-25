"""pymoo NSGA-II / NSGA-III multi-objective optimization (stub — Task 1)."""
from __future__ import annotations
from typing import Callable
import numpy as np

from .result import OptimizationResult


def nsga_optimize(forward: Callable[[np.ndarray], np.ndarray],
                  bounds: np.ndarray,
                  param_names: list[str],
                  objective_names: list[str],
                  pop_size: int = 50,
                  n_gen: int = 20,
                  seed: int | None = None,
                  variant: str = "nsga2") -> OptimizationResult:
    raise NotImplementedError("nsga_optimize not yet implemented — see Task 2")
