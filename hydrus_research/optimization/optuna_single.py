"""Optuna single-objective optimization (stub — Task 1)."""
from __future__ import annotations
from typing import Callable, Literal
import numpy as np

from .result import OptimizationResult


def optuna_optimize(forward_scalar: Callable[[np.ndarray], float],
                    bounds: np.ndarray,
                    param_names: list[str],
                    objective_name: str = "objective",
                    n_trials: int = 100,
                    sampler: Literal["tpe", "random", "cmaes"] = "tpe",
                    direction: Literal["minimize", "maximize"] = "minimize",
                    seed: int | None = None) -> OptimizationResult:
    raise NotImplementedError("optuna_optimize not yet implemented — see Task 3")
