"""OptimizationResult — typed output of nsga/optuna runs."""
from __future__ import annotations
from typing import Literal
from pydantic import BaseModel, ConfigDict


OptMethod = Literal["nsga2", "nsga3", "optuna_tpe", "optuna_random", "optuna_cmaes"]


class OptimizationResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    method: OptMethod
    param_names: list[str]
    objective_names: list[str]
    pareto_thetas: list[list[float]]       # (N_pareto, D); for single-obj, just the best as [[θ*]]
    pareto_objectives: list[list[float]]   # (N_pareto, n_obj)
    history: list[list[float]] = []         # objective trace per evaluation (single-obj) or generation (multi)
    n_evaluations: int
    wall_s: float
    diagnostics: dict = {}
