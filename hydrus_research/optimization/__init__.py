"""Decision optimization (F6, P2).

Two backends:
  - pymoo NSGA-II — multi-objective Pareto search
  - Optuna       — single-objective (TPE / random / CMA-ES)

Decision variables encode irrigation / fert event schedules.

See DOCS/superpowers/specs/2026-05-24-hydrus-research-platform-design.md §4.7.
"""
from .result import OptimizationResult
from .pymoo_nsga import nsga_optimize
from .optuna_single import optuna_optimize
from .decision_vars import encode_schedule, decode_schedule

__all__ = ["OptimizationResult", "nsga_optimize", "optuna_optimize",
           "encode_schedule", "decode_schedule"]
