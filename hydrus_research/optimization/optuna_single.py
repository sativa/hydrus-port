"""Optuna single-objective optimization (TPE / random / CMA-ES)."""
from __future__ import annotations
import time
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
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError as e:
        raise ImportError(
            "optuna_optimize requires optuna. Install with:\n"
            "    pip install 'hydrus-port[research,research-opt]'"
        ) from e

    bounds = np.asarray(bounds, dtype=float)
    D = bounds.shape[0]

    if sampler == "tpe":
        samp = optuna.samplers.TPESampler(seed=seed)
        method_label = "optuna_tpe"
    elif sampler == "random":
        samp = optuna.samplers.RandomSampler(seed=seed)
        method_label = "optuna_random"
    elif sampler == "cmaes":
        samp = optuna.samplers.CmaEsSampler(seed=seed)
        method_label = "optuna_cmaes"
    else:
        raise ValueError(f"unknown sampler {sampler!r}")

    study = optuna.create_study(direction=direction, sampler=samp)

    def _objective(trial):
        theta = np.array([
            trial.suggest_float(name, float(lo), float(hi))
            for name, (lo, hi) in zip(param_names, bounds)
        ])
        return forward_scalar(theta)

    t0 = time.time()
    study.optimize(_objective, n_trials=n_trials)
    wall = time.time() - t0

    best_theta = np.array([study.best_params[n] for n in param_names])
    history = [float(t.value) for t in study.trials if t.value is not None]

    return OptimizationResult(
        method=method_label,                  # type: ignore[arg-type]
        param_names=param_names,
        objective_names=[objective_name],
        pareto_thetas=[best_theta.tolist()],
        pareto_objectives=[[float(study.best_value)]],
        history=[[v] for v in history],
        n_evaluations=int(n_trials),
        wall_s=float(wall),
        diagnostics={"sampler": sampler, "direction": direction,
                     "best_value": float(study.best_value)},
    )
