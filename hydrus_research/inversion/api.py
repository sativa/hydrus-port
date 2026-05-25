"""fit() dispatcher with auto backend selection.

Selection rule (per spec §4.4):
  - params < 10 AND 1D simulator           → lm_scipy
  - params ≥ 10  OR  2D/3D simulator       → pyemu_ies
  - user requests posterior explicitly      → pymc_nuts (P1; raises until M9)

The two backends have different argument shapes:
  - lm_scipy needs `forward` (any callable).
  - pyemu_ies needs `scenario_dir` (path) — its workers rebuild forward
    from JSON config; passing a Python callable is not supported because
    PEST++ subprocess workers cannot share Python objects.

The dispatcher requires BOTH `forward` and `scenario_dir` and forwards
each to the relevant backend."""
from __future__ import annotations
from pathlib import Path
from typing import Callable, Literal
import numpy as np

from .base import InversionResult
from .lm_scipy import fit_lm
from .pyemu_pestpp import fit_pyemu


def fit(forward: Callable[[np.ndarray], np.ndarray] | None,
        param_map,
        obs,
        scenario_dir: str | Path | None = None,
        backend: Literal["auto", "lm", "lm_scipy",
                         "ies", "pyemu_ies", "glm", "pyemu_glm",
                         "nuts", "pymc_nuts"] = "auto",
        simulator_dimension: int = 1,
        **kwargs) -> InversionResult:
    """Dispatch to the right inversion backend."""
    if backend == "auto":
        D = len(param_map.names) if hasattr(param_map, "names") else len(param_map.specs)
        backend = "lm" if (D < 10 and simulator_dimension == 1) else "ies"

    if backend in ("lm", "lm_scipy"):
        if forward is None:
            raise ValueError("LM backend requires a `forward` callable")
        lm_kwargs = {k: v for k, v in kwargs.items() if k in ("x0", "max_nfev")}
        return fit_lm(forward=forward, param_map=param_map, obs=obs, **lm_kwargs)
    if backend in ("ies", "pyemu_ies"):
        if scenario_dir is None:
            raise ValueError(
                "PESTPP-IES backend requires a `scenario_dir` path (workers "
                "rebuild forward locally; Python callables can't cross subprocess "
                "boundaries). Pass scenario_dir=Path(...)."
            )
        pe_kwargs = {k: v for k, v in kwargs.items()
                     if k in ("n_real", "n_iter", "workspace")}
        return fit_pyemu(scenario_dir=scenario_dir, param_map=param_map, obs=obs,
                         method="ies", **pe_kwargs)
    if backend in ("glm", "pyemu_glm"):
        if scenario_dir is None:
            raise ValueError("PESTPP-GLM backend requires a `scenario_dir` path")
        pe_kwargs = {k: v for k, v in kwargs.items()
                     if k in ("n_real", "n_iter", "workspace")}
        return fit_pyemu(scenario_dir=scenario_dir, param_map=param_map, obs=obs,
                         method="glm", **pe_kwargs)
    if backend in ("nuts", "pymc_nuts"):
        from .pymc_bayes import fit_pymc
        return fit_pymc(forward=forward, param_map=param_map, obs=obs, **kwargs)
    raise ValueError(f"unknown backend {backend!r}")
