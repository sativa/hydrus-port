"""Shared InversionResult schema."""
from __future__ import annotations
from typing import Any, Literal
from pydantic import BaseModel, ConfigDict


Backend = Literal["lm_scipy", "pyemu_glm", "pyemu_ies", "pymc_nuts"]


class InversionResult(BaseModel):
    """Result of one inversion run. Fields are nullable when a backend
    can't provide them (LM gives no posterior; IES gives no jacobian)."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    backend: Backend
    best_params: dict[str, float]
    parameter_ci_lo: dict[str, float] = {}
    parameter_ci_hi: dict[str, float] = {}
    posterior_ensemble: list[list[float]] | None = None       # (N_real, D); None for LM
    posterior_param_names: list[str] = []
    objective_history: list[float] = []
    n_forward_calls: int
    wall_s: float
    pest_workspace: str | None = None
    diagnostics: dict[str, Any] = {}
