"""UQResult — shared output for MC / posterior-predict / GLUE."""
from __future__ import annotations
from typing import Literal
import numpy as np
from pydantic import BaseModel, ConfigDict


UQMethod = Literal["ptf_mc", "posterior_predict", "glue"]


class UQResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    method: UQMethod
    param_names: list[str]
    obs_names: list[str]
    ys: list[list[float]]                  # (N, M) ensemble of predictions
    weights: list[float] | None = None     # GLUE behavioral weights
    quantiles: dict[str, list[float]] = {}  # e.g. {"p2.5": [...], "p50": [...], "p97.5": [...]}
    n_samples: int
    diagnostics: dict = {}
