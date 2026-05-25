"""SensitivityResult — typed output of every sensitivity-analysis call."""
from __future__ import annotations
from typing import Literal
from pydantic import BaseModel, ConfigDict


SensMethod = Literal["morris", "sobol", "fast", "pawn"]


class SensitivityResult(BaseModel):
    """One index dict per observable (or aggregated)."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    method: SensMethod
    param_names: list[str]
    obs_names: list[str]
    # indices[index_name] -> list of length D (param indices) OR list of M lists of D
    # Common keys per method:
    #   morris: mu, mu_star, sigma, mu_star_conf
    #   sobol:  S1, S1_conf, ST, ST_conf, [S2 if calc_second_order]
    #   fast:   S1, ST
    #   pawn:   minimum, mean, median, maximum, CV
    indices: dict[str, list[list[float]] | list[float]]
    sample_size: int
    forward_cost_s: float
    diagnostics: dict = {}
