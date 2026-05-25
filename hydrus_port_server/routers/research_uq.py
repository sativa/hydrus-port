"""/research/uq/{method} — F5 REST surface."""
from __future__ import annotations
from typing import Any, Literal

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from hydrus_research.batch import BatchResult
from hydrus_research.uq import glue_filter


router = APIRouter()


class GLUEPayload(BaseModel):
    thetas: list[list[float]]
    ys: list[list[float]]
    param_names: list[str]
    obs_names: list[str]
    obs_values: list[float]
    obs_sigmas: list[float]
    likelihood_cutoff: float = 0.5


_VALID = {"mc", "ptf_mc", "posterior_predict", "posterior", "glue"}


@router.post("/glue")
def glue(p: GLUEPayload):
    br = BatchResult(
        thetas=np.array(p.thetas), ys=np.array(p.ys),
        wall_s=np.zeros(len(p.thetas)),
        converged=np.ones(len(p.thetas), dtype=bool),
        param_names=p.param_names, obs_names=p.obs_names, meta={},
    )
    r = glue_filter(br,
                    obs_values=np.array(p.obs_values),
                    obs_sigmas=np.array(p.obs_sigmas),
                    likelihood_cutoff=p.likelihood_cutoff)
    return r.model_dump()


@router.post("/{method}")
def unknown_or_other(method: str):
    if method not in _VALID:
        raise HTTPException(status_code=404,
                            detail=f"unknown UQ method {method!r}")
    raise HTTPException(status_code=501,
                        detail=f"UQ method {method!r} not exposed via REST; "
                               "use the Python API (mc / posterior_predict)")
