"""/research/dndc/* — DNDC seam REST endpoints.

Validate the schema, list / load / save crop presets. The Forcing object is
NOT exposed over REST (it contains live callables); to materialize a Forcing
for inspection, use the Python API directly.

Decoupling: this file imports only fastapi + pydantic + hydrus_research.
No direct hydrus1d / swms2d / desktop imports are permitted here, and
hydrus_research MUST NOT import from hydrus_port_server (no reverse dep).
"""
from __future__ import annotations
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ValidationError

from hydrus_research.dndc_seam import DndcSeamInputs
from hydrus_research.dndc_seam.presets import list_crop_presets, load_crop_preset


router = APIRouter()


class ValidateResponse(BaseModel):
    ok: bool
    warnings: list[str] = []


@router.post("/validate", response_model=ValidateResponse)
def validate(payload: dict[str, Any]):
    """Validate a DndcSeamInputs payload. 200 + ok=True if valid; 422 if not."""
    try:
        DndcSeamInputs.model_validate(payload)
    except ValidationError as e:
        # Pydantic v2 errors() may contain non-JSON-serializable `ctx` values
        # (e.g. ValueError instances); strip them to just type/loc/msg.
        errors = [
            {"type": err["type"], "loc": list(err["loc"]), "msg": err["msg"]}
            for err in e.errors()
        ]
        raise HTTPException(status_code=422, detail=errors)
    return ValidateResponse(ok=True)


@router.get("/crop-presets")
def crop_presets() -> dict[str, dict]:
    """Return all hardcoded crop presets as a dict {name: {feddes, root, description}}."""
    out: dict[str, dict] = {}
    for name in list_crop_presets():
        feddes, root, desc = load_crop_preset(name)
        out[name] = {
            "feddes": feddes.model_dump(),
            "root": root.model_dump(),
            "description": desc,
        }
    return out
