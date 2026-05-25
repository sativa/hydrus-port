"""/research/surrogate/* — train + evaluate."""
from __future__ import annotations
import uuid
from pathlib import Path
from typing import Any, Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from hydrus_research.batch import BatchResult
from hydrus_research.surrogate import train_gp, train_pck, evaluate


router = APIRouter()
_MODELS: dict[str, Any] = {}


class TrainRequest(BaseModel):
    batch_parquet: str
    type: Literal["gp", "pck"] = "gp"


class TrainResponse(BaseModel):
    model_id: str
    type: str


@router.post("/train", response_model=TrainResponse)
def train(req: TrainRequest):
    br = BatchResult.from_parquet(Path(req.batch_parquet))
    if req.type == "gp":
        surr = train_gp(br)
    else:
        try:
            surr = train_pck(br)
        except ImportError as e:
            raise HTTPException(status_code=503, detail=f"pck deps missing: {e}")
    mid = uuid.uuid4().hex[:12]
    _MODELS[mid] = {"surrogate": surr, "batch": br}
    return TrainResponse(model_id=mid, type=req.type)


@router.post("/{model_id}/evaluate")
def eval_route(model_id: str):
    if model_id not in _MODELS:
        raise HTTPException(status_code=404, detail="unknown model_id")
    entry = _MODELS[model_id]
    return evaluate(entry["surrogate"], entry["batch"])
