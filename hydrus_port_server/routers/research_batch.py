"""/research/batch/* — async batch-sweep REST routes (M3.7).

Job model: each POST /start creates an entry in an in-process dict (_JOBS).
This is process-local — fine for the single-user desktop GUI. Multi-process
production deployments would need an external store (Redis, SQLite on disk, etc.).

State transitions: pending → running → done | failed.
The parquet result is written to a temp file and streamed on GET /{job_id}/result.
"""
from __future__ import annotations
import tempfile
import uuid
from pathlib import Path
from typing import Any, Literal

from fastapi import APIRouter, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel


router = APIRouter()

# In-process job registry.
# WARNING: restart of the server = all job records are lost.
# This is intentional for a desktop GUI (single-user, short-lived process).
_JOBS: dict[str, dict[str, Any]] = {}


class ParamSpecPayload(BaseModel):
    name: str
    target: str
    bounds: tuple[float, float]
    transform: Literal["linear", "log", "logit"] = "linear"


class ObsSpecPayload(BaseModel):
    name: str
    kind: Literal["theta", "h", "c", "flux", "cumulative_flux", "concentration_flux"]
    location: dict
    time_day: float


class StartRequest(BaseModel):
    scenario_dir: str
    params: list[ParamSpecPayload]
    obs: list[ObsSpecPayload]
    n: int = 32
    sampler: Literal["lhs", "grid", "uniform"] = "lhs"
    workers: int = 1
    seed: int | None = None


class StartResponse(BaseModel):
    job_id: str


@router.post("/start", response_model=StartResponse)
def start(req: StartRequest, bg: BackgroundTasks):
    """Enqueue a batch sweep. Returns a job_id immediately; the sweep runs
    in a FastAPI BackgroundTask (single-process, non-blocking for the caller)."""
    job_id = uuid.uuid4().hex[:12]
    out_path = Path(tempfile.gettempdir()) / f"hydrus_batch_{job_id}.parquet"
    _JOBS[job_id] = {
        "state": "pending",
        "n_total": req.n,
        "n_done": 0,
        "out_path": str(out_path),
        "error": None,
    }
    bg.add_task(_run_job, job_id, req)
    return StartResponse(job_id=job_id)


@router.get("/{job_id}/status")
def status(job_id: str):
    """Return current job state dict."""
    if job_id not in _JOBS:
        raise HTTPException(status_code=404, detail="unknown job_id")
    return _JOBS[job_id]


@router.get("/{job_id}/result")
def result(job_id: str):
    """Stream the parquet result file. Raises 409 if the job is not yet done."""
    if job_id not in _JOBS:
        raise HTTPException(status_code=404, detail="unknown job_id")
    j = _JOBS[job_id]
    if j["state"] != "done":
        raise HTTPException(
            status_code=409,
            detail=f"job not done (state={j['state']})",
        )
    return FileResponse(
        j["out_path"],
        media_type="application/octet-stream",
        filename=f"sweep_{job_id}.parquet",
    )


# ---------------------------------------------------------------- background task

def _run_job(job_id: str, req: StartRequest) -> None:
    """Execute the batch sweep synchronously in a FastAPI BackgroundTask thread.

    Updates _JOBS[job_id]["state"] to "running" → "done" | "failed".
    Imports are lazy (inside the function) so the router can be imported
    even when hydrus_research extras are not installed — the ImportError
    would only surface when a job is actually started.
    """
    from hydrus_research.batch import BatchRunner
    from hydrus_research.batch.sampling import lhs, grid, uniform_random
    from hydrus_research.parameters import ParameterSpec, ParameterMap
    from hydrus_research.observations import ObservationSpec
    from hydrus_research.simulator import make_forward
    from hydrus_research.simulator.hydrus1d_adapter import Hydrus1DSimulator
    from hydrus_port.adapters.hydrus1d import load as _load_h1d

    j = _JOBS[job_id]
    j["state"] = "running"
    try:
        # Build ParameterMap from payload
        specs = [
            ParameterSpec(
                name=p.name,
                target=p.target,
                bounds=p.bounds,
                transform=p.transform,
            )
            for p in req.params
        ]
        pm = ParameterMap(specs)

        # Build ObservationSpec list from payload
        obs_specs = [
            ObservationSpec(
                name=o.name,
                kind=o.kind,
                location=o.location,
                time_day=o.time_day,
            )
            for o in req.obs
        ]

        # Load scenario template + build forward callable
        template = _load_h1d(Path(req.scenario_dir)).to_dict()
        sim = Hydrus1DSimulator()
        forward = make_forward(
            sim, pm,
            template_scenario=template,
            forcing=None,
            ic=None,
            obs_specs=obs_specs,
        )

        # Generate thetas
        bounds = pm.bounds_array()
        if req.sampler == "lhs":
            thetas = lhs(bounds, n=req.n, seed=req.seed)
        elif req.sampler == "grid":
            per = max(1, int(round(req.n ** (1.0 / max(1, len(specs))))))
            thetas = grid(bounds, points_per_axis=[per] * len(specs))
        else:
            thetas = uniform_random(bounds, n=req.n, seed=req.seed)

        # Run the sweep
        runner = BatchRunner(
            forward=forward,
            param_names=[s.name for s in specs],
            obs_names=[o.name for o in req.obs],
            n_workers=req.workers,
            show_progress=False,
        )
        batch_result = runner.run(thetas)

        # Persist to temp parquet
        batch_result.to_parquet(j["out_path"])

        j["state"] = "done"
        j["n_done"] = batch_result.N
        j["n_failed"] = batch_result.n_failed

    except Exception as e:
        j["state"] = "failed"
        j["error"] = f"{type(e).__name__}: {e}"
        raise
