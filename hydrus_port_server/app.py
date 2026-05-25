"""FastAPI app: backend service for the Tauri GUI.

Endpoints
---------

GET  /api/health                    -> {status, versions}
GET  /api/scenarios                 -> list builtin example scenarios (EX.1..4 etc)
POST /api/simulate/swms2d           -> launch a SWMS_2D run (returns job_id)
POST /api/simulate/hydrus1d         -> launch a HYDRUS-1D run
POST /api/simulate/richards3d       -> launch a 3D Richards run
GET  /api/jobs                      -> list jobs
GET  /api/jobs/{job_id}             -> job status / result paths
GET  /api/jobs/{job_id}/log         -> tail stdout
GET  /api/jobs/{job_id}/files/{f}   -> download a result file

The server keeps jobs in an in-memory dict and shells out simulations
into a background thread. For a single-user desktop GUI this is fine
— don't reuse it as a multi-tenant service.
"""
from __future__ import annotations
import argparse
import io
import sys
import time
import uuid
import threading
import traceback
from pathlib import Path
from typing import Optional
from contextlib import redirect_stdout, redirect_stderr

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import FileResponse, PlainTextResponse
    from pydantic import BaseModel
    import uvicorn
    _FASTAPI_OK = True
except ImportError:
    _FASTAPI_OK = False


# --------------------------------------------------------------------
# Job registry
# --------------------------------------------------------------------

class _Job:
    __slots__ = ("id", "kind", "input_dir", "output_dir", "status",
                 "started", "finished", "log_buf", "error")

    def __init__(self, kind: str, input_dir: Path, output_dir: Path):
        self.id = uuid.uuid4().hex[:12]
        self.kind = kind
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.status = "pending"   # pending | running | done | failed
        self.started: Optional[float] = None
        self.finished: Optional[float] = None
        self.log_buf = io.StringIO()
        self.error: Optional[str] = None

    def to_dict(self):
        return dict(
            id=self.id, kind=self.kind,
            input_dir=str(self.input_dir),
            output_dir=str(self.output_dir),
            status=self.status,
            started=self.started, finished=self.finished,
            error=self.error,
        )


_JOBS: dict[str, _Job] = {}
_JOBS_LOCK = threading.Lock()


def _run_job(job: _Job):
    job.status = "running"
    job.started = time.time()
    buf = job.log_buf
    try:
        with redirect_stdout(buf), redirect_stderr(buf):
            if job.kind == "swms2d":
                from swms2d.swms2d import SWMS2DSimulation
                sim = SWMS2DSimulation(job.input_dir, job.output_dir)
                sim.run(verbose=True)
            elif job.kind == "hydrus1d":
                from hydrus1d.hydrus import run_simulation
                run_simulation(input_dir=str(job.input_dir),
                               output_dir=str(job.output_dir))
            elif job.kind == "richards3d":
                # 3D needs a programmatic scenario; for the GUI's
                # smoke test we run the validation column infiltration.
                from tests.validate_richards3d import run_case  # type: ignore
                run_case("hex", lump=True, t_end=0.05)
            else:
                raise ValueError(f"Unknown job kind: {job.kind}")
        job.status = "done"
    except Exception:
        job.error = traceback.format_exc()
        job.status = "failed"
        buf.write("\n" + job.error)
    finally:
        job.finished = time.time()


# --------------------------------------------------------------------
# Schemas
# --------------------------------------------------------------------

if _FASTAPI_OK:
    class SimRequest(BaseModel):
        input_dir: str
        output_dir: Optional[str] = None


# --------------------------------------------------------------------
# App factory
# --------------------------------------------------------------------

def build_app():
    """Factory used by both `main()` (uvicorn) and tests (TestClient).

    Registers /research/dndc/* (M1) and /research/ptf/* (M2) routers if the
    corresponding hydrus_research sub-packages are installable."""
    if not _FASTAPI_OK:
        raise ImportError(
            "hydrus_port_server requires FastAPI. Install with:\n"
            "    pip install 'hydrus-port[gui]'"
        )
    app = FastAPI(title="hydrus-port server", version="0.1.0")
    # Permissive CORS so Tauri webview (file:// or localhost) can call us.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
    )

    # M2: Register the PTF research router
    try:
        from .routers.research_ptf import router as ptf_router
        app.include_router(ptf_router, prefix="/research/ptf", tags=["research", "ptf"])
    except ImportError:
        pass

    @app.get("/api/health")
    def health():
        import hydrus1d
        import swms2d
        return dict(
            status="ok",
            hydrus1d=hydrus1d.__version__,
            swms2d=swms2d.__version__,
        )

    @app.get("/api/scenarios")
    def scenarios():
        out = []
        fix = Path(__file__).resolve().parents[1] / "tests" / "fixtures"
        if fix.exists():
            for sub in sorted(fix.iterdir()):
                if sub.is_dir():
                    out.append(dict(name=sub.name, path=str(sub)))
        return out

    def _submit(kind: str, req: "SimRequest"):
        ind = Path(req.input_dir).expanduser().resolve()
        if not ind.exists():
            raise HTTPException(404, f"input_dir not found: {ind}")
        outd = (Path(req.output_dir).expanduser().resolve()
                if req.output_dir else (ind / "out"))
        outd.mkdir(parents=True, exist_ok=True)
        job = _Job(kind, ind, outd)
        with _JOBS_LOCK:
            _JOBS[job.id] = job
        threading.Thread(target=_run_job, args=(job,), daemon=True).start()
        return job.to_dict()

    @app.post("/api/simulate/swms2d")
    def sim_swms2d(req: SimRequest):
        return _submit("swms2d", req)

    @app.post("/api/simulate/hydrus1d")
    def sim_h1d(req: SimRequest):
        return _submit("hydrus1d", req)

    @app.post("/api/simulate/richards3d")
    def sim_r3d(req: SimRequest):
        return _submit("richards3d", req)

    @app.get("/api/jobs")
    def list_jobs():
        with _JOBS_LOCK:
            return [j.to_dict() for j in _JOBS.values()]

    @app.get("/api/jobs/{job_id}")
    def get_job(job_id: str):
        with _JOBS_LOCK:
            j = _JOBS.get(job_id)
        if j is None:
            raise HTTPException(404, "job not found")
        return j.to_dict()

    @app.get("/api/jobs/{job_id}/log", response_class=PlainTextResponse)
    def job_log(job_id: str):
        with _JOBS_LOCK:
            j = _JOBS.get(job_id)
        if j is None:
            raise HTTPException(404, "job not found")
        return j.log_buf.getvalue()

    @app.get("/api/jobs/{job_id}/files/{fname}")
    def job_file(job_id: str, fname: str):
        with _JOBS_LOCK:
            j = _JOBS.get(job_id)
        if j is None:
            raise HTTPException(404, "job not found")
        # Defence against path traversal
        if "/" in fname or ".." in fname:
            raise HTTPException(400, "invalid filename")
        p = j.output_dir / fname
        if not p.exists():
            raise HTTPException(404, f"file not found: {fname}")
        return FileResponse(p)

    # M1: research/dndc/* router (only if hydrus_research is installed)
    try:
        from .routers.research_dndc import router as dndc_router
        app.include_router(dndc_router, prefix="/research/dndc",
                           tags=["research", "dndc"])
    except ImportError:
        pass     # hydrus_research extra not installed

    # M3.7: research/batch/* router (only if hydrus_research is installed)
    try:
        from .routers.research_batch import router as batch_router
        app.include_router(batch_router, prefix="/research/batch",
                           tags=["research", "batch"])
    except ImportError:
        pass     # hydrus_research extra not installed

    # M4: research/sensitivity/* router
    try:
        from .routers.research_sensitivity import router as sens_router
        app.include_router(sens_router, prefix="/research/sensitivity",
                           tags=["research", "sensitivity"])
    except ImportError:
        pass

    # M5: research/inversion/* router
    try:
        from .routers.research_inversion import router as inv_router
        app.include_router(inv_router, prefix="/research/inversion",
                           tags=["research", "inversion"])
    except ImportError:
        pass

    return app


# Backward-compat alias (M2 branch added it; M1 keeps it for symmetry)
create_app = build_app


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="hydrus-port-serve")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--reload", action="store_true")
    args = p.parse_args(argv)
    if not _FASTAPI_OK:
        print("ERROR: FastAPI not installed. Run: pip install 'hydrus-port[gui]'",
              file=sys.stderr)
        return 2
    app = build_app()
    uvicorn.run(app, host=args.host, port=args.port, reload=args.reload)
    return 0


if __name__ == "__main__":
    sys.exit(main())
