"""PESTPP-IES Iterative Ensemble Smoother via PyEMU.

This wrapper builds a minimal PEST control file (.pst) in a temporary
workspace, configures PESTPP-IES to invoke our `worker_script` for each
realization, runs the binary, and parses the resulting posterior
ensemble.

The worker script receives a JSON config — no cross-process Python
serialization. See `worker_script.py`."""
from __future__ import annotations
import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Literal
import numpy as np

from .base import InversionResult


def _check_binary(name: str) -> str:
    path = shutil.which(name) or shutil.which(name.upper())
    if not path:
        raise RuntimeError(
            f"`{name}` binary not on PATH. Install PEST++ from "
            "https://github.com/usgs/pestpp/releases and add to PATH, "
            "or use backend='lm_scipy' for the in-process LM path."
        )
    return path


def fit_pyemu(scenario_dir: str | Path,
              param_map,
              obs,
              method: Literal["glm", "ies", "opt"] = "ies",
              n_real: int = 200,
              n_iter: int = 4,
              workspace: Path | None = None) -> InversionResult:
    """Calibrate via PESTPP-{IES,GLM,OPT}.

    Unlike `fit_lm`, this wrapper takes `scenario_dir` directly (not a
    forward callable) because the worker subprocess must rebuild forward
    locally from a JSON config — see `worker_script.py`."""
    import pyemu
    binary = _check_binary(f"pestpp-{method}")
    scenario_dir = Path(scenario_dir).resolve()

    ws = Path(workspace) if workspace else Path(tempfile.mkdtemp(prefix="hr_pestpp_"))
    ws.mkdir(parents=True, exist_ok=True)

    # Write the JSON config the worker script will consume
    cfg = {
        "scenario_dir": str(scenario_dir),
        "param_specs": [s.model_dump() for s in param_map.specs],
        "obs_specs": [o.model_dump() for o in obs.specs],
    }
    (ws / "config.json").write_text(json.dumps(cfg, indent=2))

    par_names = list(param_map.names)
    obs_names = [s.name for s in obs.specs]
    bounds = param_map.bounds_array()                            # internal coords

    # Build a minimal PEST control file
    pst = pyemu.Pst.from_par_obs_names(par_names=par_names, obs_names=obs_names)
    pst.parameter_data.loc[par_names, "parlbnd"] = bounds[:, 0]
    pst.parameter_data.loc[par_names, "parubnd"] = bounds[:, 1]
    pst.parameter_data.loc[par_names, "parval1"] = param_map.midpoints()
    pst.observation_data.loc[obs_names, "obsval"] = obs.values
    pst.observation_data.loc[obs_names, "weight"] = 1.0 / np.maximum(obs.sigmas, 1e-12)
    pst.control_data.noptmax = int(n_iter)
    pst.pestpp_options["ies_num_reals"] = n_real

    # model_command: invoke the worker script via -m so it's importable
    pst.model_command = [
        sys.executable, "-m", "hydrus_research.inversion.worker_script",
        "theta.dat", "y.dat", "config.json",
    ]
    pst_path = ws / "case.pst"
    pst.write(str(pst_path))

    t0 = time.time()
    rc = subprocess.run([binary, "case.pst"], cwd=str(ws), capture_output=True)
    wall = time.time() - t0
    if rc.returncode != 0:
        raise RuntimeError(
            f"PESTPP-{method.upper()} failed (rc={rc.returncode}). stderr:\n"
            + rc.stderr.decode("utf-8", errors="replace")[-2000:]
        )

    # Parse posterior from .par.csv (highest iteration number)
    candidates = sorted(ws.glob("case.*.par.csv"))
    posterior: list[list[float]] | None = None
    if candidates:
        import csv
        with candidates[-1].open() as f:
            reader = csv.DictReader(f)
            posterior = [[float(row[name]) for name in par_names] for row in reader]

    best_named: dict[str, float] = {}
    if posterior:
        arr = np.asarray(posterior)
        for j, name in enumerate(par_names):
            best_named[name] = float(arr[:, j].mean())
    else:
        for name in par_names:
            best_named[name] = float(pst.parameter_data.loc[name, "parval1"])

    return InversionResult(
        backend=f"pyemu_{method}",                                  # type: ignore[arg-type]
        best_params=best_named,
        posterior_ensemble=posterior,
        posterior_param_names=par_names,
        n_forward_calls=int(n_real * n_iter),
        wall_s=float(wall),
        pest_workspace=str(ws),
        diagnostics={"pst": str(pst_path), "binary": binary,
                     "n_real": n_real, "n_iter": n_iter},
    )
