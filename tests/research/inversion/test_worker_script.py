import json
import subprocess
import sys
import tempfile
from pathlib import Path
import numpy as np


def test_worker_script_runs_forward_and_writes_y():
    """Smoke-test the worker by invoking it like PESTPP would: with a theta
    file + an output y file + a JSON config that describes the forward."""
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        theta_path = td / "theta.dat"
        y_path = td / "y.dat"
        cfg_path = td / "config.json"

        # Use infiltr_v1; alpha at the template's nominal value
        from hydrus_port.adapters.hydrus1d import load
        template = load(Path("tests/fixtures/infiltr_v1/inputs")).to_dict()
        a0 = template["materials"][0]["alpha"]

        cfg = {
            "scenario_dir": "tests/fixtures/infiltr_v1/inputs",
            "param_specs": [
                {"name": "alpha", "target": "materials[0].alpha",
                 "bounds": [a0 * 0.5, a0 * 2.0], "transform": "log"},
            ],
            "obs_specs": [
                {"name": "theta_z30_d1", "kind": "theta",
                 "location": {"z_cm": -30.0}, "time_day": 1.0},
            ],
        }
        cfg_path.write_text(json.dumps(cfg))
        # Theta is in INTERNAL coords (log for alpha)
        theta_internal = float(np.log(a0))
        theta_path.write_text(f"{theta_internal:.10e}\n")

        r = subprocess.run(
            [sys.executable, "-m", "hydrus_research.inversion.worker_script",
             str(theta_path), str(y_path), str(cfg_path)],
            capture_output=True, text=True, timeout=120,
        )
        assert r.returncode == 0, r.stderr
        assert y_path.exists()
        y_value = float(y_path.read_text().strip())
        assert 0 <= y_value < 1.0          # plausible θ
