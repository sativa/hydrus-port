import json
import subprocess
import sys
from pathlib import Path


def test_cli_sensitize_morris(tmp_path):
    out_json = tmp_path / "morris.json"
    r = subprocess.run(
        [sys.executable, "-m", "hydrus_port.cli",
         "research", "sensitize",
         "tests/fixtures/infiltr_v1/inputs",
         "--method", "morris",
         "--param", "materials[0].alpha:0.005:0.05:log",
         "--obs", "theta@-30cm,t=1.0",
         "--n", "4", "--workers", "1", "--seed", "42",
         "--out", str(out_json)],
        capture_output=True, text=True,
    )
    assert r.returncode == 0, r.stderr
    assert out_json.exists()
    body = json.loads(out_json.read_text())
    assert body["method"] == "morris"
    assert "mu_star" in body["indices"]


def test_cli_sensitize_help_lists_methods():
    r = subprocess.run(
        [sys.executable, "-m", "hydrus_port.cli",
         "research", "sensitize", "--help"],
        capture_output=True, text=True,
    )
    assert r.returncode == 0
    out = r.stdout
    for m in ("morris", "sobol", "fast", "pawn"):
        assert m in out
