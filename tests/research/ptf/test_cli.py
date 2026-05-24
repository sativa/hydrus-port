import subprocess
import sys


# M2: Run via `python -m hydrus_port.cli` to pick up the local worktree's
# CLI code regardless of which `hydrus` binary is in PATH (which may be
# installed from the main repo while M2 develops in a worktree).
_CLI = [sys.executable, "-m", "hydrus_port.cli"]


def test_cli_soil_ptf_carsel():
    r = subprocess.run(
        _CLI + ["research", "soil", "ptf",
                "--texture", "sand=40,silt=40,clay=20",
                "--method", "carsel_parrish"],
        capture_output=True, text=True,
    )
    assert r.returncode == 0, r.stderr
    out = r.stdout.lower()
    for tag in ("theta_r", "theta_s", "alpha", "n", "ks"):
        assert tag in out
    assert "carsel_parrish" in out
