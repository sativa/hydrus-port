"""CLI tests for `hydrus research dndc {list-presets,validate,to-forcing}`.

Uses `sys.executable -m hydrus_port.cli` instead of the bare `hydrus`
entry-point to avoid the editable-install pointer racing with the M2
worktree (which shares the same site-packages).
"""
import json
import subprocess
import sys


def test_cli_dndc_list_presets():
    r = subprocess.run(
        [sys.executable, "-m", "hydrus_port.cli", "research", "dndc", "list-presets"],
        capture_output=True, text=True,
    )
    assert r.returncode == 0, r.stderr
    assert "maize" in r.stdout and "wheat" in r.stdout


def test_cli_dndc_validate_minimal(tmp_path):
    minimal = {
        "atm": {"dates": ["2026-05-01"], "precip_cm": [0.0], "pet_cm": [0.4]},
        "et":  {"mode": "lai_beer", "lai": [2.0], "extinction_k": 0.6},
        "root": {"z_max_cm": 50, "growth_curve": "logistic", "days_to_zmax": 30,
                 "density_profile": "linear_decline"},
        "feddes": {"h1": -15, "h2": -30, "h3_high": -325, "h3_low": -600, "h4": -8000,
                   "pet_high_cm_d": 0.5, "pet_low_cm_d": 0.1},
        "n_transform": {"mode": "constant_rates", "k_nitrification_d": 0.1},
        "plant_n_uptake": {"mode": "passive_with_water"},
        "state": {"z_grid_cm": [0.0]},
    }
    p = tmp_path / "in.json"
    p.write_text(json.dumps(minimal))
    r = subprocess.run(
        [sys.executable, "-m", "hydrus_port.cli", "research", "dndc", "validate", str(p)],
        capture_output=True, text=True,
    )
    assert r.returncode == 0, r.stderr
    assert "ok" in r.stdout.lower() or "valid" in r.stdout.lower()
