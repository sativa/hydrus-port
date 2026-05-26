import json, subprocess, sys
from pathlib import Path


def test_cli_agronomy_run_writes_result_json(tmp_path):
    irrig = tmp_path / "irrig.csv"
    irrig.write_text("date,depth_mm\n2026-05-10,20\n")
    fert = tmp_path / "fert.csv"
    fert.write_text("date,kg_n_ha\n")
    out = tmp_path / "out"
    res = subprocess.run(
        [sys.executable, "-m", "hydrus_port.cli", "research", "agronomy", "run",
         "--crop", "maize", "--soil", "loam", "--weather", "n_china_avg",
         "--horizon-days", "30",
         "--irrig", str(irrig), "--fert", str(fert), "--out", str(out)],
        capture_output=True, text=True, check=False,
    )
    assert res.returncode == 0, res.stderr
    payload = json.loads((out / "result.json").read_text())
    assert "theta_zt" in payload
    assert "water_balance" in payload
    assert payload["water_balance"]["irrig_mm"] == 20.0
