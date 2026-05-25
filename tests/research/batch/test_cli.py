import subprocess
import sys
from pathlib import Path


def test_cli_sweep_writes_parquet(tmp_path):
    """`hydrus research sweep` should produce a parquet file with N rows."""
    # The CLI takes a scenario path and runs N evaluations of forward(theta)
    # on it. Smallest test: use infiltr_v1 + sweep over alpha only.
    out_parquet = tmp_path / "sweep.parquet"
    r = subprocess.run(
        [
            sys.executable, "-m", "hydrus_port.cli",
            "research", "sweep",
            "tests/fixtures/infiltr_v1/inputs",
            "--param", "materials[0].alpha:0.005:0.05:log",
            "--n", "4",
            "--workers", "1",
            "--out", str(out_parquet),
        ],
        capture_output=True, text=True,
        cwd=Path(__file__).parent.parent.parent.parent,  # repo root
    )
    assert r.returncode == 0, r.stderr
    assert out_parquet.exists()

    # Read back and verify
    from hydrus_research.batch import BatchResult
    br = BatchResult.from_parquet(out_parquet)
    assert br.N == 4
    assert br.D == 1
    assert "alpha" in br.param_names[0]    # name derives from path tail
    assert br.converged.sum() >= 3         # at least 3 of 4 should converge


def test_cli_worker_help_works():
    """`hydrus research worker --help` should print without raising."""
    r = subprocess.run(
        [sys.executable, "-m", "hydrus_port.cli",
         "research", "worker", "--help"],
        capture_output=True, text=True,
    )
    assert r.returncode == 0, r.stderr
    assert "--master" in r.stdout
