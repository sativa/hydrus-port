import subprocess
import sys


def test_cli_uq_help_lists_glue():
    r = subprocess.run(
        [sys.executable, "-m", "hydrus_port.cli",
         "research", "uq", "--help"],
        capture_output=True, text=True,
    )
    assert r.returncode == 0
    assert "glue" in r.stdout
