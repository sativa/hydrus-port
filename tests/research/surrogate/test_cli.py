import subprocess
import sys

def test_cli_surrogate_help():
    r = subprocess.run([sys.executable, "-m", "hydrus_port.cli",
                        "research", "surrogate", "--help"],
                       capture_output=True, text=True)
    assert r.returncode == 0 and "train" in r.stdout
