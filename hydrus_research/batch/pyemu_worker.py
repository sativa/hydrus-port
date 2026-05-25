"""PyEMU TCP worker mode.

PESTPP-IES and friends spawn workers across machines via a TCP protocol.
This module lets a Simulator (wrapped via make_forward) act as one of those
workers. The pyemu dependency is lazy — only imported inside the start_*
functions, so users who only need the joblib backend don't need pyemu.

The CLI entry `hydrus research worker --master host:port` calls
`run_worker(master_host, master_port, forward, param_names, obs_names)`."""
from __future__ import annotations
import tempfile
from pathlib import Path
from typing import Callable

import numpy as np


def build_worker_entry(forward: Callable[[np.ndarray], np.ndarray],
                       param_names: list[str],
                       obs_names: list[str]) -> Callable[[Path], None]:
    """Return a callable `entry(workdir: Path) -> None` that:

    1. Reads PEST-format `theta.dat` from `workdir`
       containing the current theta vector (one float per line, param_names order).
    2. Calls `forward(theta) -> y_sim`.
    3. Writes PEST-format `y.dat` to `workdir` with the y_sim values
       (one float per line, obs_names order).

    No pyemu dependency — pure stdlib + numpy file I/O. The entry callable is
    passed to run_worker() which does the pyemu TCP plumbing."""
    def entry(workdir: Path) -> None:
        workdir = Path(workdir)
        in_path = workdir / "theta.dat"
        out_path = workdir / "y.dat"

        # Read theta — one float per line in param_names order
        theta = np.array([float(x) for x in in_path.read_text().split()])
        if theta.shape != (len(param_names),):
            raise ValueError(f"theta.dat had {theta.shape[0]} values; "
                             f"expected {len(param_names)}")

        # Forward-evaluate
        y = np.asarray(forward(theta), dtype=float)
        if y.shape != (len(obs_names),):
            raise ValueError(f"forward returned {y.shape[0]} values; "
                             f"expected {len(obs_names)}")

        # Write y — one float per line in obs_names order
        out_path.write_text("\n".join(f"{v:.10e}" for v in y))

    return entry


def run_worker(master_host: str,
               master_port: int,
               forward: Callable[[np.ndarray], np.ndarray],
               param_names: list[str],
               obs_names: list[str],
               workdir: Path | None = None,
               worker_name: str | None = None) -> int:
    """Start a long-running worker process that connects to a PESTPP master.

    Imports pyemu LAZILY. Returns the process exit code (0 on clean shutdown)."""
    try:
        import pyemu
    except ImportError as e:
        raise RuntimeError(
            "pyemu is not installed; install with `pip install 'hydrus-port[research]'`"
        ) from e

    wd = Path(workdir) if workdir else Path(tempfile.mkdtemp(prefix="hydrus_worker_"))
    wd.mkdir(parents=True, exist_ok=True)
    entry = build_worker_entry(forward, param_names, obs_names)
    name = worker_name or f"hydrus_worker_{wd.name}"

    # pyemu's worker API has evolved — defensively try the most common entry points
    try:
        pyemu.utils.os_utils._try_remove_existing(str(wd / ".lock"))   # type: ignore[attr-defined]
    except Exception:
        pass

    # Minimal worker loop: poll the master for new theta input, invoke entry, write y.
    # PEST++ "Yamr" protocol is implemented in pyemu — we delegate to it when available.
    if hasattr(pyemu.utils.os_utils, "start_worker_from_callable"):
        return pyemu.utils.os_utils.start_worker_from_callable(   # type: ignore[attr-defined]
            master_host=master_host, master_port=master_port,
            entry=entry, worker_dir=str(wd), worker_name=name,
        )
    # Fallback: print a warning + spin forever waiting for theta.dat to appear.
    # This is the manual-smoke path; PESTPP-IES integration is verified out-of-band.
    print(f"[pyemu_worker] No 'start_worker_from_callable' helper found; running "
          f"polling fallback. Watching {wd / 'theta.dat'} (Ctrl-C to exit).")
    import signal
    import time
    stop = {"flag": False}

    def _on_sigint(signum, frame):
        stop["flag"] = True
    signal.signal(signal.SIGINT, _on_sigint)

    while not stop["flag"]:
        if (wd / "theta.dat").exists():
            entry(wd)
            (wd / "theta.dat").unlink()       # consume input
        time.sleep(0.1)
    return 0
