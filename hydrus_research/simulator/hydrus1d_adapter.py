"""Hydrus1DSimulator — wraps hydrus1d.hydrus.run_simulation behind the
Simulator ABC. The adapter receives a fully patched canonical scenario
dict (parameter application happens upstream in make_forward), serialises
it to HYDRUS-1D ASCII files, runs the solver in a temp directory, and
parses outputs into a SimResult.

Decoupling note (M0.11):
    This is the FIRST and ONLY production file in hydrus_research/ that is
    intentionally allowed to import from hydrus_port and hydrus1d. That
    coupling is by design — the adapter's sole job is to bridge the
    abstraction layer (Simulator ABC) to the concrete solver. All other
    research modules (parameters, observations, closure, ...) must remain
    solver-agnostic. The imports are deferred inside `run()` so that the
    module itself remains importable in environments where hydrus_port /
    hydrus1d are not installed (e.g. unit-test runs that mock the adapter).
"""
from __future__ import annotations
import tempfile
from pathlib import Path
import numpy as np

from .base import Simulator, Forcing, InitialState, SimResult


class Hydrus1DSimulator(Simulator):
    name = "hydrus1d"
    dimension = 1

    def __init__(self, work_root: Path | str | None = None):
        self.work_root = Path(work_root) if work_root else Path(tempfile.gettempdir()) / "hydrus_research"
        self.work_root.mkdir(parents=True, exist_ok=True)

    def run(self, scenario, forcing, ic):
        """`scenario` is a fully patched canonical Scenario dict (i.e. what
        `Scenario.to_dict()` returns, after any parameter patching done by
        ParameterMap.apply_to_scenario). `forcing` and `ic` are reserved for
        M1 (DNDC seam); in M0 the adapter uses whatever atmospheric / initial
        data lives inside `scenario` itself.

        Decoupling note: imports from hydrus_port and hydrus1d are deferred
        here. This adapter is the designated bridge between the research
        abstraction layer and the real solvers — these imports are INTENDED.
        """
        import time as _time

        # ----- 1. round-trip through canonical schema and write HYDRUS-1D inputs
        from hydrus_port.schema import _scenario_from_dict
        from hydrus_port.adapters.hydrus1d import save as save_h1d
        sc = _scenario_from_dict(scenario)
        run_dir = Path(tempfile.mkdtemp(prefix="h1d_", dir=self.work_root))
        out_dir = run_dir / "out"
        save_h1d(sc, run_dir)
        out_dir.mkdir(exist_ok=True)

        # ----- 2. invoke the real solver
        from hydrus1d.hydrus import run_simulation
        t0 = _time.time()
        _h1d = run_simulation(input_dir=str(run_dir), output_dir=str(out_dir))
        wall = _time.time() - t0

        # ----- 3. parse outputs into a SimResult
        return self._load_outputs(out_dir, wall)

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------
    def _load_outputs(self, out_dir: Path, wall_s: float) -> SimResult:
        """Parse NOD_INF.OUT (per-node profiles per print time) + BALANCE.OUT."""
        nod_path = out_dir / "NOD_INF.OUT"
        if not nod_path.exists():
            # case-insensitive find
            for p in out_dir.iterdir():
                if p.name.lower() == "nod_inf.out":
                    nod_path = p
                    break
        times, z, theta, h = _parse_nod_inf(nod_path)
        mb = _parse_balance_total(out_dir)
        final = InitialState(z_cm=z, theta=theta[-1], h_cm=h[-1],
                             c_mg_per_L=None, t_celsius=None)
        return SimResult(
            times=times, z=z, theta=theta, h=h, c=None,
            fluxes={}, mass_balance=mb, final_state=final,
            meta={"solver": "hydrus1d", "wall_s": wall_s, "out_dir": str(out_dir)},
        )

    def observable_at(self, result, spec):
        raise NotImplementedError("implemented in Task 12")


# ---------------------------------------------------------------------------
# Module-level parsing helpers
# ---------------------------------------------------------------------------

def _parse_nod_inf(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Parse HYDRUS-1D NOD_INF.OUT into (times, z, theta(NT,NZ), h(NT,NZ)).

    File layout: repeated blocks, each preceded by 'Time:  <value>' and a
    header with columns including 'Node', 'Depth', 'Head', 'Moisture'. A
    'end' line closes each block."""
    text = path.read_text().splitlines()
    blocks: list[tuple[float, list[tuple[float, float, float]]]] = []
    i = 0
    cur_time: float | None = None
    rows: list[tuple[float, float, float]] = []
    while i < len(text):
        line = text[i].strip()
        if line.lower().startswith("time"):
            # commit previous block
            if cur_time is not None and rows:
                blocks.append((cur_time, rows))
            tokens = line.replace(":", " ").split()
            cur_time = float(tokens[1])
            rows = []
            i += 1
            continue
        if line.lower().startswith("end") or not line:
            i += 1
            continue
        parts = line.split()
        # numeric data row: Node Depth Head Moisture K C ...
        if len(parts) >= 4:
            try:
                # Node int, Depth float, Head float, Moisture float
                int(parts[0])
                depth = float(parts[1])
                head = float(parts[2])
                moist = float(parts[3])
                rows.append((depth, head, moist))
            except ValueError:
                pass
        i += 1
    if cur_time is not None and rows:
        blocks.append((cur_time, rows))

    if not blocks:
        raise ValueError(f"no time blocks found in {path}")

    # depth axis from the first block
    z = np.array([d for d, _, _ in blocks[0][1]], dtype=float)
    times = np.array([t for t, _ in blocks], dtype=float)
    NT, NZ = len(times), len(z)
    theta = np.empty((NT, NZ), dtype=float)
    h = np.empty((NT, NZ), dtype=float)
    for ti, (_, rs) in enumerate(blocks):
        for zi, (_, head, moist) in enumerate(rs):
            h[ti, zi] = head
            theta[ti, zi] = moist
    return times, z, theta, h


def _parse_balance_total(out_dir: Path) -> dict[str, float]:
    """Pull final-row 'Volume' from BALANCE.OUT; tolerant of missing file."""
    for name in ("BALANCE.OUT", "balance.out", "Balance.out"):
        p = out_dir / name
        if p.exists():
            try:
                lines = p.read_text().splitlines()
                vol_lines = [ln for ln in lines if "Volume" in ln and "[" in ln]
                if not vol_lines:
                    return {}
                last = vol_lines[-1].split()
                # final number on that line
                return {"volume_last": float(last[-1])}
            except Exception:
                return {}
    return {}
