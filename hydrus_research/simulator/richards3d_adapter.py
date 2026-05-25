"""Richards3DSimulator — wraps tests/validate_richards3d.run_case.

Key differences from the plan's assumptions:
- run_case(element, lump) returns a 3-tuple (zs, hs, ths) of 1D arrays,
  NOT 2D time-series. They represent the mid-column z-profile at the
  FINAL time step only (a single snapshot).
- We reshape to (1, Nnode) so the SimResult theta/h always have ndim==2
  and the rest of the research stack (observable_at, batch_observables)
  works without special-casing.
- Full xyz spatial lookup (tet/hex containment) is deferred: the
  validate_richards3d path does not expose mesh connectivity. Only
  location={"node": N} is supported in M8.
"""
from __future__ import annotations
import tempfile
import time
from pathlib import Path
import numpy as np

from .base import Simulator, SimResult, InitialState


class Richards3DSimulator(Simulator):
    name = "richards3d"
    dimension = 3

    def __init__(self, work_root: Path | str | None = None,
                 element_type: str = "tetra",
                 lump: bool = True):
        self.work_root = (Path(work_root) if work_root
                          else Path(tempfile.gettempdir()) / "hydrus_research_3d")
        self.work_root.mkdir(parents=True, exist_ok=True)
        self.element_type = element_type
        self.lump = lump

    def run(self, scenario, forcing, ic):
        """Run the 3D Richards solver via tests.validate_richards3d.run_case.

        The `scenario` dict is not consumed for solver parameters — the
        run_case function uses its own built-in loam material and box mesh.
        This is appropriate for M8 where the 3D adapter is a scaffold; a
        production 3D adapter would read soil parameters from the scenario.

        Returns a SimResult where theta/h have shape (1, Nnode) — a single
        final snapshot. Times is [t_end] (the end time from run_case).
        """
        from tests.validate_richards3d import run_case

        t0 = time.time()
        zs, hs, ths = run_case(self.element_type, self.lump)
        wall = time.time() - t0

        # run_case returns 1D arrays of the mid-column z-profile at t_end
        zs  = np.asarray(zs,  dtype=float)
        hs  = np.asarray(hs,  dtype=float)
        ths = np.asarray(ths, dtype=float)

        # Ensure 2D so the rest of the stack sees (NT, Nnode)
        if hs.ndim == 1:
            hs  = hs.reshape(1, -1)
            ths = ths.reshape(1, -1)

        # validate_richards3d uses t_end=0.05 (days in this problem's units)
        times = np.array([0.05], dtype=float)

        final = InitialState(z_cm=zs, theta=ths[-1], h_cm=hs[-1],
                             c_mg_per_L=None, t_celsius=None)
        return SimResult(
            times=times,
            z=zs,
            theta=ths,
            h=hs,
            c=None,
            fluxes={},
            mass_balance={},
            final_state=final,
            meta={
                "solver": "richards3d",
                "wall_s": wall,
                "element_type": self.element_type,
                "lump": self.lump,
                "note": (
                    "theta/h are (1, Nnode) final-snapshot only; "
                    "run_case does not expose intermediate time steps"
                ),
            },
        )

    def observable_at(self, result, spec):
        """Sample a scalar from the 3D SimResult.

        Only location={"node": N} is supported. xyz-based lookup requires
        the full mesh connectivity from run_case, which is not exposed.
        """
        loc = spec.location
        if "node" not in loc:
            raise NotImplementedError(
                "3D observable_at requires location={'node': N}; "
                "xyz lookup needs mesh connectivity from run_case (TBD)."
            )
        node = int(loc["node"])
        kind = spec.kind
        if kind == "theta":
            field = result.theta
        elif kind == "h":
            field = result.h
        else:
            raise NotImplementedError(f"observable kind {kind!r} not in M8 3D adapter")
        ts = field[:, node]
        return float(np.interp(spec.time_day, result.times, ts))
