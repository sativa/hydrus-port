"""Swms2DSimulator — wraps the existing swms2d.swms2d.SWMS2DSimulation.

Key differences from the plan's speculative attribute names:
- SWMS2DSimulation does NOT store theta_history / h_history in memory.
  Outputs are written to th.out / h.out files during the run; we parse
  them after the run completes.
- Mesh attributes are `mesh.nodes.x` and `mesh.nodes.y` (SWMS_2D uses y
  for the vertical/depth axis, positive upward), NOT nodes_x / nodes_z.
- Elements are 4-node quads stored in `mesh.elements.KX` with shape
  (NumEl, 4); for Delaunay spatial lookup we use the node (x, y) coords
  directly — scipy.spatial.Delaunay triangulates them on the fly.
- mesh.IJ is the number of nodes per element ROW in the output file (2
  for EX1-EX4 quad meshes), not the polygon valency.
"""
from __future__ import annotations
import copy
import tempfile
import time
from pathlib import Path
from typing import Any
import numpy as np

from .base import Simulator, SimResult, InitialState


# ---------------------------------------------------------------------------
# Output file parsers
# ---------------------------------------------------------------------------

def _parse_field_out(path: Path, num_np: int) -> tuple[np.ndarray, np.ndarray]:
    """Parse th.out or h.out into (times, field(NT, NumNP)).

    File layout: repeated blocks, each preceded by
      ' Time  ***   <value> ***'
    followed by a header line and data rows of the form:
      node_1based  x  y  val_m  val_{m+1}  ...
    where IJ values are packed per row. We read all numeric tokens after
    the first three (node, x, y) on each data row.
    """
    text = path.read_text()
    lines = text.splitlines()
    blocks: list[tuple[float, np.ndarray]] = []
    i = 0
    cur_time: float | None = None
    cur_vals: list[float] = []
    in_data = False

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        # Detect time block header
        if "Time  ***" in line:
            # commit previous block
            if cur_time is not None and cur_vals:
                arr = np.array(cur_vals[:num_np], dtype=float)
                if len(arr) == num_np:
                    blocks.append((cur_time, arr))
            # parse time value: " Time  ***   900.0000 ***"
            parts = stripped.replace("*", " ").split()
            # parts = ['Time', value]
            try:
                cur_time = float(parts[1])
            except (IndexError, ValueError):
                cur_time = None
            cur_vals = []
            in_data = False
            i += 1
            continue
        # Skip header lines (contain letters after the time line)
        if cur_time is not None and not in_data:
            if stripped and not stripped.startswith("n") and "***" not in stripped:
                # Try to parse first token as integer (node number)
                tokens = stripped.split()
                if tokens:
                    try:
                        int(tokens[0])
                        in_data = True
                    except ValueError:
                        i += 1
                        continue
            else:
                i += 1
                continue
        if in_data and stripped:
            tokens = stripped.split()
            if tokens:
                try:
                    int(tokens[0])  # node index
                    # values start at index 3 (after node, x, y)
                    for tok in tokens[3:]:
                        cur_vals.append(float(tok))
                except (ValueError, IndexError):
                    pass
        i += 1

    # commit last block
    if cur_time is not None and cur_vals:
        arr = np.array(cur_vals[:num_np], dtype=float)
        if len(arr) == num_np:
            blocks.append((cur_time, arr))

    if not blocks:
        raise RuntimeError(f"No time blocks parsed from {path}")

    times = np.array([b[0] for b in blocks], dtype=float)
    field = np.stack([b[1] for b in blocks], axis=0)  # (NT, NumNP)
    return times, field


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------

class Swms2DSimulator(Simulator):
    name = "swms2d"
    dimension = 2

    def __init__(self, work_root: Path | str | None = None):
        self.work_root = (Path(work_root) if work_root
                          else Path(tempfile.gettempdir()) / "hydrus_research_swms2d")
        self.work_root.mkdir(parents=True, exist_ok=True)

    def run(self, scenario, forcing, ic):
        from hydrus_port.schema import _scenario_from_dict
        from hydrus_port.adapters.swms2d import save as save_2d
        from swms2d.swms2d import SWMS2DSimulation

        sc = _scenario_from_dict(scenario)
        run_dir = Path(tempfile.mkdtemp(prefix="swms2d_", dir=self.work_root))
        out_dir = run_dir / "out"
        save_2d(sc, run_dir)
        out_dir.mkdir(exist_ok=True)

        t0 = time.time()
        sim = SWMS2DSimulation(input_dir=run_dir, output_dir=out_dir)

        # Cache mesh geometry before running (mesh is parsed in __init__)
        num_np = sim.mesh.NumNP
        nodes_x = np.asarray(sim.mesh.nodes.x, dtype=float).copy()
        # In SWMS_2D the vertical coordinate lives in nodes.y (positive upward)
        nodes_y = np.asarray(sim.mesh.nodes.y, dtype=float).copy()
        # quad elements: (NumEl, 4) 0-based node indices
        elements_kx = np.asarray(sim.mesh.elements.KX, dtype=int).copy()

        sim.run(verbose=False)
        wall = time.time() - t0

        # Parse output files written by the run
        th_path = out_dir / "th.out"
        h_path  = out_dir / "h.out"
        times, theta = _parse_field_out(th_path, num_np)
        _, h = _parse_field_out(h_path, num_np)

        if theta.shape != (len(times), num_np):
            raise RuntimeError(
                f"theta shape mismatch: got {theta.shape}, "
                f"expected ({len(times)}, {num_np})"
            )

        final = InitialState(z_cm=nodes_y, theta=theta[-1], h_cm=h[-1],
                             c_mg_per_L=None, t_celsius=None)
        return SimResult(
            times=times,
            z=nodes_y,          # vertical coordinate (y in SWMS_2D convention)
            theta=theta,
            h=h,
            c=None,
            fluxes={},
            mass_balance={},
            final_state=final,
            meta={
                "solver": "swms2d",
                "wall_s": wall,
                "out_dir": str(out_dir),
                "num_np": num_np,
                "nodes_x": nodes_x.tolist(),
                "nodes_y": nodes_y.tolist(),
                "elements_kx": elements_kx.tolist(),
            },
        )

    def observable_at(self, result, spec):
        """Sample a scalar observable from a 2D SimResult.

        Location: either {"node": N} for direct node index access, or
        {"xyz": [x, y_coord, z_ignored]} where x is the horizontal axis and
        the second component maps to the vertical axis (nodes.y in SWMS_2D).

        For spatial interpolation we use scipy.spatial.Delaunay on the full
        node cloud (x, y) — scipy triangulates the quad nodes automatically.
        """
        from ._interp import barycentric_2d

        kind = spec.kind
        if kind == "theta":
            field = result.theta
        elif kind == "h":
            field = result.h
        elif kind == "c":
            if result.c is None:
                raise ValueError("observable kind='c' requested but result has no solute")
            field = result.c
        else:
            raise NotImplementedError(f"observable kind {kind!r} not supported in M8")

        loc = spec.location
        if "node" in loc:
            node = int(loc["node"])
            ts = field[:, node]
        elif "xyz" in loc:
            from scipy.spatial import Delaunay
            x_nodes = np.asarray(result.meta["nodes_x"], dtype=float)
            y_nodes = np.asarray(result.meta["nodes_y"], dtype=float)
            pts_2d = np.column_stack([x_nodes, y_nodes])
            tri = Delaunay(pts_2d)
            # xyz spec uses [x, horizontal_y, vertical_z] convention;
            # map to SWMS_2D (x, y) where y is vertical
            xyz = loc["xyz"]
            query_pt = np.array([float(xyz[0]), float(xyz[1])], dtype=float)
            tri_id = int(tri.find_simplex(query_pt))
            if tri_id < 0:
                raise ValueError(f"point {query_pt} outside mesh hull")
            vertices = tri.simplices[tri_id]
            tri_verts = pts_2d[vertices]
            ts = np.array([
                barycentric_2d(tri_verts, field[ti, vertices], query_pt)
                for ti in range(field.shape[0])
            ])
        else:
            raise ValueError(f"location must have 'node' or 'xyz' key; got {loc}")

        return float(np.interp(spec.time_day, result.times, ts))
