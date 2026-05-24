"""Richards3D ↔ canonical Scenario adapter.

Unlike SWMS_2D and HYDRUS-1D, the 3D solver has **no historical ASCII
format** — canonical JSON IS the input. This adapter therefore just:

  load(input_dir)  → reads scenario.json (or builds defaults if a .py
                     demo path was passed in for back-compat)
  save(scenario, output_dir) → writes scenario.json
  run(scenario, output_dir)  → executes the solver and writes VTU
                                snapshots into output_dir
"""
from __future__ import annotations
from pathlib import Path
import numpy as np

from ..schema import (
    Scenario, ScenarioMeta, Units, Solver, HydraulicMaterial, TimeControl,
    Geometry3D,
)


def load(input_dir: Path | str) -> Scenario:
    """Read a 3D canonical scenario from a directory containing
    scenario.json, OR fall back to the box-demo defaults when the
    path is the legacy validate_richards3d.py file."""
    p = Path(input_dir)
    if p.is_file() and p.suffix == ".py":
        return default_box_scenario()
    if p.is_file() and p.suffix == ".json":
        return Scenario.from_path(p)
    if p.is_dir():
        sj = p / "scenario.json"
        if sj.exists():
            return Scenario.from_path(sj)
    # Fall back to demo defaults
    return default_box_scenario()


def save(scenario: Scenario, output_dir: Path | str) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    scenario.write(out / "scenario.json")


def default_box_scenario() -> Scenario:
    """Canonical Scenario for the synthetic box infiltration demo.

    Mirrors the hardcoded constants in tests/validate_richards3d.py:
    8×8×25 hex mesh, 0.4×0.4×1.0 m box, loam material, ponded top BC.
    """
    nx, ny, nz = 8, 8, 25
    lx, ly, lz = 0.4, 0.4, 1.0
    # Generate hex tensor mesh (same shape as MeshHex.init_tensor)
    xs = np.linspace(0.0, lx, nx)
    ys = np.linspace(0.0, ly, ny)
    zs = np.linspace(0.0, lz, nz)
    pts: list[list[float]] = []
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                pts.append([xs[i], ys[j], zs[k]])
    n_total = len(pts)
    # Connectivity (hex cells)
    cells: list[list[int]] = []
    def idx(i, j, k): return k*(nx*ny) + j*nx + i
    for k in range(nz - 1):
        for j in range(ny - 1):
            for i in range(nx - 1):
                cells.append([
                    idx(i,   j,   k),   idx(i+1, j,   k),
                    idx(i+1, j+1, k),   idx(i,   j+1, k),
                    idx(i,   j,   k+1), idx(i+1, j,   k+1),
                    idx(i+1, j+1, k+1), idx(i,   j+1, k+1),
                ])
    # Initial head: -200 cm everywhere, 0 at top
    z_top = zs[-1]
    init_h = [-200.0 if abs(p[2] - z_top) > 1e-9 else 0.0 for p in pts]
    # Boundary nodes: all nodes on top + bottom faces (Dirichlet candidates)
    bnd: list[int] = []
    for n, p in enumerate(pts):
        if abs(p[2] - 0.0) < 1e-9 or abs(p[2] - z_top) < 1e-9:
            bnd.append(n)

    geom = Geometry3D(
        x=[p[0] for p in pts],
        y=[p[1] for p in pts],
        z=[p[2] for p in pts],
        initial_h=init_h,
        mat_num=[1] * n_total,
        cells=cells,
        cell_kind="hex",
        boundary_nodes=bnd,
    )

    return Scenario(
        meta=ScenarioMeta(
            name="Synthetic 3D box: ponded infiltration into dry loam",
            description="8×8×25 hex mesh, 0.4×0.4×1.0 m box, ponded top (h=0), bottom no-flow.",
            source="hydrus_port.adapters.richards3d.default_box_scenario",
        ),
        units=Units(length="cm", time="day", mass="-"),
        solver=Solver(
            geometry_kind="vertical",
            max_picard=20,
            tol_theta=0.001,
            tol_h=0.5,
            water_flow=True,
        ),
        materials=[HydraulicMaterial(
            theta_r=0.078, theta_s=0.430,
            alpha=0.036, n=1.56, Ks=0.2496,
            l=0.5,
            theta_a=0.078, theta_m=0.430,
            theta_k=0.430, Kk=0.2496,
        )],
        time=TimeControl(
            dt=1e-3, dt_min=1e-5, dt_max=0.02,
            dt_mul=1.3, dt_mul2=0.7,
            t_init=0.0, t_max=0.12,
            print_times=[0.005, 0.015, 0.03, 0.05, 0.08, 0.12],
        ),
        geometry=geom,
    )


def run(scenario: Scenario, output_dir: Path | str) -> None:
    """Execute the 3D Richards solver on a canonical Scenario and write
    a VTU snapshot series into output_dir."""
    try:
        import skfem
        from skfem import MeshHex, MeshTet
    except ImportError as e:
        raise ImportError("3D solver needs scikit-fem; pip install scikit-fem") from e
    from swms2d.mesh_io import timeseries_to_vtk_series_3d
    from swms2d.richards3d import (RichardsState3D, evaluate_KCQ_3d,
                                   integrate_3d)
    from swms2d.dataclasses import SoilMaterial

    if scenario.dimension != "3d":
        raise ValueError(f"richards3d adapter expects 3d geometry, got {scenario.dimension}")
    g = scenario.geometry  # Geometry3D

    # Build skfem mesh
    pts = np.array(list(zip(g.x, g.y, g.z)), dtype=np.float64).T  # (3, N)
    cells = np.array(g.cells, dtype=np.int64).T                   # (verts_per_cell, NumEl)
    if g.cell_kind == "hex":
        skmesh = MeshHex(pts, cells)
    else:
        skmesh = MeshTet(pts, cells)

    # Convert canonical material → SoilMaterial
    materials = [SoilMaterial(
        thr=m.theta_r, ths=m.theta_s,
        tha=m.vc_theta_a(), thm=m.vc_theta_m(),
        alpha=m.alpha, n=m.n,
        Ks=m.Ks, Kk=m.vc_Kk(), thk=m.vc_theta_k(),
    ) for m in scenario.materials]

    # State setup
    h0 = np.array(g.initial_h, dtype=np.float64)
    matnum = np.array(g.mat_num, dtype=np.int32)
    _, _, th0 = evaluate_KCQ_3d(h0, matnum, materials)
    state = RichardsState3D(
        hNew=h0.copy(), hOld=h0.copy(),
        ThNew=th0.copy(), ThOld=th0.copy(),
        MatNum=matnum,
    )
    # Dirichlet at top (z = z_max). Read from canonical: any node in
    # boundary_nodes at the top face gets h=0; others left as no-flow.
    z_top = max(g.z)
    top_nodes = np.array([n for n in g.boundary_nodes
                           if abs(g.z[n] - z_top) < 1e-9], dtype=np.int32)
    dirich_h = np.zeros(top_nodes.size, dtype=np.float64)

    t = scenario.time
    h_final, th_final, snapshots = integrate_3d(
        skmesh, state, materials,
        t_end=t.t_max, dt_init=t.dt,
        dirich_nodes=top_nodes, dirich_h=dirich_h,
        gravity_axis=2, lump=True,
        dt_max=t.dt_max, dt_min=t.dt_min,
        dMul=t.dt_mul, dMul2=t.dt_mul2,
        max_picard=scenario.solver.max_picard,
        tol_h=scenario.solver.tol_h,
        snapshot_times=list(t.print_times),
    )
    # Write VTU series
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    series: list[tuple[float, dict]] = [(0.0, {"h": h0, "theta": th0})]
    for ts, hs, ths in snapshots:
        series.append((float(ts), {
            "h": np.asarray(hs, dtype=np.float64),
            "theta": np.asarray(ths, dtype=np.float64),
        }))
    timeseries_to_vtk_series_3d(skmesh, out, series, prefix="box3d")
