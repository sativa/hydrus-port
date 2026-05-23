"""
Modern mesh IO for SWMS_2D: gmsh reader + VTK writer.
=====================================================

The native SWMS_2D GRID.IN format is fine for the legacy verification
suite but two pain points limit modern workflows:

1. **Mesh authoring** — hand-editing GRID.IN's BLOCK H / BLOCK I /
   BLOCK J for anything more complex than a regular column is painful.
   Modern users want to build a domain in gmsh / Salome / Cubit and
   export `.msh` directly.

2. **Visualisation** — ParaView, VisIt, and the entire scientific
   pipeline expect VTK / VTU input. Hand-eyeballing `h.out` after a
   2D simulation defeats the purpose of moving to 2D.

This module bridges both gaps via `meshio`, which understands ~30
mesh formats including gmsh 2 and 4, Abaqus, MEDIT, OpenFOAM, NASTRAN,
and ANSYS.

Public API:
    read_mesh(path)              -> Mesh           # gmsh / xdmf / etc.
    write_vtk(mesh, path)        -> None           # bare mesh
    snapshot_to_vtk(mesh, fields, path)            # h, theta, conc, ...
    timeseries_to_vtk_series(mesh, path_pattern, time_field_pairs)

`meshio` is an optional runtime dependency — imported lazily so the
core swms2d package doesn't require it for the EX.1-4 verification
suite.
"""

from __future__ import annotations
from pathlib import Path
from typing import Mapping
import numpy as np
from numpy.typing import NDArray

from .dataclasses import Mesh, Node, Element


def _require_meshio():
    try:
        import meshio
        return meshio
    except ImportError as e:
        raise ImportError(
            "swms2d.mesh_io requires the 'meshio' package. Install with:\n"
            "    pip install --user --break-system-packages meshio"
        ) from e


# ============================================================================
# READ — gmsh .msh / any meshio-supported format -> swms2d.Mesh
# ============================================================================

def read_mesh(path: Path | str,
              default_material: int = 1,
              default_kode: int = 0,
              boundary_kode: int = -4,
              ) -> Mesh:
    """Read a 2D mesh from any meshio-supported format into a swms2d Mesh.

    Accepts triangle and quad elements. Tetra / hex (3D) is rejected
    pending the Stage-2 3D extension.

    Boundary nodes (those on the convex hull edges) are flagged with
    `boundary_kode` (default -4, atmospheric flux). Internal nodes get
    `default_kode` (0). The caller is expected to overwrite Kode for
    specific BC nodes after reading.

    Parameters
    ----------
    path : Path or str
        Path to the mesh file.
    default_material : int
        MatNum assigned to every node (1-based, Fortran convention).
    default_kode : int
        Kode assigned to internal nodes (0 = internal).
    boundary_kode : int
        Kode assigned to detected boundary nodes (-4 = atmospheric flux,
        a safe default — change to suit the problem).
    """
    meshio = _require_meshio()
    m = meshio.read(str(path))

    # Extract 2D points (drop z if present)
    pts = m.points
    NumNP = pts.shape[0]
    x = pts[:, 0].astype(np.float64).copy()
    y = pts[:, 1].astype(np.float64).copy()
    if pts.shape[1] > 2:
        # If there's a non-trivial z, use that as the vertical for
        # vertical-plane meshes (KAT=2). For mixed/3D users should call
        # the 3D reader instead.
        z_max = float(np.abs(pts[:, 2]).max())
        xy_max = float(max(np.abs(x).max(), np.abs(y).max()))
        if z_max > 1e-9 and z_max > 1e-6 * xy_max:
            # Looks like the mesh lives in xz plane — use x and z
            y = pts[:, 2].astype(np.float64).copy()

    # Collect 2D cells: triangle or quad
    KX_list: list[list[int]] = []
    layers: list[int] = []
    for cell_block in m.cells:
        t = cell_block.type
        data = cell_block.data
        if t == "triangle":
            for tri in data:
                # Pad to quad-shape with repeated last vertex (Fortran's
                # 2D code treats KX(*,3)==KX(*,4) as a triangle marker)
                KX_list.append([int(tri[0]), int(tri[1]),
                                int(tri[2]), int(tri[2])])
                layers.append(1)
        elif t == "quad":
            for q in data:
                KX_list.append([int(q[0]), int(q[1]),
                                int(q[2]), int(q[3])])
                layers.append(1)
        elif t in ("vertex", "line"):
            continue   # ignore 0D/1D facets
        elif t in ("tetra", "hexahedron", "wedge"):
            raise NotImplementedError(
                f"3D cell type {t!r} found — use a 3D reader "
                "(Stage 2 scikit-fem extension)."
            )
    NumEl = len(KX_list)
    if NumEl == 0:
        raise ValueError(f"No 2D triangle/quad cells found in {path}")

    KX = np.asarray(KX_list, dtype=np.int32)
    LayNum = np.asarray(layers, dtype=np.int32)
    ConAxx = np.ones(NumEl, dtype=np.float64)
    ConAzz = np.ones(NumEl, dtype=np.float64)
    ConAxz = np.zeros(NumEl, dtype=np.float64)

    # Detect boundary nodes: vertices belonging to an edge that's used
    # by only one element (the "naked" edges).
    edge_count: dict[tuple[int, int], int] = {}
    for k, q in enumerate(KX_list):
        # Skip the degenerate edge of a padded triangle (q[2]==q[3])
        n = 3 if q[2] == q[3] else 4
        for i in range(n):
            a, b = q[i], q[(i + 1) % n]
            edge = (min(a, b), max(a, b))
            edge_count[edge] = edge_count.get(edge, 0) + 1
    boundary_nodes: set[int] = set()
    KXB_list: list[int] = []
    for edge, c in edge_count.items():
        if c == 1:
            boundary_nodes.add(edge[0])
            boundary_nodes.add(edge[1])
    # KXB in boundary-traversal order: simple unique sort for now
    KXB_list = sorted(boundary_nodes)
    NumBP = len(KXB_list)
    KXB = np.asarray(KXB_list, dtype=np.int32)
    # Approximate per-boundary-segment width as the mean length of the
    # naked edges incident to the node (good for FE BC integration)
    Width = np.zeros(NumBP, dtype=np.float64)
    for edge, c in edge_count.items():
        if c != 1:
            continue
        a, b = edge
        elen = float(np.hypot(x[b] - x[a], y[b] - y[a]))
        if a in boundary_nodes:
            Width[KXB_list.index(a)] += 0.5 * elen
        if b in boundary_nodes:
            Width[KXB_list.index(b)] += 0.5 * elen

    Kode = np.full(NumNP, default_kode, dtype=np.int32)
    for n in boundary_nodes:
        Kode[n] = boundary_kode

    nodes = Node(
        Kode=Kode,
        x=x, y=y,
        hNew=np.zeros(NumNP, np.float64),
        hOld=np.zeros(NumNP, np.float64),
        hTemp=np.zeros(NumNP, np.float64),
        Q=np.zeros(NumNP, np.float64),
        Conc=np.zeros(NumNP, np.float64),
        MatNum=np.full(NumNP, default_material, dtype=np.int32),
        Beta=np.zeros(NumNP, np.float64),
        Axz=np.ones(NumNP, np.float64),
        Bxz=np.ones(NumNP, np.float64),
        Dxz=np.ones(NumNP, np.float64),
    )
    elements = Element(KX=KX, ConAxx=ConAxx, ConAzz=ConAzz, ConAxz=ConAxz,
                       LayNum=LayNum)
    mesh = Mesh(nodes=nodes, elements=elements, KXB=KXB, Width=Width,
                rLen=1.0,
                NumNP=NumNP, NumEl=NumEl, NumBP=NumBP, IJ=2, NObs=0)
    return mesh


# ============================================================================
# WRITE — Mesh + scalar fields -> VTK / VTU
# ============================================================================

def _mesh_to_meshio(mesh: Mesh):
    """Convert a swms2d Mesh into a meshio.Mesh (no data fields)."""
    meshio = _require_meshio()
    pts = np.column_stack(
        [mesh.nodes.x, mesh.nodes.y, np.zeros_like(mesh.nodes.x)]
    )
    tris = []
    quads = []
    for q in mesh.elements.KX:
        if q[2] == q[3]:
            tris.append([int(q[0]), int(q[1]), int(q[2])])
        else:
            quads.append([int(q[0]), int(q[1]), int(q[2]), int(q[3])])
    cells = []
    if tris:
        cells.append(("triangle", np.asarray(tris, dtype=np.int32)))
    if quads:
        cells.append(("quad", np.asarray(quads, dtype=np.int32)))
    return meshio.Mesh(points=pts, cells=cells)


def write_vtk(mesh: Mesh, path: Path | str) -> None:
    """Write just the mesh geometry to a VTK / VTU / XDMF file (no fields)."""
    m = _mesh_to_meshio(mesh)
    m.write(str(path))


def snapshot_to_vtk(mesh: Mesh,
                    fields: Mapping[str, NDArray[np.float64]],
                    path: Path | str) -> None:
    """Write the mesh plus node-centred scalar fields to one VTK file.

    Each `fields[name]` must be a (NumNP,) array. Typical usage:
        snapshot_to_vtk(mesh, {"h": hNew, "theta": ThNew,
                               "conc": Conc, "Kode": Kode}, "snap.vtu")
    Open the result in ParaView; the fields are then plot-and-colour
    selectable directly.
    """
    meshio = _require_meshio()
    pts = np.column_stack(
        [mesh.nodes.x, mesh.nodes.y, np.zeros_like(mesh.nodes.x)]
    )
    tris = []
    quads = []
    for q in mesh.elements.KX:
        if q[2] == q[3]:
            tris.append([int(q[0]), int(q[1]), int(q[2])])
        else:
            quads.append([int(q[0]), int(q[1]), int(q[2]), int(q[3])])
    cells = []
    if tris:
        cells.append(("triangle", np.asarray(tris, dtype=np.int32)))
    if quads:
        cells.append(("quad", np.asarray(quads, dtype=np.int32)))
    point_data = {name: np.asarray(arr).astype(np.float64)
                  for name, arr in fields.items()}
    m = meshio.Mesh(points=pts, cells=cells, point_data=point_data)
    m.write(str(path))


# ============================================================================
# 3D mesh IO — tetrahedral / hexahedral meshes
# ============================================================================

def read_mesh_3d(path: Path | str):
    """Read a 3D mesh (tetrahedral or hexahedral) into a dict + skfem mesh.

    Returns a dict with:
        'skmesh' : scikit-fem MeshTet or MeshHex
        'points' : (N, 3) coordinates
        'cells'  : (NumEl, 4 or 8) connectivity
        'boundary_nodes' : 0-based ids on the surface
        'cell_type' : 'tetra' or 'hexahedron'
    """
    meshio = _require_meshio()
    m = meshio.read(str(path))
    if m.points.shape[1] != 3:
        raise ValueError(f"Expected 3D mesh; got points shape {m.points.shape}")
    cells_tet = None
    cells_hex = None
    for cb in m.cells:
        if cb.type == "tetra":
            cells_tet = cb.data
        elif cb.type == "hexahedron":
            cells_hex = cb.data
    try:
        import skfem
    except ImportError as e:
        raise ImportError("read_mesh_3d requires scikit-fem") from e
    if cells_tet is not None:
        skmesh = skfem.MeshTet(m.points.T, cells_tet.T.astype(np.int64))
        cells = cells_tet
        cell_type = 'tetra'
    elif cells_hex is not None:
        skmesh = skfem.MeshHex(m.points.T, cells_hex.T.astype(np.int64))
        cells = cells_hex
        cell_type = 'hexahedron'
    else:
        raise ValueError(f"No tetra/hex cells found in {path}")
    # Detect surface (boundary) nodes via skfem's facets_satisfying
    bnd = skmesh.boundary_nodes()
    return dict(skmesh=skmesh, points=m.points.copy(),
                cells=cells, boundary_nodes=bnd,
                cell_type=cell_type)


def make_box_mesh_3d(nx: int = 6, ny: int = 6, nz: int = 21,
                     lx: float = 0.1, ly: float = 0.1, lz: float = 1.0,
                     element: str = "tetra"):
    """Build a structured 3D box mesh for synthetic tests.

    Origin at (0, 0, 0); extents (lx, ly, lz). Z+ is up. For tetra,
    each cube is split into 5 tetrahedra (Caendish layout). For
    hexahedra, the box is a tensor-product grid.

    Returns the same dict shape as `read_mesh_3d`.
    """
    try:
        import skfem
    except ImportError as e:
        raise ImportError("make_box_mesh_3d requires scikit-fem") from e
    x = np.linspace(0.0, lx, nx)
    y = np.linspace(0.0, ly, ny)
    z = np.linspace(0.0, lz, nz)
    if element == "hex" or element == "hexahedron":
        skmesh = skfem.MeshHex.init_tensor(x, y, z)
    else:
        # MeshTet has a tensor-product helper (subdivides each cube)
        skmesh = skfem.MeshTet.init_tensor(x, y, z)
    pts = skmesh.p.T   # (N, 3)
    cells = skmesh.t.T  # (NumEl, ?)
    bnd = skmesh.boundary_nodes()
    return dict(skmesh=skmesh, points=pts.copy(), cells=cells.copy(),
                boundary_nodes=bnd, cell_type=element)


def snapshot_to_vtk_3d(skmesh,
                       fields: Mapping[str, NDArray[np.float64]],
                       path: Path | str) -> None:
    """Write a 3D skfem mesh + node-centred scalar fields to VTU.

    `fields[name]` must be (N,) arrays where N = skmesh.nvertices.
    """
    meshio = _require_meshio()
    pts = skmesh.p.T   # (N, 3)
    cells = []
    # Detect element type from skmesh.t shape
    nverts_per_cell = skmesh.t.shape[0]
    if nverts_per_cell == 4:
        cells = [("tetra", skmesh.t.T.astype(np.int32))]
    elif nverts_per_cell == 8:
        cells = [("hexahedron", skmesh.t.T.astype(np.int32))]
    elif nverts_per_cell == 3:
        cells = [("triangle", skmesh.t.T.astype(np.int32))]
    elif nverts_per_cell == 4 and pts.shape[1] == 2:
        cells = [("quad", skmesh.t.T.astype(np.int32))]
    point_data = {name: np.asarray(arr).astype(np.float64)
                  for name, arr in fields.items()}
    m = meshio.Mesh(points=pts, cells=cells, point_data=point_data)
    m.write(str(path))


def timeseries_to_vtk_series_3d(skmesh,
                                out_dir: Path | str,
                                snapshots: list[tuple[float, dict]],
                                prefix: str = "snap3d") -> Path:
    """Write a sequence of 3D snapshots as a ParaView .pvd series."""
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    pvd_path = out / f"{prefix}.pvd"
    pvd_lines = ['<?xml version="1.0"?>',
                 '<VTKFile type="Collection" version="0.1" '
                 'byte_order="LittleEndian">',
                 '  <Collection>']
    for i, (t, fields) in enumerate(snapshots):
        fname = f"{prefix}_{i:04d}.vtu"
        snapshot_to_vtk_3d(skmesh, fields, out / fname)
        pvd_lines.append(f'    <DataSet timestep="{t}" group="" part="0" '
                         f'file="{fname}"/>')
    pvd_lines.extend(['  </Collection>', '</VTKFile>'])
    pvd_path.write_text("\n".join(pvd_lines) + "\n")
    return pvd_path


def timeseries_to_vtk_series(mesh: Mesh,
                             out_dir: Path | str,
                             snapshots: list[tuple[float, dict]],
                             prefix: str = "snap") -> Path:
    """Write a sequence of timestep snapshots as a ParaView-loadable
    ``<prefix>.pvd`` collection file plus per-step VTUs.

    Parameters
    ----------
    mesh : Mesh
    out_dir : Path or str
        Output directory; created if missing.
    snapshots : list of (t, {field_name: array}) pairs
    prefix : str
        Base filename (default 'snap').

    Returns
    -------
    Path to the ``.pvd`` collection file.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pvd_path = out / f"{prefix}.pvd"
    pvd_lines = ['<?xml version="1.0"?>',
                 '<VTKFile type="Collection" version="0.1" '
                 'byte_order="LittleEndian">',
                 '  <Collection>']
    for i, (t, fields) in enumerate(snapshots):
        fname = f"{prefix}_{i:04d}.vtu"
        snapshot_to_vtk(mesh, fields, out / fname)
        pvd_lines.append(f'    <DataSet timestep="{t}" group="" part="0" '
                         f'file="{fname}"/>')
    pvd_lines.extend(['  </Collection>', '</VTKFile>'])
    pvd_path.write_text("\n".join(pvd_lines) + "\n")
    return pvd_path
