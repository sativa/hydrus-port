"""
3D Richards equation solver on scikit-fem.
==========================================

Self-contained 3D extension of the swms2d port. There is no Fortran
reference (SWMS_2D is 2D-only), so this module validates against:

1. **1D analytical solution** (Green-Ampt / Philip infiltration) for
   homogeneous vertical infiltration on a thin column-like 3D mesh;
2. **The Stage 1 2D solver** (swms2d.watflow) run on a 2D slab that
   matches a thin 3D column's mid-plane (sanity check).

Governing equation (mixed form):

    ∂θ(h)/∂t = ∇·(K(h) ∇h) - ∂(K(h))/∂z + S

where z is the vertical coordinate (positive up). The standard
convention here: gravity points in -z direction; the flux is
q = -K(h) ∇h - K(h) ẑ.

Element types supported by scikit-fem:
- `MeshTet` (4-node linear tetrahedra) + `ElementTetP1`
- `MeshHex` (8-node trilinear hexahedra) + `ElementHexS1`
- Higher-order: `ElementTetP2` (10-node), `ElementHexS2` available
  for future accuracy boosts.

Material parameters: per-material VG via hydrus1d.material's FK / FC
/ FQ; each node carries a 1-based MatNum index.

Boundary conditions:
- Dirichlet nodes specified by index array `dirich_nodes` + values `dirich_h`
- Neumann (flux) contributions specified by RHS vector `Q_bc[N]`
- Free-drainage / atmospheric flux handled by the caller (set Q_bc[i]
  = -Width[i] * rTop and leave node out of dirich_nodes).
"""

from __future__ import annotations
from typing import Optional, Callable
from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray

try:
    import skfem
    from skfem import (Basis, ElementTetP1, ElementHex1,
                       BilinearForm, LinearForm,
                       condense, solve)
    from skfem.helpers import dot, grad
    _SKFEM_OK = True
except ImportError:
    _SKFEM_OK = False

from .dataclasses import SoilMaterial
from . import material as _mat
from hydrus1d.material import FK as _FK, FC as _FC, FQ as _FQ


def _require_skfem():
    if not _SKFEM_OK:
        raise ImportError(
            "richards3d requires scikit-fem. Install with:\n"
            "    pip install --user --break-system-packages scikit-fem"
        )


# ============================================================================
# State + material evaluation
# ============================================================================

@dataclass
class RichardsState3D:
    """Per-node state for the 3D Richards solver."""
    hNew: NDArray[np.float64]
    hOld: NDArray[np.float64]
    ThNew: NDArray[np.float64]
    ThOld: NDArray[np.float64]
    MatNum: NDArray[np.int32]      # 1-based per Fortran convention


def evaluate_KCQ_3d(h: NDArray[np.float64],
                    MatNum: NDArray[np.int32],
                    materials: list[SoilMaterial],
                    ) -> tuple[NDArray[np.float64], NDArray[np.float64],
                               NDArray[np.float64]]:
    """Per-node K(h), C(h), θ(h). Same as 2D version."""
    n = h.shape[0]
    K = np.zeros(n, np.float64)
    C = np.zeros(n, np.float64)
    Th = np.zeros(n, np.float64)
    pars = [(_mat.select_imodel(m), _mat.to_h1d_par(m)) for m in materials]
    for i in range(n):
        iM = int(MatNum[i]) - 1
        iModel, Par = pars[iM]
        hi = float(h[i])
        if hi >= 0.0:
            K[i]  = materials[iM].Ks
            C[i]  = 0.0
            Th[i] = materials[iM].ths
        else:
            K[i] = _FK(iModel, hi, Par)
            C[i] = _FC(iModel, hi, Par)
            Th[i] = _FQ(iModel, hi, Par)
    return K, C, Th


def make_basis(skmesh):
    """Build a P1 basis appropriate for the mesh's element type."""
    _require_skfem()
    if skmesh.t.shape[0] == 4:
        # 4 nodes per element: tetrahedron
        return Basis(skmesh, ElementTetP1())
    if skmesh.t.shape[0] == 8:
        return Basis(skmesh, ElementHex1())
    raise ValueError(
        f"Unsupported 3D element with {skmesh.t.shape[0]} nodes per cell"
    )


# ============================================================================
# One Picard linear solve (3D)
# ============================================================================

def picard_step_3d(skmesh,
                   basis,
                   K_n: NDArray[np.float64],
                   C_n: NDArray[np.float64],
                   Th_n: NDArray[np.float64],
                   Th_old: NDArray[np.float64],
                   h_iter: NDArray[np.float64],
                   dt: float,
                   dirich_nodes: NDArray[np.int32],
                   dirich_h: NDArray[np.float64],
                   Q_bc: NDArray[np.float64],
                   gravity_axis: int = 2,
                   ) -> NDArray[np.float64]:
    """One Picard linear solve in 3D.

    `gravity_axis` is the index of the vertical coordinate (default 2
    for z+up in (x, y, z) layout). Gravity contributes
    `∫ K · (∂z/∂x_i) · ∂φ/∂x_i dV`, which reduces to integrating
    `K · ∂φ/∂z_axis`.
    """
    _require_skfem()
    dim = skmesh.p.shape[0]   # spatial dimension

    @BilinearForm
    def stiffness(u, v, w):
        return w["K"] * dot(grad(u), grad(v))

    @BilinearForm
    def mass_lumped(u, v, w):
        return w["C_over_dt"] * u * v

    @LinearForm
    def gravity(v, w):
        # Gravity unit vector: e_z (=(0,0,1) in 3D, (0,1) in 2D).
        return w["K"] * grad(v)[gravity_axis]

    @LinearForm
    def storage_rhs(v, w):
        return w["C_h_over_dt"] * v - w["theta_diff_over_dt"] * v

    K_int    = basis.interpolate(K_n.astype(np.float64))
    C_dt_int = basis.interpolate(C_n.astype(np.float64) / dt)
    Ch_dt_int = basis.interpolate(C_n.astype(np.float64) * h_iter / dt)
    dth_dt_int = basis.interpolate((Th_n - Th_old) / dt)

    A_stiff = stiffness.assemble(basis, K=K_int)
    M_mass  = mass_lumped.assemble(basis, C_over_dt=C_dt_int)
    b_grav  = gravity.assemble(basis, K=K_int)
    b_stor  = storage_rhs.assemble(
        basis, C_h_over_dt=Ch_dt_int, theta_diff_over_dt=dth_dt_int,
    )
    A = A_stiff + M_mass
    b = b_stor - b_grav + Q_bc

    if dirich_nodes.size > 0:
        x = np.zeros(basis.N)
        x[dirich_nodes] = dirich_h
        sol = solve(*condense(A, b, x=x, D=dirich_nodes))
    else:
        sol = solve(A, b)
    return sol


# ============================================================================
# Time-step driver (one dt) — Picard outer loop with convergence check
# ============================================================================

def solve_step_3d(skmesh,
                  state: RichardsState3D,
                  materials: list[SoilMaterial],
                  dt: float,
                  dirich_nodes: NDArray[np.int32],
                  dirich_h: NDArray[np.float64],
                  Q_bc: Optional[NDArray[np.float64]] = None,
                  gravity_axis: int = 2,
                  max_picard: int = 30,
                  tol_h: float = 0.01,
                  ) -> tuple[NDArray[np.float64], NDArray[np.float64], int, bool]:
    """Solve one dt step. Returns (h_new, theta_new, n_iter, converged)."""
    _require_skfem()
    basis = make_basis(skmesh)
    if Q_bc is None:
        Q_bc = np.zeros(basis.N, np.float64)
    h_iter = state.hNew.copy()
    for it in range(max_picard):
        K_n, C_n, Th_n = evaluate_KCQ_3d(h_iter, state.MatNum, materials)
        h_new = picard_step_3d(skmesh, basis, K_n, C_n, Th_n, state.ThOld,
                               h_iter, dt, dirich_nodes, dirich_h, Q_bc,
                               gravity_axis=gravity_axis)
        np.clip(h_new, -1e10, 1e10, out=h_new)
        if np.max(np.abs(h_new - h_iter)) < tol_h:
            _, _, Th_final = evaluate_KCQ_3d(h_new, state.MatNum, materials)
            return h_new, Th_final, it + 1, True
        h_iter = h_new
    _, _, Th_final = evaluate_KCQ_3d(h_iter, state.MatNum, materials)
    return h_iter, Th_final, max_picard, False


def integrate_3d(skmesh,
                 state: RichardsState3D,
                 materials: list[SoilMaterial],
                 t_end: float,
                 dt_init: float,
                 dirich_nodes: NDArray[np.int32],
                 dirich_h: NDArray[np.float64],
                 Q_bc: Optional[NDArray[np.float64]] = None,
                 gravity_axis: int = 2,
                 dt_max: float = 1.0,
                 dt_min: float = 1e-5,
                 dMul: float = 1.3,
                 dMul2: float = 0.7,
                 max_picard: int = 30,
                 tol_h: float = 0.01,
                 snapshot_times: Optional[list[float]] = None,
                 ) -> tuple[NDArray[np.float64], NDArray[np.float64],
                            list[tuple[float, NDArray[np.float64],
                                       NDArray[np.float64]]]]:
    """Time-integrate Richards 3D from t=0 to t=t_end.

    Returns (h_final, theta_final, snapshots) where snapshots is a list
    of (t, h_snap, theta_snap) at each `snapshot_times` request.
    """
    _require_skfem()
    t = 0.0
    dt = dt_init
    snap_idx = 0
    snapshots: list[tuple[float, NDArray[np.float64], NDArray[np.float64]]] = []
    if snapshot_times is None:
        snapshot_times = []
    while t < t_end - 1e-12:
        # Snap dt to next snapshot time if necessary
        next_snap = (snapshot_times[snap_idx]
                     if snap_idx < len(snapshot_times) else t_end)
        if t + dt > min(next_snap, t_end):
            dt_use = min(next_snap, t_end) - t
        else:
            dt_use = dt
        state.ThOld[:] = state.ThNew[:]
        state.hOld[:]  = state.hNew[:]
        h_new, th_new, it, conv = solve_step_3d(
            skmesh, state, materials, dt_use,
            dirich_nodes, dirich_h, Q_bc, gravity_axis,
            max_picard=max_picard, tol_h=tol_h,
        )
        if not conv:
            if dt_use > dt_min:
                # Halve dt and retry
                dt = max(dt_use / 3.0, dt_min)
                continue
            else:
                # Accept partial result
                pass
        state.hNew[:]  = h_new
        state.ThNew[:] = th_new
        t += dt_use
        # Snapshot capture
        if snap_idx < len(snapshot_times) and abs(t - snapshot_times[snap_idx]) < 1e-9:
            snapshots.append((t, h_new.copy(), th_new.copy()))
            snap_idx += 1
        # Adaptive dt
        if it <= 3:
            dt = min(dt_use * dMul, dt_max)
        elif it >= 7:
            dt = max(dt_use * dMul2, dt_min)
    return state.hNew.copy(), state.ThNew.copy(), snapshots
