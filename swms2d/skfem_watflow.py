"""
Stage 2 prototype: 2D Richards equation on scikit-fem.
======================================================

Parallel implementation of `watflow.py` using scikit-fem's high-level
FE machinery. The motivation is to set up a 3D-capable, higher-order-
element-ready platform without abandoning the verified Stage 1
implementation.

This prototype intentionally covers a minimal feature set:

- Triangle-only mesh (auto-converted from swms2d.Mesh's quads via
  fan triangulation, mirroring the Fortran NUS-2 sub-element loop).
- Picard iteration on h-form Richards equation:

    F[i] C(h^k)/dt · h^{n+1} + L(h^k) · h^{n+1}
       = F C(h^k) h^k / dt - F (theta(h^k) - theta^n)/dt - L_grav + Q

  where L is the stiffness with K(h^k) coefficient and L_grav is the
  gravity contribution (K ∇z · ∇φ_i integrated over the domain).

- Dirichlet BC (Kode>=1) applied via skfem's `condense`.
- Flux BC (Kode<0) added to RHS via boundary integral (or zero for
  seepage Kode=-2 which is handled by the natural BC = 0).
- Time stepping via the Stage 1 driver's TmCont if invoked through
  `solve_step_skfem` (one Picard solve per call).

Verification target: water-flow only on EX.1, agreement with
swms2d.watflow output to within 1 % L2 of h field. Bit-equal is **not**
expected — the FE assembly path differs (skfem uses optimised numerical
quadrature versus the Fortran's analytic per-element integration).
"""

from __future__ import annotations
from typing import Optional
import numpy as np
from numpy.typing import NDArray

try:
    import skfem
    from skfem import (Basis, ElementTriP1, BilinearForm, LinearForm,
                       FacetBasis, condense, solve)
    from skfem.helpers import dot, grad
    _SKFEM_OK = True
except ImportError:
    _SKFEM_OK = False

from .dataclasses import Mesh, SimulationConfig, SoilMaterial
from . import material as _mat
from hydrus1d.material import FK as _FK, FC as _FC, FQ as _FQ


def _require_skfem():
    if not _SKFEM_OK:
        raise ImportError(
            "swms2d.skfem_watflow requires the 'scikit-fem' package.\n"
            "Install with: pip install --user --break-system-packages scikit-fem"
        )


# ============================================================================
# Mesh conversion: swms2d.Mesh -> skfem.MeshTri
# ============================================================================

def to_skfem_mesh(mesh: Mesh):
    """Convert a swms2d.Mesh (quads + triangles) into a skfem MeshTri.

    Quads are fan-triangulated (split into 2 triangles via the
    KX[0]-anchored fan, same convention as watflow.reset's NUS-2 loop).
    The resulting triangle ordering preserves the parent quad's element
    index in `parent_elem_idx` (returned alongside the mesh).
    """
    _require_skfem()
    KX = mesh.elements.KX
    NumEl = mesh.NumEl
    tris: list[list[int]] = []
    parent: list[int] = []
    for e in range(NumEl):
        if KX[e, 2] == KX[e, 3]:
            tris.append([int(KX[e, 0]), int(KX[e, 1]), int(KX[e, 2])])
            parent.append(e)
        else:
            tris.append([int(KX[e, 0]), int(KX[e, 1]), int(KX[e, 2])])
            parent.append(e)
            tris.append([int(KX[e, 0]), int(KX[e, 2]), int(KX[e, 3])])
            parent.append(e)
    pts = np.column_stack([mesh.nodes.x, mesh.nodes.y])    # (NumNP, 2)
    tris_arr = np.asarray(tris, dtype=np.int64).T          # (3, NumTri)
    m = skfem.MeshTri(pts.T, tris_arr)
    return m, np.asarray(parent, dtype=np.int64)


# ============================================================================
# Material property evaluation (node-wise) — wraps hydrus1d.material
# ============================================================================

def evaluate_KCQ(h: NDArray[np.float64],
                 MatNum: NDArray[np.int32],
                 materials: list[SoilMaterial],
                 ) -> tuple[NDArray[np.float64], NDArray[np.float64],
                            NDArray[np.float64]]:
    """Per-node K(h), C(h), theta(h). MatNum is 1-based."""
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
            # Saturated
            K[i] = materials[iM].Ks
            C[i] = 0.0
            Th[i] = materials[iM].ths
        else:
            K[i] = _FK(iModel, hi, Par)
            C[i] = _FC(iModel, hi, Par)
            Th[i] = _FQ(iModel, hi, Par)
    return K, C, Th


# ============================================================================
# Picard step on h-form Richards eq
# ============================================================================

def picard_step(skmesh, basis: "Basis",
                K_n: NDArray[np.float64],
                C_n: NDArray[np.float64],
                Th_n: NDArray[np.float64],
                Th_old: NDArray[np.float64],
                h_iter: NDArray[np.float64],
                dt: float,
                dirich_nodes: NDArray[np.int32],
                dirich_h: NDArray[np.float64],
                Q_bc: NDArray[np.float64],
                gravity_dir: tuple[float, float] = (0.0, 1.0),
                ) -> NDArray[np.float64]:
    """One Picard linear solve. Returns h^{k+1}.

    Forms (scikit-fem auto-quadrature):
      mass(u, v)   = (C/dt) * u * v      lumped via M_lumped diagonal
      stiff(u, v)  = K · ∇u · ∇v
      grav(v)      = K · ∇z · ∇v        (gravity vector to subtract)

    K, C, theta are passed per-node; scikit-fem interpolates to
    quadrature points internally.
    """
    _require_skfem()
    # Project node-wise K, C onto the basis as DOF arrays
    # (P1 elements -> node values directly map to DOFs).
    K_dof = K_n.astype(np.float64)
    C_dof = C_n.astype(np.float64)

    @BilinearForm
    def stiffness(u, v, w):
        return w["K"] * dot(grad(u), grad(v))

    @BilinearForm
    def mass_lumped(u, v, w):
        return w["C_over_dt"] * u * v

    @LinearForm
    def gravity(v, w):
        gx, gz = gravity_dir
        # ∂z/∂x = gx, ∂z/∂y = gz
        return w["K"] * (gx * grad(v)[0] + gz * grad(v)[1])

    @LinearForm
    def storage_rhs(v, w):
        return w["C_h_over_dt"] * v - w["theta_diff_over_dt"] * v

    # Project node-DOF arrays into the basis interpolator form
    K_int = basis.interpolate(K_dof)
    C_dt_int = basis.interpolate(C_dof / dt)
    Ch_dt_int = basis.interpolate(C_dof * h_iter / dt)
    dth_dt_int = basis.interpolate((Th_n - Th_old) / dt)

    A_stiff = stiffness.assemble(basis, K=K_int)
    M_mass  = mass_lumped.assemble(basis, C_over_dt=C_dt_int)
    b_grav  = gravity.assemble(basis, K=K_int)
    b_stor  = storage_rhs.assemble(
        basis,
        C_h_over_dt=Ch_dt_int,
        theta_diff_over_dt=dth_dt_int,
    )

    A = A_stiff + M_mass
    b = b_stor - b_grav + Q_bc

    # Dirichlet BC via condense
    if dirich_nodes.size > 0:
        x = np.zeros(basis.N)
        x[dirich_nodes] = dirich_h
        sol = solve(*condense(A, b, x=x, D=dirich_nodes))
    else:
        sol = solve(A, b)
    return sol


def solve_step_skfem(mesh: Mesh,
                    materials: list[SoilMaterial],
                    h_init: NDArray[np.float64],
                    Th_old: NDArray[np.float64],
                    dt: float,
                    max_picard: int = 20,
                    tol_h: float = 0.05,
                    Q_bc: Optional[NDArray[np.float64]] = None,
                    ) -> tuple[NDArray[np.float64], int, bool]:
    """Solve one dt time step using scikit-fem Picard.

    Returns (h_new, n_iter, converged).
    """
    _require_skfem()
    skmesh, _ = to_skfem_mesh(mesh)
    basis = Basis(skmesh, ElementTriP1())

    # Detect Dirichlet nodes from Kode>=1
    Kode = mesh.nodes.Kode
    dirich_nodes = np.where(Kode >= 1)[0].astype(np.int32)
    dirich_h = mesh.nodes.hNew[dirich_nodes].astype(np.float64)
    if Q_bc is None:
        Q_bc = np.zeros(mesh.NumNP, np.float64)

    h_iter = h_init.copy()
    for it in range(max_picard):
        K_n, C_n, Th_n = evaluate_KCQ(h_iter, mesh.nodes.MatNum, materials)
        h_new = picard_step(skmesh, basis, K_n, C_n, Th_n, Th_old, h_iter,
                            dt, dirich_nodes, dirich_h, Q_bc)
        np.clip(h_new, -1e10, 1e10, out=h_new)
        if np.max(np.abs(h_new - h_iter)) < tol_h:
            return h_new, it + 1, True
        h_iter = h_new
    return h_iter, max_picard, False
