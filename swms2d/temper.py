"""
2D heat transport for SWMS_2D Python port.
==========================================

The original SWMS_2D 1.22 source ships **no** TEMPER2.FOR — heat
transport was added by Simunek's group only in HYDRUS-2D/3D. This
module is a from-scratch 2D extension of hydrus1d.temper, sharing the
material-property formulae but using the 2D FE element-loop assembly
pattern from `solute_step`.

Governing equation (advection-conduction with optional sink):

    C_v(θ) ∂T/∂t = ∇·(κ(θ) ∇T) - ρ_w C_w v·∇T + ρ_w C_w S (T_root - T)

where:
    C_v(θ) = θ C_w + (ρ_b - θ ρ_w) C_s         volumetric heat capacity
    κ(θ)                                       Campbell thermal conductivity
    v                                          Darcy velocity (from watflow)
    S                                          sink rate (from set_snk)
    T_root                                     temperature of transpired water

Discretisation: Galerkin FE on the same triangular mesh as watflow,
two-level Crank-Nicholson (Level 1 explicit, Level 2 implicit) chosen
via `epsi` (1.0 = fully implicit, 0.5 = C-N, 0.0 = explicit Euler).

Boundary conditions per node (`KodeT` array, parallel to `Kode`):
    +1  Dirichlet T = T_bc           e.g. surface heating
    -1  prescribed heat flux Qh
     0  internal node                 no BC modification

Thermal parameters per material (`ParT[j, M]` for j = 0..7):
    [0] θ_wr     residual water content for κ(θ) curve
    [1] θ_ws     saturated water content
    [2] λ_d      dry-soil thermal conductivity
    [3] λ_s      saturated thermal conductivity (or δ_λ)
    [4] δ_λ      λ_s - λ_d (Campbell wetness coefficient)
    [5] ρ_b      bulk density
    [6] C_s      specific heat of soil solids
    [7] C_w      specific heat of water (typically 4180 J/(kg·K))

`compute_thermal_props` returns per-node (κ, C_v) following Campbell
(1985) — the same model used in hydrus1d.temper.
"""

from __future__ import annotations
from typing import Optional
import numpy as np
from numpy.typing import NDArray
from scipy.sparse import coo_matrix, csr_matrix, csc_matrix
from scipy.sparse.linalg import spsolve

from .dataclasses import Mesh, SimulationConfig


# ============================================================================
# Material properties — Campbell (1985) thermal model
# ============================================================================

def thermal_conductivity_node(theta: float,
                              ParT: NDArray[np.float64],
                              IKappa: int = 1) -> float:
    """Campbell wet-θ-dependent thermal conductivity κ(θ).

    IKappa=0: constant (returns ParT[2])
    IKappa=1: Campbell: κ = λ_d + δ_λ * sqrt((θ - θ_wr)/(θ_ws - θ_wr))
    """
    if IKappa == 0:
        return float(ParT[2])
    theta_wr = ParT[0]
    theta_ws = ParT[1]
    lambda_d = ParT[2]
    delta_l  = ParT[4]
    if theta <= theta_wr:
        return lambda_d
    Pf = (theta - theta_wr) / max(theta_ws - theta_wr, 1e-10)
    return lambda_d + delta_l * np.sqrt(max(Pf, 0.0))


def volumetric_heat_capacity_node(theta: float,
                                  ParT: NDArray[np.float64]) -> float:
    """C_v(θ) = θ·C_w + (ρ_b - θ·ρ_w)·C_s  (Campbell 1985, eq. 4.18).

    Here we treat ρ_w = 1 (water density factor in θ already), so this
    simplifies to C_v = θ·C_w + (ρ_b - θ)·C_s.
    """
    rho_b = ParT[5]
    Cs    = ParT[6]
    Cw    = ParT[7]
    return theta * Cw + max(rho_b - theta, 0.0) * Cs


def compute_thermal_props(theta: NDArray[np.float64],
                          MatNum: NDArray[np.int32],
                          ParT: NDArray[np.float64],
                          IKappa: int = 1,
                          ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Per-node (κ, C_v) for the 2D FE assembly.

    ParT has shape (8, NMat). MatNum is 1-based per Fortran convention.
    """
    n = theta.shape[0]
    kappa = np.zeros(n, np.float64)
    Cv    = np.zeros(n, np.float64)
    for i in range(n):
        M = int(MatNum[i]) - 1
        kappa[i] = thermal_conductivity_node(theta[i], ParT[:, M], IKappa)
        Cv[i]    = volumetric_heat_capacity_node(theta[i], ParT[:, M])
    return kappa, Cv


# ============================================================================
# Heat transport step (one dt — same Crank-Nicholson pattern as solute_step)
# ============================================================================

def heat_step(mesh: Mesh, cfg: SimulationConfig,
              t: float, dt: float,
              ThNew: NDArray[np.float64], ThOld: NDArray[np.float64],
              Vx: NDArray[np.float64], Vz: NDArray[np.float64],
              TempO: NDArray[np.float64],
              ParT: NDArray[np.float64],
              KodeT: NDArray[np.int32],
              T_bc: NDArray[np.float64],
              Qh_bc: Optional[NDArray[np.float64]] = None,
              Sink: Optional[NDArray[np.float64]] = None,
              T_root: float = 0.0,
              epsi: float = 0.5,
              IKappa: int = 1,
              ) -> NDArray[np.float64]:
    """Solve one time step of the heat transport equation.

    Parameters
    ----------
    ThNew, ThOld : per-node water content at t^{n+1}, t^n (from watflow)
    Vx, Vz       : per-node Darcy velocity (from solute.veloc)
    TempO        : per-node temperature at t^n
    ParT         : thermal parameters, shape (8, NMat)
    KodeT        : per-node BC code: +1 Dirichlet, -1 flux, 0 internal
    T_bc         : per-node Dirichlet temperature (only used where KodeT==+1)
    Qh_bc        : per-node heat flux (only used where KodeT==-1); zeros if None
    Sink         : water sink array (m³/m³/T) — root extraction
    T_root       : temperature of transpired water (root sink)
    epsi         : temporal weight (0=explicit, 0.5=C-N, 1=fully implicit)
    IKappa       : 0=constant κ, 1=Campbell

    Returns
    -------
    TempNew : per-node temperature at t^{n+1}
    """
    NumNP = mesh.NumNP
    NumEl = mesh.NumEl
    KAT = cfg.KAT
    KX = mesh.elements.KX
    x  = mesh.nodes.x
    y  = mesh.nodes.y
    MatNum = mesh.nodes.MatNum
    if Qh_bc is None:
        Qh_bc = np.zeros(NumNP, np.float64)
    if Sink is None:
        Sink = np.zeros(NumNP, np.float64)

    alf = 1.0 - epsi
    NLevel = 1 if epsi >= 0.999 else 2

    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    B = np.zeros(NumNP, np.float64)

    Cw_default = float(ParT[7, 0])   # water specific heat — same all materials

    for Level in range(1, NLevel + 1):
        if Level == NLevel:
            kappa, Cv = compute_thermal_props(ThNew, MatNum, ParT, IKappa)
        else:
            kappa, Cv = compute_thermal_props(ThOld, MatNum, ParT, IKappa)

        # Reaction coefficients per node (analogous to solute Ac/Fc/Gc):
        #   storage:   Ac = -C_v
        #   sink:      Fc = ρ_w C_w Sink     (heat removed by root uptake)
        #   source:    Gc = ρ_w C_w Sink T_root   (heat carried in by water)
        # For now ρ_w = 1 (water density factor absorbed in θ).
        Ac = -Cv.copy() if Level == NLevel else None
        Fc = Cw_default * Sink                         # 1st-order sink term
        Gc = Cw_default * Sink * T_root                # 0th-order source

        F = np.zeros(NumNP, np.float64)
        DS = np.zeros(NumNP, np.float64) if Level == NLevel else None

        for n in range(NumEl):
            CAxx = mesh.elements.ConAxx[n]
            CAzz = mesh.elements.ConAzz[n]
            NUS = 3 if KX[n, 2] == KX[n, 3] else 4
            for sk in range(NUS - 2):
                i = KX[n, 0]; j = KX[n, sk + 1]; l = KX[n, sk + 2]
                LIST = (i, j, l)
                Ci0 = x[l] - x[j]; Ci1 = x[i] - x[l]; Ci2 = x[j] - x[i]
                Bi0 = y[j] - y[l]; Bi1 = y[l] - y[i]; Bi2 = y[i] - y[j]
                Bi = (Bi0, Bi1, Bi2)
                Ci = (Ci0, Ci1, Ci2)
                AE = (Ci2 * Bi1 - Ci1 * Bi2) / 2.0

                # Element-averaged thermal conductivity (isotropic at element)
                kE = (kappa[i] + kappa[j] + kappa[l]) / 3.0
                # Velocity at element centroid (heat advection)
                VxE = [Vx[i] * Cw_default, Vx[j] * Cw_default, Vx[l] * Cw_default]
                VzE = [Vz[i] * Cw_default, Vz[j] * Cw_default, Vz[l] * Cw_default]
                VxEE = (VxE[0] + VxE[1] + VxE[2]) / 3.0
                VzEE = (VzE[0] + VzE[1] + VzE[2]) / 3.0

                xMul = 1.0
                if KAT == 1:
                    xMul = 2.0 * 3.1416 * (x[i] + x[j] + x[l]) / 3.0
                FMul  = xMul * AE / 4.0
                SMul1 = -1.0 / AE / 4.0 * xMul
                SMul2 = AE / 20.0 * xMul

                GcE = (Gc[i] + Gc[j] + Gc[l]) / 3.0
                FcE = (Fc[i] + Fc[j] + Fc[l]) / 3.0
                AcE = (Ac[i] + Ac[j] + Ac[l]) / 3.0 if Level == NLevel else 0.0

                for j1 in range(3):
                    i1 = LIST[j1]
                    F[i1] += FMul * (GcE + Gc[i1] / 3.0)
                    if Level == NLevel:
                        DS[i1] += FMul * (AcE + Ac[i1] / 3.0)
                    for j2 in range(3):
                        i2 = LIST[j2]
                        # Conduction term (isotropic κ on a tensor of CAxx, CAzz)
                        S = SMul1 * kE * (
                            CAxx * Bi[j1] * Bi[j2] + CAzz * Ci[j1] * Ci[j2]
                        )
                        # Advection (heat carried by water)
                        S -= (Bi[j2] / 8.0 * (VxEE + VxE[j1] / 3.0)
                              + Ci[j2] / 8.0 * (VzEE + VzE[j1] / 3.0)) * xMul
                        # Reaction (Fc · T storage in mass-matrix lumped form)
                        ic = 2 if i1 == i2 else 1
                        S += SMul2 * ic * (FcE + (Fc[i1] + Fc[i2]) / 3.0)
                        if Level != NLevel:
                            B[i1] -= alf * S * TempO[i2]
                        else:
                            rows.append(int(i1))
                            cols.append(int(i2))
                            vals.append(epsi * S)

        if Level == NLevel:
            # Add DS/dt to diagonal and form RHS contribution
            B += DS / dt * TempO - epsi * F
            for i in range(NumNP):
                rows.append(int(i))
                cols.append(int(i))
                vals.append(DS[i] / dt)
        else:
            B -= alf * F

    # ---- Build sparse matrix
    A = coo_matrix(
        (np.asarray(vals, np.float64),
         (np.asarray(rows, np.int32), np.asarray(cols, np.int32))),
        shape=(NumNP, NumNP),
    ).tocsr()
    A.sum_duplicates()

    # ---- Apply boundary conditions
    A_lil = A.tolil()
    for i in range(NumNP):
        if KodeT[i] == 1:
            # Dirichlet T = T_bc[i]
            A_lil.rows[i] = [int(i)]
            A_lil.data[i] = [1.0]
            B[i] = T_bc[i]
        elif KodeT[i] == -1:
            # Neumann: add prescribed flux to RHS
            B[i] += Qh_bc[i]
    A = A_lil.tocsr()
    A.sum_duplicates()

    return spsolve(csc_matrix(A), B)


# ============================================================================
# Convenience: default parameter set
# ============================================================================

def default_ParT(NMat: int = 1,
                 theta_wr: float = 0.05,
                 theta_ws: float = 0.40,
                 lambda_d: float = 0.4,
                 lambda_s: float = 1.6,
                 rho_b: float = 1.5,
                 Cs: float = 0.85e6,   # J / m³ / K — typical mineral soil
                 Cw: float = 4.18e6,   # J / m³ / K — water
                 ) -> NDArray[np.float64]:
    """Build a (8, NMat) ParT array with reasonable Campbell defaults."""
    ParT = np.zeros((8, NMat), np.float64)
    for M in range(NMat):
        ParT[0, M] = theta_wr
        ParT[1, M] = theta_ws
        ParT[2, M] = lambda_d
        ParT[3, M] = lambda_s
        ParT[4, M] = lambda_s - lambda_d
        ParT[5, M] = rho_b
        ParT[6, M] = Cs
        ParT[7, M] = Cw
    return ParT
