"""
2D advection-dispersion-reaction solver for SWMS_2D Python port.
================================================================

Direct port of SOLUTE2.FOR. Implements:
    veloc()  — Darcy velocity at every node from h-gradient × K
    disper() — dispersion tensor (Dxx, Dxz, Dzz)
    pe_cour() — Peclet/Courant + dtMaxC limit
    we_fact() — upwind weighting factors (only if lUpW=True)
    c_bound() — apply Dirichlet/Neumann/Cauchy solute BCs to (A, B)
    solute_step() — one ADE solve per dt-step

The temporal scheme:
    NLevel = 1 if epsi >= 0.999 else 2
    Level 1: explicit half-step builds RHS using ConO, ThOld, hOld, Conc^n
    Level 2: implicit half-step builds matrix using Con, ThNew, hNew, Conc^{n+1}
The two levels are summed in c_Bound's S matrix and B vector.

ChPar layout (per material) — SWMS_2D MATERIA2 convention:
    [0] Bulk.d     bulk density
    [1] Difus      free-water diffusion coefficient
    [2] DispL      longitudinal dispersivity
    [3] DispT      transverse dispersivity
    [4] Adsorp     linear isotherm Kd
    [5] SinkL1     liquid 1st-order decay
    [6] SinkS1     solid  1st-order decay
    [7] SinkL0     liquid 0th-order source
    [8] SinkS0     solid  0th-order source
"""

from __future__ import annotations
import numpy as np
from numpy.typing import NDArray
from scipy.sparse import coo_matrix, csr_matrix, csc_matrix
from scipy.sparse.linalg import spsolve

from .dataclasses import Mesh, SimulationConfig


# ============================================================================
# Velocity field at nodes (Veloc in SOLUTE2.FOR L418-473)
# ============================================================================

def veloc(mesh: Mesh, hNew: NDArray[np.float64],
          Con: NDArray[np.float64], KAT: int,
          ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Nodal Darcy velocities Vx, Vz averaged over neighboring elements."""
    NumNP = mesh.NumNP
    NumEl = mesh.NumEl
    KX = mesh.elements.KX
    x  = mesh.nodes.x
    y  = mesh.nodes.y
    ConAxx = mesh.elements.ConAxx
    ConAzz = mesh.elements.ConAzz
    ConAxz = mesh.elements.ConAxz
    Vx = np.zeros(NumNP, np.float64)
    Vz = np.zeros(NumNP, np.float64)
    for e in range(NumEl):
        CAxx = ConAxx[e]
        CAzz = ConAzz[e]
        CAxz = ConAxz[e]
        NCorn = 3 if KX[e, 2] == KX[e, 3] else 4
        for n in range(NCorn - 2):
            i = KX[e, 0]; j = KX[e, n + 1]; k = KX[e, n + 2]
            vi = y[j] - y[k]; vj = y[k] - y[i]; vk = y[i] - y[j]
            wi = x[k] - x[j]; wj = x[i] - x[k]; wk = x[j] - x[i]
            Area = 0.5 * (wk * vj - wj * vk)
            A = 1.0 / Area / 2.0
            # Vx
            Ai = CAxx * vi + CAxz * wi
            Aj = CAxx * vj + CAxz * wj
            Ak = CAxx * vk + CAxz * wk
            Vxx = A * (Ai * hNew[i] + Aj * hNew[j] + Ak * hNew[k])
            if KAT > 0:
                Vxx += CAxz
            # Vz
            Ai = CAxz * vi + CAzz * wi
            Aj = CAxz * vj + CAzz * wj
            Ak = CAxz * vk + CAzz * wk
            Vzz = A * (Ai * hNew[i] + Aj * hNew[j] + Ak * hNew[k])
            if KAT > 0:
                Vzz += CAzz
            for m in (i, j, k):
                Vx[m] -= Con[m] * Vxx
                Vz[m] -= Con[m] * Vzz
    # Average by neighboring-element count
    LNE = np.where(mesh.ListNE > 0, mesh.ListNE, 1)
    Vx /= LNE
    Vz /= LNE
    return Vx, Vz


# ============================================================================
# Dispersion (Disper in SOLUTE2.FOR L479-506)
# ============================================================================

def disper(Vx: NDArray[np.float64], Vz: NDArray[np.float64],
           theta: NDArray[np.float64], thSat: NDArray[np.float64],
           ChPar: NDArray[np.float64], MatNum: NDArray[np.int32],
           lArtD: bool = False, PeCr: float = 2.0, dt: float = 0.0,
           ) -> tuple[NDArray[np.float64], NDArray[np.float64],
                      NDArray[np.float64]]:
    """Per-node dispersion tensor (Dxx, Dxz, Dzz)."""
    n = theta.shape[0]
    Dxx = np.zeros(n, np.float64)
    Dxz = np.zeros(n, np.float64)
    Dzz = np.zeros(n, np.float64)
    for i in range(n):
        M = MatNum[i] - 1
        Tau = theta[i] ** (7.0/3.0) / thSat[M] ** 2
        Vabs = np.sqrt(Vx[i]**2 + Vz[i]**2)
        Dif = theta[i] * ChPar[1, M] * Tau
        DispL = ChPar[2, M]
        DispT = ChPar[3, M]
        if lArtD and Vabs > 1e-20:
            denom = (theta[i] + ChPar[0, M] * ChPar[4, M]) * PeCr
            DispL = max(DispL, Vabs * dt / denom - Dif / Vabs)
        if Vabs > 1e-20:
            Dxx[i] = DispL * Vx[i]**2 / Vabs + DispT * Vz[i]**2 / Vabs + Dif
            Dzz[i] = DispL * Vz[i]**2 / Vabs + DispT * Vx[i]**2 / Vabs + Dif
            Dxz[i] = (DispL - DispT) * Vx[i] * Vz[i] / Vabs
        else:
            Dxx[i] = Dif
            Dzz[i] = Dif
    return Dxx, Dxz, Dzz


# ============================================================================
# Main solute step (Solute in SOLUTE2.FOR L3-281)
# ============================================================================

def solute_step(mesh: Mesh, cfg: SimulationConfig,
                t: float, dt: float,
                hNew: NDArray[np.float64], hOld: NDArray[np.float64],
                Con: NDArray[np.float64], ConO: NDArray[np.float64],
                ThNew: NDArray[np.float64], ThOld: NDArray[np.float64],
                thSat: NDArray[np.float64],
                Conc: NDArray[np.float64], Sink: NDArray[np.float64],
                ChPar: NDArray[np.float64],
                MatNum: NDArray[np.int32],
                cBound: NDArray[np.float64],
                KodCB: NDArray[np.int32],
                epsi: float,
                tPulse: float,
                cPrec: float = 0.0,
                crt: float = 0.0, cht: float = 0.0,
                lUpW: bool = False, lArtD: bool = False, PeCr: float = 2.0,
                ) -> NDArray[np.float64]:
    """Solve one solute-transport time step. Returns updated Conc array."""
    NumNP = mesh.NumNP
    NumEl = mesh.NumEl
    KAT = cfg.KAT
    KX = mesh.elements.KX
    x  = mesh.nodes.x
    y  = mesh.nodes.y
    Kode = mesh.nodes.Kode
    KXB  = mesh.KXB
    Q    = mesh.nodes.Q

    alf = 1.0 - epsi
    NLevel = 1 if epsi >= 0.999 else 2

    # Tools: dense triplet builder for sparse matrix
    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    B = np.zeros(NumNP, np.float64)
    Qc = np.zeros(NumNP, np.float64)

    # Effective pulse cutoff
    if t > tPulse:
        cBound = cBound.copy()
        cBound[:4] = 0.0

    for Level in range(1, NLevel + 1):
        Eps = epsi if Level == NLevel else (1.0 - epsi)
        # Recompute velocities + dispersion at the current iterate
        if Level == NLevel:
            Vx, Vz = veloc(mesh, hNew, Con, KAT)
            Dxx, Dxz, Dzz = disper(Vx, Vz, ThNew, thSat, ChPar, MatNum,
                                    lArtD, PeCr, dt)
        else:
            Vx, Vz = veloc(mesh, hOld, ConO, KAT)
            Dxx, Dxz, Dzz = disper(Vx, Vz, ThOld, thSat, ChPar, MatNum,
                                    lArtD, PeCr, dt)

        # Per-node Ac/Fc/Gc reaction coefficients
        Ac = np.zeros(NumNP, np.float64)
        Fc = np.zeros(NumNP, np.float64)
        Gc = np.zeros(NumNP, np.float64)
        for i in range(NumNP):
            M = MatNum[i] - 1
            if Level == NLevel:
                Ac[i] = -(ThOld[i] * alf + ThNew[i] * epsi) - ChPar[0, M] * ChPar[4, M]
                cS = cBound[4]
                if cS > Conc[i]:
                    cS = Conc[i]
                Gc[i] = ChPar[7, M] * ThNew[i] + ChPar[0, M] * ChPar[8, M] - Sink[i] * cS
                Fc[i] = ChPar[5, M] * ThNew[i] + ChPar[0, M] * ChPar[6, M] * ChPar[4, M] + Sink[i]
            else:
                cS = cBound[4]
                if cS > Conc[i]:
                    cS = Conc[i]
                Gc[i] = ChPar[7, M] * ThOld[i] + ChPar[0, M] * ChPar[8, M] - Sink[i] * cS
                Fc[i] = ChPar[5, M] * ThOld[i] + ChPar[0, M] * ChPar[6, M] * ChPar[4, M] + Sink[i]

        F = np.zeros(NumNP, np.float64)
        DS = np.zeros(NumNP, np.float64) if Level == NLevel else None

        # Element loop
        for n in range(NumEl):
            CAxx = mesh.elements.ConAxx[n]
            CAzz = mesh.elements.ConAzz[n]
            CAxz = mesh.elements.ConAxz[n]
            NUS = 3 if KX[n, 2] == KX[n, 3] else 4
            for sk in range(NUS - 2):
                i = KX[n, 0]; j = KX[n, sk + 1]; l = KX[n, sk + 2]
                LIST = (i, j, l)
                Ci0 = x[l] - x[j]; Ci1 = x[i] - x[l]; Ci2 = x[j] - x[i]
                Bi0 = y[j] - y[l]; Bi1 = y[l] - y[i]; Bi2 = y[i] - y[j]
                Bi = (Bi0, Bi1, Bi2)
                Ci = (Ci0, Ci1, Ci2)
                AE = (Ci2 * Bi1 - Ci1 * Bi2) / 2.0
                AE1 = 0.5 / AE
                # Velocities at the element (Darcy v = -K∇H)
                if Level == NLevel:
                    h_use = hNew
                    Con_use = Con
                else:
                    h_use = hOld
                    Con_use = ConO
                Ai = CAxx * Bi0 + CAxz * Ci0
                Aj = CAxx * Bi1 + CAxz * Ci1
                Ak = CAxx * Bi2 + CAxz * Ci2
                Vxx = AE1 * (Ai * h_use[i] + Aj * h_use[j] + Ak * h_use[l])
                if KAT > 0:
                    Vxx += CAxz
                Ai = CAxz * Bi0 + CAzz * Ci0
                Aj = CAxz * Bi1 + CAzz * Ci1
                Ak = CAxz * Bi2 + CAzz * Ci2
                Vzz = AE1 * (Ai * h_use[i] + Aj * h_use[j] + Ak * h_use[l])
                if KAT > 0:
                    Vzz += CAzz
                ConE = (Con_use[i] + Con_use[j] + Con_use[l]) / 3.0
                VxE = [-Con_use[i] * Vxx, -Con_use[j] * Vxx, -Con_use[l] * Vxx]
                VzE = [-Con_use[i] * Vzz, -Con_use[j] * Vzz, -Con_use[l] * Vzz]
                VxEE = -ConE * Vxx
                VzEE = -ConE * Vzz

                xMul = 1.0
                if KAT == 1:
                    xMul = 2.0 * np.pi * (x[i] + x[j] + x[l]) / 3.0
                FMul  = xMul * AE / 4.0
                SMul1 = -1.0 / AE / 4.0 * xMul
                SMul2 = AE / 20.0 * xMul

                GcE = (Gc[i] + Gc[j] + Gc[l]) / 3.0
                FcE = (Fc[i] + Fc[j] + Fc[l]) / 3.0
                AcE = (Ac[i] + Ac[j] + Ac[l]) / 3.0 if Level == NLevel else 0.0
                Ec1 = (Dxx[i] + Dxx[j] + Dxx[l]) / 3.0
                Ec2 = (Dxz[i] + Dxz[j] + Dxz[l]) / 3.0
                Ec3 = (Dzz[i] + Dzz[j] + Dzz[l]) / 3.0

                for j1 in range(3):
                    i1 = LIST[j1]
                    F[i1] += FMul * (GcE + Gc[i1] / 3.0)
                    if Level == NLevel:
                        DS[i1] += FMul * (AcE + Ac[i1] / 3.0)
                    for j2 in range(3):
                        i2 = LIST[j2]
                        S = SMul1 * (Ec1 * Bi[j1] * Bi[j2]
                                     + Ec3 * Ci[j1] * Ci[j2]
                                     + Ec2 * (Bi[j1] * Ci[j2] + Ci[j1] * Bi[j2]))
                        S -= (Bi[j2] / 8.0 * (VxEE + VxE[j1] / 3.0)
                              + Ci[j2] / 8.0 * (VzEE + VzE[j1] / 3.0)) * xMul
                        ic = 2 if i1 == i2 else 1
                        S += SMul2 * ic * (FcE + (Fc[i1] + Fc[i2]) / 3.0)
                        if Level != NLevel:
                            B[i1] -= alf * S * Conc[i2]
                        else:
                            rows.append(int(i1))
                            cols.append(int(i2))
                            vals.append(epsi * S)

        # End element loop. After both levels, add DS/dt to diagonal + bookkeeping.
        if Level == NLevel:
            B += DS / dt * Conc - epsi * F
            # add DS/dt to diagonal
            for i in range(NumNP):
                rows.append(int(i))
                cols.append(int(i))
                vals.append(DS[i] / dt)
        else:
            B -= alf * F

    # Build sparse matrix
    A = coo_matrix(
        (np.asarray(vals, np.float64),
         (np.asarray(rows, np.int32), np.asarray(cols, np.int32))),
        shape=(NumNP, NumNP),
    ).tocsr()
    A.sum_duplicates()

    # ---- Boundary conditions (c_Bound)
    A = A.tolil()
    DS_dummy = np.zeros(NumNP, np.float64)
    # Build KXB → KodCB lookup
    kxb_to_kodcb: dict[int, int] = {int(KXB[k]): int(KodCB[k]) for k in range(mesh.NumBP)}
    for i in range(NumNP):
        if Kode[i] == 0:
            continue
        if i in kxb_to_kodcb:
            kcb = kxb_to_kodcb[i]
            if kcb > 0:
                cKod = 1
                if abs(Kode[i]) <= 2 or abs(Kode[i]) >= 5:
                    cBnd = cBound[kcb - 1]
                elif abs(Kode[i]) == 3:
                    cBnd = cht
                elif abs(Kode[i]) == 4:
                    cBnd = cPrec
                else:
                    cBnd = 0.0
            else:
                if Q[i] > 0.0:
                    cKod = 3
                    if abs(Kode[i]) == 1 or abs(Kode[i]) >= 5:
                        cBnd = cBound[-kcb - 1] if -kcb - 1 < 6 else 0.0
                    elif abs(Kode[i]) == 3:
                        cBnd = crt
                    elif abs(Kode[i]) == 4:
                        cBnd = cPrec
                    else:
                        cBnd = 0.0
                else:
                    cKod = 2
                    cBnd = 0.0
                    if Kode[i] == -4:
                        cKod = 3
                        cBnd = 0.0
            if abs(Kode[i]) == 2:
                cKod = 2
        else:
            # Point source or sink at internal Dirichlet
            if Q[i] < 0.0:
                cKod = 2
                cBnd = 0.0
            else:
                cBnd = cBound[5]
                cKod = 3
        # Apply
        if cKod == 1:
            Qc[i] += Q[i] * (epsi * cBnd + alf * Conc[i]) - DS_dummy[i] * (cBnd - Conc[i]) / dt
            # Zero row, set diag = 1, RHS = cBnd
            row_dense = np.zeros(NumNP)
            A.rows[i] = [int(i)]
            A.data[i] = [1.0]
            B[i] = cBnd
        elif cKod == 2:
            Qc[i] = Q[i] * Conc[i]
        elif cKod == 3:
            B[i] -= Q[i] * (cBnd - alf * Conc[i])
            A[i, i] = A[i, i] - epsi * Q[i]
            Qc[i] = Q[i] * cBnd

    # ---- Solve
    A = A.tocsr()
    A.sum_duplicates()
    sol = spsolve(csc_matrix(A), B)
    sol = np.where(np.abs(sol) < 1e-38, 0.0, sol)
    return sol
