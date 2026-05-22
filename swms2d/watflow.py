"""
2D Richards equation solver for SWMS_2D Python port.
====================================================

Direct port of WATFLOW2.FOR. Galerkin finite element discretisation on
linear triangular elements (quadrilaterals split into 2 triangles via
fan from node 1, mirroring Fortran NUS-2 sub-element loop).

Functional decomposition matches Fortran exactly:
    set_mat()  — node-wise K(h), C(h), θ(h) (replaces SetMat + table lookup)
    reset()    — element loop building global sparse A and RHS B
    shift()    — modify Kode for seepage / atm / drain / free-drainage BCs
    dirich()   — zero rows/cols and set diagonal for Dirichlet nodes
    solve_water_flow() — Picard iteration loop with adaptive dt fallback

Differences from Fortran (documented inline):
    1. Banded matrix A(MBandD, NumNP) → scipy.sparse.csr_matrix.
       The Solve banded Gauss elimination is replaced by spsolve.
       Numerically equivalent up to fill-in ordering.
    2. Material property table (hTab/ConTab/CapTab/TheTab) → direct
       FK/FC/FQ calls via swms2d.material adapter.
       Slightly slower, identical accuracy to within 1/NTab=1% (the
       table is linear interpolation with NTab=100 points anyway).
    3. ORTHOMIN/ILU path (lOrt=true) is not implemented — we always
       use direct sparse solve.
"""

from __future__ import annotations
from typing import Optional
import numpy as np
from numpy.typing import NDArray
from scipy.sparse import coo_matrix, csr_matrix, csc_matrix
from scipy.sparse.linalg import spsolve

from .dataclasses import (
    Mesh, SoilMaterial, SimulationConfig, TimeControl,
)
from . import material as _mat


# ============================================================================
# Material property update at all nodes (mirrors SetMat in WATFLOW2.FOR)
# ============================================================================

# Log-spaced K/C/θ tables, built once per simulation, indexed by material.
# Mirrors Fortran GenMat (INPUT2.FOR L117-145) so SetMat's linear-interp
# rounding error is reproduced bit-equal.
_TABLE_CACHE: dict[int, tuple] = {}


def build_material_tables(materials: list[SoilMaterial],
                          hTab1: float, hTabN: float,
                          NTab: int = 100) -> tuple:
    """Build (hTab, ConTab, CapTab, TheTab) following Fortran GenMat.

    hTab1, hTabN: positive numbers (read straight from BLOCK B); Fortran
    flips sign on them internally (`hTab(i)=-10**alh`).
    """
    from hydrus1d.material import FK as _FK, FC as _FC, FQ as _FQ
    h1 = -abs(hTab1)
    hN = -abs(hTabN)
    alh1 = np.log10(-h1)
    dlh  = (np.log10(-hN) - alh1) / (NTab - 1)
    hTab = -10.0 ** (alh1 + np.arange(NTab) * dlh)
    nm = len(materials)
    ConTab = np.zeros((NTab, nm), np.float64)
    CapTab = np.zeros((NTab, nm), np.float64)
    TheTab = np.zeros((NTab, nm), np.float64)
    for M, mat in enumerate(materials):
        iModel = _mat.select_imodel(mat)
        Par = _mat.to_h1d_par(mat)
        for i in range(NTab):
            ConTab[i, M] = _FK(iModel, float(hTab[i]), Par)
            CapTab[i, M] = _FC(iModel, float(hTab[i]), Par)
            TheTab[i, M] = _FQ(iModel, float(hTab[i]), Par)
    return hTab, ConTab, CapTab, TheTab, alh1, dlh


def _table_lookup(h: float, tab_y: NDArray[np.float64],
                  hTab: NDArray[np.float64],
                  alh1: float, dlh: float) -> float:
    """Linear interpolation matching Fortran SetMat L455-457."""
    # iT = floor((log10(-h) - alh1)/dlh)  [Fortran 1-based; here 0-based]
    iT = int((np.log10(-h) - alh1) / dlh)
    if iT < 0:
        iT = 0
    elif iT >= hTab.shape[0] - 1:
        iT = hTab.shape[0] - 2
    S1 = (tab_y[iT + 1] - tab_y[iT]) / (hTab[iT + 1] - hTab[iT])
    return tab_y[iT] + S1 * (h - hTab[iT])


def set_mat(mesh: Mesh, materials: list[SoilMaterial],
            thR: NDArray[np.float64], thSat: NDArray[np.float64],
            hSat: NDArray[np.float64], ConSat: NDArray[np.float64],
            Explic: bool = False,
            tables: Optional[tuple] = None,
            ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Evaluate Con, Cap, Theta at every node (WATFLOW2.FOR L433-482).

    If tables is provided ((hTab, ConTab, CapTab, TheTab, alh1, dlh)) we use
    log-spaced linear interpolation matching Fortran exactly; otherwise we
    fall back to direct FK/FC/FQ calls.
    """
    n = mesh.NumNP
    Con = np.zeros(n, np.float64)
    Cap = np.zeros(n, np.float64)
    Theta = np.zeros(n, np.float64)

    nodes = mesh.nodes
    pars = [(_mat.select_imodel(m), _mat.to_h1d_par(m)) for m in materials]
    from hydrus1d.material import FK as _FK, FC as _FC, FQ as _FQ

    if tables is not None:
        hTab, ConTab, CapTab, TheTab, alh1, dlh = tables
        hTab_first = float(hTab[0])
        hTab_last  = float(hTab[-1])
    else:
        hTab_first = hTab_last = 0.0

    for i in range(n):
        M = nodes.MatNum[i] - 1
        iModel, Par = pars[M]
        hSat_m = hSat[M]
        # K(h) at blended iterate
        hi1 = min(hSat_m, nodes.hTemp[i] / nodes.Axz[i])
        hi2 = min(hSat_m, nodes.hNew[i]  / nodes.Axz[i])
        if Explic:
            hi2 = hi1
        hiM = 0.1 * hi1 + 0.9 * hi2
        if hi1 >= hSat_m and hi2 >= hSat_m:
            Ci = ConSat[M]
        elif (tables is not None
              and hiM >= hTab_last and hiM <= hTab_first):
            Ci = _table_lookup(hiM, ConTab[:, M], hTab, alh1, dlh)
        else:
            Ci = _FK(iModel, hiM, Par)
        Con[i] = nodes.Bxz[i] * Ci
        # Cap, Theta at hNew (or hOld if explicit)
        hi2 = nodes.hNew[i] / nodes.Axz[i]
        if Explic:
            hi2 = nodes.hOld[i] / nodes.Axz[i]
        if hi2 >= hSat_m:
            Ci = 0.0
            Ti = thSat[M]
        elif (tables is not None
              and hi2 >= hTab_last and hi2 <= hTab_first):
            Ci = _table_lookup(hi2, CapTab[:, M], hTab, alh1, dlh)
            Ti = _table_lookup(hi2, TheTab[:, M], hTab, alh1, dlh)
        else:
            Ci = _FC(iModel, hi2, Par)
            Ti = _FQ(iModel, hi2, Par)
        Cap[i]   = Ci * nodes.Dxz[i] / nodes.Axz[i]
        Theta[i] = thR[M] + (Ti - thR[M]) * nodes.Dxz[i]
    return Con, Cap, Theta


# ============================================================================
# Galerkin element assembly (mirrors Reset in WATFLOW2.FOR)
# ============================================================================

def reset(mesh: Mesh, cfg: SimulationConfig,
          Con: NDArray[np.float64], Cap: NDArray[np.float64],
          ThNew: NDArray[np.float64], ThOld: NDArray[np.float64],
          DS: NDArray[np.float64], dt: float,
          Iter: int,
          Sink: Optional[NDArray[np.float64]] = None,
          P3: float = 0.0,
          ) -> tuple[csr_matrix, NDArray[np.float64], NDArray[np.float64],
                     NDArray[np.float64]]:
    """
    Assemble global stiffness + mass matrix and RHS for Richards eq.

    Returns
    -------
    A_csr   : (NumNP, NumNP) sparse — effective matrix M/dt + K with
              mass-matrix lumping added on diagonal
    B       : (NumNP,) RHS vector
    F       : (NumNP,) lumped mass (∫φ_i dA) per node — kept for output
    Q_intern: (NumNP,) "internal flux" recovered at Dirichlet nodes —
              equals B(n) of the original Reset before mass-matrix is added,
              used by the boundary-flux output and the Shift seepage logic.

    KAT meaning:
        0 = horizontal plane (no gravity)
        1 = axisymmetric vertical (multiply by 2πr, gravity in z)
        2 = vertical plane (no 2π, gravity in z)
    Sink and root extraction are NOT included here for Stage 1 (SinkF=False).
    """
    NumNP = mesh.NumNP
    NumEl = mesh.NumEl
    KAT = cfg.KAT
    KX = mesh.elements.KX
    x  = mesh.nodes.x
    y  = mesh.nodes.y
    ConAxx = mesh.elements.ConAxx
    ConAzz = mesh.elements.ConAzz
    ConAxz = mesh.elements.ConAxz

    # Triplet builder
    rows_list: list[int] = []
    cols_list: list[int] = []
    vals_list: list[float] = []
    B = np.zeros(NumNP, np.float64)
    F = np.zeros(NumNP, np.float64)
    Bi = np.zeros(3, np.float64)
    Ci = np.zeros(3, np.float64)

    Beta = mesh.nodes.Beta
    hNew = mesh.nodes.hNew
    use_sink = cfg.SinkF and Sink is not None and Iter == 0
    for n in range(NumEl):
        CondI = ConAxx[n]
        CondJ = ConAzz[n]
        CondK = ConAxz[n]
        NUS = 3 if KX[n, 2] == KX[n, 3] else 4
        for k in range(NUS - 2):           # 1 iter for triangle, 2 for quad
            i = KX[n, 0]
            j = KX[n, k + 1]
            l = KX[n, k + 2]
            iLoc = (0, k + 1, k + 2)
            Ci[0] = x[l] - x[j]
            Ci[1] = x[i] - x[l]
            Ci[2] = x[j] - x[i]
            Bi[0] = y[j] - y[l]
            Bi[1] = y[l] - y[i]
            Bi[2] = y[i] - y[j]
            AE = (Ci[2] * Bi[1] - Ci[1] * Bi[2]) / 2.0
            ConE = (Con[i] + Con[j] + Con[l]) / 3.0
            xMul = 1.0
            if KAT == 1:
                xMul = 2.0 * np.pi * (x[i] + x[j] + x[l]) / 3.0
            AMul = xMul * ConE / 4.0 / AE
            BMul = xMul * ConE / 2.0
            FMul = xMul * AE / 12.0
            # Sink contribution to DS (only on Iter==0, only if Beta>0)
            if use_sink:
                BetaE = (Beta[i] + Beta[j] + Beta[l]) / 3.0
                if BetaE > 0.0:
                    SinkE = (Sink[i] + Sink[j] + Sink[l]) / 3.0
                    if hNew[i] > P3:
                        DS[i] += FMul * (3.0 * SinkE + Sink[i])
                    if hNew[j] > P3:
                        DS[j] += FMul * (3.0 * SinkE + Sink[j])
                    if hNew[l] > P3:
                        DS[l] += FMul * (3.0 * SinkE + Sink[l])
            # Local node loop
            for ii in range(3):
                iG = KX[n, iLoc[ii]]
                F[iG] += FMul * 4.0                       # lumped mass
                if KAT >= 1:
                    B[iG] += BMul * (CondK * Bi[ii] + CondJ * Ci[ii])  # gravity
                for jj in range(3):
                    jG = KX[n, iLoc[jj]]
                    E_ij = (CondI * Bi[ii] * Bi[jj]
                            + CondK * (Bi[ii] * Ci[jj] + Ci[ii] * Bi[jj])
                            + CondJ * Ci[ii] * Ci[jj])
                    rows_list.append(int(iG))
                    cols_list.append(int(jG))
                    vals_list.append(AMul * E_ij)

    # Build stiffness matrix K (without the M/dt yet)
    K_coo = coo_matrix(
        (np.asarray(vals_list, np.float64),
         (np.asarray(rows_list, np.int32), np.asarray(cols_list, np.int32))),
        shape=(NumNP, NumNP),
    )
    K_csr: csr_matrix = K_coo.tocsr()           # type: ignore[assignment]
    K_csr.sum_duplicates()

    # Boundary flux recovery at Dirichlet nodes (Kode >= 1) and propagation
    # of prescribed Q at flux-BC nodes (Kode < 1) into the linear system.
    #
    # Fortran's Reset (WATFLOW2.FOR L247-269) only overwrites Q(n) for
    # Kode>=1 — for Kode<1 (atmospheric, seepage, etc.) the Q in the global
    # array was already set by SetAtm/Shift and the L271-279 effective-RHS
    # loop uses that prescribed flux directly via "B(i)=... + Q(i) - ...".
    # In Python we mirror this by seeding Q_eff with mesh.nodes.Q (which
    # carries the SetAtm-set atm flux), then overwriting Kode>=1 entries
    # with the newly-computed Q_intern flux.
    Kode = mesh.nodes.Kode
    hNew = mesh.nodes.hNew
    Q_eff = mesh.nodes.Q.astype(np.float64, copy=True)
    Q_intern = np.zeros(NumNP, np.float64)
    K_h = K_csr @ hNew                          # one matvec
    for nn in range(NumNP):
        if Kode[nn] < 1:
            continue
        Q_intern[nn] = (B[nn] + DS[nn]
                        + F[nn] * (ThNew[nn] - ThOld[nn]) / dt
                        + K_h[nn])
        Q_eff[nn] = Q_intern[nn]

    # Now add M/dt to diagonal and form effective RHS
    diag_add = F * Cap / dt
    K_csr = K_csr + csr_matrix(
        (diag_add, (np.arange(NumNP), np.arange(NumNP))),
        shape=(NumNP, NumNP),
    )
    B_eff = (F * Cap * hNew / dt
             - F * (ThNew - ThOld) / dt
             + Q_eff
             - B
             - DS)

    return K_csr, B_eff, F, Q_intern


# ============================================================================
# Boundary condition Shift (seepage / atm / drain / free drain)
# ============================================================================

def shift(mesh: Mesh, cfg: SimulationConfig,
          NSeep: int, NSP: list[int], NP_seep: list[list[int]],
          rTop: float, hCritA: float, hCritS: float,
          GWL0L: float, Aqh: float, Bqh: float,
          Con: NDArray[np.float64], ConO: NDArray[np.float64],
          Iter: int, Explic: bool,
          ) -> None:
    """Modify mesh.nodes.Kode and mesh.nodes.Q based on current state.

    Mirrors Shift in WATFLOW2.FOR L319-428. Stage 1 supports SeepF + AtmInF
    + FreeD + qGWLF. DrainF (subsurface drains) is rare and not implemented.
    """
    nodes = mesh.nodes
    hNew = nodes.hNew
    Q    = nodes.Q
    Kode = nodes.Kode

    # ---- Seepage face: switch -2 (flux=0) ↔ +2 (h=0)
    if cfg.SeepF:
        for i in range(NSeep):
            for j in range(NSP[i]):
                n = NP_seep[i][j]
                if Kode[n] == -2:
                    if hNew[n] >= 0.0:
                        Kode[n] = 2
                        hNew[n] = 0.0
                else:
                    if Q[n] >= 0.0:
                        Kode[n] = -2
                        Q[n] = 0.0

    # ---- DrainF: not implemented (no Stage 1 example uses it)
    if cfg.DrainF:
        raise NotImplementedError("DrainF (subsurface drains) not implemented.")

    # ---- AtmInF: surface flux ↔ critical pressure switching, GWL flux
    if cfg.AtmInF:
        KXB = mesh.KXB
        Width = mesh.Width
        for i in range(mesh.NumBP):
            n = int(KXB[i])
            k = int(Kode[n])
            if Explic and abs(k) == 4:
                Kode[n] = -abs(k)
                continue
            # Critical surface pressure on the head-controlled side
            if k == 4:
                if (abs(Q[n]) > abs(-rTop * Width[i])
                        or Q[n] * (-rTop) <= 0.0):
                    Kode[n] = -4
                    Q[n] = -rTop * Width[i]
                continue
            # Surface flux on, flip to head if exceeding the critical envelope
            if k == -4 and Iter != 0:
                if hNew[n] <= hCritA:
                    Kode[n] = 4
                    hNew[n] = hCritA
                    continue
                if hNew[n] >= hCritS:
                    Kode[n] = 4
                    hNew[n] = hCritS
            # Time-variable flux at GWL nodes (k == -3)
            if k == -3 and cfg.qGWLF:
                # Fqh(GWL,Aqh,Bqh) = -Aqh * exp(Bqh * |GWL|)
                GWL = nodes.hOld[n] - GWL0L
                Q[n] = -Width[i] * (-Aqh * np.exp(Bqh * abs(GWL)))

    # ---- Free drainage: bottom flux = -K(h)
    if cfg.FreeD:
        KXB = mesh.KXB
        Width = mesh.Width
        for i in range(mesh.NumBP):
            n = int(KXB[i])
            if Kode[n] == -3:
                Q[n] = -Width[i] * ConO[n]


# ============================================================================
# Dirichlet application
# ============================================================================

def dirich(A: csr_matrix, B: NDArray[np.float64],
           Kode: NDArray[np.int32], hNew: NDArray[np.float64]) -> csr_matrix:
    """
    For every node with Kode >= 1: set row/col interactions so the
    solution at that node equals hNew[n].

    Implementation: zero out the row, place 1 on diagonal, set B[n]=hNew[n].
    The Fortran banded version also zeroes out the column (and propagates
    to B) — for a direct sparse solve we don't need that, but doing it
    keeps the matrix symmetric and matches Fortran's eliminated form.
    """
    A_lil = A.tolil()
    n_dir = np.where(Kode >= 1)[0]
    for n in n_dir:
        # Save column entries to subtract from RHS, then zero column
        col_data = A_lil.getcol(n).toarray().flatten()
        # Update B for non-dirichlet rows (those rows lose A[r,n]*h_n)
        for r in range(A_lil.shape[0]):
            if r == n: continue
            if Kode[r] >= 1: continue
            if col_data[r] != 0.0:
                B[r] -= col_data[r] * hNew[n]
        # Zero out column n
        A_lil[:, n] = 0
        # Zero out row n, set diagonal to 1
        A_lil.rows[n] = [int(n)]
        A_lil.data[n] = [1.0]
        B[n] = hNew[n]
    return A_lil.tocsr()


# ============================================================================
# Main Picard iteration driver (mirrors WatFlow in WATFLOW2.FOR L3-151)
# ============================================================================

def solve_water_flow(mesh: Mesh, cfg: SimulationConfig, time: TimeControl,
                     materials: list[SoilMaterial],
                     thR: NDArray[np.float64], thSat: NDArray[np.float64],
                     hSat: NDArray[np.float64], ConSat: NDArray[np.float64],
                     ThNew: NDArray[np.float64], ThOld: NDArray[np.float64],
                     ConO: NDArray[np.float64],
                     NSeep: int, NSP: list[int], NP_seep: list[list[int]],
                     dt: float, dtMin: float, dtOld: float, tOld: float,
                     rTop: float = 0.0, hCritA: float = -1e6,
                     hCritS: float = 0.0,
                     GWL0L: float = 0.0, Aqh: float = 0.0, Bqh: float = 0.0,
                     Sink: Optional[NDArray[np.float64]] = None,
                     P3: float = 0.0,
                     tables: Optional[tuple] = None,
                     ) -> tuple[float, float, int, bool,
                                NDArray[np.float64], NDArray[np.float64],
                                NDArray[np.float64], NDArray[np.float64]]:
    """
    Solve one time step of Richards equation using Picard iteration.

    Returns
    -------
    dt_used     : the dt actually used (may be reduced if convergence failed)
    t_new       : tOld + dt_used
    n_iter      : iterations consumed
    converged   : True if converged within MaxIt
    Con, Cap, ThNew, Q_intern : updated nodal arrays

    Side-effects: mesh.nodes.hNew, .hOld, .hTemp, .Kode, .Q are mutated.
    """
    nodes = mesh.nodes
    NumNP = mesh.NumNP

    # Predict hNew from previous step (skip on TLevel==1, handled by caller)
    # The TLevel==1 setup is done in the main driver; here we always treat
    # dt-step as a re-entry case.

    DS = np.zeros(NumNP, np.float64)    # sink (Stage 1: zero, SinkF=False)
    Explic = False
    Con = Cap = None
    Q_intern = np.zeros(NumNP, np.float64)
    dt_current = dt

    while True:
        # Save state for restart on convergence failure
        hOld_iter = nodes.hOld.copy()
        Iter = 0
        Explic = False

        while True:
            # SetMat
            Con, Cap, ThNew_new = set_mat(mesh, materials, thR, thSat, hSat,
                                          ConSat, Explic=Explic, tables=tables)
            ThNew[:] = ThNew_new
            # Reset
            A_csr, B_eff, F, Q_intern_new = reset(mesh, cfg, Con, Cap,
                                                  ThNew, ThOld, DS,
                                                  dt_current, Iter,
                                                  Sink=Sink, P3=P3)
            Q_intern[:] = Q_intern_new
            # Only Kode >= 1 (Dirichlet) gets its flux re-derived; flux-BC
            # nodes (Kode < 1) keep the Q value set by SetAtm / SetSnk.
            dir_mask = nodes.Kode >= 1
            nodes.Q[dir_mask] = Q_intern_new[dir_mask]
            # Shift (seepage / atmospheric / free-drain — may flip Kode/hNew/Q)
            shift(mesh, cfg, NSeep, NSP, NP_seep, rTop, hCritA, hCritS,
                  GWL0L, Aqh, Bqh, Con, ConO, Iter, Explic)
            # Dirich
            A_dir = dirich(A_csr, B_eff, nodes.Kode, nodes.hNew)
            # Solve
            try:
                sol = spsolve(csc_matrix(A_dir), B_eff)
            except RuntimeError:
                # singular: stuck, force dt reduction
                break

            # Update hTemp ← old hNew (previous iterate), hNew ← solution
            nodes.hTemp[:] = nodes.hNew
            nodes.hNew[:]  = sol
            np.clip(nodes.hNew, -1e10, 1e10, out=nodes.hNew)
            Iter += 1
            time.ItCum += 1
            if Explic:
                break

            # Convergence test (per node, drop out as soon as one fails)
            ItCrit = True
            for i in range(NumNP):
                M = nodes.MatNum[i] - 1
                EpsTh = 0.0
                EpsH = 0.0
                if nodes.hTemp[i] < hSat[M] and nodes.hNew[i] < hSat[M]:
                    Th_predict = (ThNew[i]
                                  + Cap[i] * (nodes.hNew[i] - nodes.hTemp[i])
                                  / (thSat[M] - thR[M]) / nodes.Dxz[i])
                    EpsTh = abs(ThNew[i] - Th_predict)
                else:
                    EpsH = abs(nodes.hNew[i] - nodes.hTemp[i])
                if EpsTh > cfg.TolTh or EpsH > cfg.TolH:
                    ItCrit = False
                    break

            if ItCrit:
                # Converged
                return (dt_current, tOld + dt_current, Iter, True,
                        Con, Cap, ThNew, Q_intern)

            if Iter >= cfg.MaxIt:
                # Not converged within MaxIt — try dt reduction or explicit
                break

        # Inner loop exited without convergence
        if dt_current > dtMin:
            # Cut dt by 3 and restart from hOld
            nodes.hNew[:]  = hOld_iter
            nodes.hTemp[:] = hOld_iter
            dt_current = max(dt_current / 3.0, dtMin)
            time.dt = dt_current
            continue
        else:
            # Force explicit fallback
            nodes.hNew[:]  = hOld_iter
            nodes.hTemp[:] = hOld_iter
            Explic = True
            # Run once more in explicit mode then return
            Con, Cap, ThNew_new = set_mat(mesh, materials, thR, thSat, hSat,
                                          ConSat, Explic=True, tables=tables)
            ThNew[:] = ThNew_new
            return (dt_current, tOld + dt_current, Iter, False,
                    Con, Cap, ThNew, Q_intern)
