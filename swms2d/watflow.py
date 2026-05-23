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
from scipy.sparse.linalg import spsolve, splu

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
    # Fortran's FK/FC/FQ return REAL*4 (float32) and Fortran's ConTab/
    # CapTab/TheTab are REAL*4 arrays, so each table entry is truncated
    # to ~7 sig figs at table-build time. Python's float64 storage
    # gives ~15 sig figs and the extra precision compounds through
    # Picard iterations as a 0.01-1.1 hPa drift. Truncate-then-promote
    # to mirror Fortran's storage precision exactly.
    ConTab = ConTab.astype(np.float32).astype(np.float64)
    CapTab = CapTab.astype(np.float32).astype(np.float64)
    TheTab = TheTab.astype(np.float32).astype(np.float64)
    # hTab also stored as REAL*4 in Fortran via `hTab(i)=-10**alh` where
    # alh is double precision but assignment to real array truncates.
    hTab = hTab.astype(np.float32).astype(np.float64)
    return hTab, ConTab, CapTab, TheTab, alh1, dlh


def _table_lookup(h: float, tab_y: NDArray[np.float64],
                  hTab: NDArray[np.float64],
                  alh1: float, dlh: float) -> float:
    """Linear interpolation matching Fortran SetMat L455-457.

    To bit-match Fortran's behaviour, the log10 and division here are
    performed in float32 (Fortran's REAL*4 default for alh1/dlh),
    so the integer index `iT` rounds identically to Fortran near
    integer boundaries.
    """
    # Match Fortran's alh1, dlh, log10 precision (all REAL*4):
    log_neg_h = float(np.float32(np.log10(-h)))
    alh1_32 = float(np.float32(alh1))
    dlh_32  = float(np.float32(dlh))
    iT = int(np.float32((log_neg_h - alh1_32) / dlh_32))
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
    # Fortran stores Con/Cap/Theta as REAL*4 (single-precision arrays). Truncate
    # to float32 precision so downstream matrix assembly sees the same values.
    Con   = Con.astype(np.float32).astype(np.float64)
    Cap   = Cap.astype(np.float32).astype(np.float64)
    Theta = Theta.astype(np.float32).astype(np.float64)
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
          newton_diag_corr: Optional[NDArray[np.float64]] = None,
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
                xMul = 2.0 * 3.1416 * (x[i] + x[j] + x[l]) / 3.0
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

    # Now add M/dt to diagonal and form effective RHS.
    # If newton_diag_corr is provided, also add the Newton diagonal Jacobian
    # term ∂(F*C*h/dt)/∂h_i = F[i] * dC/dh|h_i * (h_i - h_n_i)/dt. This makes
    # the linear system a modified-Newton update (the off-diagonal ∂L/∂h_j
    # K-derivative terms are dropped — diagonal-only quasi-Newton).
    diag_add = F * Cap / dt
    if newton_diag_corr is not None:
        diag_add = diag_add + newton_diag_corr
    K_csr = K_csr + csr_matrix(
        (diag_add, (np.arange(NumNP), np.arange(NumNP))),
        shape=(NumNP, NumNP),
    )
    # B_eff RHS: standard Picard mixed form. The Newton correction enters
    # only through diag_add (left-hand side); the RHS is unchanged because
    # the linearization is around h^k, so the correction terms F[i] * dC/dh
    # * (h_i - h_n_i)/dt cancel against the same expression evaluated at
    # h^k on both sides.
    B_eff = (F * Cap * hNew / dt
             - F * (ThNew - ThOld) / dt
             + Q_eff
             - B
             - DS)
    if newton_diag_corr is not None:
        # Add the Newton correction's contribution to the RHS so the Jacobian
        # corresponds to the actual residual being solved:
        #   J · δh = -R(h^k)   where J = A_Picard + diag(Newton-corr)
        # Equivalently:
        #   J · h^{k+1} = J · h^k - R(h^k)
        # The Picard form already gives A_Picard·h^{k+1} = A_Picard·h^k - R;
        # we add diag(Newton-corr) · h^k on the RHS to balance the LHS extra
        # term diag(Newton-corr) · h^{k+1}.
        B_eff = B_eff + newton_diag_corr * hNew

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
          NDr: int = 0,
          ND_drain: Optional[NDArray[np.int32]] = None,
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

    # ---- DrainF: subsurface drain switching (Kode=-5 ↔ +5)
    if cfg.DrainF and NDr > 0 and ND_drain is not None:
        from .drain import shift_drain
        shift_drain(mesh, NDr, ND_drain)

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
# Fortran's banded Gauss elimination solver — 1:1 port from WATFLOW2.FOR
# L486-519 (subroutine Solve). For symmetric positive-definite banded
# matrices stored as the upper triangle in (MBand, NumNP) format:
#   A_band[m-1, n] = A[n, n+m-1]   for m = 1..MBand, n = 0..NumNP-1
# Eliminates without pivoting (relies on diagonal dominance).
#
# Why this exists: SuperLU (used by scipy.spsolve) and Fortran banded
# Gauss take different paths through the floating-point space. At
# ill-conditioned regimes (e.g. EX.2 dry-spell day 210-212) the two
# paths produce solutions that differ by O(εκ) and accumulate over
# Picard iters, eventually exceeding the TolH=0.05 tolerance ball.
# Using the same algorithm class as Fortran eliminates this drift.
# ============================================================================

def _solve_banded_fortran(A: csr_matrix,
                          b: NDArray[np.float64],
                          ) -> NDArray[np.float64]:
    """Solve A · x = b via Fortran-style upper-triangular banded Gauss.

    Assumes A is symmetric. Detects the bandwidth from the sparse pattern,
    extracts the upper triangle to banded form, then runs the in-place
    elimination identical to WATFLOW2.FOR's Solve subroutine.

    Falls back to scipy.spsolve if A turns out to be too wide-bandwidth
    for banded storage to be efficient (MBand > 50).
    """
    A_coo = A.tocoo()
    NumNP = A.shape[0]
    # Compute half-bandwidth
    rows = A_coo.row
    cols = A_coo.col
    MBand_minus1 = int(np.max(np.abs(rows - cols))) if A_coo.nnz else 0
    MBand = MBand_minus1 + 1
    if MBand > 50:
        # Bandwidth too large — banded form would be 50× redundant
        return spsolve(csc_matrix(A), b)
    # Build banded storage (upper triangle): A_band[m, n] = A[n, n+m]
    A_band = np.zeros((MBand, NumNP), dtype=np.float64)
    for r, c, v in zip(rows, cols, A_coo.data):
        if c >= r and (c - r) < MBand:
            # store at A_band[c - r, r] = A[r, c]
            A_band[c - r, r] += v
    # Make a working copy of b (Fortran's algorithm mutates)
    B = b.astype(np.float64, copy=True)
    # Forward elimination (Fortran 1-based loops translated to 0-based)
    # for n in 0..NumNP-1: divide row n's diagonal entry into the
    # rows below it within the band, then store the multiplier in
    # A_band[m, n] so it can be reused in the elimination of B.
    for n in range(NumNP - 1):
        diag = A_band[0, n]
        if abs(diag) < 1e-30:
            continue
        for m in range(1, MBand):
            a_mn = A_band[m, n]
            if abs(a_mn) < 1e-30:
                continue
            i = n + m
            if i >= NumNP:
                break
            C = a_mn / diag
            # row i: subtract C * row n's tail starting at column n+m
            # row i has entries A_band[j, i] = A[i, i+j], j=0..MBand-1
            # row n's tail entries are A_band[k, n] = A[n, n+k], k=m..MBand-1
            # offset: A[i, i+j] -= C * A[n, n+k] where i+j = n+k → j = k - m
            for k in range(m, MBand):
                j = k - m
                A_band[j, i] -= C * A_band[k, n]
            A_band[m, n] = C
            B[i] -= C * B[n]
        # B[n] /= diag — but we still need diag for back-sub, so divide later
    # Forward-substitute the diagonal division pass
    for n in range(NumNP):
        diag = A_band[0, n]
        if abs(diag) > 1e-30:
            B[n] /= diag
    # Back substitution: row n -= sum_{k=1..MBand-1} A_band[k, n] * x[n+k] / diag
    # but A_band[0, n] was already divided, so just subtract A_band[k, n]*B[n+k]
    for n in range(NumNP - 2, -1, -1):
        diag = A_band[0, n]
        if abs(diag) < 1e-30:
            continue
        s = 0.0
        for k in range(1, MBand):
            i = n + k
            if i >= NumNP:
                break
            # Note: A_band[k, n] was overwritten with the multiplier C above,
            # which is exactly A[n, n+k]/diag(at the time of elimination).
            # For back-sub Fortran uses A(k,n) (the multiplier) * B[i].
            s += A_band[k, n] * B[i]
        B[n] -= s
    return B


# ============================================================================
# Iterative refinement of the sparse LU solve.
#
# scipy.sparse.linalg.spsolve returns x with the precision of one LU back-
# substitution — for ill-conditioned A (large condition number) the
# accuracy can be much worse than working precision. Two LU implementations
# (e.g. SuperLU vs banded Gauss) take different pivoting/fill-in paths and
# produce x values that differ in the lower bits.
#
# Iterative refinement (Wilkinson 1963; LAPACK *RFS family) brings the
# solution to ~working precision regardless of the underlying LU path:
#     x_0 = LU \ b
#     r_k = b - A x_k      (residual)
#     dx_k = LU \ r_k       (cheap — reuses LU factors)
#     x_{k+1} = x_k + dx_k
# Iterate until ||r_k|| stops decreasing. Three iters is usually enough
# to recover near-machine precision.
# ============================================================================

def _solve_refined(A: csc_matrix, b: NDArray[np.float64],
                   max_iters: int = 3,
                   rel_tol: float = 1e-12,
                   ) -> NDArray[np.float64]:
    """LU solve with iterative refinement. Drop-in for spsolve.

    Returns x ≈ A⁻¹b accurate to ~machine precision in the residual
    sense, regardless of the LU implementation's internal path.
    """
    lu = splu(A)
    x = lu.solve(b)
    bn = float(np.max(np.abs(b)))
    if bn == 0.0:
        return x
    for _ in range(max_iters):
        r = b - A @ x
        rn = float(np.max(np.abs(r)))
        if rn < rel_tol * bn:
            break
        dx = lu.solve(r)
        x = x + dx
    return x


# ============================================================================
# Anderson acceleration for the Picard fixed-point iteration.
#
# References:
#   - Walker & Ni (2011) "Anderson acceleration for fixed-point iterations"
#     SIAM J. Numer. Anal. 49(4), 1715-1735.
#   - Lott et al. (2012) "An accelerated Picard method for nonlinear systems
#     related to variably saturated flow." Adv. Water Resour. 38, 92-101.
#
# Implementation: Type-II Anderson with QR-updated normal equations.
# Treats the Picard iterate as a fixed-point map h_{k+1} = G(h_k); when m
# previous (h, G(h)) pairs are available, we extrapolate to a better next
# iterate by solving a small least-squares problem over residuals.
# ============================================================================

class _AndersonState:
    """Rolling buffer of (h_k, G(h_k)) pairs for Anderson acceleration."""

    __slots__ = ("m_max", "h_hist", "g_hist")

    def __init__(self, m_max: int = 3):
        self.m_max = m_max
        self.h_hist: list[NDArray[np.float64]] = []
        self.g_hist: list[NDArray[np.float64]] = []

    def reset(self) -> None:
        self.h_hist.clear()
        self.g_hist.clear()

    def step(self, h_k: NDArray[np.float64],
             g_k: NDArray[np.float64]) -> NDArray[np.float64]:
        """Push (h_k, g_k) and return the Anderson-accelerated next iterate.

        On the first call returns g_k (plain Picard step). On subsequent
        calls solves a least-squares problem over the last m residuals
        F_i = G(h_i) - h_i to extrapolate. If the LSQ system is rank-
        deficient (e.g. all residuals collinear at hyperextreme dry h),
        we fall back to plain Picard for that step.
        """
        self.h_hist.append(h_k.copy())
        self.g_hist.append(g_k.copy())
        # Drop oldest pair if we exceed buffer
        if len(self.h_hist) > self.m_max + 1:
            self.h_hist.pop(0)
            self.g_hist.pop(0)
        m = len(self.h_hist) - 1
        if m == 0:
            return g_k.copy()
        # Build dF (m × N) and dG (m × N) as row-differences of stacked arrays
        F = np.stack(self.g_hist) - np.stack(self.h_hist)
        G = np.stack(self.g_hist)
        dF = np.diff(F, axis=0)
        dG = np.diff(G, axis=0)
        try:
            alpha, *_ = np.linalg.lstsq(dF.T, F[-1], rcond=1e-10)
        except np.linalg.LinAlgError:
            return g_k.copy()
        if not np.all(np.isfinite(alpha)):
            return g_k.copy()
        return G[-1] - dG.T @ alpha


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
                     use_anderson: bool = False,
                     anderson_m: int = 3,
                     refine_solve: bool = False,
                     use_newton: bool = False,
                     use_lscheme: bool = False,
                     lscheme_L: float = 0.0,
                     use_banded: bool = False,
                     debug_file: Optional[object] = None,
                     debug_TLevel: int = 0,
                     NDr: int = 0,
                     ND_drain: Optional[NDArray[np.int32]] = None,
                     hyst_state: Optional[object] = None,
                     hyst_materials: Optional[list] = None,
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
    anderson = _AndersonState(m_max=anderson_m) if use_anderson else None

    while True:
        # Save state for restart on convergence failure
        hOld_iter = nodes.hOld.copy()
        Iter = 0
        Explic = False
        if anderson is not None:
            anderson.reset()

        while True:
            # SetMat
            Con, Cap, ThNew_new = set_mat(mesh, materials, thR, thSat, hSat,
                                          ConSat, Explic=Explic, tables=tables)
            ThNew[:] = ThNew_new

            # Hysteresis override: when active, replace the single-curve
            # theta(h) and C(h) values from set_mat with branch-aware
            # hysteresis evaluations. K(h) is left as set_mat's value
            # (Scott 1983 assumes K depends only on theta, not branch).
            if hyst_state is not None and hyst_materials is not None:
                from .hysteresis import step_state as _hyst_step
                th_h, cap_h = _hyst_step(
                    hyst_state, mesh.nodes.hNew, mesh.nodes.hOld,
                    mesh.nodes.MatNum, hyst_materials,
                )
                ThNew[:] = th_h
                # Cap was Ci*Dxz/Axz in set_mat; hysteresis cap is bare
                # dtheta/dh per material curve. Apply same scaling.
                Axz = mesh.nodes.Axz; Dxz = mesh.nodes.Dxz
                Cap[:] = cap_h * Dxz / Axz

            # Newton diagonal correction: F[i] * dC/dh|h_i * (h_i - h_n_i)/dt.
            # Captures the dominant Jacobian term ∂R/∂h_i without the more
            # expensive off-diagonal ∂L/∂h_j K-derivative coupling. Skipped
            # under Explic mode (which is itself a recovery fallback).
            newton_corr: Optional[NDArray[np.float64]] = None
            if use_newton and not Explic:
                from .material import dC_dh_numeric as _dC_dh
                from . import material as _mat
                pars_n = [(_mat.select_imodel(m), _mat.to_h1d_par(m))
                          for m in materials]
                newton_corr = np.zeros(NumNP, np.float64)
                MatNum = nodes.MatNum
                Axz = nodes.Axz
                for i in range(NumNP):
                    M = MatNum[i] - 1
                    iModel, Par = pars_n[M]
                    h_i = nodes.hNew[i] / Axz[i]
                    if h_i >= hSat[M]:
                        continue
                    # ΔΘ-residual contribution: F * d²θ/dh² * dh/dt
                    dCdh = _dC_dh(iModel, h_i, Par, hSat=hSat[M])
                    dhdt = (nodes.hNew[i] - nodes.hOld[i]) / dt_current
                    newton_corr[i] = F_dummy = 0.0  # placeholder; F not yet built
                # We need F (lumped mass) to scale the correction. Build a
                # cheap pass over the mesh to compute it. The element loop in
                # reset will fill F again — to avoid double-assembly we
                # capture F here from a previous Reset's result via cache.
                if hasattr(solve_water_flow, '_F_cache') and \
                        solve_water_flow._F_cache[0] is mesh and \
                        solve_water_flow._F_cache[1].shape == (NumNP,):
                    F_cached = solve_water_flow._F_cache[1]
                else:
                    F_cached = None
                if F_cached is None:
                    # First iteration: do a probe reset without Newton corr
                    # to get F, then redo with corr below.
                    newton_corr = None
                else:
                    # Compute correction using cached F
                    for i in range(NumNP):
                        M = MatNum[i] - 1
                        iModel, Par = pars_n[M]
                        h_i = nodes.hNew[i] / Axz[i]
                        if h_i >= hSat[M]:
                            newton_corr[i] = 0.0
                            continue
                        dCdh = _dC_dh(iModel, h_i, Par, hSat=hSat[M])
                        dhdt = (nodes.hNew[i] - nodes.hOld[i]) / dt_current
                        # F[i] is lumped mass at node i; correction = F * dCdh * dhdt
                        newton_corr[i] = (F_cached[i] * dCdh * dhdt
                                          * nodes.Dxz[i] / Axz[i])

            # L-scheme stabiliser: add L_param * F[i] to the diagonal of A
            # and L_param * F[i] * h^k to the RHS. Provably contractive at
            # any L_param ≥ Lipschitz constant of θ(h). See List & Radu (2016).
            lscheme_corr: Optional[NDArray[np.float64]] = None
            if use_lscheme and not Explic:
                if not hasattr(solve_water_flow, '_F_cache') or \
                        solve_water_flow._F_cache[0] is not mesh:
                    pass   # F not yet built — skip on iter 0
                else:
                    F_cached = solve_water_flow._F_cache[1]
                    L = lscheme_L if lscheme_L > 0.0 else 0.1
                    lscheme_corr = L * F_cached

            # Reset
            A_csr, B_eff, F, Q_intern_new = reset(
                mesh, cfg, Con, Cap, ThNew, ThOld, DS,
                dt_current, Iter,
                Sink=Sink, P3=P3,
                newton_diag_corr=(newton_corr if newton_corr is not None
                                  else lscheme_corr),
            )
            # Cache F for next iter's Newton/L-scheme correction
            if use_newton or use_lscheme:
                solve_water_flow._F_cache = (mesh, F)
            Q_intern[:] = Q_intern_new
            # Only Kode >= 1 (Dirichlet) gets its flux re-derived; flux-BC
            # nodes (Kode < 1) keep the Q value set by SetAtm / SetSnk.
            dir_mask = nodes.Kode >= 1
            nodes.Q[dir_mask] = Q_intern_new[dir_mask]
            # Shift (seepage / atmospheric / free-drain / drain — may flip
            # Kode/hNew/Q at flux-BC nodes)
            shift(mesh, cfg, NSeep, NSP, NP_seep, rTop, hCritA, hCritS,
                  GWL0L, Aqh, Bqh, Con, ConO, Iter, Explic,
                  NDr=NDr, ND_drain=ND_drain)
            # Dirich
            A_dir = dirich(A_csr, B_eff, nodes.Kode, nodes.hNew)
            # Solve
            try:
                A_csc = csc_matrix(A_dir)
                if use_banded:
                    sol = _solve_banded_fortran(A_dir, B_eff)
                elif refine_solve:
                    sol = _solve_refined(A_csc, B_eff)
                else:
                    sol = spsolve(A_csc, B_eff)
            except RuntimeError:
                # singular: stuck, force dt reduction
                break

            # The Picard fixed-point map is G(h) = sol. Plain Picard sets
            # h_{k+1} = sol; Anderson extrapolates using last m residuals.
            h_prev = nodes.hNew.copy()                # x_k
            if anderson is not None and not Explic:
                h_next = anderson.step(h_prev, sol)   # Anderson update
                np.clip(h_next, -1e10, 1e10, out=h_next)
            else:
                h_next = sol
                np.clip(h_next, -1e10, 1e10, out=h_next)

            # Fortran does `hNew(i)=sngl(B(i))` (truncate to REAL*4) after
            # each linear solve. To bit-match its iterate trajectory, we
            # also truncate the iterate through float32 here.
            h_next = h_next.astype(np.float32).astype(np.float64)
            nodes.hTemp[:] = h_prev      # previous iterate (for convergence test)
            nodes.hNew[:]  = h_next
            Iter += 1
            time.ItCum += 1

            # DEBUG: per-iter state dump (matches Fortran's IT/CV lines)
            if debug_file is not None:
                t_dbg = tOld + dt_current
                # IT line: TLevel, Iter, t, dt, h1, h33, h65, Q1, Q65, K1, K65
                debug_file.write(
                    f"IT {debug_TLevel:5d} {Iter:3d} {t_dbg:10.4f} "
                    f"{dt_current:12.5E} "
                    f"{nodes.hNew[0]:15.8E} {nodes.hNew[32]:15.8E} "
                    f"{nodes.hNew[64]:15.8E} "
                    f"{nodes.Q[0]:15.8E} {nodes.Q[64]:15.8E} "
                    f"{int(nodes.Kode[0]):3d} {int(nodes.Kode[64]):3d}\n"
                )

            if Explic:
                break

            # Convergence test (per node, drop out as soon as one fails).
            # When Anderson is on, use the *Picard* residual (sol - h_prev)
            # for the test — the Anderson extrapolate is only the next
            # iterate, not the convergence signal.
            h_test_new = sol if anderson is not None else nodes.hNew
            ItCrit = True
            EpsThM = 0.0
            EpsHM = 0.0
            ifail = 0
            for i in range(NumNP):
                M = nodes.MatNum[i] - 1
                EpsTh = 0.0
                EpsH = 0.0
                if h_prev[i] < hSat[M] and h_test_new[i] < hSat[M]:
                    Th_predict = (ThNew[i]
                                  + Cap[i] * (h_test_new[i] - h_prev[i])
                                  / (thSat[M] - thR[M]) / nodes.Dxz[i])
                    EpsTh = abs(ThNew[i] - Th_predict)
                else:
                    EpsH = abs(h_test_new[i] - h_prev[i])
                if EpsTh > EpsThM: EpsThM = EpsTh
                if EpsH > EpsHM: EpsHM = EpsH
                if EpsTh > cfg.TolTh or EpsH > cfg.TolH:
                    if ItCrit:
                        ifail = i + 1  # 1-based to match Fortran
                    ItCrit = False
                    break

            if debug_file is not None:
                debug_file.write(
                    f"CV {Iter:3d} {EpsThM:12.5E} {EpsHM:12.5E} "
                    f"{ifail:4d} {'T' if ItCrit else 'F'}\n"
                )

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
