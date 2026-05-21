"""
Water flow solver for HYDRUS-1D.
================================

Core 1D variably saturated water flow solver.
Direct port of WATFLOW.FOR subroutines.

Implements:
- SetMat   : Compute soil hydraulic properties at each node
- Reset    : Assemble tridiagonal system for water flow
- Gauss    : Solve tridiagonal system (Gauss elimination)
- Shift    : Adjust boundary conditions
- Fqh      : Groundwater level fluctuation flux
- FqDrain  : Drainage flux (SWAP model)

Governing equation (Richards):
    C(h) * dh/dt = d/dz[K(h) * (dh/dz + 1)] + S

Discretized using mass-lumping finite elements with implicit time scheme.
"""

from __future__ import annotations
import numpy as np
from numpy.typing import NDArray
from typing import Tuple

from .material import FK, FC, FQ, FH, FS
from .utils import solve_tridiagonal


# ============================================================================
# Main water flow solver
# ============================================================================

def solve_water_flow(
    N: int,
    x: NDArray[np.float64],
    hNew: NDArray[np.float64],
    hOld: NDArray[np.float64],
    hTemp: NDArray[np.float64],
    MatNum: NDArray[np.int64],
    ParD: NDArray[np.float64],
    ParW: NDArray[np.float64],
    iModel: int,
    iHyst: int,
    iDualPor: int,
    dt: float,
    KodTop: int,
    KodBot: int,
    rTop: float,
    rBot: float,
    Sink: NDArray[np.float64],
    SinkIm: NDArray[np.float64],
    Ah: NDArray[np.float64],
    AK: NDArray[np.float64],
    ATh: NDArray[np.float64],
    Con: NDArray[np.float64],
    Cap: NDArray[np.float64],
    theta: NDArray[np.float64],
    CosAlf: float = 1.0,
    lCentrif: bool = False,
    Radius: float = 0.0,
    lGeom: bool = False,
    lDensity: bool = False,
    lVapor: bool = False,
    lWTDep: bool = False,
    Temp: NDArray[np.float64] | None = None,
    ConLT: NDArray[np.float64] | None = None,
    ConVT: NDArray[np.float64] | None = None,
    ConVh: NDArray[np.float64] | None = None,
    ThVOld: NDArray[np.float64] | None = None,
    ThVNew: NDArray[np.float64] | None = None,
    hSeep: float = 0.0,
    SeepF: bool = False,
    TopInf: bool = False,
    hCritA: float = 0.0,
    WLayer: bool = False,
    qGWLF: bool = False,
    GWL0L: float = 0.0,
    Aqh: float = 0.0,
    Bqh: float = 0.0,
    qDrain: bool = False,
    hTop_in: float | None = None,
    hBot_in: float | None = None,
    TolTh: float = 0.001,
    TolH: float = 1.0,
    MaxIt_in: int = 20,
) -> Tuple[NDArray[np.float64], float, float, int, int, int, bool]:
    """
    Solve water flow equation for one time step.
    
    Newton-Raphson iteration on Richards equation.
    
    Parameters
    ----------
    N : int
        Number of nodes
    x : array, shape (N,)
        Node coordinates (depth, increasing downward)
    hNew : array, shape (N,)
        Current pressure head estimate (updated in place)
    hOld : array, shape (N,)
        Previous time step pressure heads
    hTemp : array, shape (N,)
        Temporary heads (for hysteresis)
    MatNum : array, shape (N,)
        Material zone index for each node
    ParD : array, shape (11, NMat)
        Drying hydraulic parameters
    ParW : array, shape (11, NMat)
        Wetting hydraulic parameters
    iModel : int
        Hydraulic model code
    iHyst : int
        Hysteresis flag
    iDualPor : int
        Dual-porosity flag
    dt : float
        Time step
    KodTop : int
        Top boundary condition code
    KodBot : int
        Bottom boundary condition code
    rTop : float
        Top boundary flux value
    rBot : float
        Bottom boundary flux value
    Sink : array, shape (N,)
        Root water uptake (mobile domain)
    SinkIm : array, shape (N,)
        Root water uptake (immobile domain)
    Ah : array, shape (N,)
        Hysteretic head adjustment
    AK : array, shape (N,)
        Hysteretic conductivity adjustment
    ATh : array, shape (N,)
        Hysteretic capacity adjustment
    Con : array, shape (N,)
        Hydraulic conductivity at each node
    Cap : array, shape (N,)
        Specific water capacity at each node
    theta : array, shape (N,)
        Water content at each node
    CosAlf : float
        Cosine of centrifuge angle
    lCentrif : bool
        Centrifuge flag
    Radius : float
        Centrifuge radius
    lGeom : bool
        Geometric mean for conductivity averaging
    lDensity : bool
        Density-dependent flow flag
    lVapor : bool
        Vapor flow flag
    lWTDep : bool
        Temperature-dependent flow flag
    Temp : array, shape (N,), optional
        Temperature at each node
    ConLT : array, shape (N,), optional
        Liquid thermal conductivity
    ConVT : array, shape (N,), optional
        Vapor thermal conductivity
    ConVh : array, shape (N,), optional
        Vapor hydraulic conductivity
    ThVOld : array, shape (N,), optional
        Previous vapor water content
    ThVNew : array, shape (N,), optional
        Current vapor water content
    hSeep : float
        Seepage face head
    SeepF : bool
        Seepage face flag
    TopInf : bool
        Atmospheric BC flag
    hCritA : float
        Critical head for atmospheric BC
    WLayer : bool
        Surface water layer flag
    qGWLF : bool
        Groundwater level fluctuation flag
    GWL0L : float
        Base groundwater level
    Aqh : float
        GWLF amplitude parameter
    Bqh : float
        GWLF exponent parameter
    qDrain : bool
        Drainage flag
    
    Returns
    -------
    hNew : array, shape (N,)
        Updated pressure heads
    vTop : float
        Top flux
    vBot : float
        Bottom flux
    KodTop : int
        Updated top BC code
    KodBot : int
        Updated bottom BC code
    """
    # Newton-Raphson iteration loop (Fortran: label 12..18). Tolerances come
    # from Selector.in (TolTh defaults to 0.001, TolH to 1.0 in HYDRUS).
    MaxIt = MaxIt_in
    # rMax matches Fortran WATFLOW.FOR:35 (rMax=1e10). The original Python
    # used 1e4 which forced non-convergence as soon as hNew reached hCritA
    # (typically −1e5 for atmospheric BCs).
    rMax = 1.0e10
    Rate = 1.0  # relaxation factor (Fortran: Rate based on TauW)
    
    # Save old water content for RHS computation
    theta_old = theta.copy()
    
    # Save old BC codes in case we need to revert
    KTOld = KodTop
    KBOld = KodBot
    
    Iter = 0
    converged = False

    # Pre-step BC switch. Calling shift_bc *once* at the start of the time
    # step (with the previous step's converged hNew) avoids the Picard
    # chattering between flux/head BC types that the Fortran code somehow
    # survives through subtle ordering tricks.
    KodTop, KodBot, rTop, _ = shift_bc(
        N, KodTop, rTop, rBot, 0.0, 0.0, hCritA,
        CosAlf, WLayer, Con, hNew, x, TopInf, KodBot,
        theta, theta_old, Sink, dt, lVapor, lWTDep,
        ConLT, Temp, ConVh, ConVT, iDualPor, SinkIm,
        lDensity, np.zeros((1, N)), 0, lCentrif, Radius,
        hSeep, SeepF,
    )

    while Iter < MaxIt:
        Iter += 1
        
        # Save current heads as temporary (Fortran: hTemp = hNew before solve)
        hTemp[:] = hNew.copy()
        
        # Compute soil hydraulic properties
        _set_mat_properties(
            N, x, hNew, hOld, hTemp, MatNum, ParD, ParW,
            iModel, Con, Cap, theta, Ah, AK, ATh,
            CosAlf, lCentrif, Radius, lGeom, lWTDep, Temp,
            ConLT, ConVT, ConVh, lDensity,
        )
        
        # Assemble tridiagonal system
        P, R, S, PB, RB, SB, PT, RT, ST, vTop_out = _reset_assemble(
            N, x, hNew, hOld, theta, theta_old, MatNum, ParD, ParW,
            iModel, iHyst, iDualPor, dt, Con, Cap, Sink, SinkIm,
            Ah, AK, ATh, CosAlf, lCentrif, Radius, lGeom,
            lDensity, lVapor, lWTDep, Temp, ConLT, ConVT,
            ConVh, ThVOld, ThVNew, WLayer,
            rTop=rTop, rBot=rBot, FreeD=(KodBot == -5), qGWLF=qGWLF,
            qDrain=qDrain, GWL0L=GWL0L, Aqh=Aqh, Bqh=Bqh,
        )

        # NOTE: shift_bc is now called *once* before the Picard loop, not
        # per iteration.  That keeps the BC type fixed across the inner
        # iterations and prevents the chattering observed when the surface
        # crosses hCritA.
        # When Shift has just promoted KodTop to +4 (head BC at hCritA)
        # the Gauss step must know the prescribed head. For BCs that were
        # already positive (e.g. KodTop=+1 with a pre-specified head from
        # Profile.dat), use the caller-supplied ``hTop_in`` instead.
        if KodTop > 0:
            if abs(KodTop) == 4:
                hTop_val = hCritA
            else:
                hTop_val = hTop_in if hTop_in is not None else hNew[-1]
        else:
            hTop_val = 0.0
        if KodBot > 0:
            hBot_val = hBot_in if hBot_in is not None else hNew[0]
        else:
            hBot_val = 0.0
        hNew = solve_tridiagonal(
            N, P, R, S, hNew.copy(),
            KodTop, KodBot, hTop_val, hBot_val,
            PB=PB, RB=RB, SB_coef=SB, PT=PT, RT=RT, ST=ST,
        )
        
        # Check convergence (Fortran: label 14..15)
        ItCrit = True
        for i in range(N):
            M = MatNum[i]
            EpsTh = 0.0
            EpsH = 0.0
            hSat = ParW[10, M]  # saturated head
            
            if hTemp[i] < hSat and hNew[i] < hSat:
                # Compute water content change
                Th = theta[i] + Cap[i] * (hNew[i] - hTemp[i]) / \
                     max(ParW[1, M] - ParW[0, M], 1e-10) / ATh[i] * Rate
                EpsTh = abs(theta[i] - Th)
            else:
                EpsH = abs(hNew[i] - hTemp[i])
            
            if EpsTh > TolTh or EpsH > TolH or abs(hNew[i]) > rMax * 0.999:
                ItCrit = False
                if abs(hNew[i]) > rMax * 0.999:
                    Iter = MaxIt  # force exit
                break
        
        if ItCrit:
            converged = True
            break

        # Do NOT update theta_old here — it must stay frozen at the previous
        # *time-step* value so the storage term (theta - theta_old)/dt is
        # not nulled out across Picard iterations.  The earlier version had
        # ``theta_old = theta.copy()`` at this point which made the time
        # derivative vanish and let hNew drift far past the BC.
    
    # If converged, update water content (Fortran: label 18)
    if converged:
        for i in range(N):
            theta[i] = theta[i] + Cap[i] * (hNew[i] - hTemp[i]) * Rate
    
    # Compute boundary fluxes.  For *flux*-prescribed BCs (KodTop ≤ 0 with
    # rTop given, or FreeD/qGWLF at the bottom) we want the actual imposed
    # flux at the surface — not the Darcy expression evaluated at the
    # adjacent internal edge.  The two coincide only at steady state;
    # during a transient the difference shows up as mass-balance drift.
    if KodTop <= 0 and abs(KodTop) != 4 and abs(KodTop) != 1:
        vTop_out = _compute_top_flux(
            N, x, hNew, hOld, theta, MatNum, ParD, iModel, iHyst,
            iDualPor, dt, Con, Cap, Sink, SinkIm, Ah, AK, ATh,
            CosAlf, lCentrif, Radius, lGeom, lDensity, lVapor,
            lWTDep, Temp, ConLT, ConVT, ConVh, ThVOld, ThVNew,
            WLayer,
        )
    elif KodTop <= 0:
        # Flux-type atmospheric BC: rTop is the prescribed surface flux.
        # HYDRUS convention: rTop > 0 = water out (evap), so vTop ≡ rTop.
        vTop_out = rTop
    else:
        # Head-prescribed BC: compute the actual flux that the head pin
        # imposes via the top-edge Darcy expression.
        vTop_out = _compute_top_flux(
            N, x, hNew, hOld, theta, MatNum, ParD, iModel, iHyst,
            iDualPor, dt, Con, Cap, Sink, SinkIm, Ah, AK, ATh,
            CosAlf, lCentrif, Radius, lGeom, lDensity, lVapor,
            lWTDep, Temp, ConLT, ConVT, ConVh, ThVOld, ThVNew,
            WLayer,
        )

    if KodBot == -5:
        # Free drainage: vBot = -K(h_bot).
        M_bot = int(MatNum[0])
        ConB_bot = (Con[0] + Con[1]) / 2.0
        vBot_out = -ConB_bot * CosAlf
    else:
        vBot_out = _compute_bottom_flux(
            N, x, hNew, hOld, theta, MatNum, ParD, iModel, iHyst,
            iDualPor, dt, Con, Cap, Sink, SinkIm, Ah, AK, ATh,
            CosAlf, lCentrif, Radius, lGeom, lDensity, lVapor,
            lWTDep, Temp, ConLT, ConVT, ConVh, ThVOld, ThVNew,
        )
    
    return hNew, vTop_out, vBot_out, KodTop, KodBot, Iter, converged


def _set_mat_properties(
    N: int,
    x: NDArray[np.float64],
    hNew: NDArray[np.float64],
    hOld: NDArray[np.float64],
    hTemp: NDArray[np.float64],
    MatNum: NDArray[np.int64],
    ParD: NDArray[np.float64],
    ParW: NDArray[np.float64],
    iModel: int,
    Con: NDArray[np.float64],
    Cap: NDArray[np.float64],
    theta: NDArray[np.float64],
    Ah: NDArray[np.float64],
    AK: NDArray[np.float64],
    ATh: NDArray[np.float64],
    CosAlf: float = 1.0,
    lCentrif: bool = False,
    Radius: float = 0.0,
    lGeom: bool = False,
    lWTDep: bool = False,
    Temp: NDArray[np.float64] | None = None,
    ConLT: NDArray[np.float64] | None = None,
    ConVT: NDArray[np.float64] | None = None,
    ConVh: NDArray[np.float64] | None = None,
    lDensity: bool = False,
):
    """
    Compute soil hydraulic properties at each node.
    
    Matches SetMat subroutine in WATFLOW.FOR.
    """
    for i in range(N):
        M = MatNum[i]
        
        # Temperature correction
        AT = 1.0
        BT = 1.0
        if lWTDep and Temp is not None:
            TempR = 20.0
            AT = (75.6 - 0.1425 * Temp[i] - 2.38e-4 * Temp[i] ** 2) / \
                 (75.6 - 0.1425 * TempR - 2.38e-4 * TempR ** 2)
            BT = (1.787 - 0.007 * TempR) / (1.0 + 0.03225 * TempR) / \
                 ((1.787 - 0.007 * Temp[i]) / (1.0 + 0.03225 * Temp[i]))
        
        # Compute head for property evaluation
        hi1 = min(ParW[10, M], hTemp[i] / Ah[i] / AT) if Ah[i] != 0 else hTemp[i]
        hi2 = min(ParW[10, M], hNew[i] / Ah[i] / AT) if Ah[i] != 0 else hNew[i]
        hiM = 0.1 * hi1 + 0.9 * hi2
        
        # Conductivity (pass full 11-element parameter array for material M)
        Coni = FK(iModel, hiM, ParD[:, M])
        
        # Capacity and water content
        Capi = FC(iModel, hiM, ParD[:, M])
        Thei = FQ(iModel, hiM, ParD[:, M])
        
        # Apply hysteretic corrections
        Con[i] = Coni * AK[i] * BT
        Cap[i] = Capi * ATh[i] / Ah[i] / AT
        theta[i] = ParD[0, M] + (Thei - ParD[0, M]) * ATh[i]


def _reset_assemble(
    N: int,
    x: NDArray[np.float64],
    hNew: NDArray[np.float64],
    hOld: NDArray[np.float64],
    theta: NDArray[np.float64],
    theta_old: NDArray[np.float64],
    MatNum: NDArray[np.int64],
    ParD: NDArray[np.float64],
    ParW: NDArray[np.float64],
    iModel: int,
    iHyst: int,
    iDualPor: int,
    dt: float,
    Con: NDArray[np.float64],
    Cap: NDArray[np.float64],
    Sink: NDArray[np.float64],
    SinkIm: NDArray[np.float64],
    Ah: NDArray[np.float64],
    AK: NDArray[np.float64],
    ATh: NDArray[np.float64],
    CosAlf: float = 1.0,
    lCentrif: bool = False,
    Radius: float = 0.0,
    lGeom: bool = False,
    lDensity: bool = False,
    lVapor: bool = False,
    lWTDep: bool = False,
    Temp: NDArray[np.float64] | None = None,
    ConLT: NDArray[np.float64] | None = None,
    ConVT: NDArray[np.float64] | None = None,
    ConVh: NDArray[np.float64] | None = None,
    ThVOld: NDArray[np.float64] | None = None,
    ThVNew: NDArray[np.float64] | None = None,
    WLayer: bool = False,
    rTop: float = 0.0,
    rBot: float = 0.0,
    FreeD: bool = False,
    qGWLF: bool = False,
    qDrain: bool = False,
    GWL0L: float = 0.0,
    Aqh: float = 0.0,
    Bqh: float = 0.0,
) -> Tuple:
    """
    Assemble tridiagonal system for water flow.
    
    Matches Reset subroutine in WATFLOW.FOR.
    
    Returns
    -------
    P, R, S : arrays, shape (N,)
        Tridiagonal matrix coefficients
    PB, RB, SB : floats
        Bottom boundary coefficients
    PT, RT, ST : floats
        Top boundary coefficients
    vTop : float
        Top flux
    """
    P = np.zeros(N, dtype=np.float64)
    R = np.zeros(N, dtype=np.float64)
    S = np.zeros(N, dtype=np.float64)
    
    fRE = 1.0
    Grav = CosAlf
    
    # Bottom node (node 1, index 0)
    dx = x[1] - x[0]
    dxB = dx / 2.0
    ConB = (Con[0] + Con[1]) / 2.0
    if lGeom:
        ConB = np.sqrt(Con[0] * Con[1])
    if lCentrif:
        Grav = CosAlf * (Radius + abs((x[0] + x[1]) / 2.0))
    B = ConB * Grav
    
    if lVapor and ConVh is not None:
        ConB = ConB + (ConVh[0] + ConVh[1]) / 2.0
    
    S[0] = -ConB / dxB
    F2 = Cap[0] * dx / dt * fRE
    RB = ConB / dxB + F2
    SB = -ConB / dxB

    # Bottom-boundary inflow term (Fortran WATFLOW.FOR:222). Notes:
    # - FreeD: rBot = -ConB*Grav so the gravity contribution in B cancels;
    #   the equation reduces to dh/dz = 0 → gravity-only drainage.
    # - qGWLF: rBot = Fqh(h_bot - GWL0L). qDrain: would be FqDrain(...).
    # - Otherwise rBot is the caller-supplied prescribed flux.
    if FreeD:
        rBot_eff = -ConB * Grav
    elif qGWLF:
        rBot_eff = Fqh(hNew[0] - GWL0L, Aqh, Bqh)
    else:
        rBot_eff = rBot

    PB = (B - Sink[0] * dx + F2 * hNew[0]
          - (theta[0] - theta_old[0]) * dx / dt * fRE
          + rBot_eff)
    if iDualPor > 0:
        PB = PB - SinkIm[0] * dx
    
    # Interior nodes (nodes 2..N-1, indices 1..N-2)
    for i in range(1, N - 1):
        dxA = x[i] - x[i - 1]
        dxB = x[i + 1] - x[i]
        dx = (dxA + dxB) / 2.0
        
        ConA = (Con[i] + Con[i - 1]) / 2.0
        ConB = (Con[i] + Con[i + 1]) / 2.0
        if lGeom:
            ConA = np.sqrt(Con[i] * Con[i - 1])
            ConB = np.sqrt(Con[i] * Con[i + 1])
        
        if lCentrif:
            Grav = CosAlf * (Radius + abs(x[i]))
        
        B = (ConA - ConB) * Grav
        if lCentrif:
            B = B + CosAlf * Con[i] * dx
        
        if lVapor and ConVh is not None:
            ConA = ConA + (ConVh[i] + ConVh[i - 1]) / 2.0
            ConB = ConB + (ConVh[i] + ConVh[i + 1]) / 2.0
        
        A2 = ConA / dxA + ConB / dxB
        A3 = -ConB / dxB
        F2 = Cap[i] * dx / dt * fRE
        
        R[i] = A2 + F2
        P[i] = F2 * hNew[i] - (theta[i] - theta_old[i]) * dx / dt * fRE - B - Sink[i] * dx
        if iDualPor > 0:
            P[i] = P[i] - SinkIm[i] * dx
        S[i] = A3
    
    # Top node (node N, index N-1)
    dxA = x[N - 1] - x[N - 2]
    dx = dxA / 2.0
    ConA = (Con[N - 1] + Con[N - 2]) / 2.0
    if lGeom:
        ConA = np.sqrt(Con[N - 1] * Con[N - 2])
    
    if lCentrif:
        Grav = CosAlf * (Radius + abs((x[N - 1] + x[N - 2]) / 2.0))
    
    B = ConA * Grav
    if lVapor and ConVh is not None:
        ConA = ConA + (ConVh[N - 1] + ConVh[N - 2]) / 2.0
    
    F2 = Cap[N - 1] * dx / dt * fRE
    RT = ConA / dxA + F2
    ST = -ConA / dxA
    PT = F2 * hNew[N - 1] - (theta[N - 1] - theta_old[N - 1]) * dx / dt * fRE - Sink[N - 1] * dx - B

    if iDualPor > 0:
        PT = PT - SinkIm[N - 1] * dx

    # Top flux before applying the boundary correction. Mirrors WATFLOW.FOR:294
    # which computes vTop *before* subtracting rTop.
    vTop = -ST * hNew[N - 2] - RT * hNew[N - 1] + PT

    # Apply prescribed top flux (Fortran: ``PT = PT - rTop``).  This is the
    # single most important line for atmospheric / specified-flux BCs — the
    # original Python port skipped it entirely.
    PT = PT - rTop

    if WLayer:
        if hNew[N - 1] > 0.0:
            RT = RT + 1.0 / dt
            PT = PT + max(theta[N - 1], 0.0) / dt
        else:
            PT = PT + max(theta[N - 1], 0.0) / dt

    return P, R, S, PB, RB, SB, PT, RT, ST, vTop


def _compute_top_flux(
    N: int,
    x: NDArray[np.float64],
    hNew: NDArray[np.float64],
    hOld: NDArray[np.float64],
    theta: NDArray[np.float64],
    MatNum: NDArray[np.int64],
    ParD: NDArray[np.float64],
    iModel: int,
    iHyst: int,
    iDualPor: int,
    dt: float,
    Con: NDArray[np.float64],
    Cap: NDArray[np.float64],
    Sink: NDArray[np.float64],
    SinkIm: NDArray[np.float64],
    Ah: NDArray[np.float64],
    AK: NDArray[np.float64],
    ATh: NDArray[np.float64],
    CosAlf: float = 1.0,
    lCentrif: bool = False,
    Radius: float = 0.0,
    lGeom: bool = False,
    lDensity: bool = False,
    lVapor: bool = False,
    lWTDep: bool = False,
    Temp: NDArray[np.float64] | None = None,
    ConLT: NDArray[np.float64] | None = None,
    ConVT: NDArray[np.float64] | None = None,
    ConVh: NDArray[np.float64] | None = None,
    ThVOld: NDArray[np.float64] | None = None,
    ThVNew: NDArray[np.float64] | None = None,
    WLayer: bool = False,
) -> float:
    """Compute top boundary flux."""
    dx = x[N - 1] - x[N - 2]
    ConA = (Con[N - 1] + Con[N - 2]) / 2.0
    if lGeom:
        ConA = np.sqrt(Con[N - 1] * Con[N - 2])
    
    Grav = CosAlf
    if lCentrif:
        Grav = CosAlf * (Radius + abs((x[N - 1] + x[N - 2]) / 2.0))
    
    vTop = -ConA * ((hNew[N - 1] - hNew[N - 2]) / dx + Grav)
    
    if iDualPor > 0:
        vTop = vTop - SinkIm[N - 1] * dx / 2.0
    
    return vTop


def _compute_bottom_flux(
    N: int,
    x: NDArray[np.float64],
    hNew: NDArray[np.float64],
    hOld: NDArray[np.float64],
    theta: NDArray[np.float64],
    MatNum: NDArray[np.int64],
    ParD: NDArray[np.float64],
    iModel: int,
    iHyst: int,
    iDualPor: int,
    dt: float,
    Con: NDArray[np.float64],
    Cap: NDArray[np.float64],
    Sink: NDArray[np.float64],
    SinkIm: NDArray[np.float64],
    Ah: NDArray[np.float64],
    AK: NDArray[np.float64],
    ATh: NDArray[np.float64],
    CosAlf: float = 1.0,
    lCentrif: bool = False,
    Radius: float = 0.0,
    lGeom: bool = False,
    lDensity: bool = False,
    lVapor: bool = False,
    lWTDep: bool = False,
    Temp: NDArray[np.float64] | None = None,
    ConLT: NDArray[np.float64] | None = None,
    ConVT: NDArray[np.float64] | None = None,
    ConVh: NDArray[np.float64] | None = None,
    ThVOld: NDArray[np.float64] | None = None,
    ThVNew: NDArray[np.float64] | None = None,
) -> float:
    """Compute bottom boundary flux."""
    dx = x[1] - x[0]
    ConB = (Con[0] + Con[1]) / 2.0
    if lGeom:
        ConB = np.sqrt(Con[0] * Con[1])
    
    Grav = CosAlf
    if lCentrif:
        Grav = CosAlf * (Radius + abs((x[0] + x[1]) / 2.0))
    
    vBot = -ConB * ((hNew[1] - hNew[0]) / dx + Grav)
    
    if iDualPor > 0:
        vBot = vBot - SinkIm[0] * dx / 2.0
    
    return vBot


# ============================================================================
# Boundary condition adjustment
# ============================================================================

def shift_bc(
    N: int,
    KodTop: int,
    rTop: float,
    rBot: float,
    hTop: float,
    hBot: float,
    hCritA: float,
    CosAlf: float,
    WLayer: bool,
    Con: NDArray[np.float64],
    hNew: NDArray[np.float64],
    x: NDArray[np.float64],
    TopInf: bool,
    KodBot: int,
    ThNew: NDArray[np.float64],
    ThOld: NDArray[np.float64],
    Sink: NDArray[np.float64],
    dt: float,
    lVapor: bool,
    lWTDep: bool,
    ConLT: NDArray[np.float64],
    Temp: NDArray[np.float64],
    ConVh: NDArray[np.float64],
    ConVT: NDArray[np.float64],
    iDualPor: int,
    SinkIm: NDArray[np.float64],
    lDensity: bool,
    Conc: NDArray[np.float64],
    NSD: int,
    lCentrif: bool,
    Radius: float,
    hSeep: float,
    SeepF: bool,
) -> Tuple[int, int, float, float]:
    """
    Adjust boundary conditions based on current solution.
    
    Matches Shift subroutine in WATFLOW.FOR.
    
    Handles:
    - Seepage face at bottom
    - Atmospheric boundary at top
    - Free drainage
    
    Returns
    -------
    KodTop, KodBot : int
        Updated BC codes
    rTop, rBot : float
        Updated BC flux values
    """
    fRE = 1.0
    Grav = CosAlf
    
    # Seepage face at bottom
    if SeepF:
        dx = x[1] - x[0]
        if lDensity and NSD > 0:
            fRE = 1.0  # fRo(1, Conc[1, 0])
        if lCentrif:
            Grav = CosAlf * (Radius + abs((x[1] + x[0]) / 2.0))
        
        vBot = -(Con[0] + Con[1]) / 2.0 * ((hNew[1] - hNew[0]) / dx + Grav * fRE) - \
                dx / 2.0 * fRE * ((ThNew[0] - ThOld[0]) / dt + Sink[0])
        
        if KodBot >= 0:
            if vBot > 0.0:
                KodBot = -2
                rBot = 0.0
        else:
            if hNew[0] >= hSeep:
                KodBot = 2
                hBot = hSeep
    
    # Atmospheric boundary condition at top
    if TopInf and (abs(KodTop) == 4 or (abs(KodTop) == 1 and rTop > 0.0)):
        if KodTop > 0:
            M = N - 2          # one node below the surface
            dx = x[N - 1] - x[M]
            if lCentrif:
                Grav = CosAlf * (Radius + abs((x[N - 1] + x[M]) / 2.0))
            
            vTop = -(Con[N - 1] + Con[M]) / 2.0 * ((hNew[N - 1] - hNew[M]) / dx + Grav * fRE) - \
                    (ThNew[N - 1] - ThOld[N - 1]) * fRE * dx / 2.0 / dt - Sink[N - 1] * dx / 2.0
            
            if iDualPor > 0:
                vTop = vTop - SinkIm[N - 1] * dx / 2.0
            
            if lWTDep and ConLT is not None:
                vTop = vTop - (ConLT[N - 1] + ConLT[M]) / 2.0 * (Temp[N - 1] - Temp[M]) / dx
            
            if lVapor and ConVh is not None:
                vTop = vTop - (ConVh[N - 1] + ConVh[M]) / 2.0 * (hNew[N - 1] - hNew[M]) / dx
                if ConVT is not None:
                    vTop = vTop - (ConVT[N - 1] + ConVT[M]) / 2.0 * (Temp[N - 1] - Temp[M]) / dx
            
            if abs(vTop) > abs(rTop) or vTop * rTop <= 0.0:
                if abs(KodTop) == 4:
                    KodTop = -4
            
            if KodTop == 4 and hNew[N - 1] <= 0.99 * hCritA and rTop < 0.0:
                KodTop = -4
        else:
            if not WLayer:
                if hNew[N - 1] > 0.0:
                    if abs(KodTop) == 4:
                        KodTop = 4
                    if abs(KodTop) == 1:
                        KodTop = 1
                    hTop = 0.0
            
            if hNew[N - 1] <= hCritA:
                if abs(KodTop) == 4:
                    KodTop = 4
                if abs(KodTop) == 1:
                    KodTop = 1
                hTop = hCritA
    
    return KodTop, KodBot, rTop, rBot


# ============================================================================
# Groundwater level fluctuation
# ============================================================================

def Fqh(GWL: float, Aqh: float, Bqh: float) -> float:
    """
    Groundwater level fluctuation flux.
    
    Parameters
    ----------
    GWL : float
        Groundwater level
    Aqh : float
        Amplitude parameter
    Bqh : float
        Exponent parameter
    
    Returns
    -------
    flux : float
        GWLF flux
    """
    return Aqh * np.exp(Bqh * abs(GWL))


# ============================================================================
# Drainage flux (SWAP model)
# ============================================================================

def FqDrain(
    GWL: float,
    zBotDr: float,
    BaseGW: float,
    rSpacing: float,
    iPosDr: int,
    KhTop: float,
    KhBot: float,
    KvTop: float,
    KvBot: float,
    Entres: float,
    WetPer: float,
    zInTF: float,
    GeoFac: float,
) -> float:
    """
    Drainage flux based on SWAP model (van Dam et al. 1997).
    
    Parameters
    ----------
    GWL : float
        Groundwater level
    zBotDr : float
        Depth to drain
    BaseGW : float
        Depth to impervious layer
    rSpacing : float
        Drain spacing
    iPosDr : int
        Drain position code (1-5)
    KhTop : float
        Horizontal conductivity above drain
    KhBot : float
        Horizontal conductivity below drain
    KvTop : float
        Vertical conductivity above drain
    KvBot : float
        Vertical conductivity below drain
    Entres : float
        Entrance resistance
    WetPer : float
        Wet perimeter
    zInTF : float
        Transition level
    GeoFac : float
        Geometry factor
    
    Returns
    -------
    flux : float
        Drainage flux
    """
    dh = GWL - zBotDr
    
    # No infiltration allowed
    if dh < 1e-10:
        return 0.0
    
    pi = 3.14159
    
    if iPosDr == 1:
        # Homogeneous, on top of impervious layer
        TotRes = rSpacing ** 2 / (4.0 * KhTop * abs(dh)) + Entres
    
    elif iPosDr in (2, 3):
        # Homogeneous profile or at interface
        zImp = max(BaseGW, zBotDr - 0.25 * rSpacing)
        dBot = zBotDr - zImp
        if dBot < 0.0:
            raise ValueError("dBot negative in FqDrain")
        
        x_val = 2.0 * pi * dBot / rSpacing
        
        if x_val > 0.5:
            fx = 0.0
            for i in range(1, 11, 2):
                fx += (4.0 * np.exp(-2.0 * i * x_val)) / (i * (1.0 - np.exp(-2.0 * i * x_val)))
            EqD = pi * rSpacing / 8.0 / (np.log(rSpacing / WetPer) + fx)
        elif x_val < 1e-6:
            EqD = dBot
        else:
            fx = pi ** 2 / (4.0 * x_val) + np.log(x_val / (2.0 * pi))
            EqD = pi * rSpacing / 8.0 / (np.log(rSpacing / WetPer) + fx)
        
        EqD = min(EqD, dBot)
        
        if iPosDr == 2:
            TotRes = rSpacing ** 2 / (8.0 * KhTop * EqD + 4.0 * KhTop * abs(dh)) + Entres
        else:
            TotRes = rSpacing ** 2 / (8.0 * KhBot * EqD + 4.0 * KhTop * abs(dh)) + Entres
    
    elif iPosDr == 4:
        # Drain in bottom layer
        if zBotDr > zInTF:
            raise ValueError("Check zInTF and zBotDr")
        RVer = max(GWL - zInTF, 0.0) / KvTop + (min(zInTF, GWL) - zBotDr) / KvBot
        RHor = rSpacing ** 2 / (8.0 * KhBot * dBot)
        RRad = rSpacing / (pi * np.sqrt(KhBot * KvBot)) * np.log(dBot / WetPer)
        TotRes = RVer + RHor + RRad + Entres
    
    elif iPosDr == 5:
        # Drain in top layer
        if zInTF < zBotDr:
            raise ValueError("Check zInTF and zBotDr")
        RVer = (zInTF - zBotDr) / KvTop
        dBot = zBotDr - BaseGW
        x_val = 2.0 * pi * dBot / rSpacing
        if x_val > 0.5:
            fx = 0.0
            for i in range(1, 11, 2):
                fx += (4.0 * np.exp(-2.0 * i * x_val)) / (i * (1.0 - np.exp(-2.0 * i * x_val)))
            EqD = pi * rSpacing / 8.0 / (np.log(GeoFac * rSpacing / WetPer) + fx)
        elif x_val < 1e-6:
            EqD = dBot
        else:
            fx = pi ** 2 / (4.0 * x_val) + np.log(x_val / (2.0 * pi))
            EqD = pi * rSpacing / 8.0 / (np.log(GeoFac * rSpacing / WetPer) + fx)
        EqD = min(EqD, dBot)
        RHor = rSpacing ** 2 / (8.0 * KhTop * EqD)
        RRad = rSpacing / (pi * np.sqrt(KhTop * KvTop)) * np.log((zInTF - zBotDr) / WetPer)
        TotRes = RVer + RHor + RRad + Entres
    
    else:
        return 0.0
    
    return dh / TotRes
