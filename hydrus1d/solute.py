"""
Solute transport solver for HYDRUS-1D.
======================================

Convection-dispersion equation with:
- Mass-lumping finite elements
- Sequential decay chains
- Linear/non-linear sorption
- First-order kinetic reactions
- Retardation factors
- Upstream weighting for stability

Direct port of SOLUTE.FOR subroutines:
- Coeff    : Compute dispersion, retardation, Peclet/Courant numbers
- MassTran : Assemble and solve solute transport equation
- BanSol   : Banbury solver (tridiagonal)
"""

from __future__ import annotations
import numpy as np
from numpy.typing import NDArray
from typing import Tuple

from .utils import solve_banbury, compute_peclet, compute_courant


# ============================================================================
# Coefficient computation
# ============================================================================

def compute_coefficients(
    jS: int,
    Level: int,
    NLevel: int,
    N: int,
    x: NDArray[np.float64],
    Disp: NDArray[np.float64],
    vO: NDArray[np.float64],
    vN: NDArray[np.float64],
    thO: NDArray[np.float64],
    thN: NDArray[np.float64],
    thSat: NDArray[np.float64],
    ChPar: NDArray[np.float64],
    MatNum: NDArray[np.int64],
    TempN: NDArray[np.float64],
    TempO: NDArray[np.float64],
    TDep: NDArray[np.float64],
    Retard: NDArray[np.float64],
    Conc: NDArray[np.float64],
    cNew: NDArray[np.float64],
    cPrevO: NDArray[np.float64],
    dt: float,
    lEquil: bool,
    lUpW: bool,
    lArtD: bool,
    Iter: int,
    lTort: bool,
    iDualPor: int,
    ThNIm: NDArray[np.float64],
    ThOIm: NDArray[np.float64],
    SinkIm: NDArray[np.float64],
    iTort: int = 1,
    xConv: float = 1.0,
    tConv: float = 1.0,
) -> Tuple[float, float, float]:
    """
    Compute dispersion coefficients, retardation factors, Peclet and
    Courant numbers, upstream weighting factors.
    
    Direct port of Coeff subroutine in SOLUTE.FOR.
    
    Parameters
    ----------
    jS : int
        Species index (1-based)
    Level : int
        Current iteration level
    NLevel : int
        Total number of iteration levels
    N : int
        Number of nodes
    x : array, shape (N,)
        Node coordinates
    Disp : array, shape (N,)
        Dispersion coefficients
    vO, vN : array, shape (N,)
        Old/new velocities
    thO, thN : array, shape (N,)
        Old/new water contents
    thSat : array, shape (NMat,)
        Saturated water content
    ChPar : array, shape (NSD*16+4, NMat)
        Chemical parameters
    MatNum : array, shape (N,)
        Material zone indices
    TempN, TempO : array, shape (N,)
        Old/new temperatures
    TDep : array, shape (NSD*16+4,)
        Temperature dependence parameters
    Retard : array, shape (N,)
        Retardation factors
    Conc : array, shape (NSD, N)
        Concentrations
    cNew : array, shape (N,)
        New concentration estimate
    cPrevO : array, shape (N,)
        Previous concentration estimate
    dt : float
        Time step
    lEquil : bool
        Equilibrium sorption flag
    lUpW : bool
        Upstream weighting flag
    lArtD : bool
        Artificial dispersion flag
    Iter : int
        Iteration counter
    lTort : bool
        Tortuosity flag
    iDualPor : int
        Dual-porosity flag
    ThNIm, ThOIm : array, shape (N,)
        Immobile water contents
    SinkIm : array, shape (N,)
        Immobile sink term
    iTort : int
        Tortuosity model code
    xConv : float
        Length conversion factor
    tConv : float
        Time conversion factor
    
    Returns
    -------
    Peclet : float
        Maximum Peclet number
    Courant : float
        Maximum Courant number
    dtMaxC : float
        Maximum stable time step for concentration
    """
    jjj = (jS - 1) * 16
    Peclet = 0.0
    Courant = 0.0
    CourMax = 1.0
    dtMaxC = 1e30
    Tr = 293.15
    R = 8.314
    
    for i in range(N - 1, 0, -1):
        j = i + 1
        k = i - 1
        M = MatNum[i]
        
        if Level == NLevel:
            ThW = thN[i]
            ThWO = thO[i]
            ThG = max(0.0, thSat[M] - ThW)
            v = vN[i]
            TT = (TempN[i] + 273.15 - Tr) / R / (TempN[i] + 273.15) / Tr
        else:
            ThW = thO[i]
            ThG = max(0.0, thSat[M] - ThW)
            v = vO[i]
            TT = (TempO[i] + 273.15 - Tr) / R / (TempO[i] + 273.15) / Tr
        
        # Temperature-dependent parameters
        Dw = ChPar[jjj + 5, M] * np.exp(TDep[jjj + 5] * TT)
        Dg = ChPar[jjj + 6, M] * np.exp(TDep[jjj + 6] * TT)
        xKs = ChPar[jjj + 7, M] * np.exp(TDep[jjj + 7] * TT)
        xNu = ChPar[jjj + 8, M] * np.exp(TDep[jjj + 8] * TT)
        fExp = ChPar[jjj + 9, M]
        Henry = ChPar[jjj + 10, M] * np.exp(TDep[jjj + 10] * TT)
        
        # Tortuosity correction
        if lTort and iTort == 1:
            Dw = Dw * (ThW / thSat[M]) ** 1.5
            Dg = Dg * (ThG / max(thSat[M] - 0.001, 0.001)) ** 1.5
        
        # Dispersion coefficient
        dx = x[j] - x[i]
        alpha_L = max(abs(v) * xConv * tConv, 1e-10)
        
        # Molecular diffusion
        D_mol = Dw * max(ThW, 0.001)
        
        # Mechanical dispersion
        D_mech = alpha_L * abs(v) / xConv
        
        # Total dispersion
        Disp[i] = D_mol + D_mech
        
        # Artificial dispersion for stability
        if lArtD and Iter < 2:
            if abs(v) > 0:
                Disp[i] = max(Disp[i], abs(v) * dx / (2.0 * xConv))
        
        # Retardation factor
        R_f = 1.0
        if lEquil:
            R_f = 1.0 + xKs * xNu / max(ThW, 0.001)
        Retard[i] = max(R_f, 1.0)
        
        # Peclet number
        if Disp[i] > 0:
            Pe = abs(v) * dx / (2.0 * Disp[i] * xConv)
            Peclet = max(Peclet, Pe)
        
        # Courant number
        if dx > 0:
            Co = abs(v) * dt / (dx * xConv * tConv)
            Courant = max(Courant, Co)
            dtMaxC = min(dtMaxC, dx * xConv / max(abs(v), 1e-10) / tConv)
    
    return Peclet, Courant, dtMaxC


# ============================================================================
# Solute transport equation assembly and solution
# ============================================================================

def solve_solute_transport(
    jS: int,
    N: int,
    x: NDArray[np.float64],
    Conc: NDArray[np.float64],
    vN: NDArray[np.float64],
    thN: NDArray[np.float64],
    thO: NDArray[np.float64],
    Disp: NDArray[np.float64],
    Retard: NDArray[np.float64],
    dt: float,
    kTopCh: int,
    kBotCh: int,
    cTop: float,
    cBot: float,
    g0: NDArray[np.float64],
    g1: NDArray[np.float64],
    sSink: NDArray[np.float64],
    lUpW: bool,
    lEquil: bool,
    epsi: float = 1.0,
    rMin: float = 1e-37,
) -> Tuple[NDArray[np.float64], float, float]:
    """
    Assemble and solve solute transport equation.
    
    Direct port of MassTran subroutine in SOLUTE.FOR.
    
    Governing equation:
        R * dc/dt = d/dz[D * dc/dz] - d/dz[v * c] - g0 * c - g1 + S
    
    Parameters
    ----------
    jS : int
        Species index (1-based)
    N : int
        Number of nodes
    x : array, shape (N,)
        Node coordinates
    Conc : array, shape (NSD, N)
        Concentrations (updated in place)
    vN : array, shape (N,)
        Velocities
    thN : array, shape (N,)
        New water content
    thO : array, shape (N,)
        Old water content
    Disp : array, shape (N,)
        Dispersion coefficients
    Retard : array, shape (N,)
        Retardation factors
    dt : float
        Time step
    kTopCh : int
        Top chemical BC code
    kBotCh : int
        Bottom chemical BC code
    cTop : float
        Top BC concentration
    cBot : float
        Bottom BC concentration
    g0 : array, shape (N,)
        First-order decay coefficient
    g1 : array, shape (N,)
        Source/sink term
    sSink : array, shape (N,)
        Root solute uptake
    lUpW : bool
        Upstream weighting flag
    lEquil : bool
        Equilibrium sorption flag
    epsi : float
        Porosity factor for vapor
    rMin : float
        Minimum value floor
    
    Returns
    -------
    cNew : array, shape (N,)
        Updated concentrations
    cvTop : float
        Top flux concentration
    cvBot : float
        Bottom flux concentration
    """
    # Initialize tridiagonal system
    B = np.zeros(N, dtype=np.float64)  # Lower diagonal
    D = np.zeros(N, dtype=np.float64)  # Main diagonal
    E = np.zeros(N, dtype=np.float64)  # Upper diagonal
    F = np.zeros(N, dtype=np.float64)  # RHS
    
    cvTop = 0.0
    cvBot = 0.0
    
    # Bottom node (index 0)
    dx = x[1] - x[0]
    dxB = dx / 2.0
    D1 = (Disp[0] + Disp[1]) / 2.0
    v1 = (vN[0] + vN[1]) / 2.0
    th1 = (thN[0] + thN[1]) / 2.0
    
    # Upstream weighting for convection
    if lUpW:
        if v1 > 0:
            wc = 1.0
        else:
            wc = 0.0
    else:
        wc = 0.5
    
    A2 = D1 / dxB
    D[0] = A2 / dt + Retard[0] * (thN[0] - thO[0]) / dt + g0[0] + sSink[0]
    E[0] = -A2 / dt
    F[0] = Retard[0] * thO[0] / dt * Conc[jS - 1, 0] - g1[0]
    
    # Apply bottom BC
    if kBotCh < 0:
        # Flux BC
        if vN[0] >= 0:
            cvBot = epsi * cBot * vN[0]
        else:
            cvBot = epsi * Conc[jS - 1, 0] * vN[0]
        E[0] = rMin
        D[0] = D[0] + abs(E[0])
    elif kBotCh == 0:
        # Variable head
        cvBot = epsi * Conc[jS - 1, 0] * vN[0]
    else:
        # Concentration BC
        F[0] = F[0] + A2 * cBot / dt
        D[0] = D[0] + A2 / dt
        E[0] = rMin
    
    # Interior nodes
    for i in range(1, N - 1):
        dxA = x[i] - x[i - 1]
        dxB = x[i + 1] - x[i]
        dx = (dxA + dxB) / 2.0
        
        DA = (Disp[i] + Disp[i - 1]) / 2.0
        DB = (Disp[i] + Disp[i + 1]) / 2.0
        vA = (vN[i] + vN[i - 1]) / 2.0
        vB = (vN[i] + vN[i + 1]) / 2.0
        
        if lUpW:
            wcA = 1.0 if vA > 0 else 0.0
            wcB = 1.0 if vB > 0 else 0.0
        else:
            wcA = wcB = 0.5
        
        A1 = DA / dxA
        A2 = DB / dxB
        A3 = -DB / dxB
        
        # Mass-lumping: convection term goes to RHS
        conv_A = vA * wcA * Conc[jS - 1, i - 1] + vA * (1 - wcA) * Conc[jS - 1, i]
        conv_B = vB * wcB * Conc[jS - 1, i] + vB * (1 - wcB) * Conc[jS - 1, i + 1]
        
        R_i = Retard[i]
        D[i] = (A1 + A2) / dt + R_i * (thN[i] - thO[i]) / dt + g0[i] + sSink[i]
        B[i] = -A1 / dt
        E[i] = A3 / dt
        F[i] = R_i * thO[i] / dt * Conc[jS - 1, i] - g1[i] + conv_A / dxA - conv_B / dxB
    
    # Top node (index N-1)
    dxA = x[N - 1] - x[N - 2]
    dx = dxA / 2.0
    DA = (Disp[N - 1] + Disp[N - 2]) / 2.0
    vA = (vN[N - 1] + vN[N - 2]) / 2.0
    
    if lUpW:
        wcA = 1.0 if vA > 0 else 0.0
    else:
        wcA = 0.5
    
    A1 = DA / dxA
    conv_A = vA * wcA * Conc[jS - 1, N - 2] + vA * (1 - wcA) * Conc[jS - 1, N - 1]
    
    R_i = Retard[N - 1]
    D[N - 1] = A1 / dt + R_i * (thN[N - 1] - thO[N - 1]) / dt + g0[N - 1] + sSink[N - 1]
    B[N - 1] = -A1 / dt
    E[N - 1] = rMin
    F[N - 1] = R_i * thO[N - 1] / dt * Conc[jS - 1, N - 1] - g1[N - 1] + conv_A / dxA
    
    # Apply top BC
    if kTopCh < 0:
        # Flux BC
        if vN[N - 1] >= 0:
            cvTop = epsi * cTop * vN[N - 1]
        else:
            cvTop = epsi * Conc[jS - 1, N - 1] * vN[N - 1]
        B[N - 1] = rMin
        D[N - 1] = D[N - 1] + abs(B[N - 1])
    else:
        # Concentration BC
        F[N - 1] = F[N - 1] + A1 * cTop / dt
        D[N - 1] = D[N - 1] + A1 / dt
        B[N - 1] = rMin
    
    # Solve tridiagonal system
    cNew = solve_banbury(N, B, D, E, F)
    
    # Update concentrations
    Conc[jS - 1, :] = cNew
    
    return cNew, cvTop, cvBot


# ============================================================================
# Decay chain source terms
# ============================================================================

def compute_decay_source(
    jS: int,
    N: int,
    Conc: NDArray[np.float64],
    ChPar: NDArray[np.float64],
    MatNum: NDArray[np.int64],
    TempN: NDArray[np.float64],
    TDep: NDArray[np.float64],
    g0: NDArray[np.float64],
    g1: NDArray[np.float64],
    lEquil: bool = True,
) -> None:
    """
    Compute first-order decay chain source terms.
    
    For sequential decay: C1 -> C2 -> C3 -> ...
    g0 = decay rate of current species
    g1 = production rate from parent species
    
    Parameters
    ----------
    jS : int
        Species index (1-based)
    N : int
        Number of nodes
    Conc : array, shape (NSD, N)
        Concentrations
    ChPar : array, shape (NSD*16+4, NMat)
        Chemical parameters
    MatNum : array, shape (N,)
        Material zone indices
    TempN : array, shape (N,)
        Temperature
    TDep : array, shape (NSD*16+4,)
        Temperature dependence
    g0 : array, shape (N,)
        Decay rate (output)
    g1 : array, shape (N,)
        Production rate (output)
    lEquil : bool
        Equilibrium sorption flag
    """
    jjj = (jS - 1) * 16
    Tr = 293.15
    R = 8.314
    
    for i in range(N):
        M = MatNum[i]
        TT = (TempN[i] + 273.15 - Tr) / R / (TempN[i] + 273.15) / Tr
        
        # Decay rate of current species
        GamL = ChPar[jjj + 11, M] * np.exp(TDep[jjj + 11] * TT)
        GamS = ChPar[jjj + 12, M] * np.exp(TDep[jjj + 12] * TT)
        GamG = ChPar[jjj + 13, M] * np.exp(TDep[jjj + 13] * TT)
        
        g0[i] = GamL + GamS + GamG
        
        # Production from parent
        g1[i] = 0.0
        if jS > 1:
            jj1 = jjj - 16
            GamL1P = ChPar[jj1 + 14, M] * np.exp(TDep[jj1 + 14] * TT)
            GamS1P = ChPar[jj1 + 15, M] * np.exp(TDep[jj1 + 15] * TT)
            GamG1P = ChPar[jj1 + 16, M] * np.exp(TDep[jj1 + 16] * TT)
            
            # Production = parent decay rate * parent concentration
            g1[i] = (GamL1P + GamS1P + GamG1P) * Conc[jS - 2, i]


# ============================================================================
# Freundlich sorption
# ============================================================================

def freundlich_sorption(
    c: float,
    xKs: float,
    xNu: float,
    fExp: float,
) -> float:
    """
    Freundlich sorption isotherm.
    
    S = Ks * c^Nu * (1 + c)^(fExp - 1)
    
    Parameters
    ----------
    c : float
        Concentration
    xKs : float
        Freundlich K value
    xNu : float
        Freundlich exponent
    fExp : float
        Exponential parameter
    
    Returns
    -------
    S : float
        Sorbed concentration
    """
    if c <= 0:
        return 0.0
    return xKs * c ** xNu * (1.0 + c) ** max(fExp - 1.0, 0.0)


def freundlich_retardation(
    c: float,
    xKs: float,
    xNu: float,
    fExp: float,
    theta: float,
) -> float:
    """
    Retardation factor from Freundlich isotherm.
    
    R = 1 + (rho_b / theta) * dS/dc
    
    Parameters
    ----------
    c : float
        Concentration
    xKs : float
        Freundlich K value
    xNu : float
        Freundlich exponent
    fExp : float
        Exponential parameter
    theta : float
        Water content
    
    Returns
    -------
    R : float
        Retardation factor
    """
    if c <= 0 or theta <= 0:
        return 1.0
    
    # dS/dc for Freundlich
    dSdc = xKs * xNu * c ** (xNu - 1.0) * (1.0 + c) ** max(fExp - 1.0, 0.0)
    
    return 1.0 + dSdc / max(theta, 0.001)
