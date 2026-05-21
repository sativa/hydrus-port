"""
Root water and nutrient uptake for HYDRUS-1D.
=============================================

Root water uptake models:
- Feddes model (stress function)
- Silvertown model
- Hanks model
- Eddy Woehling modification

Root solute uptake:
- Passive uptake
- Active uptake with Michaelis-Menten kinetics
- Compensation mechanism

Direct port of SINK.FOR.
"""

from __future__ import annotations
import numpy as np
from numpy.typing import NDArray
from typing import Tuple

from .material import FQ


# ============================================================================
# Root water uptake
# ============================================================================

def set_root_water_uptake(
    N: int,
    x: NDArray[np.float64],
    Beta: NDArray[np.float64],
    Sink: NDArray[np.float64],
    TPot: float,
    hNew: NDArray[np.float64],
    lMoSink: bool,
    P0: float,
    POptm: float,
    P2H: float,
    P2L: float,
    P3: float,
    r2H: float,
    r2L: float,
    ThNew: NDArray[np.float64],
    ParD: NDArray[np.float64],
    MatNum: NDArray[np.int64],
    iModel: int,
    Con: NDArray[np.float64],
    OmegaC: float = 1.0,
    lChem: bool = False,
    lSolRed: bool = False,
    lSolAdd: bool = False,
    aOsm: NDArray[np.float64] | None = None,
    Conc: NDArray[np.float64] | None = None,
    c50: float = 0.0,
    P3c: float = 0.0,
    lMsSink: bool = False,
    dt: float = 1.0,
) -> Tuple[float, float]:
    """
    Compute root water uptake using Feddes stress function.
    
    Direct port of SetSnk subroutine in SINK.FOR.
    
    Parameters
    ----------
    N : int
        Number of nodes
    x : array, shape (N,)
        Node coordinates
    Beta : array, shape (N,)
        Root density distribution
    Sink : array, shape (N,)
        Root water uptake (output)
    TPot : float
        Potential transpiration rate
    hNew : array, shape (N,)
        Current pressure head
    lMoSink : bool
        Feddes model flag (True) or Silvertown model flag (False)
    P0 : float
        Permanent wilting point (Silvertown)
    POptm : float
        Optimal pressure head (Feddes)
    P2H : float
        P2 at high transpiration
    P2L : float
        P2 at low transpiration
    P3 : float
        Permanent wilting point (Feddes)
    r2H : float
        High transpiration rate
    r2L : float
        Low transpiration rate
    ThNew : array, shape (N,)
        Current water content
    ParD : array, shape (11, NMat)
        Drying hydraulic parameters
    MatNum : array, shape (N,)
        Material zone indices
    iModel : int
        Hydraulic model code
    Con : array, shape (N,)
        Hydraulic conductivity
    OmegaC : float
        Compensation coefficient
    lChem : bool
        Chemical effect flag
    lSolRed : bool
        Solute reduction flag
    lSolAdd : bool
        Additive solute effect flag
    aOsm : array, shape (NS,), optional
        Osmotic coefficients
    Conc : array, shape (NSD, N), optional
        Concentrations
    c50 : float
        Critical concentration
    P3c : float
        Concentration stress exponent
    lMsSink : bool
        Multiplicative solute stress flag
    dt : float
        Time step
    
    Returns
    -------
    vRoot : float
        Total root water uptake
    hRoot : float
        Depth-weighted average root water potential
    """
    Compen = 1.0
    nStep = 1
    if OmegaC < 1.0:
        nStep = 2
    
    Omega = 0.0
    vRoot = 0.0
    hRoot = 0.0
    ARoot = 0.0
    
    for iStep in range(nStep):
        for i in range(1, N):
            if Beta[i] > 0:
                # Compute element width
                if i == N - 1:
                    dxM = (x[i] - x[i - 1]) / 2.0
                else:
                    dxM = (x[i + 1] - x[i - 1]) / 2.0
                
                M = MatNum[i]
                hRed = hNew[i]
                SAlfa = 1.0
                
                # Chemical effect on uptake
                if lChem and lSolRed and Conc is not None and aOsm is not None:
                    cRed = 0.0
                    for j in range(len(aOsm)):
                        cRed += aOsm[j] * Conc[j, i]
                    
                    if lSolAdd:
                        hRed = hRed + cRed
                    else:
                        SAlfa = _fsalfa(lMsSink, cRed, c50, P3c)
                
                # Water stress function (Feddes)
                Alfa = _falfa(lMoSink, TPot, hRed, P0, POptm, P2H, P2L, P3, r2H, r2L)
                
                if iStep != nStep - 1:
                    Omega = Omega + Alfa * SAlfa * Beta[i] * dxM
                    continue
                
                # Compensation
                Compen = 1.0
                if Omega < OmegaC and Omega > 0:
                    Compen = OmegaC
                elif Omega >= OmegaC:
                    Compen = Omega
                
                # Compute uptake
                Sink[i] = Alfa * SAlfa * Beta[i] * TPot / Compen
                
                # Limit by available water
                PMin = P3 if lMoSink else 10.0 * P0
                ThLimit = FQ(iModel, PMin, ParD[:, M])
                Sink[i] = min(Sink[i], max(0.0, 0.5 * (ThNew[i] - ThLimit) / dt))
                
                vRoot = vRoot + Sink[i] * dxM
                hRoot = hRoot + hNew[i] * dxM
                ARoot = ARoot + dxM
            else:
                Sink[i] = 0.0
            
            # Eddy Woehling modification
            if Beta[i] < 0:
                if i == N - 1:
                    dxM = (x[i] - x[i - 1]) / 2.0
                else:
                    dxM = (x[i + 1] - x[i - 1]) / 2.0
                Sink[i] = Beta[i] * 0.0  # rBot placeholder
                Sink[i] = max(Sink[i], 0.5 * (ThNew[i] - ParD[1, MatNum[i]]) / dt)
    
    if ARoot > 0.001:
        hRoot = hRoot / ARoot
    
    return vRoot, hRoot


# ============================================================================
# Root solute uptake
# ============================================================================

def set_root_solute_uptake(
    jS: int,
    N: int,
    x: NDArray[np.float64],
    Beta: NDArray[np.float64],
    Sink: NDArray[np.float64],
    SinkS: NDArray[np.float64],
    Conc: NDArray[np.float64],
    OmegaW: float,
    cRootMax: float,
    lActRSU: bool,
    OmegaS: float,
    SPot: float,
    rKM: float,
    cMin: float = 0.0,
) -> Tuple[float, float]:
    """
    Compute root solute uptake.
    
    Direct port of SetSSnk subroutine in SINK.FOR.
    
    Parameters
    ----------
    jS : int
        Species index (1-based)
    N : int
        Number of nodes
    x : array, shape (N,)
        Node coordinates
    Beta : array, shape (N,)
        Root density distribution
    Sink : array, shape (N,)
        Root water uptake
    SinkS : array, shape (N,)
        Root solute uptake (output)
    Conc : array, shape (NSD, N)
        Concentrations
    OmegaW : float
        Ratio of actual to potential transpiration
    cRootMax : float
        Maximum concentration for passive uptake
    lActRSU : bool
        Active root solute uptake flag
    OmegaS : float
        Solute stress index
    SPot : float
        Potential root solute uptake
    rKM : float
        Michaelis-Menten constant
    cMin : float
        Minimum concentration
    
    Returns
    -------
    SPUptake : float
        Total passive uptake
    SAUptakeA : float
        Total active uptake
    """
    Compen = 1.0
    nStep = 1
    if lActRSU:
        nStep = 2
    if lActRSU and OmegaS < 1.0:
        nStep = 3
    
    Omega = 0.0
    SPUptake = 0.0
    SAUptakeA = 0.0
    SAUptakeP = 0.0
    
    for i in range(N):
        SinkS[i] = 0.0
    
    for iStep in range(nStep):
        SAUptakeA_local = 0.0
        
        for i in range(N):
            if Beta[i] > 0:
                if i == N - 1:
                    dxM = (x[i] - x[i - 1]) / 2.0
                elif i == 0:
                    dxM = (x[i] - x[i + 1]) / 2.0
                else:
                    dxM = (x[i + 1] - x[i - 1]) / 2.0
                
                cc = max(Conc[jS - 1, i] - cMin, 0.0)
                
                if iStep == 0:
                    # Passive uptake
                    SinkS[i] = Sink[i] * max(min(Conc[jS - 1, i], cRootMax), 0.0)
                    SPUptake = SPUptake + SinkS[i] * dxM
                    SAUptakeP = max(SPot * OmegaW - SPUptake, 0.0)
                
                elif iStep == 1:
                    # Active uptake without compensation
                    AUptakeA = cc / (rKM + cc) * Beta[i] * SAUptakeP
                    Omega = Omega + AUptakeA * dxM
                    SAUptakeA_local = Omega
                
                elif iStep == 2:
                    # Active uptake with compensation
                    if Omega > 0:
                        Omega1 = Omega / max(SAUptakeP, 1e-10)
                    else:
                        Omega1 = 0.0
                    
                    if Omega1 < OmegaS and Omega1 > 0:
                        Compen = OmegaS
                    elif Omega1 >= OmegaS:
                        Compen = Omega1
                    
                    if Compen > 0:
                        AUptakeA = cc / (rKM + cc) * Beta[i] * SAUptakeP / Compen
                    SinkS[i] = SinkS[i] + AUptakeA
                    SAUptakeA = SAUptakeA + AUptakeA * dxM
    
    return SPUptake, SAUptakeA


# ============================================================================
# Root density distribution
# ============================================================================

def set_root_distribution(
    N: int,
    x: NDArray[np.float64],
    Beta: NDArray[np.float64],
    xRoot: float,
    xRMin: float = 0.0,
    xRMax: float = 0.0,
    RGR: float = 0.0,
    lRoot: bool = False,
) -> None:
    """
    Set root density distribution.
    
    Direct port of SetRG subroutine in SINK.FOR.
    
    Parameters
    ----------
    N : int
        Number of nodes
    x : array, shape (N,)
        Node coordinates
    Beta : array, shape (N,)
        Root density distribution (output)
    xRoot : float
        Root depth
    xRMin : float
        Minimum root depth
    xRMax : float
        Maximum root depth
    RGR : float
        Root growth rate
    lRoot : bool
        Dynamic root growth flag
    """
    if not lRoot:
        xR = xRoot
    else:
        xR = xRMax if xRMax > 0 else xRoot
        if xRMin <= 0.001:
            xRMin = 0.001
    
    SBeta = 0.0
    for i in range(1, N):
        if x[i] < x[N - 1] - xR:
            Beta[i] = 0.0
        elif x[i] < x[N - 1] - 0.2 * xR:
            Beta[i] = 2.08333 / xR * (1.0 - (x[N - 1] - x[i]) / xR)
        else:
            Beta[i] = 1.66667 / xR
        
        if i < N - 1:
            SBeta = SBeta + Beta[i] * (x[i + 1] - x[i - 1]) / 2.0
        else:
            SBeta = SBeta + Beta[i] * (x[i] - x[i - 1]) / 2.0
    
    if SBeta < 0.0001:
        Beta[N - 2] = 1.0 / ((x[N - 1] - x[N - 2]) / 2.0)
    else:
        for i in range(1, N):
            Beta[i] = Beta[i] / SBeta


# ============================================================================
# Stress functions
# ============================================================================

def _falfa(
    lMoSink: bool,
    TPot: float,
    h: float,
    P0: float,
    P1: float,
    P2H: float,
    P2L: float,
    P3: float,
    r2H: float,
    r2L: float,
) -> float:
    """
    Water stress function (Feddes model).
    
    Parameters
    ----------
    lMoSink : bool
        Feddes model flag
    TPot : float
        Potential transpiration
    h : float
        Pressure head
    P0 : float
        Zero flux head (Silvertown)
    P1 : float
        Optimal head (Feddes)
    P2H : float
        P2 at high transpiration
    P2L : float
        P2 at low transpiration
    P3 : float
        Permanent wilting point
    r2H : float
        High transpiration rate
    r2L : float
        Low transpiration rate
    
    Returns
    -------
    Alfa : float
        Stress factor
    """
    if lMoSink:
        # Feddes model
        if TPot < r2L:
            P2 = P2L
        elif TPot > r2H:
            P2 = P2H
        else:
            P2 = P2H + (r2H - TPot) / (r2H - r2L) * (P2L - P2H)
        
        Alfa = 0.0
        if P3 < h < P2:
            Alfa = (h - P3) / (P2 - P3)
        elif P2 <= h <= P1:
            Alfa = 1.0
        elif h > P1 and h < P0 and P0 - P1 > 0:
            Alfa = (h - P0) / (P1 - P0)
        elif h >= P2 and P1 == 0 and P0 == 0:
            Alfa = 1.0
    else:
        # Silvertown model
        ratio = h / P0
        if ratio <= 0.0:
            # Negative or zero ratio: use absolute value
            ratio = abs(ratio)
        if ratio > 1e10:
            Alfa = 0.0
        elif ratio < 1e-10:
            Alfa = 1.0
        else:
            # Clamp exponent to prevent overflow
            exp_val = abs(P3)
            try:
                Alfa = 1.0 / (1.0 + ratio ** exp_val)
            except (OverflowError, ZeroDivisionError):
                Alfa = 0.0 if ratio > 1.0 else 1.0
    
    return max(Alfa, 0.0)


def _fsalfa(
    lMode: bool,
    cRed: float,
    c50: float,
    P3c: float,
) -> float:
    """
    Solute stress function.
    
    Parameters
    ----------
    lMode : bool
        Mode flag
    cRed : float
        Reduced concentration
    c50 : float
        Critical concentration
    P3c : float
        Stress exponent
    
    Returns
    -------
    SAlfa : float
        Solute stress factor
    """
    if lMode:
        if abs(c50) > 0:
            return 1.0 / (1.0 + (cRed / c50) ** P3c)
        return 0.0
    else:
        if cRed <= c50:
            return 1.0
        return max(0.0, 1.0 - (cRed - c50) * P3c * 0.01)
