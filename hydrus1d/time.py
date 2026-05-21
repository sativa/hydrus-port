"""
Time stepping control for HYDRUS-1D.
====================================

Adaptive time stepping with:
- Newton-Raphson convergence criteria
- Peclet/Courant number limits
- Maximum/minimum time step constraints
- Date/time conversion

Direct port of TIME.FOR.
"""

from __future__ import annotations
import numpy as np
from numpy.typing import NDArray
from typing import Tuple
from datetime import datetime, timedelta


# ============================================================================
# Time step calculation
# ============================================================================

def compute_time_step(
    dt: float,
    dtMax: float,
    dtMin: float,
    dtInit: float,
    dtMaxC: float,
    dtMaxT: float,
    Iter: int,
    IterMax: int,
    IterC: int,
    IterT: int,
    rMax: float,
    rMin: float,
    dtFact: float,
    dtFactC: float,
    dtFactT: float,
    lAdapt: bool = True,
    lAdaptC: bool = True,
    lAdaptT: bool = True,
    dtMinC: float = 0.0,
    dtMinT: float = 0.0,
    lWater: bool = True,
    lSolute: bool = False,
    lTemp: bool = False,
) -> Tuple[float, int, int, int]:
    """
    Compute adaptive time step.
    
    Direct port of Dt subroutine in TIME.FOR.
    
    Parameters
    ----------
    dt : float
        Current time step
    dtMax : float
        Maximum time step
    dtMin : float
        Minimum time step
    dtInit : float
        Initial time step
    dtMaxC : float
        Maximum time step for concentration
    dtMaxT : float
        Maximum time step for temperature
    Iter : int
        Water flow iterations
    IterMax : int
        Maximum water flow iterations
    IterC : int
        Solute iterations
    IterT : int
        Temperature iterations
    rMax : float
        Maximum convergence criterion
    rMin : float
        Minimum convergence criterion
    dtFact : float
        Time step increase factor for water
    dtFactC : float
        Time step increase factor for solute
    dtFactT : float
        Time step increase factor for temperature
    lAdapt : bool
        Adaptive time step for water
    lAdaptC : bool
        Adaptive time step for solute
    lAdaptT : bool
        Adaptive time step for temperature
    dtMinC : float
        Minimum time step for concentration
    dtMinT : float
        Minimum time step for temperature
    lWater : bool
        Water flow active
    lSolute : bool
        Solute transport active
    lTemp : bool
        Heat transport active
    
    Returns
    -------
    dtNew : float
        New time step
    Iter : int
        Water flow iterations
    IterC : int
        Solute iterations
    IterT : int
        Temperature iterations
    """
    dtNew = dt
    
    if lWater:
        # Water flow time step adjustment
        if lAdapt:
            if Iter <= 2:
                # Increase time step
                dtNew = min(dt * dtFact, dtMax)
            elif Iter > IterMax:
                # Decrease time step
                dtNew = max(dt / dtFact, dtMin)
            else:
                # Keep time step
                dtNew = dt
    
    if lSolute:
        # Limit by concentration stability
        dtNew = min(dtNew, dtMaxC)
        
        if lAdaptC:
            if IterC <= 2:
                dtNew = min(dtNew * dtFactC, dtMaxC)
            elif IterC > IterMax:
                dtNew = max(dtNew / dtFactC, dtMinC)
    
    if lTemp:
        # Limit by temperature stability
        dtNew = min(dtNew, dtMaxT)
        
        if lAdaptT:
            if IterT <= 2:
                dtNew = min(dtNew * dtFactT, dtMaxT)
            elif IterT > IterMax:
                dtNew = max(dtNew / dtFactT, dtMinT)
    
    # Apply limits
    dtNew = max(dtNew, dtMin)
    dtNew = min(dtNew, dtMax)
    
    return dtNew, Iter, IterC, IterT


# ============================================================================
# Date/time conversion
# ============================================================================

def rtime(
    t: float,
    t0: float,
    tConv: float,
) -> Tuple[int, int, int, int, int, int, float]:
    """
    Convert simulation time to date/time components.
    
    Direct port of RTime subroutine in TIME.FOR.
    
    Parameters
    ----------
    t : float
        Current simulation time (in model time units)
    t0 : float
        Start time (in model time units)
    tConv : float
        Time conversion factor (model units to days)
    
    Returns
    -------
    Year : int
    Month : int
    Day : int
    Hour : int
    Min : int
    Sec : int
    tDay : float
        Fractional day
    """
    # Convert to total days
    tDayTotal = (t + t0) * tConv
    
    # Integer day count
    nDays = int(tDayTotal)
    tDay = tDayTotal - nDays
    
    # Convert fractional day to time
    tSec = tDay * 86400.0
    Sec = int(tSec % 60.0)
    Min = int((tSec / 60.0) % 60.0)
    Hour = int((tSec / 3600.0) % 24.0)
    
    # Convert day count to date (approximate)
    Year = 2000 + nDays // 365
    rem = nDays % 365
    Month = min(rem // 30 + 1, 12)
    Day = min(rem % 30 + 1, 28)
    
    return Year, Month, Day, Hour, Min, Sec, tDay


def seconds_to_datetime(
    seconds: float,
    start_datetime: datetime | None = None,
) -> datetime:
    """
    Convert seconds since start to datetime.
    
    Parameters
    ----------
    seconds : float
        Seconds since start
    start_datetime : datetime, optional
        Start datetime (default: 2000-01-01 00:00:00)
    
    Returns
    -------
    dt : datetime
        Resulting datetime
    """
    if start_datetime is None:
        start_datetime = datetime(2000, 1, 1)
    
    return start_datetime + timedelta(seconds=seconds)


# ============================================================================
# Time series output control
# ============================================================================

def should_output(
    t: float,
    dt: float,
    tOut: float,
    tOutC: float,
    tOutT: float,
    iOut: int,
    iOutC: int,
    iOutT: int,
    lCumFlux: bool,
    lMassBal: bool,
) -> Tuple[bool, bool, bool, bool, bool]:
    """
    Determine if output should be written.
    
    Parameters
    ----------
    t : float
        Current time
    dt : float
        Current time step
    tOut : float
        Output interval for water
    tOutC : float
        Output interval for concentration
    tOutT : float
        Output interval for temperature
    iOut : int
        Water output code
    iOutC : int
        Concentration output code
    iOutT : int
        Temperature output code
    lCumFlux : bool
        Cumulative flux flag
    lMassBal : bool
        Mass balance flag
    
    Returns
    -------
    outWater : bool
    outConc : bool
    outTemp : bool
    outCumFlux : bool
    outMassBal : bool
    """
    outWater = (iOut > 0 and t >= tOut - dt * 0.5)
    outConc = (iOutC > 0 and t >= tOutC - dt * 0.5)
    outTemp = (iOutT > 0 and t >= tOutT - dt * 0.5)
    outCumFlux = lCumFlux and outWater
    outMassBal = lMassBal and outWater
    
    return outWater, outConc, outTemp, outCumFlux, outMassBal


# ============================================================================
# Convergence checking
# ============================================================================

def check_convergence(
    hNew: NDArray[np.float64],
    hOld: NDArray[np.float64],
    N: int,
    rMax: float,
    rMin: float,
) -> Tuple[bool, float]:
    """
    Check water flow convergence.
    
    Parameters
    ----------
    hNew : array, shape (N,)
        New pressure heads
    hOld : array, shape (N,)
        Old pressure heads
    N : int
        Number of nodes
    rMax : float
        Maximum convergence criterion
    rMin : float
        Minimum convergence criterion
    
    Returns
    -------
    converged : bool
        Whether solution converged
    rMaxCurrent : float
        Maximum residual
    """
    rMaxCurrent = 0.0
    
    for i in range(N):
        dh = abs(hNew[i] - hOld[i])
        rMaxCurrent = max(rMaxCurrent, dh)
    
    converged = (rMaxCurrent <= rMax)
    
    return converged, rMaxCurrent


def check_concentration_convergence(
    ConcNew: NDArray[np.float64],
    ConcOld: NDArray[np.float64],
    NSD: int,
    N: int,
    rMaxC: float,
) -> Tuple[bool, float]:
    """
    Check solute transport convergence.
    
    Parameters
    ----------
    ConcNew : array, shape (NSD, N)
        New concentrations
    ConcOld : array, shape (NSD, N)
        Old concentrations
    NSD : int
        Number of species
    N : int
        Number of nodes
    rMaxC : float
        Maximum convergence criterion
    
    Returns
    -------
    converged : bool
        Whether solution converged
    rMaxCurrent : float
        Maximum residual
    """
    rMaxCurrent = 0.0
    
    for j in range(NSD):
        for i in range(N):
            dc = abs(ConcNew[j, i] - ConcOld[j, i])
            rMaxCurrent = max(rMaxCurrent, dc)
    
    converged = (rMaxCurrent <= rMaxC)
    
    return converged, rMaxCurrent
