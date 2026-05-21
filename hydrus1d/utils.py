"""
Utility functions for HYDRUS-1D Python port.
=============================================

Numerical helpers including:
- Tridiagonal matrix solver (Gauss elimination / Thomas algorithm)
- Banbury solver (block tridiagonal)
- Unit conversions
- Mathematical helpers matching Fortran intrinsics
"""

from __future__ import annotations
import numpy as np
from numpy.typing import NDArray
from typing import Tuple


# ============================================================================
# Tridiagonal Solver (Gauss elimination)
# ============================================================================

def solve_tridiagonal(
    N: int,
    P: NDArray[np.float64],
    R: NDArray[np.float64],
    S: NDArray[np.float64],
    hNew: NDArray[np.float64],
    KodTop: int,
    KodBot: int,
    hTop: float,
    hBot: float,
    rMin: float = 1e-100,
    PB: float = 0.0,
    RB: float = 1.0,
    SB_coef: float = 0.0,
    PT: float = 0.0,
    RT: float = 1.0,
    ST: float = 0.0,
) -> NDArray[np.float64]:
    """
    Solve tridiagonal system using Gauss elimination.
    
    Direct Python equivalent of the Gauss subroutine in WATFLOW.FOR.
    
    Uses boundary coefficients PB,RB,SB (bottom) and PT,RT,ST (top)
    from the Reset subroutine for head BCs.
    
    Parameters
    ----------
    N : int
        Number of nodes (equations)
    P : array, shape (N,)
        Lower diagonal (P[i] multiplies h[i-1])
    R : array, shape (N,)
        Main diagonal (R[i] multiplies h[i])
    S : array, shape (N,)
        Upper diagonal (S[i] multiplies h[i+1])
    hNew : array, shape (N,)
        Right-hand side on input; solution on output
    KodTop : int
        Top BC code: >=0 flux, <0 head
    KodBot : int
        Bottom BC code: >=0 flux, <0 head
    hTop : float
        Top boundary head/flux value
    hBot : float
        Bottom boundary head/flux value
    rMin : float
        Minimum diagonal element to prevent division by zero
    PB, RB, SB_coef : float
        Bottom boundary coefficients from Reset
    PT, RT, ST : float
        Top boundary coefficients from Reset
    
    Returns
    -------
    hNew_out : array, shape (N,)
        Solution vector (pressure heads at all nodes)
    
    Notes
    -----
    Fortran: nodes 1..N (1=bottom, N=top)
    Python:  0..N-1 (0=bottom, N-1=top)
    
    Mapping:
        Fortran P(i),R(i),S(i) -> Python P[i-1],R[i-1],S[i-1]
        Fortran node 1 -> Python index 0
        Fortran node N -> Python index N-1
    """
    # Work on copies (avoid mutating caller's arrays).  Convert P/R/S to
    # plain Python lists for tight scalar arithmetic — numpy ndarray
    # element access goes through __getitem__ each time and is ~4× slower
    # than a list lookup in the Thomas algorithm's tight inner loops.
    Pw = P.tolist()
    Rw = R.tolist()
    Sw = S.tolist()

    # Forward elimination — first eliminate the boundary node.
    if KodBot >= 0:
        Pw[1] = Pw[1] - Sw[0] * hBot
    else:
        if abs(RB) < rMin:
            RB = rMin if RB >= 0 else -rMin
        ratio = Sw[0] / RB
        Pw[1] = Pw[1] - PB * ratio
        Rw[1] = Rw[1] - SB_coef * ratio

    # Interior forward elimination — purely sequential.  We explicitly
    # cache R_prev and S_prev to avoid two list lookups per step.
    R_prev = Rw[1]
    S_prev = Sw[1]
    P_prev = Pw[1]
    for i in range(2, N - 1):
        if R_prev > -rMin and R_prev < rMin:
            R_prev = rMin if R_prev >= 0 else -rMin
            Rw[i - 1] = R_prev
        S_im1 = Sw[i - 1]
        m = S_im1 / R_prev
        Pw[i] = Pw[i] - P_prev * m
        Rw[i] = Rw[i] - S_im1 * m
        R_prev = Rw[i]
        P_prev = Pw[i]

    # Top BC.
    if KodTop > 0:
        Pw[N - 2] = Pw[N - 2] - Sw[N - 2] * hTop
    else:
        rN2 = Rw[N - 2]
        if -rMin < rN2 < rMin:
            rN2 = rMin if rN2 >= 0 else -rMin
            Rw[N - 2] = rN2
        ratio = ST / rN2
        Pw[N - 1] = PT - Pw[N - 2] * ratio
        Rw[N - 1] = RT - Sw[N - 2] * ratio

    # Back substitution.
    hNew_out = np.empty(N, dtype=np.float64)
    rN2 = Rw[N - 2]
    if -rMin < rN2 < rMin:
        rN2 = rMin if rN2 >= 0 else -rMin
        Rw[N - 2] = rN2

    if KodTop > 0:
        hNew_out[N - 1] = hTop
        hNew_out[N - 2] = Pw[N - 2] / rN2
    else:
        rN1 = Rw[N - 1]
        if -rMin < rN1 < rMin:
            rN1 = rMin if rN1 >= 0 else -rMin
        h_top = Pw[N - 1] / rN1
        hNew_out[N - 1] = h_top
        hNew_out[N - 2] = (Pw[N - 2] - Sw[N - 2] * h_top) / rN2

    # Interior back substitution.  Carry h_next in a local for speed.
    h_next = hNew_out[N - 2]
    for i in range(N - 3, 0, -1):
        r = Rw[i]
        if -rMin < r < rMin:
            r = rMin if r >= 0 else -rMin
        h_next = (Pw[i] - Sw[i] * h_next) / r
        hNew_out[i] = h_next

    # Bottom node.
    if KodBot >= 0:
        hNew_out[0] = hBot
    else:
        if -rMin < RB < rMin:
            RB = rMin if RB >= 0 else -rMin
        hNew_out[0] = (PB - SB_coef * hNew_out[1]) / RB

    return hNew_out


def solve_banbury(
    N: int,
    B: NDArray[np.float64],
    D: NDArray[np.float64],
    E: NDArray[np.float64],
    F: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Banbury solver for tridiagonal system.
    
    Solves B*D*E*F where:
        B = lower diagonal
        D = main diagonal
        E = upper diagonal
        F = right-hand side (input) / solution (output)
    
    This is the direct Python equivalent of the BanSol subroutine in SOLUTE.FOR.
    Uses the standard Thomas algorithm (TDMA).
    
    Parameters
    ----------
    N : int
        Number of equations
    B : array, shape (N,)
        Lower diagonal
    D : array, shape (N,)
        Main diagonal
    E : array, shape (N,)
        Upper diagonal
    F : array, shape (N,)
        Right-hand side on input; solution on output
    
    Returns
    -------
    F : array, shape (N,)
        Solution vector
    """
    # Work on copies
    DN = D.copy()
    FN = F.copy()
    
    # Forward elimination
    for i in range(1, N):
        if abs(DN[i - 1]) < 1e-100:
            factor = 0.0
        else:
            factor = B[i] / DN[i - 1]
        DN[i] = DN[i] - factor * E[i - 1]
        FN[i] = FN[i] - factor * FN[i - 1]
    
    # Back substitution
    FN[N - 1] = FN[N - 1] / DN[N - 1]
    for i in range(N - 2, -1, -1):
        FN[i] = (FN[i] - E[i] * FN[i + 1]) / DN[i]
    
    return FN


# ============================================================================
# Unit Conversions
# ============================================================================

def get_length_conversion(LUnit: str) -> float:
    """
    Get length conversion factor from meters to simulation unit.
    
    Matches the Conversion subroutine in INPUT.FOR.
    
    Parameters
    ----------
    LUnit : str
        Length unit string: 'm   ', 'cm  ', 'mm  '
    
    Returns
    -------
    xConv : float
        Conversion factor (1.0 for m, 100.0 for cm, 1000.0 for mm)
    """
    unit = LUnit.strip().lower()
    if unit == 'cm':
        return 100.0
    elif unit == 'mm':
        return 1000.0
    else:  # 'm'
        return 1.0


def get_time_conversion(TUnit: str) -> float:
    """
    Get time conversion factor from seconds to simulation unit.
    
    Matches the Conversion subroutine in INPUT.FOR.
    
    Parameters
    ----------
    TUnit : str
        Time unit string: 's   ', 'min ', 'hours', 'days', 'years'
    
    Returns
    -------
    tConv : float
        Conversion factor
    """
    unit = TUnit.strip().lower()
    if unit == 'min':
        return 1.0 / 60.0
    elif unit == 'hours':
        return 1.0 / (60.0 * 60.0)
    elif unit == 'days':
        return 1.0 / (60.0 * 60.0 * 24.0)
    elif unit == 'years':
        return 1.0 / (60.0 * 60.0 * 24.0 * 365.0)
    else:  # 's'
        return 1.0


def get_mass_conversion(MUnit: str) -> float:
    """
    Get mass conversion factor.
    
    Parameters
    ----------
    MUnit : str
        Mass unit string: 'mol ', 'mmol', 'mg  ', 'g   '
    
    Returns
    -------
    mConv : float
        Conversion factor
    """
    unit = MUnit.strip().lower()
    if unit == 'mmol':
        return 1000.0
    elif unit == 'mg':
        return 1000000.0
    elif unit == 'g':
        return 1000.0
    else:  # 'mol'
        return 1.0


def conversion_factors(LUnit: str, TUnit: str, MUnit: str = 'mol   ') -> Tuple[float, float, float]:
    """
    Get all conversion factors.
    
    Parameters
    ----------
    LUnit : str
        Length unit
    TUnit : str
        Time unit
    MUnit : str
        Mass unit
    
    Returns
    -------
    xConv, tConv, mConv : tuple of floats
        Conversion factors
    """
    return (
        get_length_conversion(LUnit),
        get_time_conversion(TUnit),
        get_mass_conversion(MUnit),
    )


# ============================================================================
# Mathematical Helpers
# ============================================================================

def fortran_amax1(*args: float) -> float:
    """Fortran AMAX1 equivalent - maximum of arguments."""
    return max(args)


def fortran_amin1(*args: float) -> float:
    """Fortran AMIN1 equivalent - minimum of arguments."""
    return min(args)


def fortran_sign(mag: float, sign_val: float) -> float:
    """Fortran SIGN equivalent - mag with sign of sign_val."""
    return mag * np.sign(sign_val) if sign_val != 0 else 0.0


def safe_log(x: float, min_val: float = 1e-300) -> float:
    """Safe logarithm that prevents log(0) or log(negative)."""
    return np.log(max(x, min_val))


def safe_power(base: float, exp: float, min_base: float = 1e-300) -> float:
    """Safe power that prevents 0^negative."""
    if base <= 0:
        base = min_base
    return base ** exp


def clamp(value: float, lo: float, hi: float) -> float:
    """Clamp value to [lo, hi]."""
    return max(lo, min(hi, value))


# ============================================================================
# Date/Time Utilities (matching RTime in TIME.FOR)
# ============================================================================

def rtime(
    iMonth: int,
    iDay: int,
    iHours: int,
    iMins: int,
    iSecs: int,
    i100th: int = 0,
) -> float:
    """
    Convert date/time components to seconds from start of month.
    
    Matches RTime function in TIME.FOR.
    
    Parameters
    ----------
    iMonth : int
        Month (1-12)
    iDay : int
        Day of month (0-31, 0 = beginning of month)
    iHours : int
        Hours (0-23)
    iMins : int
        Minutes (0-59)
    iSecs : int
        Seconds (0-59)
    i100th : int
        Hundredths of seconds
    
    Returns
    -------
    time_seconds : float
        Time in seconds from start of month
    """
    # Days in month (non-leap year, matching Fortran)
    if iMonth in (1, 3, 5, 7, 8, 10, 12):
        NoDay = 31
    elif iMonth in (4, 6, 9, 11):
        NoDay = 30
    else:  # February
        NoDay = 28
    
    nMonth = NoDay * 24.0 * 60.0 * 60.0
    return (
        nMonth
        + iDay * 24.0 * 60.0 * 60.0
        + iHours * 60.0 * 60.0
        + iMins * 60.0
        + iSecs
        + i100th / 100.0
    )


def seconds_to_datetime(t: float) -> Tuple[int, int, int, int, int, int]:
    """
    Convert seconds to date/time components.
    
    Inverse of rtime (approximate, for output formatting).
    
    Parameters
    ----------
    t : float
        Time in seconds
    
    Returns
    -------
    (day, hours, mins, secs, hundredths, fraction) : tuple
    """
    total_secs = int(t)
    day = total_secs // 86400
    remainder = total_secs - day * 86400
    hours = remainder // 3600
    remainder = remainder - hours * 3600
    mins = remainder // 60
    secs = remainder - mins * 60
    hundredths = int((t - int(t)) * 100)
    return (day, hours, mins, secs, hundredths, 0)


# ============================================================================
# Profile Utilities
# ============================================================================

def interpolate_profile(
    x_new: float,
    x_tab: NDArray[np.float64],
    y_tab: NDArray[np.float64],
    n_tab: int,
    constant_below: bool = False,
    constant_above: bool = True,
) -> float:
    """
    Linear interpolation with optional constant extrapolation.
    
    Matches the interpolation logic used throughout HYDRUS for
    tabular K-S-P relationships and initial conditions.
    
    Parameters
    ----------
    x_new : float
        x value at which to interpolate
    x_tab : array
        x values of table (sorted)
    y_tab : array
        y values of table
    n_tab : int
        Number of valid table entries
    constant_below : bool
        If True, return y_tab[0] for x < x_tab[0]
    constant_above : bool
        If True, return y_tab[n_tab-1] for x > x_tab[n_tab-1]
    
    Returns
    -------
    y : float
        Interpolated y value
    """
    if n_tab < 1:
        return 0.0
    
    if x_new <= x_tab[0]:
        return y_tab[0] if constant_below else y_tab[0]
    
    if x_new >= x_tab[n_tab - 1]:
        return y_tab[n_tab - 1] if constant_above else y_tab[n_tab - 1]
    
    # Find bracketing interval
    for i in range(n_tab - 1):
        if x_tab[i] <= x_new <= x_tab[i + 1]:
            if x_tab[i + 1] == x_tab[i]:
                return y_tab[i]
            frac = (x_new - x_tab[i]) / (x_tab[i + 1] - x_tab[i])
            return y_tab[i] + frac * (y_tab[i + 1] - y_tab[i])
    
    return y_tab[n_tab - 1]


# ============================================================================
# Peclet and Courant Number Utilities
# ============================================================================

def compute_peclet(
    v: float,
    dx: float,
    D: float,
) -> float:
    """
    Compute cell Peclet number.
    
    Parameters
    ----------
    v : float
        Velocity
    dx : float
        Cell size
    D : float
        Dispersion coefficient
    
    Returns
    -------
    Pe : float
        Peclet number (|v|*dx/D)
    """
    if D <= 0:
        return 1e30
    return abs(v) * dx / (2.0 * D)


def compute_courant(
    v: float,
    dx: float,
    dt: float,
) -> float:
    """
    Compute cell Courant number.
    
    Parameters
    ----------
    v : float
        Velocity
    dx : float
        Cell size
    dt : float
        Time step
    
    Returns
    -------
    Co : float
        Courant number (|v|*dt/dx)
    """
    if dx <= 0:
        return 1e30
    return abs(v) * dt / dx
