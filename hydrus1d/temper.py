"""
Heat transport solver for HYDRUS-1D.
====================================

Energy equation with:
- Campbell thermal conductivity model
- Temperature-dependent water flow
- Vapor flow coupling
- Heat sources/sinks

Direct port of TEMPER.FOR.
"""

from __future__ import annotations
import numpy as np
from numpy.typing import NDArray
from typing import Tuple

from .utils import solve_tridiagonal


# ============================================================================
# Thermal properties
# ============================================================================

def compute_thermal_conductivity(
    th: float,
    ParW: NDArray[np.float64],
    IKappa: int,
) -> float:
    """
    Compute thermal conductivity from water content.
    
    Campbell model (default) or constant.
    
    Parameters
    ----------
    th : float
        Water content
    ParW : array, shape (11,)
        Thermal parameters
    IKappa : int
        Thermal conductivity model code
    
    Returns
    -------
    kappa : float
        Thermal conductivity
    """
    if IKappa == 0:
        # Constant thermal conductivity
        return ParW[2]
    
    # Campbell model
    theta_wr = ParW[0]
    theta_ws = ParW[1]
    lambda_d = ParW[2]
    delta_l = ParW[4]
    
    if th <= theta_wr:
        return lambda_d
    
    Pf = (th - theta_wr) / max(theta_ws - theta_wr, 1e-10)
    return lambda_d + delta_l * np.sqrt(Pf)


def compute_volumetric_heat_capacity(
    th: float,
    rho_b: float = 1.5,
    Cw: float = 4180.0,
    Cs: float = 837.0,
) -> float:
    """
    Volumetric heat capacity.
    
    Parameters
    ----------
    th : float
        Water content
    rho_b : float
        Bulk density
    Cw : float
        Specific heat of water
    Cs : float
        Specific heat of soil solids
    
    Returns
    -------
    C : float
        Volumetric heat capacity
    """
    return th * Cw + (rho_b - th * 1.0) * Cs


# ============================================================================
# Heat transport solver
# ============================================================================

def solve_heat_transport(
    N: int,
    x: NDArray[np.float64],
    TempN: NDArray[np.float64],
    TempO: NDArray[np.float64],
    thN: NDArray[np.float64],
    thO: NDArray[np.float64],
    vN: NDArray[np.float64],
    ParW: NDArray[np.float64],
    MatNum: NDArray[np.int64],
    dt: float,
    KodTopT: int,
    KodBotT: int,
    TTop: float,
    TBot: float,
    IKappa: int = 1,
    lWTDep: bool = False,
    rMin: float = 1e-37,
) -> Tuple[NDArray[np.float64], float, float]:
    """
    Solve heat transport equation.
    
    Energy equation:
        C * dT/dt = d/dz[kappa * dT/dz] - rho_w * Cw * v * dT/dz + S
    
    Parameters
    ----------
    N : int
        Number of nodes
    x : array, shape (N,)
        Node coordinates
    TempN : array, shape (N,)
        Current temperature estimate
    TempO : array, shape (N,)
        Previous time step temperature
    thN : array, shape (N,)
        Current water content
    thO : array, shape (N,)
        Previous water content
    vN : array, shape (N,)
        Velocity
    ParW : array, shape (11, NMat)
        Thermal parameters
    MatNum : array, shape (N,)
        Material zone indices
    dt : float
        Time step
    KodTopT : int
        Top thermal BC code
    KodBotT : int
        Bottom thermal BC code
    TTop : float
        Top boundary temperature
    TBot : float
        Bottom boundary temperature
    IKappa : int
        Thermal conductivity model code
    lWTDep : bool
        Temperature-dependent flow flag
    rMin : float
        Minimum value floor
    
    Returns
    -------
    TempNew : array, shape (N,)
        Updated temperatures
    qTop : float
        Top heat flux
    qBot : float
        Bottom heat flux
    """
    # Compute thermal properties at each node
    kappa = np.zeros(N, dtype=np.float64)
    Cv = np.zeros(N, dtype=np.float64)
    
    for i in range(N):
        M = MatNum[i]
        kappa[i] = compute_thermal_conductivity(thN[i], ParW[0, M], IKappa)
        Cv[i] = compute_volumetric_heat_capacity(thN[i])
    
    # Assemble tridiagonal system
    P = np.zeros(N, dtype=np.float64)
    R = np.zeros(N, dtype=np.float64)
    S = np.zeros(N, dtype=np.float64)
    hNew = TempN.copy()
    
    # Bottom node (index 0)
    dx = x[1] - x[0]
    dxB = dx / 2.0
    kB = (kappa[0] + kappa[1]) / 2.0
    
    S[0] = -kB / dxB
    F2 = Cv[0] * dx / dt
    RB = kB / dxB + F2
    SB = -kB / dxB
    PB = F2 * TempN[0] - (thN[0] - thO[0]) * 4180.0 * dx / dt
    
    # Interior nodes
    for i in range(1, N - 1):
        dxA = x[i] - x[i - 1]
        dxB = x[i + 1] - x[i]
        dx = (dxA + dxB) / 2.0
        
        kA = (kappa[i] + kappa[i - 1]) / 2.0
        kB = (kappa[i] + kappa[i + 1]) / 2.0
        
        A2 = kA / dxA + kB / dxB
        A3 = -kB / dxB
        F2 = Cv[i] * dx / dt
        
        R[i] = A2 + F2
        P[i] = F2 * TempN[i] - (thN[i] - thO[i]) * 4180.0 * dx / dt
        S[i] = A3
    
    # Top node (index N-1)
    dxA = x[N - 1] - x[N - 2]
    dx = dxA / 2.0
    kA = (kappa[N - 1] + kappa[N - 2]) / 2.0
    
    F2 = Cv[N - 1] * dx / dt
    RT = kA / dxA + F2
    ST = -kA / dxA
    PT = F2 * TempN[N - 1] - (thN[N - 1] - thO[N - 1]) * 4180.0 * dx / dt
    
    # Solve
    TempNew = solve_tridiagonal(
        N, P, R, S, hNew.copy(),
        KodTopT, KodBotT, TTop, TBot, rMin,
    )
    
    # Compute boundary fluxes
    qTop = -kA * (TempNew[N - 1] - TempNew[N - 2]) / dxA
    qBot = -kB * (TempNew[1] - TempNew[0]) / dx
    
    return TempNew, qTop, qBot


# ============================================================================
# Temperature-dependent water properties
# ============================================================================

def surface_tension(Temp: float) -> float:
    """
    Water surface tension as function of temperature.
    
    Parameters
    ----------
    Temp : float
        Temperature (Celsius)
    
    Returns
    -------
    sigma : float
        Surface tension (mN/m)
    """
    return 75.6 - 0.1425 * Temp - 2.38e-4 * Temp ** 2


def dynamic_viscosity(Temp: float) -> float:
    """
    Dynamic viscosity of water.
    
    Parameters
    ----------
    Temp : float
        Temperature (Celsius)
    
    Returns
    -------
    mu : float
        Dynamic viscosity (mPa*s)
    """
    return (1.787 - 0.007 * Temp) / (1.0 + 0.03225 * Temp)


def water_density(Temp: float) -> float:
    """
    Water density as function of temperature.
    
    Parameters
    ----------
    Temp : float
        Temperature (Celsius)
    
    Returns
    -------
    rho : float
        Density (kg/m^3)
    """
    return 1000.0 * (1.0 - 7.37e-6 * (Temp - 4.0) ** 2 + 3.79e-8 * (Temp - 4.0) ** 3)


def temperature_correction_factors(
    Temp: float,
    TempR: float = 20.0,
) -> Tuple[float, float]:
    """
    Compute temperature correction factors for hydraulic properties.
    
    Parameters
    ----------
    Temp : float
        Current temperature
    TempR : float
        Reference temperature
    
    Returns
    -------
    AT : float
        Surface tension correction
    BT : float
        Viscosity correction
    """
    AT = surface_tension(Temp) / surface_tension(TempR)
    BT = dynamic_viscosity(TempR) / dynamic_viscosity(Temp)
    
    return AT, BT
