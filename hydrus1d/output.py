"""
Output formatting for HYDRUS-1D.
================================

Output file generation:
- Profile.out  (concentration profiles)
- CumFlux.out  (cumulative fluxes)
- MassBal.out  (mass balance)
- Time.out     (time series)
- Node.out     (node data)

Direct port of OUTPUT.FOR.
"""

from __future__ import annotations
import os
import numpy as np
from numpy.typing import NDArray
from typing import List, Optional, Dict, Any
from datetime import datetime


# ============================================================================
# Profile output
# ============================================================================

def write_profile_output(
    filename: str,
    t: float,
    N: int,
    x: NDArray[np.float64],
    hNew: NDArray[np.float64],
    thNew: NDArray[np.float64],
    Con: NDArray[np.float64],
    vN: NDArray[np.float64],
    Conc: NDArray[np.float64],
    TempN: NDArray[np.float64] | None = None,
    NSD: int = 0,
    lTemp: bool = False,
    xConv: float = 1.0,
    tConv: float = 1.0,
    iOut: int = 0,
    lCumFlux: bool = False,
    CumQ: float = 0.0,
    CumE: float = 0.0,
    CumT: float = 0.0,
    CumQCh: NDArray[np.float64] | None = None,
    lMassBal: bool = False,
    MassOld: NDArray[np.float64] | None = None,
    MassNew: NDArray[np.float64] | None = None,
    MassFlux: NDArray[np.float64] | None = None,
    MassBal: NDArray[np.float64] | None = None,
) -> None:
    """
    Write profile output file.
    
    Direct port of OutPrf subroutine in OUTPUT.FOR.
    
    Parameters
    ----------
    filename : str
        Output file path
    t : float
        Current simulation time
    N : int
        Number of nodes
    x : array, shape (N,)
        Node coordinates
    hNew : array, shape (N,)
        Pressure heads
    thNew : array, shape (N,)
        Water contents
    Con : array, shape (N,)
        Hydraulic conductivities
    vN : array, shape (N,)
        Velocities
    Conc : array, shape (NSD, N)
        Concentrations
    TempN : array, shape (N,), optional
        Temperatures
    NSD : int
        Number of chemical species
    lTemp : bool
        Temperature output flag
    xConv : float
        Length conversion factor
    tConv : float
        Time conversion factor
    iOut : int
        Output format code
    lCumFlux : bool
        Cumulative flux flag
    CumQ : float
        Cumulative bottom flux
    CumE : float
        Cumulative evaporation
    CumT : float
        Cumulative transpiration
    CumQCh : array, shape (NSD,), optional
        Cumulative chemical fluxes
    lMassBal : bool
        Mass balance flag
    MassOld : array, shape (NSD,), optional
        Old mass
    MassNew : array, shape (NSD,), optional
        New mass
    MassFlux : array, shape (NSD,), optional
        Mass flux
    MassBal : array, shape (NSD,), optional
        Mass balance error
    """
    with open(filename, 'a') as f:
        if iOut == 0:
            # Compact format
            f.write(f"  t = {t * tConv:14.6e}\n")
            for i in range(N):
                line = f"   x = {x[i] * xConv:10.4f}  h = {hNew[i]:12.6e}  "
                line += f"theta = {thNew[i]:10.6f}  K = {Con[i]:12.6e}  "
                line += f"v = {vN[i]:12.6e}"
                if NSD > 0:
                    for j in range(NSD):
                        line += f"  c{j+1} = {Conc[j, i]:12.6e}"
                if lTemp and TempN is not None:
                    line += f"  T = {TempN[i]:10.4f}"
                f.write(line + "\n")
        elif iOut == 1:
            # Full format
            f.write(f"  Time = {t * tConv:14.6e}\n")
            f.write(f"  {'x':>10}  {'h':>14}  {'theta':>10}  {'K':>14}  {'v':>14}")
            if NSD > 0:
                for j in range(NSD):
                    f.write(f"  {'c{j+1}':>14}")
            if lTemp:
                f.write(f"  {'T':>10}")
            f.write("\n")
            for i in range(N):
                line = f"  {x[i] * xConv:10.4f}  {hNew[i]:14.6e}  {thNew[i]:10.6f}  "
                line += f"{Con[i]:14.6e}  {vN[i]:14.6e}"
                if NSD > 0:
                    for j in range(NSD):
                        line += f"  {Conc[j, i]:14.6e}"
                if lTemp and TempN is not None:
                    line += f"  {TempN[i]:10.4f}"
                f.write(line + "\n")
        
        # Cumulative fluxes
        if lCumFlux:
            f.write(f"  CumQ = {CumQ:14.6e}  CumE = {CumE:14.6e}  CumT = {CumT:14.6e}\n")
            if CumQCh is not None:
                for j in range(NSD):
                    f.write(f"  CumQCh{j+1} = {CumQCh[j]:14.6e}\n")
        
        # Mass balance
        if lMassBal and MassOld is not None and MassNew is not None:
            for j in range(NSD):
                f.write(f"  Mass{j+1}: old={MassOld[j]:12.6e}  new={MassNew[j]:12.6e}  "
                        f"flux={MassFlux[j]:12.6e}  bal={MassBal[j]:12.6e}\n")


# ============================================================================
# Time series output
# ============================================================================

def write_time_series_output(
    filename: str,
    t: float,
    vTop: float,
    vBot: float,
    thTop: float,
    thBot: float,
    hTop: float,
    hBot: float,
    SinkTop: float,
    vTopCh: NDArray[np.float64] | None = None,
    vBotCh: NDArray[np.float64] | None = None,
    qTopT: float = 0.0,
    qBotT: float = 0.0,
    NSD: int = 0,
    lTemp: bool = False,
    lSink: bool = False,
    xConv: float = 1.0,
    tConv: float = 1.0,
) -> None:
    """
    Write time series output file.
    
    Direct port of OutTm subroutine in OUTPUT.FOR.
    
    Parameters
    ----------
    filename : str
        Output file path
    t : float
        Current simulation time
    vTop : float
        Top flux
    vBot : float
        Bottom flux
    thTop : float
        Top water content
    thBot : float
        Bottom water content
    hTop : float
        Top pressure head
    hBot : float
        Bottom pressure head
    SinkTop : float
        Root uptake at top
    vTopCh : array, shape (NSD,), optional
        Top chemical fluxes
    vBotCh : array, shape (NSD,), optional
        Bottom chemical fluxes
    qTopT : float
        Top heat flux
    qBotT : float
        Bottom heat flux
    NSD : int
        Number of species
    lTemp : bool
        Temperature output flag
    lSink : bool
        Sink output flag
    xConv : float
        Length conversion
    tConv : float
        Time conversion
    """
    with open(filename, 'a') as f:
        line = f"  {t * tConv:14.6e}  {vTop:12.6e}  {vBot:12.6e}  "
        line += f"{thTop:10.6f}  {thBot:10.6f}  {hTop:12.6e}  {hBot:12.6e}"
        
        if lSink:
            line += f"  {SinkTop:12.6e}"
        
        if NSD > 0 and vTopCh is not None:
            for j in range(NSD):
                line += f"  {vTopCh[j]:12.6e}  {vBotCh[j]:12.6e}"
        
        if lTemp:
            line += f"  {qTopT:12.6e}  {qBotT:12.6e}"
        
        f.write(line + "\n")


def write_time_series_header(
    filename: str,
    NSD: int = 0,
    lTemp: bool = False,
    lSink: bool = False,
) -> None:
    """Write time series output header."""
    with open(filename, 'w') as f:
        f.write(f"  {'t':>14}  {'vTop':>12}  {'vBot':>12}  "
                f"{'thTop':>10}  {'thBot':>10}  {'hTop':>12}  {'hBot':>12}")
        
        if lSink:
            f.write(f"  {'SinkTop':>12}")
        
        if NSD > 0:
            for j in range(NSD):
                f.write(f"  {'vTopCh{j+1}':>12}  {'vBotCh{j+1}':>12}")
        
        if lTemp:
            f.write(f"  {'qTopT':>12}  {'qBotT':>12}")
        
        f.write("\n")


# ============================================================================
# Cumulative flux output
# ============================================================================

def write_cumulative_flux(
    filename: str,
    t: float,
    CumQ: float,
    CumE: float,
    CumT: float,
    CumQCh: NDArray[np.float64] | None = None,
    NSD: int = 0,
    tConv: float = 1.0,
) -> None:
    """
    Write cumulative flux output file.
    
    Direct port of OutCum subroutine in OUTPUT.FOR.
    
    Parameters
    ----------
    filename : str
        Output file path
    t : float
        Current simulation time
    CumQ : float
        Cumulative bottom flux
    CumE : float
        Cumulative evaporation
    CumT : float
        Cumulative transpiration
    CumQCh : array, shape (NSD,), optional
        Cumulative chemical fluxes
    NSD : int
        Number of species
    tConv : float
        Time conversion
    """
    with open(filename, 'a') as f:
        f.write(f"  {t * tConv:14.6e}  {CumQ:14.6e}  {CumE:14.6e}  {CumT:14.6e}")
        if CumQCh is not None:
            for j in range(NSD):
                f.write(f"  {CumQCh[j]:14.6e}")
        f.write("\n")


# ============================================================================
# Mass balance output
# ============================================================================

def compute_mass_balance(
    N: int,
    x: NDArray[np.float64],
    Conc: NDArray[np.float64],
    thN: NDArray[np.float64],
    thO: NDArray[np.float64],
    vTop: float,
    vBot: float,
    cTop: NDArray[np.float64],
    cBot: NDArray[np.float64],
    Sink: NDArray[np.float64],
    SinkS: NDArray[np.float64],
    dt: float,
) -> Dict[str, NDArray[np.float64]]:
    """
    Compute mass balance for all species.
    
    Parameters
    ----------
    N : int
        Number of nodes
    x : array, shape (N,)
        Node coordinates
    Conc : array, shape (NSD, N)
        Concentrations
    thN : array, shape (N,)
        New water content
    thO : array, shape (N,)
        Old water content
    vTop : float
        Top flux
    vBot : float
        Bottom flux
    cTop : array, shape (NSD,)
        Top concentration
    cBot : array, shape (NSD,)
        Bottom concentration
    Sink : array, shape (N,)
        Root water uptake
    SinkS : array, shape (N,)
        Root solute uptake
    dt : float
        Time step
    
    Returns
    -------
    balance : dict
        Mass balance components
    """
    NSD = Conc.shape[0]
    
    # Element widths
    dx = np.zeros(N, dtype=np.float64)
    for i in range(N):
        if i == 0:
            dx[i] = (x[1] - x[0]) / 2.0
        elif i == N - 1:
            dx[i] = (x[i] - x[i - 1]) / 2.0
        else:
            dx[i] = (x[i + 1] - x[i - 1]) / 2.0
    
    # Mass in each element
    MassNew = np.zeros(NSD, dtype=np.float64)
    MassOld = np.zeros(NSD, dtype=np.float64)
    
    for j in range(NSD):
        for i in range(N):
            MassNew[j] += Conc[j, i] * thN[i] * dx[i]
            MassOld[j] += Conc[j, i] * thO[i] * dx[i]
    
    # Fluxes
    FluxTop = cTop * vTop * dt
    FluxBot = cBot * vBot * dt
    
    # Root uptake
    Uptake = np.zeros(NSD, dtype=np.float64)
    for j in range(NSD):
        for i in range(N):
            Uptake[j] += SinkS[i] * dx[i]
    
    # Balance
    MassFlux = FluxTop - FluxBot - Uptake
    MassBal = MassNew - MassOld - MassFlux
    
    return {
        'MassOld': MassOld,
        'MassNew': MassNew,
        'FluxTop': FluxTop,
        'FluxBot': FluxBot,
        'Uptake': Uptake,
        'MassFlux': MassFlux,
        'MassBal': MassBal,
    }


# ============================================================================
# Initialize output files
# ============================================================================

def initialize_output_files(
    output_dir: str,
    NSD: int = 0,
    lTemp: bool = False,
    lSink: bool = False,
    lCumFlux: bool = False,
    lMassBal: bool = False,
) -> Dict[str, str]:
    """
    Initialize all output files.
    
    Parameters
    ----------
    output_dir : str
        Output directory
    NSD : int
        Number of species
    lTemp : bool
        Temperature output flag
    lSink : bool
        Sink output flag
    lCumFlux : bool
        Cumulative flux flag
    lMassBal : bool
        Mass balance flag
    
    Returns
    -------
    files : dict
        File paths
    """
    os.makedirs(output_dir, exist_ok=True)
    
    files = {
        'profile': os.path.join(output_dir, 'Profile.out'),
        'time': os.path.join(output_dir, 'Time.out'),
        'cumflux': os.path.join(output_dir, 'CumFlux.out'),
        'massbal': os.path.join(output_dir, 'MassBal.out'),
    }
    
    # Write headers
    write_time_series_header(files['time'], NSD, lTemp, lSink)
    
    if lCumFlux:
        with open(files['cumflux'], 'w') as f:
            f.write(f"  {'t':>14}  {'CumQ':>14}  {'CumE':>14}  {'CumT':>14}")
            if NSD > 0:
                for j in range(NSD):
                    f.write(f"  {'CumQCh{j+1}':>14}")
            f.write("\n")
    
    if lMassBal:
        with open(files['massbal'], 'w') as f:
            f.write(f"  {'t':>14}")
            for j in range(NSD):
                f.write(f"  {'M{j+1}old':>12}  {'M{j+1}new':>12}  "
                        f"{'Fl{j+1}':>12}  {'B{j+1}':>12}")
            f.write("\n")
    
    return files
