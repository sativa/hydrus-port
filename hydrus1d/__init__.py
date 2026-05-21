"""
HYDRUS-1D Python Port
=====================

Numerical model of one-dimensional variably saturated water flow,
heat transport, and transport of solutes involved in sequential
first-order decay reactions.

Ported from Fortran source (HYDRUS version 7.0 / 4.08) to Python
with exact 1:1 functional parity.

Original authors: J. Simunek, M. Sejna, M. Th. van Genuchten
Python port: 2026
"""

__version__ = "7.0.0"
__author__ = "HYDRUS-1D Python Port"

from .dataclasses import (
    SimulationConfig,
    GridState,
    SoilMaterial,
    ChemicalSpecies,
    BoundaryConditions,
    TimeControl,
    HysteresisState,
    RootUptakeState,
)
from .utils import (
    solve_tridiagonal,
    solve_banbury,
    conversion_factors,
)

__all__ = [
    # Data classes
    "SimulationConfig",
    "GridState",
    "SoilMaterial",
    "ChemicalSpecies",
    "BoundaryConditions",
    "TimeControl",
    "HysteresisState",
    "RootUptakeState",
    # Utilities
    "solve_tridiagonal",
    "solve_banbury",
    "conversion_factors",
]
