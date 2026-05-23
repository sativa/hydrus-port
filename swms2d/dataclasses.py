"""
Data structures for SWMS_2D Python port.
========================================

Maps Fortran 77 dimensioned arrays to Python dataclasses + NumPy arrays.
Mirrors hydrus1d/dataclasses.py style for cross-package consistency.

Fortran reference: see /Users/zhangfeng/CODE_BLOCK_DNDC/SWMS_2D_Src/SOURCE.FOR/

Conventions:
    - Fortran 1-based indexing → NumPy 0-based with explicit conversion
      at I/O boundaries (read/write of .IN / .OUT files)
    - SWMS_2D uses x = horizontal, y = vertical (positive upward).
      Some literature uses (x, z) — we keep SWMS_2D's (x, y).
    - Pressure head h: cm (negative = unsaturated, 0 = saturated, positive = ponded)
    - Volumetric water content θ: cm³/cm³
    - Time: configurable units (sec/min/hour/day) via SimulationConfig
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
from numpy.typing import NDArray


# ============================================================================
# Dimensional limits (mirror SWMS_2D.FOR parameter declarations)
# ============================================================================

NumNPD = 5000       # Max number of nodes
NumElD = 6000       # Max number of elements
NumBPD = 250        # Max number of boundary nodes
MBandD = 20         # Max bandwidth of global matrix (legacy banded solver)
NSeepD = 2          # Max number of seepage faces
NumSPD = 50         # Max nodes per seepage face
NDrD = 2            # Max number of drains
NMatD = 20          # Max number of materials
NTabD = 100         # Max material lookup table entries
NumKD = 6           # Max number of boundary code types
NObsD = 4           # Max number of observation nodes


# ============================================================================
# SimulationConfig — global flags (mirrors BasInf in INPUT2.FOR)
# ============================================================================

@dataclass
class SimulationConfig:
    """Global simulation switches and convergence parameters."""
    # Physics enable flags
    lWat: bool = True       # Solve Richards equation
    lChem: bool = False     # Solve solute transport
    SinkF: bool = False     # Root water uptake active
    qGWLF: bool = False     # GWL-dependent bottom flux
    FreeD: bool = False     # Free drainage bottom BC
    SeepF: bool = False     # Seepage face present
    DrainF: bool = False    # Drain present
    AtmInF: bool = False    # Atmospheric BC time series
    ShortF: bool = False    # Short-form output
    CheckF: bool = False    # Print mesh check info
    FluxF: bool = False     # Print element flux output
    Explic: bool = False    # Explicit time stepping (vs implicit)
    lUpW: bool = False      # Upstream weighting in solute transport
    lArtD: bool = False     # Artificial dispersion for solute
    lOrt: bool = True       # Use ORTHOMIN solver (vs Gauss elim)
    lHyst: bool = False     # Soil-water retention hysteresis (Scott 1983)
    lTemp: bool = False     # Couple heat transport (temper.py)

    # Convergence
    MaxIt: int = 20         # Max Picard iterations per timestep
    TolTh: float = 0.001    # Theta tolerance
    TolH: float = 1.0       # Head tolerance (cm)

    # Coordinate system
    KAT: int = 0            # 0=cartesian, 1=axisymmetric, 2=plane radial


# ============================================================================
# SoilMaterial — van Genuchten parameters (mirrors MATERIA2.FOR)
# ============================================================================

@dataclass
class SoilMaterial:
    """
    van Genuchten-Mualem hydraulic parameters for one material.

    SWMS_2D stores material parameters in Par(10, NMat) — 10 slots per material.
    Slots 1-10 map to:
        1: thr   residual water content
        2: ths   saturated water content
        3: tha   imaginary value (for hysteresis, unused if lWat-only)
        4: thm   imaginary value (for hysteresis)
        5: alpha van Genuchten α (1/cm)
        6: n     van Genuchten n
        7: Ks    saturated hydraulic conductivity (cm/T)
        8: Kk    relative K at θk
        9: thk   water content at which Kk applies
        10: (reserved)
    """
    thr: float
    ths: float
    alpha: float
    n: float
    Ks: float
    tha: float = 0.0
    thm: float = 0.0
    Kk: float = 0.0
    thk: float = 0.0


# ============================================================================
# TimeControl — time stepping (mirrors TIME2.FOR + portions of INPUT2.FOR)
# ============================================================================

@dataclass
class TimeControl:
    """Time discretization and adaptive step parameters."""
    t: float = 0.0          # Current time
    tInit: float = 0.0
    tMax: float = 0.0
    dt: float = 0.001       # Current timestep
    dtOpt: float = 0.001    # Running optimum dt (TmCont state; clipped to dt near tPrint)
    dtInit: float = 0.001
    dtMin: float = 1e-5
    dtMaxW: float = 1.0     # Max dt allowed by water flow
    dtMaxC: float = 1.0     # Max dt allowed by solute (Courant)
    dMul: float = 1.3       # Multiplier when iterations are few
    dMul2: float = 0.7      # Multiplier when iterations are many
    TLevel: int = 1         # Current time step counter
    ItCum: int = 0          # Cumulative iteration count


# ============================================================================
# BoundaryConditions — atmospheric + dirichlet/neumann (INPUT2.FOR BoundInf)
# ============================================================================

@dataclass
class BoundaryConditions:
    """
    Boundary node lists and time-variable atmospheric inputs.

    SWMS_2D uses Kode(n) per-node integer code:
        Kode = 0   internal node
        Kode = ±1  Dirichlet (prescribed h), ±2 atmospheric, ±3 free drainage,
                   ±4 seepage face, ±5 deep drainage, ±6 ponded infiltration
        Sign indicates if it is currently flux-imposed (+) or head-imposed (-)
    """
    # Boundary node lookup (1D)
    KodCB: NDArray[np.int32] = field(default_factory=lambda: np.zeros(0, np.int32))
    KXB:   NDArray[np.int32] = field(default_factory=lambda: np.zeros(0, np.int32))
    Width: NDArray[np.float64] = field(default_factory=lambda: np.zeros(0, np.float64))

    # Atmospheric time series (when AtmInF=True)
    tAtm: NDArray[np.float64] = field(default_factory=lambda: np.zeros(0, np.float64))
    Prec: NDArray[np.float64] = field(default_factory=lambda: np.zeros(0, np.float64))
    rSoil: NDArray[np.float64] = field(default_factory=lambda: np.zeros(0, np.float64))
    rRoot: NDArray[np.float64] = field(default_factory=lambda: np.zeros(0, np.float64))
    hCritA: float = -1e6
    hCritS: float = 0.0


# ============================================================================
# CumulativeFlux — mass-balance tracking (mirrors CumQ in main program)
# ============================================================================

@dataclass
class CumulativeFlux:
    """Cumulative fluxes for mass balance verification."""
    CumQ: NDArray[np.float64] = field(
        default_factory=lambda: np.zeros(12, np.float64)
    )
    """
    Slots (1-based in Fortran, 0-based here):
        0: CumQAP   cumulative atmospheric precipitation
        1: CumQAE   cumulative actual evaporation
        2: CumQAT   cumulative actual transpiration
        3: CumQrT   cumulative transpiration (root)
        4: CumQrR   cumulative root extraction
        5: CumQvR   cumulative variable BC inflow/outflow
        6-11: per-boundary-type cumulative fluxes
    """


# ============================================================================
# ════════════════════════════════════════════════════════════════════════════
# ►►► USER CONTRIBUTION POINT — design the FE mesh data classes
# ════════════════════════════════════════════════════════════════════════════
#
# Below is the most important design decision in the entire SWMS_2D port.
# 5-10 lines of code here shapes the memory layout, IO format, assembly loop,
# and every downstream module. Please fill in.
#
# CONTEXT: SWMS_2D Fortran uses these per-node and per-element arrays
# (verified from INPUT2.FOR lines 255-450):
#
#   Per-node arrays (1..NumNP, max NumNPD=5000):
#       Kode(n)              integer  : BC type code (see BoundaryConditions)
#       x(n), y(n)           real     : Cartesian coordinates (cm)
#       hNew(n), hOld(n)     real     : pressure head at current/previous step
#       hTemp(n)             real     : iteration scratch
#       Q(n)                 real     : nodal flux / Dirichlet value
#       Conc(n)              real     : solute concentration
#       MatNum(n)            integer  : material index (1..NMat)  ⚠ PER-NODE!
#       Beta(n)              real     : root distribution weight
#       Axz(n), Bxz(n), Dxz(n) real   : anisotropy ratios for K tensor at node
#
#   Per-element arrays (1..NumEl, max NumElD=6000):
#       KX(e, 1..4)          integer  : 4 node indices per element
#                                       (if KX(e,3) == KX(e,4) → triangle)
#       ConAxx(e), ConAzz(e), ConAxz(e) real
#                                     : per-element anisotropy tensor components
#
# DESIGN CHOICES YOU NEED TO MAKE (these are real choices, not boilerplate):
#
#  (a) Per-node Material — Fortran style (MatNum per NODE) vs modern FEM
#      (material per ELEMENT). Fortran lets the same element span 2 materials,
#      which is physically weird but happens in SWMS_2D's input format. Decide.
#
#  (b) Triangle representation — keep Fortran "degenerate quad" trick (4th
#      node = 3rd node) or use separate triangles/quads arrays. The former
#      makes 1:1 verification trivial but is ugly Python.
#
#  (c) Indexing — store coordinates as separate x[], y[] (Fortran-like) or
#      as a single (N, 2) array? Either works; one is faster for vectorized
#      assembly, the other matches Fortran source line-by-line.
#
#  (d) Element class vs structure-of-arrays — for 6000 elements at most,
#      both work. Element-as-dataclass is Pythonic but slow if you instantiate
#      one per assembly step. SoA NumPy arrays match Fortran exactly.
#
# Recommendation for Stage 1 (1:1 port): match Fortran exactly (SoA, per-node
# material, 4-column KX with degenerate quad for triangles, separate x/y).
# This makes line-by-line verification possible. Stage 2 (scikit-fem) can
# re-design freely.
#
# Please fill in the three classes below. Aim for 5-10 lines per class.
# After you fill these, I will continue with mesh.py, watflow.py, etc.
# ════════════════════════════════════════════════════════════════════════════


@dataclass
class Node:
    """
    Per-node state for SWMS_2D 2D FE mesh (Structure-of-Arrays).

    All arrays are 0-indexed in Python but the values they store are
    semantically 1-based ids matching Fortran (e.g., MatNum[i] = 1 means
    material 1, even though i = 0 is the first node).

    Fields directly correspond to NodInf reader in INPUT2.FOR L255-345:
        read(32,*) n,Kode(n),x(n),y(n),hOld(n),Conc(n),Q(n),
                   MatNum(n),Beta(n),Axz(n),Bxz(n),Dxz(n)
    """
    Kode:   NDArray[np.int32]   = field(default_factory=lambda: np.zeros(0, np.int32))
    x:      NDArray[np.float64] = field(default_factory=lambda: np.zeros(0, np.float64))
    y:      NDArray[np.float64] = field(default_factory=lambda: np.zeros(0, np.float64))
    hNew:   NDArray[np.float64] = field(default_factory=lambda: np.zeros(0, np.float64))
    hOld:   NDArray[np.float64] = field(default_factory=lambda: np.zeros(0, np.float64))
    hTemp:  NDArray[np.float64] = field(default_factory=lambda: np.zeros(0, np.float64))
    Q:      NDArray[np.float64] = field(default_factory=lambda: np.zeros(0, np.float64))
    Conc:   NDArray[np.float64] = field(default_factory=lambda: np.zeros(0, np.float64))
    MatNum: NDArray[np.int32]   = field(default_factory=lambda: np.zeros(0, np.int32))
    Beta:   NDArray[np.float64] = field(default_factory=lambda: np.zeros(0, np.float64))
    Axz:    NDArray[np.float64] = field(default_factory=lambda: np.zeros(0, np.float64))
    Bxz:    NDArray[np.float64] = field(default_factory=lambda: np.zeros(0, np.float64))
    Dxz:    NDArray[np.float64] = field(default_factory=lambda: np.zeros(0, np.float64))


@dataclass
class Element:
    """
    Per-element connectivity + anisotropy tensor for SWMS_2D FE mesh.

    KX[e, 0..3] = 4 node indices (0-based). For triangles, KX[e,3] == KX[e,2]
    (degenerate quad — Fortran convention preserved for 1:1 verification).

    ConAxx/Azz/Axz: full 2x2 anisotropy tensor in GLOBAL coordinates.
    GRID.IN stores them as (Angle, Aniz1, Aniz2); the rotation
        ConAxx = Aniz1*cos²(α) + Aniz2*sin²(α)
        ConAzz = Aniz1*sin²(α) + Aniz2*cos²(α)
        ConAxz = (Aniz1 - Aniz2)*sin(α)*cos(α)
    is applied in input.py before storage (mirrors INPUT2.FOR L368-376).
    """
    KX:     NDArray[np.int32]   = field(default_factory=lambda: np.zeros((0, 4), np.int32))
    ConAxx: NDArray[np.float64] = field(default_factory=lambda: np.zeros(0, np.float64))
    ConAzz: NDArray[np.float64] = field(default_factory=lambda: np.zeros(0, np.float64))
    ConAxz: NDArray[np.float64] = field(default_factory=lambda: np.zeros(0, np.float64))
    LayNum: NDArray[np.int32]   = field(default_factory=lambda: np.zeros(0, np.int32))


@dataclass
class Mesh:
    """
    Top-level FE mesh container: nodes + elements + boundary metadata.

    Boundary representation matches GRID.IN Block J:
        KXB    boundary node indices (0-based, length NumBP)
        Width  boundary segment width per node (length NumBP)
        rLen   total boundary length scalar

    Counts cache the array sizes for Fortran-style validation.
    Derived caches (ListNE = element count per node) populated by mesh.py
    after element connectivity is read.
    """
    nodes: Node = field(default_factory=Node)
    elements: Element = field(default_factory=Element)

    KXB:    NDArray[np.int32]   = field(default_factory=lambda: np.zeros(0, np.int32))
    Width:  NDArray[np.float64] = field(default_factory=lambda: np.zeros(0, np.float64))
    rLen:   float = 0.0

    NumNP:  int = 0     # number of nodes
    NumEl:  int = 0     # number of elements
    NumBP:  int = 0     # number of boundary nodes
    IJ:     int = 0     # half-bandwidth indicator (legacy banded solver)
    NObs:   int = 0     # number of observation nodes

    ListNE: NDArray[np.int32] = field(default_factory=lambda: np.zeros(0, np.int32))
    # Per-node count of elements containing this node — derived by mesh.build_listne()


# ============================================================================
# FullSimulationState — convenience aggregate (filled after Mesh is designed)
# ============================================================================

@dataclass
class FullSimulationState:
    """Top-level container passed between modules. Mirrors H1D's FullSimulationState."""
    config: SimulationConfig = field(default_factory=SimulationConfig)
    time: TimeControl = field(default_factory=TimeControl)
    bc: BoundaryConditions = field(default_factory=BoundaryConditions)
    cum_flux: CumulativeFlux = field(default_factory=CumulativeFlux)
    materials: list[SoilMaterial] = field(default_factory=list)
    mesh: Optional[Mesh] = None  # populated after Mesh is designed
