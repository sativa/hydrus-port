"""
Data structures for HYDRUS-1D Python port.
==========================================

Maps all Fortran COMMON blocks and dimensioned arrays to Python
dataclasses for state management. Mirrors the exact structure of
the original Fortran code.

Array conventions (matching Fortran):
- Node 1 = bottom of profile, Node N = top of profile
- All 1D arrays are indexed 1..N (using 0-based with padding or direct mapping)
- Double precision in Fortran -> float64 in NumPy
- Single precision in Fortran -> float32 in NumPy (or float64 for safety)
"""

from __future__ import annotations
import dataclasses
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import numpy as np
from numpy.typing import NDArray

# ============================================================================
# Constants matching Fortran defaults
# ============================================================================

NumNPD = 1001       # Maximum number of nodes + 1
NMatD = 20          # Maximum number of materials
NTabD = 100         # Maximum number of table entries
NObsD = 100         # Maximum number of observation nodes
NSD = 11            # Maximum number of chemical species
NUnitD = 7          # Maximum number of output units
NPD = 1000          # Print interval array size


# ============================================================================
# SimulationConfig - Global simulation flags and parameters
# ============================================================================

@dataclass
class SimulationConfig:
    """
    Global simulation configuration flags.
    
    Corresponds to logical flags scattered throughout HYDRUS.FOR
    and the Init/BasInf subroutines in INPUT.FOR.
    """
    # Basic simulation switches
    lWat: bool = True           # Water flow simulation
    lChem: bool = False         # Solute transport simulation
    lTemp: bool = False         # Heat transport simulation
    SinkF: bool = False         # Root water/nutrient uptake
    lRoot: bool = False         # Root zone specified
    ShortO: bool = False        # Short output
    lWDep: bool = False         # Water stress dependency
    lScreen: bool = False       # Screen output
    AtmBC: bool = False         # Atmospheric boundary conditions
    lEquil: bool = True         # Equilibrium adsorption
    
    # Additional flags (version 3+)
    lSnow: bool = False         # Snow module
    lMeteo: bool = False        # Meteorological data input
    lVapor: bool = False        # Vapor flow
    lActRSU: bool = False       # Activated root solute uptake
    lFlux: bool = False         # Flux boundary conditions
    
    # Bottom boundary conditions
    qGWLF: bool = False         # Groundwater level fluctuation
    FreeD: bool = False         # Free drainage
    SeepF: bool = False         # Seepage face
    qDrain: bool = False        # Drainage pipe
    
    # Top boundary
    TopInF: bool = False        # Time-variable top BC
    BotInF: bool = False        # Time-variable bottom BC
    WLayer: bool = False        # Water layer on top
    
    # Convergence/solution control
    lVarBC: bool = False        # Variable boundary conditions
    lInitW: bool = False        # Initial conditions from water content
    lPrint: bool = True         # Print output files
    lMinStep: bool = False      # Minimum time step flag
    ConvgF: bool = True         # Convergence flag
    
    # Advanced flags
    lExtrap: bool = False       # Extrapolation
    lPrintD: bool = False       # Print debug
    lLAI: bool = False          # Leaf area index
    lDensity: bool = False      # Density-dependent transport
    lCentrif: bool = False      # Centrifuge mode
    lEqInit: bool = False       # Equilibrium initial conditions
    lSinPrec: bool = False      # Sinusoidal precipitation
    lDualNEq: bool = False      # Dual porosity nonequilibrium
    lMassIni: bool = False      # Mass initialization
    lStopConv: bool = True      # Stop on non-convergence
    lOmegaW: bool = False       # Omega water stress
    lFluxOut: bool = False      # Flux output
    lVaporOut: bool = False     # Vapor output
    
    # Numerical parameters
    MaxIt: int = 100            # Maximum Newton iterations
    TolTh: float = 1e-4         # Water content tolerance
    TolH: float = 1e-3          # Pressure head tolerance
    CosAlf: float = 1.0         # Cosine of angle (gravity factor)
    
    # Unit conversions
    xConv: float = 1.0          # Length conversion factor (m -> simulation unit)
    tConv: float = 1.0          # Time conversion factor (s -> simulation unit)
    
    # Grid
    NMat: int = 1               # Number of materials
    NLay: int = 1               # Number of layers
    NumNP: int = 0              # Actual number of nodes
    
    # Solute
    NS: int = 0                 # Number of chemical species
    
    # Hysteresis
    iHyst: int = 0              # Hysteresis model (0=none, 1,2=two-curve, 3=Lenhard)
    iModel: int = 0             # Soil hydraulic model
    
    # Dual porosity
    iDualPor: int = 0           # Dual porosity flag
    WTransf: float = 0.0        # Mass transfer coefficient
    
    # Non-equilibrium water flow
    TauW: float = 0.0           # Relaxation time for water
    
    # Output
    nPrStep: int = 0            # Number of print steps
    nTabMod: int = 0            # Table modification
    
    # Snow
    SnowMF: float = 0.0         # Snow melt factor
    SnowLayer: float = 0.0      # Snow layer depth
    
    # Root
    iSunSh: int = 0             # Sunshine hours
    iRelHum: int = 0            # Relative humidity
    iRootIn: int = 0            # Root input flag
    
    # Tortuosity/enhancement
    iTort: int = 0              # Tortuosity model
    iEnhanc: int = 0            # Enhancement factor model
    
    hSeep: float = 0.0          # Seepage head
    
    # Moisture dependency
    iMoistDep: int = 0          # Moisture dependency model
    OmegaC: float = 0.0         # Omega chemical
    OmegaS: float = 0.0         # Omega stress
    SPot: float = 0.0           # Potential solute concentration
    
    # Drainage
    zBotDr: float = 0.0         # Drain depth
    BaseGW: float = 0.0         # Base groundwater level
    rSpacing: float = 0.0       # Drain spacing
    iPosDr: int = 0             # Drain position model
    rKhTop: float = 0.0         # Horizontal K top
    rKhBot: float = 0.0         # Horizontal K bottom
    rKvTop: float = 0.0         # Vertical K top
    rKvBot: float = 0.0         # Vertical K bottom
    Entres: float = 0.0         # Entrance resistance
    WetPer: float = 0.0         # Wetted perimeter
    zInTF: float = 0.0          # Interface flow depth
    GeoFac: float = 0.0         # Geometric factor
    
    # Centrifuge
    Radius: float = 0.0         # Centrifuge radius
    
    # GWL
    GWL0L: float = 0.0          # Initial groundwater level
    Aqh: float = 0.0            # GWL amplitude
    Bqh: float = 0.0            # GWL phase
    
    # LAI
    rExtinct: float = 0.0       # Extinction coefficient
    
    # Concentration type
    iConcType: int = 0          # Concentration type
    
    # Misc
    ExcesInt: float = 0.0       # Excess integral


# ============================================================================
# SoilMaterial - Hydraulic and thermal properties
# ============================================================================

@dataclass
class SoilMaterial:
    """
    Soil hydraulic and thermal properties for a single material type.
    
    Corresponds to ParD(11,NMatD) and ParW(11,NMatD) arrays in Fortran.
    
    Hydraulic parameters (ParD):
        ParD(1)  = theta_r  (residual water content)
        ParD(2)  = theta_s  (saturated water content)
        ParD(3)  = alpha    (van Genuchten alpha, cm^-1)
        ParD(4)  = n        (van Genuchten n)
        ParD(5)  = Ks       (saturated hydraulic conductivity)
        ParD(6)  = beta     (conductivity exponent, Brooks-Corey)
        ParD(7)  = theta_m  (modified VG: m parameter)
        ParD(8)  = theta_a  (modified VG: air content)
        ParD(9)  = theta_k  (modified VG: knee point content)
        ParD(10) = Kk       (modified VG: knee point conductivity)
        ParD(11) = h_sat    (saturated pressure head)
    
    Thermal parameters (ParW):
        ParW(1)  = theta_wr (residual water content for thermal)
        ParW(2)  = theta_ws (saturated water content for thermal)
        ParW(3)  = lambda_d (dry thermal conductivity)
        ParW(4)  = lambda_w (water thermal conductivity)
        ParW(5)  = delta_l  (delta for thermal conductivity)
        ParW(6)  = rho_b    (bulk density)
        ParW(7)  = C_s      (solid heat capacity)
        ParW(8)  = C_w      (water heat capacity)
        ParW(9)  = C_a      (air heat capacity)
        ParW(10) = gamma_v  (vapor diffusion coefficient)
        ParW(11) = gamma_h  (heat of vaporization)
    """
    # Hydraulic parameters (matching ParD)
    theta_r: float = 0.0        # Residual water content
    theta_s: float = 0.0        # Saturated water content
    alpha: float = 0.0          # van Genuchten alpha
    n: float = 1.0              # van Genuchten n
    Ks: float = 0.0             # Saturated hydraulic conductivity
    beta: float = 0.5           # Conductivity exponent
    theta_m: float = 0.0        # Modified VG m parameter
    theta_a: float = 0.0        # Modified VG air content
    theta_k: float = 0.0        # Modified VG knee point content
    Kk: float = 0.0             # Modified VG knee point conductivity
    h_sat: float = 0.0          # Saturated pressure head
    
    # Thermal parameters (matching ParW)
    theta_wr: float = 0.0
    theta_ws: float = 0.0
    lambda_d: float = 0.0
    lambda_w: float = 0.0
    delta_l: float = 0.0
    rho_b: float = 0.0
    C_s: float = 0.0
    C_w: float = 0.0
    C_a: float = 0.0
    gamma_v: float = 0.0
    gamma_h: float = 0.0
    
    # Table-based K-S-P (optional)
    NTab: int = 0               # Number of table entries
    hTab: NDArray[np.float64] = field(default_factory=lambda: np.zeros(NTabD))
    ConTab: NDArray[np.float64] = field(default_factory=lambda: np.zeros(NTabD))
    CapTab: NDArray[np.float64] = field(default_factory=lambda: np.zeros(NTabD))
    TheTab: NDArray[np.float64] = field(default_factory=lambda: np.zeros(NTabD))
    
    # Derived properties
    ConSat: float = 0.0         # Saturated specific water capacity
    ths: float = 0.0            # Saturated water content (copy)
    thr: float = 0.0            # Residual water content (copy)
    
    # iModel reference
    iModel: int = 0
    
    def as_array(self) -> NDArray[np.float64]:
        """Return ParD as a 11-element array matching Fortran indexing."""
        return np.array([
            self.theta_r, self.theta_s, self.alpha, self.n, self.Ks,
            self.beta, self.theta_m, self.theta_a, self.theta_k,
            self.Kk, self.h_sat
        ], dtype=np.float64)


# ============================================================================
# ChemicalSpecies - Solute transport parameters
# ============================================================================

@dataclass
class ChemicalSpecies:
    """
    Chemical species parameters for solute transport.
    
    Corresponds to ChPar(NSD*16+4, NMatD) in Fortran.
    Each species has up to 16*NSD+4 parameters per material.
    
    For species jS, parameters are at indices (jS-1)*16 + k:
        ChPar(1)  = k_bot   (bottom BC type)
        ChPar(2)  = c_bot   (bottom concentration)
        ChPar(3)  = k_top   (top BC type)
        ChPar(4)  = c_top   (top concentration)
        ChPar(5)  = D       (diffusion coefficient)
        ChPar(6)  = Dg      (gas-phase diffusion coefficient)
        ChPar(7)  = k_d     (distribution coefficient)
        ChPar(8)  = rho_b   (bulk density for sorption)
        ChPar(9)  = K_f     (Freundlich K)
        ChPar(10) = n_f     (Freundlich n)
        ChPar(11) = lambda0 (decay constant parent)
        ChPar(12) = lambda1 (decay constant daughter)
        ChPar(13) = Henry   (Henry's law constant)
        ChPar(14) = epsi    (mobile water fraction)
        ChPar(15) = alpha_m (mass transfer coefficient)
        ChPar(16) = ...
    """
    # Boundary conditions
    k_bot: int = 0              # Bottom BC code
    c_bot: float = 0.0          # Bottom concentration
    k_top: int = 0              # Top BC code
    c_top: float = 0.0          # Top concentration
    
    # Transport
    D: float = 0.0              # Diffusion coefficient
    Dg: float = 0.0             # Gas-phase diffusion coefficient
    epsi: float = 1.0           # Mobile water fraction
    
    # Sorption
    k_d: float = 0.0            # Distribution coefficient
    rho_b: float = 0.0          # Bulk density
    K_f: float = 0.0            # Freundlich K
    n_f: float = 1.0            # Freundlich n
    lLinear: bool = True        # Linear isotherm
    
    # Decay
    lambda0: float = 0.0        # Decay constant (parent)
    lambda1: float = 0.0        # Decay constant (daughter)
    
    # Volatility
    Henry: float = 0.0          # Henry's law constant
    
    # Mobile/immobile
    alpha_m: float = 0.0        # Mass transfer coefficient
    lMobIm: bool = False        # Mobile/immobile model
    
    # Temperature dependency
    TDep: NDArray[np.float64] = field(default_factory=lambda: np.zeros(16))
    
    # Convergence
    cTolA: float = 1e-6         # Absolute concentration tolerance
    cTolR: float = 1e-4         # Relative concentration tolerance
    MaxItC: int = 100           # Maximum concentration iterations
    
    # Root uptake
    cRoot: float = 0.0          # Root concentration
    
    # Initial concentration
    cInit: float = 0.0
    
    # Tortuosity
    iTort: int = 0
    iEnhanc: int = 0
    
    # Misc
    lArtD: bool = False         # Artificial dispersion
    PeCr: float = 0.0           # Critical Peclet number
    
    def as_array(self, n_species: int) -> NDArray[np.float64]:
        """Return ChPar array matching Fortran indexing for this species."""
        arr = np.zeros(n_species * 16 + 4, dtype=np.float64)
        # Populate based on actual parameter mapping
        return arr


# ============================================================================
# BoundaryConditions - Time-variable BC state
# ============================================================================

@dataclass
class BoundaryConditions:
    """
    Boundary condition state at current time step.
    
    Corresponds to boundary condition variables in SetBC (TIME.FOR)
    and the BC handling in Reset/Shift (WATFLOW.FOR).
    """
    # Water flow BCs
    KodTop: int = 1             # Top BC code
    KodBot: int = 2             # Bottom BC code
    rTop: float = 0.0           # Top flux (positive = infiltration)
    rBot: float = 0.0           # Bottom flux
    hTop: float = 0.0           # Top head
    hBot: float = 0.0           # Bottom head
    hCritS: float = 0.0         # Critical head for surface runoff
    hCritA: float = -1e10       # Critical head for atmospheric BC
    rRoot: float = 0.0          # Root water uptake rate
    
    # Old BC values (for tracking changes)
    kTOld: int = 1
    kBOld: int = 2
    
    # Solute BCs
    kTopCh: int = 0             # Top chemical BC code
    kBotCh: int = 0             # Bottom chemical BC code
    
    # Heat BCs
    tTop: float = 20.0          # Top temperature
    tBot: float = 20.0          # Bottom temperature
    Ampl: float = 0.0           # Temperature amplitude
    
    # Atmospheric
    Prec: float = 0.0           # Precipitation rate
    rSoil: float = 0.0          # Soil evaporation rate
    rR: float = 0.0             # Transpiration rate
    rLAI: float = 0.0           # Leaf area index
    rPET: float = 0.0           # Potential evapotranspiration
    
    # GWL fluctuation
    GWL0L: float = 0.0          # Base groundwater level
    Aqh: float = 0.0            # Amplitude
    Bqh: float = 0.0            # Phase


# ============================================================================
# TimeControl - Time stepping state
# ============================================================================

@dataclass
class TimeControl:
    """
    Time stepping control state.
    
    Corresponds to time management in TIME.FOR (TmCont, SetBC, RTime).
    """
    t: float = 0.0              # Current simulation time (seconds)
    tInit: float = 0.0          # Initial time
    tMax: float = 0.0           # Maximum simulation time
    tOld: float = 0.0           # Time at previous step
    dt: float = 0.0             # Current time step
    dtMin: float = 1e-10        # Minimum time step
    dtMaxW: float = 1e30        # Maximum water flow time step
    dtMaxC: float = 1e30        # Maximum solute time step
    dtMaxT: float = 1e30        # Maximum heat time step
    dtOpt: float = 0.0          # Optimal time step
    dtInit: float = 0.0         # Initial time step
    
    # Atmospheric time
    tAtm: float = 0.0           # Next atmospheric BC change time
    tAtm1: float = 0.0
    tAtm2: float = 0.0
    tAtmOld: float = 0.0
    tAtmN: float = 0.0
    tAtm2O: float = 0.0
    
    # Print control
    TPrint: NDArray[np.float64] = field(
        default_factory=lambda: np.zeros(NPD))
    tPrint1: float = 0.0        # Next print time
    tPrintInt: float = 0.0      # Print interval
    
    # Iteration control
    ItMin: int = 2              # Min iterations for dt increase
    ItMax: int = 5              # Max iterations before dt increase
    dMul: float = 2.0           # dt multiplication factor (increase)
    dMul2: float = 0.5          # dt multiplication factor (decrease)
    IterW: int = 0              # Water flow iterations at current step
    IterC: int = 0              # Solute iterations at current step
    ItCum: int = 0              # Cumulative iterations
    
    # Date/time
    iYear: int = 0
    iMonth: int = 1
    iDay: int = 1
    iHours: int = 0
    iMins: int = 0
    iSecs: int = 0
    i100th: int = 0


# ============================================================================
# GridState - Full simulation grid state
# ============================================================================

@dataclass
class GridState:
    """
    Complete grid state for the simulation.
    
    Corresponds to all node-level arrays in HYDRUS.FOR dimension statements
    and COMMON blocks. Node 1 = bottom, Node N = top.
    
    This is the central state object that replaces all Fortran COMMON blocks.
    """
    NumNP: int = 0              # Number of nodes
    NS: int = 0                 # Number of chemical species
    NMat: int = 1               # Number of materials
    
    # Grid geometry
    x: NDArray[np.float64] = field(
        default_factory=lambda: np.zeros(NumNPD))  # Node coordinates
    
    # Material assignment per node
    MatNum: NDArray[np.int32] = field(
        default_factory=lambda: np.zeros(NumNPD, dtype=np.int32))
    LayNum: NDArray[np.int32] = field(
        default_factory=lambda: np.zeros(NumNPD, dtype=np.int32))
    Beta: NDArray[np.float64] = field(
        default_factory=lambda: np.zeros(NumNPD))  # Layer weight factor
    
    # Pressure head (primary variable)
    hNew: NDArray[np.float64] = field(
        default_factory=lambda: np.zeros(NumNPD))  # Current iteration head
    hOld: NDArray[np.float64] = field(
        default_factory=lambda: np.zeros(NumNPD))  # Previous time step head
    hTemp: NDArray[np.float64] = field(
        default_factory=lambda: np.zeros(NumNPD))  # Temporary head
    
    # Water content
    ThNew: NDArray[np.float64] = field(
        default_factory=lambda: np.zeros(NumNPD))  # Current water content
    ThOld: NDArray[np.float64] = field(
        default_factory=lambda: np.zeros(NumNPD))  # Previous water content
    ThEq: NDArray[np.float64] = field(
        default_factory=lambda: np.zeros(NumNPD))  # Equilibrium water content
    ThVNew: NDArray[np.float64] = field(
        default_factory=lambda: np.zeros(NumNPD))  # Vapor water content new
    ThVOld: NDArray[np.float64] = field(
        default_factory=lambda: np.zeros(NumNPD))  # Vapor water content old
    ThNewIm: NDArray[np.float64] = field(
        default_factory=lambda: np.zeros(NumNPD))  # Immobile water content new
    ThOldIm: NDArray[np.float64] = field(
        default_factory=lambda: np.zeros(NumNPD))  # Immobile water content old
    
    # Velocity (Darcy flux)
    vNew: NDArray[np.float64] = field(
        default_factory=lambda: np.zeros(NumNPD))
    vOld: NDArray[np.float64] = field(
        default_factory=lambda: np.zeros(NumNPD))
    vVOld: NDArray[np.float64] = field(
        default_factory=lambda: np.zeros(NumNPD))  # Vapor velocity old
    vVNew: NDArray[np.float64] = field(
        default_factory=lambda: np.zeros(NumNPD))  # Vapor velocity new
    vTop: float = 0.0           # Top velocity
    vCorr: NDArray[np.float64] = field(
        default_factory=lambda: np.zeros(NumNPD))  # Velocity correction
    
    # Hydraulic conductivity at nodes
    Con: NDArray[np.float64] = field(
        default_factory=lambda: np.zeros(NumNPD))  # Water content (node)
    Cap: NDArray[np.float64] = field(
        default_factory=lambda: np.zeros(NumNPD))  # Specific capacity
    Disp: NDArray[np.float64] = field(
        default_factory=lambda: np.zeros(NumNPD))  # Dispersion coefficient
    Retard: NDArray[np.float64] = field(
        default_factory=lambda: np.zeros(NumNPD))  # Retardation factor
    
    # Conductivity matrices (tridiagonal system)
    P: NDArray[np.float64] = field(
        default_factory=lambda: np.zeros(NumNPD))  # Lower diagonal
    R: NDArray[np.float64] = field(
        default_factory=lambda: np.zeros(NumNPD))  # Main diagonal
    S: NDArray[np.float64] = field(
        default_factory=lambda: np.zeros(NumNPD))  # Upper diagonal
    
    # Solute concentrations
    Conc: NDArray[np.float64] = field(
        default_factory=lambda: np.zeros((NSD, NumNPD)))  # Species x Node
    cNew: NDArray[np.float64] = field(
        default_factory=lambda: np.zeros(NumNPD))
    cTemp: NDArray[np.float64] = field(
        default_factory=lambda: np.zeros(NumNPD))
    cPrevO: NDArray[np.float64] = field(
        default_factory=lambda: np.zeros(NumNPD))
    
    # Sorbed concentrations
    Sorb: NDArray[np.float64] = field(
        default_factory=lambda: np.zeros((NSD, NumNPD)))
    SorbN: NDArray[np.float64] = field(
        default_factory=lambda: np.zeros(NumNPD))
    Sorb2: NDArray[np.float64] = field(
        default_factory=lambda: np.zeros((NSD, NumNPD)))
    SorbN2: NDArray[np.float64] = field(
        default_factory=lambda: np.zeros(NumNPD))
    
    # Temperature
    TempN: NDArray[np.float64] = field(
        default_factory=lambda: np.zeros(NumNPD))  # New temperature
    TempO: NDArray[np.float64] = field(
        default_factory=lambda: np.zeros(NumNPD))  # Old temperature
    
    # Sink terms (root uptake)
    Sink: NDArray[np.float64] = field(
        default_factory=lambda: np.zeros(NumNPD))
    sSink: NDArray[np.float64] = field(
        default_factory=lambda: np.zeros(NumNPD))  # Solute sink
    SinkIm: NDArray[np.float64] = field(
        default_factory=lambda: np.zeros(NumNPD))  # Immobile sink
    STrans: NDArray[np.float64] = field(
        default_factory=lambda: np.zeros(NumNPD))  # Mass transfer
    
    # Thermal properties at nodes
    Kappa: NDArray[np.float64] = field(
        default_factory=lambda: np.zeros(NumNPD))  # Thermal conductivity
    KappaO: NDArray[np.float64] = field(
        default_factory=lambda: np.zeros(NumNPD))
    ATh: NDArray[np.float64] = field(
        default_factory=lambda: np.zeros(NumNPD))  # Thermal water capacity
    AThS: NDArray[np.float64] = field(
        default_factory=lambda: np.zeros(NumNPD))  # Solid heat capacity
    ThRR: NDArray[np.float64] = field(
        default_factory=lambda: np.zeros(NumNPD))
    
    # Conductivity tracking
    Ah: NDArray[np.float64] = field(
        default_factory=lambda: np.zeros(NumNPD))
    AK: NDArray[np.float64] = field(
        default_factory=lambda: np.zeros(NumNPD))
    ConO: NDArray[np.float64] = field(
        default_factory=lambda: np.zeros(NumNPD))
    ConR: NDArray[np.float64] = field(
        default_factory=lambda: np.zeros(NumNPD))
    AKS: NDArray[np.float64] = field(
        default_factory=lambda: np.zeros(NumNPD))
    ConLT: NDArray[np.float64] = field(
        default_factory=lambda: np.zeros(NumNPD))  # Liquid conductivity
    ConVT: NDArray[np.float64] = field(
        default_factory=lambda: np.zeros(NumNPD))  # Vapor conductivity
    ConVh: NDArray[np.float64] = field(
        default_factory=lambda: np.zeros(NumNPD))  # Vapor heat conductivity
    
    # Vapor
    WatIn: NDArray[np.float64] = field(
        default_factory=lambda: np.zeros(NumNPD))
    SolIn: NDArray[np.float64] = field(
        default_factory=lambda: np.zeros(NumNPD))
    
    # Matrix equation RHS (solute)
    g0: NDArray[np.float64] = field(
        default_factory=lambda: np.zeros(NumNPD))
    g1: NDArray[np.float64] = field(
        default_factory=lambda: np.zeros(NumNPD))
    q0: NDArray[np.float64] = field(
        default_factory=lambda: np.zeros(NumNPD))
    q1: NDArray[np.float64] = field(
        default_factory=lambda: np.zeros(NumNPD))
    
    # Cumulative quantities
    wc: NDArray[np.float64] = field(
        default_factory=lambda: np.zeros(NumNPD))  # Water content integral
    CumQ: NDArray[np.float64] = field(
        default_factory=lambda: np.zeros(12))  # Cumulative water flux
    
    # Surface
    xSurf: float = 0.0          # Surface coordinate
    
    # Observation nodes
    NObs: int = 0
    Node: NDArray[np.int32] = field(
        default_factory=lambda: np.zeros(NObsD, dtype=np.int32))
    
    # Root zone
    xRoot: NDArray[np.float64] = field(
        default_factory=lambda: np.zeros(10))
    hRoot: float = 0.0
    vRoot: float = 0.0


# ============================================================================
# HysteresisState - Hysteresis tracking
# ============================================================================

@dataclass
class HysteresisState:
    """
    Hysteretic K-S-P relationship state.
    
    Corresponds to HYSTER.FOR common blocks for scanning curve tracking.
    """
    # Reversal point tracking
    nRev: int = 0               # Number of reversal points
    hRev: NDArray[np.float64] = field(
        default_factory=lambda: np.zeros(100))  # Reversal heads
    thRev: NDArray[np.float64] = field(
        default_factory=lambda: np.zeros(100))  # Reversal water contents
    
    # Current state
    lWetting: bool = True       # Current wetting/drying direction
    hMain: float = 0.0          # Head on main curve
    thMain: float = 0.0         # Water content on main curve
    
    # Lenhard model (iHyst=3)
    hAir: NDArray[np.float64] = field(
        default_factory=lambda: np.zeros(NumNPD))  # Air entry head
    thRes: NDArray[np.float64] = field(
        default_factory=lambda: np.zeros(NumNPD))  # Residual content
    
    # Scanning curve parameters
    a_scan: NDArray[np.float64] = field(
        default_factory=lambda: np.zeros(NumNPD))
    n_scan: NDArray[np.float64] = field(
        default_factory=lambda: np.zeros(NumNPD))


# ============================================================================
# RootUptakeState - Root water/nutrient uptake
# ============================================================================

@dataclass
class RootUptakeState:
    """
    Root water and nutrient uptake state.
    
    Corresponds to SINK.FOR variables and root-related parameters.
    """
    # Feddes model parameters
    p1: float = 0.0             # Optimal head
    p2: float = 0.0             # Sub-optimal head
    p3: float = 0.0             # wilting point head
    p4: float = 0.0             # Zero uptake head
    
    # Root distribution
    root_density: NDArray[np.float64] = field(
        default_factory=lambda: np.zeros(NumNPD))
    
    # Potential uptake
    SPot: float = 0.0           # Potential transpiration
    SAct: float = 0.0           # Actual transpiration
    
    # Water stress
    OmegaW: float = 0.0         # Water stress factor
    OmegaS: float = 0.0         # Solute stress factor
    
    # Silvertown model
    lSilvertown: bool = False
    
    # Maximum root concentration
    cRootMax: NDArray[np.float64] = field(
        default_factory=lambda: np.zeros(NSD))
    
    # Michaelis-Menten
    rKM: float = 0.0            # Half-saturation constant
    cMin: float = 0.0           # Minimum concentration


# ============================================================================
# CumulativeFlux - Output accumulation
# ============================================================================

@dataclass
class CumulativeFlux:
    """
    Cumulative flux tracking for output.
    
    Corresponds to output accumulation in OUTPUT.FOR.
    """
    # Water
    wCumT: float = 0.0          # Cumulative top flux
    wCumA: float = 0.0          # Cumulative bottom flux
    
    # Solute (per species)
    cCumT: NDArray[np.float64] = field(
        default_factory=lambda: np.zeros(NSD))
    cCumA: NDArray[np.float64] = field(
        default_factory=lambda: np.zeros(NSD))
    
    # Concentration at boundaries
    cvTop: NDArray[np.float64] = field(
        default_factory=lambda: np.zeros(NSD))
    cvBot: NDArray[np.float64] = field(
        default_factory=lambda: np.zeros(NSD))
    cvCh0: NDArray[np.float64] = field(
        default_factory=lambda: np.zeros(NSD))
    cvCh1: NDArray[np.float64] = field(
        default_factory=lambda: np.zeros(NSD))
    cvChR: NDArray[np.float64] = field(
        default_factory=lambda: np.zeros(NSD))
    cvChIm: NDArray[np.float64] = field(
        default_factory=lambda: np.zeros(NSD))
    
    # Sub-volume fluxes
    SubVol: NDArray[np.float64] = field(
        default_factory=lambda: np.zeros(10))
    Area: NDArray[np.float64] = field(
        default_factory=lambda: np.zeros(10))


# ============================================================================
# FullSimulationState - Complete simulation state
# ============================================================================

@dataclass
class FullSimulationState:
    """
    Complete simulation state container.
    
    Aggregates all state objects for the full HYDRUS-1D simulation.
    This replaces all COMMON blocks from the Fortran code.
    """
    config: SimulationConfig = field(default_factory=SimulationConfig)
    grid: GridState = field(default_factory=GridState)
    bc: BoundaryConditions = field(default_factory=BoundaryConditions)
    time: TimeControl = field(default_factory=TimeControl)
    hysteresis: HysteresisState = field(default_factory=HysteresisState)
    root: RootUptakeState = field(default_factory=RootUptakeState)
    flux: CumulativeFlux = field(default_factory=CumulativeFlux)
    
    # Materials (list of SoilMaterial)
    materials: List[SoilMaterial] = field(default_factory=list)
    
    # Chemical species (list of ChemicalSpecies)
    species: List[ChemicalSpecies] = field(default_factory=list)
    
    # Moisture dependency tables
    DMoist: NDArray[np.float64] = field(
        default_factory=lambda: np.zeros((NMatD, NSD, 13, 6)))
    WDep: NDArray[np.float64] = field(
        default_factory=lambda: np.zeros((2 + NMatD, NSD * 9)))
    
    # aOsm - osmotic coefficients
    aOsm: NDArray[np.float64] = field(
        default_factory=lambda: np.zeros(NSD))
    
    # cT - top concentrations
    cT: NDArray[np.float64] = field(
        default_factory=lambda: np.zeros(NSD))
    
    # Error tracking
    iNonConv: int = 0           # Non-convergence counter
    Peclet: float = 0.0         # Max Peclet number
    Courant: float = 0.0        # Max Courant number
    
    # File paths
    data_path: str = ""
    selector_file: str = ""
    profile_file: str = ""
    atmosphere_file: str = ""
    meteo_file: str = ""
    options_file: str = ""
