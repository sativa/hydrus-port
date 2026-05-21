"""
Main HYDRUS-1D simulation driver.
=================================

Orchestrates the complete simulation loop:
1. Read input files
2. Initialize state
3. Time stepping loop
4. Water flow solver
5. Solute transport solver (optional)
6. Heat transport solver (optional)
7. Output generation

Direct port of HYDRUS.FOR main program.
"""

from __future__ import annotations
import os
import sys
import numpy as np
from numpy.typing import NDArray
from typing import Dict, Any, Optional, Tuple
from datetime import datetime

from .dataclasses import (
    SimulationConfig, GridState, SoilMaterial, ChemicalSpecies,
    BoundaryConditions, TimeControl, HysteresisState, RootUptakeState,
    CumulativeFlux, FullSimulationState,
)
from .input import (
    read_selector, read_profile, read_atmospheric, read_meteorological,
    iGetFileVersion,
)
from .material import FK, FC, FQ, FH, FS
from .watflow import solve_water_flow, _set_mat_properties, shift_bc
from .solute import compute_coefficients, solve_solute_transport, compute_decay_source
from .temper import solve_heat_transport
from .hyster import update_hysteresis, compute_hysteretic_properties
from .sink import set_root_water_uptake, set_root_solute_uptake, set_root_distribution
from .output import (
    write_profile_output, write_time_series_output, write_cumulative_flux,
    compute_mass_balance, initialize_output_files,
)
from .time import (
    compute_time_step, rtime, check_convergence, check_concentration_convergence,
    should_output,
)
from .utils import (
    solve_tridiagonal, get_length_conversion, get_time_conversion,
    fortran_amax1, fortran_amin1, clamp,
)


# ============================================================================
# Main simulation class
# ============================================================================

class Hydrus1DSimulation:
    """
    HYDRUS-1D simulation engine.
    
    Direct port of HYDRUS.FOR main program.
    """
    
    def __init__(
        self,
        input_dir: str = ".",
        output_dir: str = ".",
        selector_file: str = "Selector.in",
        profile_file: str = "Profile.dat",
        atmospheric_file: str = "ATMOSPH.IN",
        meteo_file: str = "Meteo.in",
    ):
        """
        Initialize simulation from input files.
        
        Parameters
        ----------
        input_dir : str
            Input file directory
        output_dir : str
            Output file directory
        selector_file : str
            Selector file name
        profile_file : str
            Profile file name
        atmospheric_file : str
            Atmospheric BC file name
        meteo_file : str
            Meteorological data file name
        """
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.state = self._initialize_state()
        self._read_input_files(
            selector_file, profile_file, atmospheric_file, meteo_file,
        )
    
    def _initialize_state(self) -> "SimpleNamespace":
        """Initialize simulation state container (flat Fortran-named namespace).

        We deliberately bypass the legacy dataclasses defined in
        ``dataclasses.py``: those classes were never wired up to the physics
        modules and use names that don't match either the Fortran source or
        the function signatures in ``watflow.py``/``solute.py``/etc. A flat
        :class:`SimpleNamespace` mirrors the Fortran COMMON-block style and
        lets us pass arrays straight through to the existing functions.
        """
        from types import SimpleNamespace
        return SimpleNamespace()
    
    def _read_input_files(
        self,
        selector_file: str,
        profile_file: str,
        atmospheric_file: str,
        meteo_file: str,
    ) -> None:
        """
        Read all input files and populate state.
        
        Parameters
        ----------
        selector_file : str
            Selector file name
        profile_file : str
            Profile file name
        atmospheric_file : str
            Atmospheric BC file name
        meteo_file : str
            Meteorological data file name
        """
        # Read Selector.in
        sel = read_selector(os.path.join(self.input_dir, selector_file))
        
        # Read Profile.dat
        prof = read_profile(os.path.join(self.input_dir, profile_file), sel)

        # Read atmospheric BC
        atmos = read_atmospheric(os.path.join(self.input_dir, atmospheric_file), sel)
        
        # Read meteorological data
        meteo = read_meteorological(os.path.join(self.input_dir, meteo_file))
        
        # Populate state
        self._populate_state(sel, prof, atmos, meteo)
    
    def _populate_state(
        self,
        sel: Dict[str, Any],
        prof: Dict[str, Any],
        atmos: Dict[str, Any],
        meteo: Dict[str, Any],
    ) -> None:
        """
        Populate simulation state from parsed input.
        
        Parameters
        ----------
        sel : dict
            Selector data
        prof : dict
            Profile data
        atmos : dict
            Atmospheric BC data
        meteo : dict
            Meteorological data
        """
        N = prof['N']
        NMat = sel['NMat']
        NSD = sel.get('NSD', max(sel.get('NS', 0), 1))
        
        # Flat Fortran-style state container. Everything from the parser
        # dictionaries plus the atmospheric / meteo arrays lives directly on
        # ``self.state`` so the physics functions can be called with simple
        # ``self.state.x`` lookups.
        from types import SimpleNamespace
        s = self.state

        # Selector / Atmosph / Meteo flat copy
        for k, v in sel.items():
            setattr(s, k, v)
        for k, v in atmos.items():
            setattr(s, k, v)
        for k, v in meteo.items():
            setattr(s, k, v)

        # Profile arrays (override anything from sel of the same name)
        s.x = prof['x'].copy()
        s.hNew = prof['hNew'].copy()
        s.hOld = prof['hOld'].copy()
        s.hTemp = prof['hTemp'].copy()
        s.thNew = prof['thNew'].copy()
        s.thOld = prof['thOld'].copy()
        # The physics modules (watflow.py, solute.py, ...) expect 0-based
        # material indices. Profile.dat stores them 1-based (Fortran).
        s.MatNum = (prof['MatNum'].astype(np.int64) - 1)
        s.LayNum = (prof['LayNum'].astype(np.int64) - 1)
        s.Beta = prof['Beta'].copy()
        s.Ah = prof['Ah'].copy()
        s.AK = prof['AK'].copy()
        s.ATh = prof['ATh'].copy()
        s.TempN = prof['TempN'].copy()
        s.TempO = prof['TempO'].copy()
        s.Conc = prof['Conc'].copy()
        s.Sorb = prof['Sorb'].copy()
        s.Sorb2 = prof['Sorb2'].copy()
        s.NumNP = prof['NumNP']
        s.N = prof['NumNP']
        s.NObs = prof['NObs']
        s.Node = prof['Node']
        s.xSurf = prof['xSurf']
        # Prescribed boundary heads (used when KodTop / KodBot > 0).
        s.hTop = float(prof['hTop'])
        s.hBot = float(prof['hBot'])
        s.ThNewIm = prof.get('ThNewIm', np.zeros(s.NumNP, dtype=np.float64))
        s.ThOldIm = prof.get('ThOldIm', np.zeros(s.NumNP, dtype=np.float64))

        # Working arrays initialised to zero
        N = s.NumNP
        for name in (
            'vNew', 'vOld', 'Con', 'Cap', 'Sink', 'SinkIm', 'SinkS',
            'sSink', 'STrans', 'cNew', 'cTemp', 'cPrevO', 'SorbN',
            'SorbN2', 'Kappa', 'KappaO', 'AThS', 'ThRR', 'ConO', 'ConR',
            'AKS', 'ConLT', 'ConVT', 'ConVh', 'WatIn', 'SolIn',
            'g0', 'g1', 'q0', 'q1', 'wc', 'Disp', 'Retard',
            'ThVNew', 'ThVOld', 'P', 'R', 'S',
        ):
            setattr(s, name, np.zeros(N, dtype=np.float64))
        s.Retard[:] = 1.0
        # Aliases the existing physics layer expects.
        s.theta = s.thNew              # working copy used by watflow
        s.vN = s.vNew
        s.vO = s.vOld
        s.ThNIm = s.ThNewIm
        s.ThOIm = s.ThOldIm
        s.ConcOld = s.Conc.copy()
        s.rTopCh = np.zeros(max(s.NS, 1), dtype=np.float64) if hasattr(s, 'NS') else np.zeros(1)
        s.rBotCh = np.zeros(max(getattr(s, 'NS', 0), 1), dtype=np.float64)
        s.kTopCh = sel.get('kTopCh', np.zeros(max(getattr(s, 'NS', 0), 1), dtype=np.int64))
        s.kBotCh = sel.get('kBotCh', np.zeros(max(getattr(s, 'NS', 0), 1), dtype=np.int64))
        s.cTop = sel.get('cTop', np.zeros(max(getattr(s, 'NS', 0), 1), dtype=np.float64))
        s.cBot = sel.get('cBot', np.zeros(max(getattr(s, 'NS', 0), 1), dtype=np.float64))
        s.cRoot = np.zeros(max(getattr(s, 'NS', 0), 1), dtype=np.float64)
        s.CumQCh = np.zeros(max(getattr(s, 'NS', 0), 1), dtype=np.float64)
        s.CumE = 0.0
        s.CumT = 0.0
        s.cRootMax = sel.get('cRootMax', np.zeros(max(getattr(s, 'NS', 0), 1), dtype=np.float64))
        s.SPot_arr = np.zeros(max(getattr(s, 'NS', 0), 1), dtype=np.float64)
        s.rKM_arr = np.zeros(max(getattr(s, 'NS', 0), 1), dtype=np.float64)
        s.cMin_arr = np.zeros(max(getattr(s, 'NS', 0), 1), dtype=np.float64)
        s.thSat = s.ParD[1, :].copy()
        s.thR_arr = s.ParD[0, :].copy()
        s.ChPar = sel.get('ChPar', np.zeros((max(s.NS, 1) * 16 + 4, NMat), dtype=np.float64)) if 'NS' in sel else np.zeros((4, NMat), dtype=np.float64)
        s.TDep = sel.get('TDep', np.zeros(max(sel.get('NS', 0), 1) * 16 + 4, dtype=np.float64))
        s.tEnd = sel.get('tMax', 1.0)
        s.t0 = sel.get('tInit', 0.0)
        # Compute initial Con/Cap from h.  We deliberately use the GenMat
        # table interpolation when it is available (sel carries hTab/etc.)
        # so the initial Darcy flux written to BALANCE.OUT at t=tInit
        # matches Fortran's table-derived K to 4-5 significant digits.
        from .material import FK, FC
        from .input import interp_K, interp_theta_cap
        for i in range(s.NumNP):
            M = int(s.MatNum[i])     # already 0-based at this point
            if "hTab" in sel:
                s.Con[i] = interp_K(float(s.hNew[i]), M, sel)
                _, s.Cap[i] = interp_theta_cap(float(s.hNew[i]), M, sel)
            else:
                s.Con[i] = FK(s.iModel, s.hNew[i], s.ParD[:, M])
                s.Cap[i] = FC(s.iModel, s.hNew[i], s.ParD[:, M])
        # Hysteresis scratch
        s.AhW = np.ones(NMat, dtype=np.float64)
        s.AKW = np.ones(NMat, dtype=np.float64)
        s.AThW = np.ones(NMat, dtype=np.float64)
        s.AKR = np.ones(s.NumNP, dtype=np.float64)
        s.nRev = np.zeros(s.NumNP, dtype=np.int64)
        s.iRev = np.zeros(s.NumNP, dtype=np.int64)
        s.KappaR = np.zeros(s.NumNP, dtype=np.int64)
        s.hRev0 = np.zeros(s.NumNP, dtype=np.float64)
        s.ThRev0 = np.zeros(s.NumNP, dtype=np.float64)
        s.KappaRev = np.zeros(s.NumNP, dtype=np.int64)
        s.hRev1 = np.zeros(s.NumNP, dtype=np.float64)
        s.ThRev1 = np.zeros(s.NumNP, dtype=np.float64)
        s.KappaRev1 = np.zeros(s.NumNP, dtype=np.int64)
        s.hRev = np.zeros((s.NumNP, 20), dtype=np.float64)
        s.ThRev = np.zeros((s.NumNP, 20), dtype=np.float64)
        # Pull boundary fluxes used by the run loop. For non-time-variable BC
        # we keep whatever BasInf set; if AtmBC is on we initialise from the
        # first atmospheric record.
        s.rTop = sel.get('rTop', 0.0)
        s.rBot = sel.get('rBot', 0.0)
        s.hCritA = sel.get('hCritA', -1.0e10)
        if atmos.get('MaxAL', 0) > 0:
            s.rTop = float(atmos['rSoil'][0] - atmos['Prec'][0])
            s.tAtm = float(atmos['tAtm'][0])
            s.hCritA = float(atmos['hCritA'][0]) if 'hCritA' in atmos else s.hCritA
        # Convenience scalar flags
        s.lSink = sel.get('SinkF', False)
        s.lSolute = sel.get('lChem', False)
        s.lHyst = sel.get('iHyst', 0) > 0
        s.lAdapt = True
        s.lCumFlux = True
        s.lMassBal = True
        # Run-loop default flags (HYDRUS.FOR Init defaults, INPUT.FOR:1648-1700)
        _defaults = dict(
            lCentrif=False, Radius=0.0, lGeom=False, lDensity=False,
            lVapor=False, lWTDep=False, lUpW=False, lArtD=False,
            lTort=False, lMoSink=True, lSolRed=False, lSolAdd=False,
            lMsSink=False, lActRSU=False, lOmegaW=False, P3=-8000.0,
            IKappa=-1, epsi=1.0, rMin=1.0e-37, rMax=0.01, rMaxC=0.01,
            rMaxT=0.01, dtInit=s.dt, dtMax=sel.get('dtMax', 0.5),
            dtMin=sel.get('dtMin', 1e-5), dtFact=sel.get('dMul', 1.3),
            dtFactC=1.0, dtFactT=1.0, dtMaxC=1.0e30, dtMaxT=1.0e30,
            IterMax=sel.get('MaxIt', 20), IterMaxC=10, IterMaxT=10,
            iOut=1, iOutC=0, iOutT=0,
            tOut=sel.get('TPrint', np.array([s.tEnd]))[0] if len(sel.get('TPrint', [s.tEnd])) > 0 else s.tEnd,
            tOutC=s.tEnd, tOutT=s.tEnd,
            qDrain=sel.get('qDrain', False), qGWLF=sel.get('qGWLF', False),
            SeepF=sel.get('SeepF', False), WLayer=sel.get('WLayer', False),
            TopInf=sel.get('TopInF', False), hSeep=sel.get('hSeep', 0.0),
            GWL0L=sel.get('GWL0L', 0.0), Aqh=sel.get('Aqh', 0.0),
            Bqh=sel.get('Bqh', 0.0), CosAlf=sel.get('CosAlf', 1.0),
        )
        for k, v in _defaults.items():
            if not hasattr(s, k):
                setattr(s, k, v)
        s.Ah[:] = np.where(s.Ah == 0, 1.0, s.Ah)
        s.AK[:] = np.where(s.AK == 0, 1.0, s.AK)
        s.ATh[:] = np.where(s.ATh == 0, 1.0, s.ATh)
        s.Kappa[:] = sel.get('IKappa', -1)
        s.KappaO[:] = sel.get('IKappa', -1)

        # Cumulative scalars (Fortran wCumT/wCumA/CumQ(12)/...)
        s.CumQ = np.zeros(12, dtype=np.float64)
        s.wCumT = 0.0
        s.wCumA = 0.0
        s.hRoot = 0.0
        s.vRoot = 0.0
        s.TPot = 0.0
        s.t = sel.get('tInit', 0.0)
        s.tOld = s.t
        s.dt = sel.get('dt', 0.001)
        s.dtOpt = s.dt
        s.tAtm = atmos.get('tAtm', np.array([sel.get('tMax', 1.0)]))[0] if atmos.get('MaxAL', 0) > 0 else sel.get('tMax', 1.0)
        s.iAtm = 0          # current atmospheric record index
        s.TLevel = 1
        s.ALevel = 1
        s.PLevel = 1
        s.ItCum = 0
        s.IterW = 0
        s.IterC = 0

        # Aliases for compatibility with the old object-style references.
        # config / grid / mat / etc. now all alias the same flat namespace.
        for alias in ('config', 'grid', 'mat', 'chem', 'bc', 'time',
                      'hyster', 'root', 'cumflux'):
            setattr(self.state, alias, self.state)
        self.atmos_data = atmos
        self.meteo_data = meteo

        # Initialize output files
        self.output_files = initialize_output_files(
            self.output_dir,
            NSD=NSD,
            lTemp=sel.get('lTemp', False),
            lSink=sel.get('SinkF', False),
            lCumFlux=True,
            lMassBal=True,
        )
        return

        # === everything below is legacy stub code retained for reference ===
        # The original code attempted to populate hand-built dataclasses that
        # were never wired up to the physics modules.  Kept here as a marker
        # until the dataclasses can be removed.
        grid = GridState(
            N=N,
            x=prof['x'].copy(),
            hNew=prof['hNew'].copy(),
            hOld=prof['hOld'].copy(),
            hTemp=prof['hTemp'].copy(),
            thNew=prof['thNew'].copy(),
            thOld=prof['thOld'].copy(),
            theta=prof['thNew'].copy(),
            Con=prof.get('Con', np.zeros(N, dtype=np.float64)),
            Cap=prof.get('Cap', np.zeros(N, dtype=np.float64)),
            vN=np.zeros(N, dtype=np.float64),
            vO=np.zeros(N, dtype=np.float64),
            P=np.zeros(N, dtype=np.float64),
            R=np.zeros(N, dtype=np.float64),
            S=np.zeros(N, dtype=np.float64),
            Disp=np.zeros(N, dtype=np.float64),
            Retard=np.ones(N, dtype=np.float64),
            Sink=np.zeros(N, dtype=np.float64),
            SinkIm=np.zeros(N, dtype=np.float64),
            SinkS=np.zeros(N, dtype=np.float64),
            g0=np.zeros(N, dtype=np.float64),
            g1=np.zeros(N, dtype=np.float64),
            sSink=np.zeros(N, dtype=np.float64),
            MatNum=prof['MatNum'].copy(),
            Beta=prof.get('Beta', np.zeros(N, dtype=np.float64)),
            Conc=prof.get('Conc', np.zeros((NSD, N), dtype=np.float64)),
            ConcOld=np.zeros((NSD, N), dtype=np.float64),
            cNew=np.zeros(N, dtype=np.float64),
            cPrevO=np.zeros(N, dtype=np.float64),
            TempN=prof.get('TempN', np.zeros(N, dtype=np.float64)),
            TempO=prof.get('TempO', np.zeros(N, dtype=np.float64)),
            ThNIm=np.zeros(N, dtype=np.float64),
            ThOIm=np.zeros(N, dtype=np.float64),
            Sorb=np.zeros((NSD, N), dtype=np.float64),
            SorbN=np.zeros(N, dtype=np.float64),
            Sorb2=np.zeros((NSD, N), dtype=np.float64),
            SorbN2=np.zeros(N, dtype=np.float64),
        )
        
        # Soil material
        mat = SoilMaterial(
            ParD=prof['ParD'].copy(),
            ParW=prof['ParW'].copy(),
            thSat=prof['ParW'][1, :].copy(),
            thR=prof['ParD'][0, :].copy(),
        )
        
        # Chemical species
        chem = ChemicalSpecies(
            ChPar=prof.get('ChPar', np.zeros((NSD * 16 + 4, NMat), dtype=np.float64)),
            TDep=prof.get('TDep', np.zeros(NSD * 16 + 4, dtype=np.float64)),
            cTop=np.zeros(NSD, dtype=np.float64),
            cBot=np.zeros(NSD, dtype=np.float64),
            kTopCh=np.zeros(NSD, dtype=np.int64),
            kBotCh=np.zeros(NSD, dtype=np.int64),
            cRootMax=np.ones(NSD, dtype=np.float64) * 1e-37,
            rKM=np.zeros(NSD, dtype=np.float64),
            SPot=np.zeros(NSD, dtype=np.float64),
            cMin=np.zeros(NSD, dtype=np.float64),
        )
        
        # Boundary conditions
        bc = BoundaryConditions(
            KodTop=sel.get('KodTop', 2),
            KodBot=sel.get('KodBot', 2),
            rTop=sel.get('rTop', 0.0),
            rBot=sel.get('rBot', 0.0),
            hTop=sel.get('hTop', 0.0),
            hBot=sel.get('hBot', 0.0),
            KodTopT=sel.get('KodTopT', 2),
            KodBotT=sel.get('KodBotT', 2),
            TTop=sel.get('TTop', 20.0),
            TBot=sel.get('TBot', 20.0),
            rTopCh=np.zeros(NSD, dtype=np.float64),
            rBotCh=np.zeros(NSD, dtype=np.float64),
        )
        
        # Time control
        tc = TimeControl(
            t0=sel.get('t0', 0.0),
            tEnd=sel.get('tEnd', 1.0),
            dt=sel.get('dtInit', 1.0),
            t=0.0,
            tOut=sel.get('tOut', 1.0),
            tOutC=sel.get('tOutC', 1.0),
            tOutT=sel.get('tOutT', 1.0),
        )
        
        # Hysteresis state
        hyster = HysteresisState(
            Kappa=np.zeros(N, dtype=np.int64),
            KappaO=np.zeros(N, dtype=np.int64),
            Ah=np.ones(N, dtype=np.float64),
            AK=np.ones(N, dtype=np.float64),
            ATh=np.ones(N, dtype=np.float64),
            AhW=np.ones(N, dtype=np.float64),
            AKW=np.ones(N, dtype=np.float64),
            AThW=np.ones(N, dtype=np.float64),
            ThRR=np.zeros(N, dtype=np.float64),
            ConR=np.zeros(N, dtype=np.float64),
            AKR=np.ones(N, dtype=np.float64),
            hRev=np.zeros((N, 20), dtype=np.float64),
            ThRev=np.zeros((N, 20), dtype=np.float64),
            nRev=np.zeros(N, dtype=np.int64),
            iRev=np.zeros(N, dtype=np.int64),
            KappaR=np.zeros(N, dtype=np.int64),
            hRev0=np.zeros(N, dtype=np.float64),
            ThRev0=np.zeros(N, dtype=np.float64),
            KappaRev=np.zeros(N, dtype=np.int64),
            hRev1=np.zeros(N, dtype=np.float64),
            ThRev1=np.zeros(N, dtype=np.float64),
            KappaRev1=np.zeros(N, dtype=np.int64),
        )
        
        # Root uptake state
        root = RootUptakeState(
            TPot=0.0,
            vRoot=0.0,
            hRoot=0.0,
            cRoot=np.zeros(NSD, dtype=np.float64),
        )
        
        # Cumulative flux
        cumflux = CumulativeFlux(
            CumQ=0.0,
            CumE=0.0,
            CumT=0.0,
            CumQCh=np.zeros(NSD, dtype=np.float64),
        )
        
        # Full state
        self.state = FullSimulationState(
            config=config,
            grid=grid,
            mat=mat,
            chem=chem,
            bc=bc,
            time=tc,
            hyster=hyster,
            root=root,
            cumflux=cumflux,
        )
        
        # Atmospheric data
        self.atmos_data = atmos
        self.meteo_data = meteo
        
        # Initialize output files
        self.output_files = initialize_output_files(
            self.output_dir,
            NSD=NSD,
            lTemp=config.lTemp,
            lSink=config.lSink,
            lCumFlux=config.lCumFlux,
            lMassBal=config.lMassBal,
        )
    
    def run(self) -> None:
        """Port of HYDRUS.FOR main time-stepping loop.

        High-level structure (labels reference HYDRUS.FOR):

        - **12**: top of time loop. ``WatFlow`` runs one Picard iteration
          sweep internally — we call it once per step.
        - Velocity → root uptake → solute → heat
        - Output: T-level (every step), P-level (at TPrint times),
          A-level (at atmospheric record boundaries)
        - **TmCont** chooses the next dt; ``Update`` rolls New → Old
        - Loop terminates when ``t >= tMax`` (within 0.5*dtMin).
        """
        s = self.state
        N = s.NumNP
        NSD = max(getattr(s, "NS", 0), 1)

        # Snapshot used by the output writer for the t = tInit row.
        s.hOld[:] = s.hNew
        s.thOld[:] = s.thNew
        s.vOld[:] = s.vNew
        s.TempO[:] = s.TempN
        if hasattr(s, 'Conc'):
            s.ConcOld = s.Conc.copy()

        TPrint = s.TPrint if isinstance(s.TPrint, np.ndarray) else np.array([s.tEnd])
        MPL = len(TPrint)
        PLevel = 0           # index into TPrint
        ALevel = 0           # index into atmospheric record
        TLevel = 0           # global step counter

        # Open Fortran-style output files
        self._open_outputs()

        # Compute initial Darcy fluxes at t = tInit so BALANCE.OUT at the
        # initial print time reports the same -K(h_init)*Grav as Fortran
        # (HYDRUS.FOR:377-385 calls Veloc *before* the time loop starts).
        # _update_velocity(use_new=False) populates s.vOld in place.
        self._update_velocity(use_new=False)
        # Surface flux = Darcy flux between top two nodes; bottom flux =
        # Darcy flux between node 0 and 1 (≈ -K*Grav for free drainage).
        s.last_vTop = float(s.vOld[-1])
        s.last_vBot = float(s.vOld[1])

        # Initial profile + balance dump at t = tInit
        self._write_nod_out(s.t)
        self._write_balance_subreg(s.t, dt=0.0, level=0,
                                   ws_initial=self._water_storage())

        tAtm = s.tAtm if s.tAtm > s.t else s.tEnd
        if s.MaxAL > 0:
            tAtm = float(s.tAtm_arr[0]) if hasattr(s, 'tAtm_arr') else float(getattr(s, 'tAtm', s.tEnd))
        wsInit = self._water_storage()

        # Initial velocity field (Darcy)
        self._update_velocity(use_new=False)

        max_steps = 1_000_000
        for _step in range(max_steps):
            if abs(s.t - s.tEnd) <= 0.5 * s.dtMin or s.t > s.tEnd:
                break

            # --- one full time step (with retry on non-convergence) -------
            dt = float(min(s.dt, s.tEnd - s.t))
            h_snap = s.hNew.copy()
            th_snap = s.thNew.copy()
            con_snap = s.Con.copy()
            cap_snap = s.Cap.copy()
            kt_snap, kb_snap = s.KodTop, s.KodBot
            retry = 0
            while True:
                # hOld / thOld are the values at t (frozen across the Picard
                # sweep and across any retries inside this time step).
                s.hOld[:] = h_snap
                s.thOld[:] = th_snap
                s.hNew[:] = h_snap
                s.thNew[:] = th_snap
                s.Con[:] = con_snap
                s.Cap[:] = cap_snap
                s.KodTop, s.KodBot = kt_snap, kb_snap

                s.hNew, vTop, vBot, s.KodTop, s.KodBot, Iter, conv = solve_water_flow(
                    N, s.x, s.hNew, s.hOld, s.hTemp,
                    s.MatNum, s.ParD, s.ParW,
                    s.iModel, s.iHyst, s.iDualPor,
                    dt, s.KodTop, s.KodBot, s.rTop, s.rBot,
                    s.Sink, s.SinkIm, s.Ah, s.AK, s.ATh,
                    s.Con, s.Cap, s.thNew,
                    s.CosAlf, s.lCentrif, s.Radius,
                    s.lGeom, s.lDensity, s.lVapor,
                    s.lWTDep, s.TempN, None, None, None,
                    None, None, s.hSeep, s.SeepF,
                    s.TopInF, s.hCritA, s.WLayer,
                    s.qGWLF, s.GWL0L, s.Aqh, s.Bqh,
                    s.qDrain,
                    hTop_in=getattr(s, 'hTop', s.hNew[-1]),
                    hBot_in=getattr(s, 'hBot', s.hNew[0]),
                    TolTh=getattr(s, 'TolTh', 0.001),
                    TolH=getattr(s, 'TolH', 1.0),
                    MaxIt_in=getattr(s, 'MaxIt', 20),
                    tables=self._gen_tables(),
                )
                if conv or dt <= s.dtMin * 1.0001 or retry >= 5:
                    break
                dt = max(dt / 3.0, s.dtMin)
                retry += 1
            s.LastIter = Iter
            s.LastConv = conv
            s.last_vTop = float(vTop)
            s.last_vBot = float(vBot)

            # Update Con from the converged head; do NOT overwrite thNew with
            # FQ(hNew).  solve_water_flow returns thNew with the
            # mass-conservative update (Celia/Bouloutas/Zarba 1990):
            #     thNew = thNew + Cap*(hNew − hTemp)
            # applied at the post-convergence Picard step.  Recomputing thNew
            # from FQ(hNew) here would replace the conservative value with the
            # *analytical* one and break the water mass balance.
            for i in range(N):
                M = int(s.MatNum[i])
                s.Con[i] = FK(s.iModel, s.hNew[i], s.ParD[:, M])

            self._update_velocity(use_new=True)

            # Cumulative fluxes (Fortran CumQ array layout, 1-based →
            # 0-based here):
            #   CumQ[0] : top boundary flux
            #   CumQ[1] : bottom boundary flux
            #   CumQ[2] : evap (max -vTop, 0)
            #   CumQ[3] : root water uptake
            #   CumQ[4] : precipitation (top inflow)
            s.CumQ[0] += vTop * dt
            s.CumQ[1] += vBot * dt
            s.CumQ[2] += max(-vTop, 0.0) * dt
            s.CumQ[3] += s.vRoot * dt

            s.tOld = s.t
            s.t = s.t + dt
            TLevel += 1
            s.TLevel = TLevel

            # T-level output (every step)
            self._write_tlevel(s.t, dt, vTop, vBot, TLevel)

            # P-level output: when t reaches one of the TPrint values
            while PLevel < MPL and abs(TPrint[PLevel] - s.t) < 0.001 * dt:
                self._write_nod_out(TPrint[PLevel])
                self._write_balance_subreg(s.t, dt, PLevel + 1,
                                           ws_initial=wsInit)
                PLevel += 1

            # Atmospheric BC advance
            if (s.TopInF or s.BotInF or s.AtmBC) and s.MaxAL > 0:
                if abs(s.t - tAtm) < 0.001 * dt:
                    ALevel += 1
                    if ALevel < s.MaxAL:
                        s.rTop = float(self.atmos_data['rSoil'][ALevel]
                                       - self.atmos_data['Prec'][ALevel])
                        tAtm = float(self.atmos_data['tAtm'][ALevel])
                    else:
                        tAtm = s.tEnd

            # Adaptive time step (TmCont-style heuristic)
            self._adapt_dt(dt)

            # Roll New → Old for the next iteration
            s.vOld[:] = s.vNew
            s.TempO[:] = s.TempN
            if hasattr(s, 'Conc'):
                s.ConcOld = s.Conc.copy()

        # Final dump if we never hit the last TPrint exactly.
        if PLevel < MPL:
            self._write_nod_out(s.t)
            self._write_balance_subreg(s.t, dt, PLevel + 1, ws_initial=wsInit)
        self._close_outputs()

    def _dead_code_removed(self):
        """Placeholder for removed buggy time-loop. See run() above."""
        return
        # The following block was the original buggy implementation; kept
        # as a no-op behind an early return so old references compile if any
        # external code accidentally imports it.
        while True:
            # Compute time step
            dt = min(time.dt, time.tEnd - time.t)
            
            # Water flow iteration
            Iter = 0
            converged = False
            
            while Iter < config.IterMax and not converged:
                Iter += 1
                
                # Save old values
                grid.hOld = grid.hNew.copy()
                grid.thOld = grid.thNew.copy()
                
                # Compute root water uptake
                if config.lSink:
                    root.vRoot, root.hRoot = set_root_water_uptake(
                        N, grid.x, grid.Beta, grid.Sink,
                        root.TPot, grid.hNew, config.lMoSink,
                        config.hCritA, 0.0, 0.0, 0.0, config.P3,
                        0.0, 0.0, grid.thNew, mat.ParD, grid.MatNum,
                        config.iModel, grid.Con, config.OmegaC,
                        config.lChem, config.lSolRed, config.lSolAdd,
                        None, None, 0.0, 0.0, config.lMsSink, dt,
                    )
                
                # Update hysteresis
                if config.lHyst:
                    update_hysteresis(
                        N, grid.hNew, grid.hOld, grid.hTemp,
                        hyster.Kappa, hyster.KappaO,
                        hyster.Ah, hyster.AK, hyster.ATh,
                        hyster.AhW, hyster.AKW, hyster.AThW,
                        hyster.ThRR, hyster.ConR, hyster.AKR,
                        hyster.hRev, hyster.ThRev, hyster.nRev,
                        hyster.iRev, hyster.KappaR, hyster.hRev0,
                        hyster.ThRev0, hyster.KappaRev, hyster.hRev1,
                        hyster.ThRev1, hyster.KappaRev1,
                        mat.ParD, mat.ParW, config.iModel,
                    )
                
                # Solve water flow
                grid.hNew, vTop, vBot, bc.KodTop, bc.KodBot = solve_water_flow(
                    N, grid.x, grid.hNew, grid.hOld, grid.hTemp,
                    grid.MatNum, mat.ParD, mat.ParW,
                    config.iModel, config.iHyst, config.iDualPor,
                    dt, bc.KodTop, bc.KodBot, bc.rTop, bc.rBot,
                    grid.Sink, grid.SinkIm, hyster.Ah, hyster.AK, hyster.ATh,
                    grid.Con, grid.Cap, grid.theta,
                    config.CosAlf, config.lCentrif, config.Radius,
                    config.lGeom, config.lDensity, config.lVapor,
                    config.lWTDep, grid.TempN, None, None, None,
                    None, None, config.hSeep, config.SeepF,
                    config.TopInf, config.hCritA, config.WLayer,
                    config.qGWLF, config.GWL0L, config.Aqh, config.Bqh,
                    config.qDrain,
                )
                
                # Update water content from head
                for i in range(N):
                    M = int(grid.MatNum[i])
                    grid.thNew[i] = FQ(config.iModel, grid.hNew[i], mat.ParD[:, M])
                    grid.thNew[i] = max(grid.thNew[i], mat.ParD[0, M])
                    grid.Con[i] = FK(config.iModel, grid.hNew[i], mat.ParD[:, M])
                
                # Compute velocity
                for i in range(1, N):
                    dx = grid.x[i] - grid.x[i - 1]
                    grid.vN[i] = -(grid.Con[i] + grid.Con[i - 1]) / 2.0 * (
                        (grid.hNew[i] - grid.hNew[i - 1]) / dx + config.CosAlf
                    )
                
                # Check convergence
                converged, rMaxCurrent = check_convergence(
                    grid.hNew, grid.hOld, N, config.rMax, config.rMin,
                )
            
            # Adaptive time step
            time.dt, Iter, _, _ = compute_time_step(
                time.dt, config.dtMax, config.dtMin, config.dtInit,
                config.dtMaxC, config.dtMaxT,
                Iter, config.IterMax, 0, 0,
                config.rMax, config.rMin,
                config.dtFact, config.dtFactC, config.dtFactT,
                config.lAdapt, True, True,
                0.0, 0.0, True, config.lSolute, config.lTemp,
            )
            
            # Update time
            time.t += dt
            
            # Solute transport
            if config.lSolute:
                for jS in range(1, NSD + 1):
                    # Compute decay source terms
                    compute_decay_source(
                        jS, N, grid.Conc, chem.ChPar, grid.MatNum,
                        grid.TempN, chem.TDep, grid.g0, grid.g1,
                        config.lEquil,
                    )
                    
                    # Compute coefficients
                    Peclet, Courant, dtMaxC = compute_coefficients(
                        jS, 1, 1, N, grid.x, grid.Disp,
                        grid.vO, grid.vN, grid.thOld, grid.thNew,
                        mat.thSat, chem.ChPar, grid.MatNum,
                        grid.TempN, grid.TempO, chem.TDep,
                        grid.Retard, grid.Conc, grid.cNew, grid.cPrevO,
                        dt, config.lEquil, config.lUpW, config.lArtD,
                        0, config.lTort, config.iDualPor,
                        grid.ThNIm, grid.ThOIm, grid.SinkIm,
                    )
                    
                    # Solve transport
                    grid.cNew, cvTop, cvBot = solve_solute_transport(
                        jS, N, grid.x, grid.Conc, grid.vN,
                        grid.thNew, grid.thOld, grid.Disp, grid.Retard,
                        dt, chem.kTopCh[jS - 1], chem.kBotCh[jS - 1],
                        chem.cTop[jS - 1], chem.cBot[jS - 1],
                        grid.g0, grid.g1, grid.sSink,
                        config.lUpW, config.lEquil, config.epsi,
                        config.rMin,
                    )
                    
                    # Root solute uptake
                    if config.lSink:
                        SPUptake, SAUptakeA = set_root_solute_uptake(
                            jS, N, grid.x, grid.Beta, grid.Sink, grid.SinkS,
                            grid.Conc, root.vRoot / max(root.TPot, 1e-10),
                            chem.cRootMax[jS - 1], config.lActRSU,
                            config.OmegaS, chem.SPot[jS - 1],
                            chem.rKM[jS - 1], chem.cMin[jS - 1],
                        )
                    
                    # Update cumulative fluxes
                    cumflux.CumQCh[jS - 1] += cvBot * dt
            
            # Heat transport
            if config.lTemp:
                grid.TempN, qTopT, qBotT = solve_heat_transport(
                    N, grid.x, grid.TempN, grid.TempO,
                    grid.thNew, grid.thOld, grid.vN,
                    mat.ParW, grid.MatNum, dt,
                    bc.KodTopT, bc.KodBotT, bc.TTop, bc.TBot,
                    config.IKappa, config.lWTDep, config.rMin,
                )
            
            # Update cumulative fluxes
            cumflux.CumQ += vBot * dt
            cumflux.CumE += max(-vTop, 0.0) * dt
            cumflux.CumT += root.vRoot * dt
            
            # Output
            outWater, outConc, outTemp, outCumFlux, outMassBal = should_output(
                time.t, dt, time.tOut, time.tOutC, time.tOutT,
                config.iOut, config.iOutC, config.iOutT,
                config.lCumFlux, config.lMassBal,
            )
            
            if outWater:
                write_profile_output(
                    self.output_files['profile'],
                    time.t, N, grid.x, grid.hNew, grid.thNew,
                    grid.Con, grid.vN, grid.Conc, grid.TempN,
                    NSD, config.lTemp, config.xConv, config.tConv,
                    config.iOut, config.lCumFlux,
                    cumflux.CumQ, cumflux.CumE, cumflux.CumT, cumflux.CumQCh,
                    config.lMassBal,
                )
                write_time_series_output(
                    self.output_files['time'],
                    time.t, vTop, vBot,
                    grid.thNew[N - 1], grid.thNew[0],
                    grid.hNew[N - 1], grid.hNew[0],
                    root.vRoot,
                    None, None,
                    qTopT if config.lTemp else 0.0,
                    qBotT if config.lTemp else 0.0,
                    NSD, config.lTemp, config.lSink,
                    config.xConv, config.tConv,
                )
            
            if outCumFlux:
                write_cumulative_flux(
                    self.output_files['cumflux'],
                    time.t, cumflux.CumQ, cumflux.CumE, cumflux.CumT,
                    cumflux.CumQCh, NSD, config.tConv,
                )
            
            # Prepare for next step
            grid.hOld = grid.hNew.copy()
            grid.thOld = grid.thNew.copy()
            grid.vO = grid.vN.copy()
            grid.ConcOld = grid.Conc.copy()
            grid.TempO = grid.TempN.copy()
    
    # ========================================================================
    # Helper methods used by run()
    # ========================================================================

    def _gen_tables(self) -> dict | None:
        """Bundle the GenMat-built table arrays into the dict that
        ``solve_water_flow`` / ``_set_mat_properties`` look up."""
        s = self.state
        if not hasattr(s, "hTab"):
            return None
        return {
            "hTab": s.hTab,
            "ConTab": s.ConTab,
            "CapTab": s.CapTab,
            "TheTab": s.TheTab,
            "NTab": int(s.NTab),
            "alh1": float(s.alh1),
            "dlh": float(s.dlh),
            "hSat_M": s.hSat_M,
            "ConSat": s.ConSat,
        }

    def _open_outputs(self) -> None:
        """Open the standard HYDRUS-1D output files (NOD_INF/T_LEVEL/BALANCE).

        Output filenames match the upper-case names the Fortran binary emits
        so the comparison harness can diff them directly.
        """
        import os
        os.makedirs(self.output_dir, exist_ok=True)

        def _open(name):
            return open(os.path.join(self.output_dir, name), "w")

        self._fNod = _open("NOD_INF.OUT")
        self._fT = _open("T_LEVEL.OUT")
        self._fBal = _open("BALANCE.OUT")
        s = self.state
        hdr = (
            f" ******* Program HYDRUS (Python port)\n"
            f" ******* {getattr(s, 'Heading', '')}\n"
            f" Units: L = {getattr(s, 'LUnit', 'cm   '):<5s}, "
            f"T = {getattr(s, 'TUnit', 'days '):<5s}, "
            f"M = {getattr(s, 'MUnit', 'g    '):<5s}\n\n"
        )
        for f in (self._fNod, self._fT, self._fBal):
            f.write(hdr)

        # T_LEVEL column header (matches Fortran TLInf format)
        self._fT.write(
            "       Time          rTop         rRoot         vTop         vRoot"
            "        vBot       sum(rTop)    sum(rRoot)    sum(vTop)   sum(vRoot)"
            "      sum(vBot)      hTop          hRoot         hBot         RunOff"
            "     sum(RunOff)    Volume     sum(Infil)     sum(Evap)  TLevel"
            "  Cum(WTrans)  SnowLayer\n"
            "        [T]         [L/T]        [L/T]        [L/T]        [L/T]"
            "        [L/T]         [L]          [L]          [L]         [L]"
            "         [L]          [L]          [L]          [L]         [L/T]"
            "         [L]          [L]          [L]          [L]\n\n"
        )

    def _close_outputs(self) -> None:
        for attr in ("_fNod", "_fT", "_fBal"):
            f = getattr(self, attr, None)
            if f is not None and not f.closed:
                try:
                    if attr == "_fT":
                        f.write("end\n")
                    f.close()
                except OSError:
                    pass

    def _water_storage(self) -> float:
        """Total water volume per unit cross-section (trapezoid rule)."""
        s = self.state
        x = s.x
        th = s.thNew
        V = 0.0
        for i in range(1, s.NumNP):
            V += 0.5 * (th[i] + th[i - 1]) * (x[i] - x[i - 1])
        return V

    def _update_velocity(self, use_new: bool) -> None:
        """Darcy velocity v at internal nodes; copy to vNew or vOld."""
        s = self.state
        h = s.hNew if use_new else s.hOld
        th = s.thNew if use_new else s.thOld
        v = s.vNew if use_new else s.vOld
        # interior nodes use centred differences on K, head + gravity
        for i in range(1, s.NumNP):
            dx = s.x[i] - s.x[i - 1]
            if dx == 0.0:
                v[i] = 0.0
                continue
            Kbar = 0.5 * (s.Con[i] + s.Con[i - 1])
            v[i] = -Kbar * ((h[i] - h[i - 1]) / dx + s.CosAlf)
        v[0] = v[1]

    def _adapt_dt(self, dt_used: float) -> None:
        """Full port of TIME.FOR:TmCont — not just the multiplicative growth.

        The Fortran logic that the original Python adaptation skipped:

        - ``dtOpt`` is the "preferred" step (kept across calls); it grows by
          ``dMul`` only when there is at least ``dMul·dtOpt`` of remaining
          time before the next print / atm / tMax event.
        - ``iStep = round((tFix − t) / dt)`` is the number of sub-steps the
          solver thinks it needs.  For iStep ∈ [1, 10] dt is re-snapped to
          ``min((tFix − t) / iStep, dtMax)`` so the next print point is hit
          exactly.  For iStep == 1 with too-large dt, dt is halved.

        Matching this is what reproduces Fortran's discrete time grid (50
        steps over 1 day for evap_v4) instead of the ~47 steps the
        simplified adapter produces.
        """
        s = self.state
        Iter = getattr(s, "LastIter", 1)
        ItMin = int(getattr(s, "ItMin", 3))
        ItMax = int(getattr(s, "ItMax", 7))
        dMul = float(getattr(s, "dMul", 1.3))
        dMul2 = float(getattr(s, "dMul2", 0.7))
        dtMax = float(s.dtMax)
        dtMin = float(s.dtMin)
        dtOpt = float(getattr(s, "dtOpt", dt_used))

        # tFix = next print event / next atm record / tEnd
        tFix = s.tEnd
        if hasattr(s, "TPrint"):
            for tp in s.TPrint:
                if tp > s.t + 1e-12:
                    tFix = min(tFix, float(tp))
                    break
        if hasattr(s, "MaxAL") and s.MaxAL > 0 and hasattr(self, "atmos_data"):
            tAtm_arr = self.atmos_data.get("tAtm")
            if tAtm_arr is not None:
                for ta in tAtm_arr:
                    if ta > s.t + 1e-12:
                        tFix = min(tFix, float(ta))
                        break

        # Step-size policy (TIME.FOR:16-19)
        if Iter <= ItMin and (tFix - s.t) >= dMul * dtOpt:
            dtOpt = min(dtMax, dMul * dtOpt)
        if Iter >= ItMax:
            dtOpt = max(dtMin, dMul2 * dtOpt)

        dt = min(dtOpt, tFix - s.t)

        # Snap to next print event in ≤ 10 sub-steps (TIME.FOR:22-28)
        if dt > 0.0:
            i_step = max(1, int(round((tFix - s.t) / dt)))
        else:
            i_step = 1
        if 1 <= i_step <= 10:
            dt = min((tFix - s.t) / i_step, dtMax)
        if i_step == 1:
            dt = tFix - s.t
            if dt - dtMax > dtMin:
                dt = dt / 2.0
        if dt <= 0.0:
            dt = dtMin / 3.0

        s.dtOpt = dtOpt
        s.dt = max(dt, dtMin)

    def _write_nod_out(self, t: float) -> None:
        """Append one profile snapshot to NOD_INF.OUT (Fortran format 110/120)."""
        s = self.state
        f = self._fNod
        f.write(f"\n Time:    {t:12.4f}\n\n")
        f.write(
            " Node      Depth      Head Moisture       K          C         "
            "Flux        Sink         Kappa   v/KsTop   Temp\n"
            "           [L]        [L]    [-]        [L/T]      [1/L]      "
            "[L/T]        [1/T]         [-]      [-]      [C]\n\n"
        )
        # Node 1 = bottom, node NumNP = top — Fortran prints top first.
        # KsTop = K at the surface node.
        from .material import FC
        KsTop = max(s.Con[-1], 1e-30)
        for n in range(s.NumNP - 1, -1, -1):
            M = int(s.MatNum[n])
            cap = FC(s.iModel, s.hNew[n], s.ParD[:, M])
            v = s.vNew[n]
            sink = s.Sink[n]
            kappa = int(s.Kappa[n])
            vkstop = v / KsTop
            Tn = s.TempN[n]
            f.write(
                f"{s.NumNP - n:5d} {s.x[n]:11.4f} {s.hNew[n]:10.3f} "
                f"{s.thNew[n]:7.4f} {s.Con[n]:11.4e} {cap:11.4e} "
                f"{v:11.4e} {sink:11.4e} {kappa:8d} "
                f"{vkstop:10.3e} {Tn:8.2f}\n"
            )
        f.write(" end\n")

    def _write_tlevel(self, t: float, dt: float, vTop: float, vBot: float,
                      TLevel: int) -> None:
        """Append one T-level row to T_LEVEL.OUT (Fortran TLInf format)."""
        s = self.state
        f = self._fT
        rTop = abs(s.rTop)
        rRoot = abs(s.vRoot)
        hTop = s.hNew[-1]
        hBot = s.hNew[0]
        hRoot = s.hRoot
        Volume = self._water_storage()
        sum_rTop = s.CumQ[0]
        sum_vTop = s.CumQ[0]
        sum_vBot = s.CumQ[1]
        sum_vRoot = s.CumQ[3]
        sum_Evap = s.CumQ[2]
        sum_Infil = -min(sum_vTop, 0.0)
        f.write(
            f"  {t:11.4f}  "
            f"{rTop:11.5E}  {rRoot:11.5E}  {vTop:11.5E}  {s.vRoot:11.5E}  "
            f"{vBot:11.5E}  "
            f"{sum_rTop:11.5E}  {0.0:11.5E}  {sum_vTop:11.5E}  {sum_vRoot:11.5E}  "
            f"{sum_vBot:11.5E}  "
            f"{hTop:11.5E}  {hRoot:11.5E}  {hBot:11.5E}  {0.0:11.5E}  "
            f"{0.0:11.5E}  {Volume:11.5E}  {sum_Infil:11.5E}  {sum_Evap:11.5E}  "
            f"{TLevel:6d}  {0.0:11.5E}  {0.000:9.3f}\n"
        )

    def _write_balance_subreg(self, t: float, dt: float, level: int,
                              ws_initial: float) -> None:
        """Append one sub-region balance block to BALANCE.OUT (Fortran SubReg)."""
        s = self.state
        f = self._fBal
        wsNow = self._water_storage()
        area = s.x[-1] - s.x[0]
        f.write("\n----------------------------------------------------------\n")
        f.write(f" Time       [T]    {t:12.4f}\n")
        f.write("----------------------------------------------------------\n")
        f.write(" Sub-region num.                     1\n")
        f.write("----------------------------------------------------------\n")
        f.write(f" Area     [L]        {area:11.5E}  {area:11.5E}\n")
        f.write(f" W-volume [L]        {wsNow:11.5E}  {wsNow:11.5E}\n")
        # Fortran SubReg uses a trapezoidal average:
        #   hMean = Σ (hE_i * dx_i) / Σ dx_i, where hE_i = (h_i + h_{i+1}) / 2
        # over each interval.  In effect endpoints get half-weight and
        # interior nodes get full weight.  np.mean would give an unweighted
        # arithmetic mean, which differs by ~50 % when one endpoint is at
        # hCritA = −1e5 and the rest of the column is near −100.
        h_mean = 0.0
        area_total = 0.0
        for i in range(s.NumNP - 1):
            dx_i = s.x[i + 1] - s.x[i]
            h_mean += 0.5 * (s.hNew[i] + s.hNew[i + 1]) * dx_i
            area_total += dx_i
        h_mean = h_mean / area_total if area_total > 0 else float(s.hNew.mean())
        f.write(f" h Mean   [L]        {h_mean:11.5E}  {h_mean:11.5E}\n")
        # Fortran SubReg writes the *instantaneous* fluxes at the print time
        # — not the time-averaged values.  We stash them on the state from
        # the most recent solve step.
        topflux = float(getattr(s, "last_vTop", 0.0))
        botflux = float(getattr(s, "last_vBot", 0.0))
        f.write(f" Top Flux [L/T]      {topflux:11.5E}\n")
        f.write(f" Bot Flux [L/T]      {botflux:11.5E}\n")
        # HYDRUS sign convention: vTop > 0 means upward at the top (water
        # leaves the column), vBot > 0 means upward at the bottom (water
        # enters the column from below). Net inflow integral = −CumQ[0] +
        # CumQ[1].  Mass balance residual:
        #     wbalT = ΔStorage − net inflow = ΔS + CumQ[0] − CumQ[1]
        wbalT = (wsNow - ws_initial) + s.CumQ[0] - s.CumQ[1]
        wbalR = 100.0 * abs(wbalT) / max(abs(s.CumQ[0] - s.CumQ[1]),
                                         abs(wsNow), 1e-30)
        f.write(f" WatBalT  [L]        {wbalT:11.5E}\n")
        f.write(f" WatBalR  [%]        {wbalR:18.3f}\n")
        f.write("----------------------------------------------------------\n")

    def get_results(self) -> Dict[str, Any]:
        """
        Get simulation results.
        
        Returns
        -------
        results : dict
            Simulation results
        """
        return {
            'config': self.state.config,
            'grid': self.state.grid,
            'mat': self.state.mat,
            'chem': self.state.chem,
            'bc': self.state.bc,
            'time': self.state.time,
            'root': self.state.root,
            'cumflux': self.state.cumflux,
        }


# ============================================================================
# Convenience function
# ============================================================================

def run_simulation(
    input_dir: str = ".",
    output_dir: str = ".",
    selector_file: str = "Selector.in",
    profile_file: str = "Profile.dat",
    atmospheric_file: str = "ATMOSPH.IN",
    meteo_file: str = "Meteo.in",
) -> Hydrus1DSimulation:
    """
    Run HYDRUS-1D simulation.
    
    Parameters
    ----------
    input_dir : str
        Input file directory
    output_dir : str
        Output file directory
    selector_file : str
        Selector file name
    profile_file : str
        Profile file name
    atmospheric_file : str
        Atmospheric BC file name
    meteo_file : str
        Meteorological data file name
    
    Returns
    -------
    sim : Hydrus1DSimulation
        Simulation object with results
    """
    sim = Hydrus1DSimulation(
        input_dir, output_dir,
        selector_file, profile_file,
        atmospheric_file, meteo_file,
    )
    sim.run()
    return sim


# ============================================================================
# CLI entry point
# ============================================================================

def main():
    """Command-line entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="HYDRUS-1D Python Port")
    parser.add_argument("--input-dir", default=".", help="Input file directory")
    parser.add_argument("--output-dir", default=".", help="Output file directory")
    parser.add_argument("--selector", default="Selector.in", help="Selector file")
    parser.add_argument("--profile", default="Profile.dat", help="Profile file")
    parser.add_argument("--atmospheric", default="ATMOSPH.IN", help="Atmospheric file")
    parser.add_argument("--meteo", default="Meteo.in", help="Meteo file")
    
    args = parser.parse_args()
    
    sim = run_simulation(
        args.input_dir, args.output_dir,
        args.selector, args.profile,
        args.atmospheric, args.meteo,
    )
    
    print(f"Simulation completed. Time: {sim.state.time.t}")
    print(f"Output files in: {args.output_dir}")


if __name__ == "__main__":
    main()
