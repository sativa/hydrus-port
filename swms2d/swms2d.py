"""
Main simulation driver for SWMS_2D Python port.
===============================================

Wires together input → init → time loop → output. Mirrors the main
program SWMS_2D.FOR.

Usage:
    from swms2d.swms2d import SWMS2DSimulation
    sim = SWMS2DSimulation(input_dir="path/to/SWMS_2D.IN",
                           output_dir="path/to/SWMS_2D.OUT")
    sim.run()

For now (Stage 1 EXAMPLE.1 milestone) only water flow + seepage face
boundaries are supported. Atm / Drain / Sink / Solute / Free drainage
will be added in subsequent passes.
"""

from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass, field
import numpy as np

from .dataclasses import (
    SimulationConfig, SoilMaterial, TimeControl, Mesh, FullSimulationState,
)
from .input import parse_example
from .material import saturated_values
from .watflow import solve_water_flow, set_mat, build_material_tables
from .output import HOutWriter, ThOutWriter, ConcOutWriter
from .sink import set_snk, normalize_beta
from .solute import solute_step


class SWMS2DSimulation:
    """One SWMS_2D run end-to-end."""

    def __init__(self, input_dir: Path | str, output_dir: Path | str,
                 use_anderson: bool = False, anderson_m: int = 3,
                 refine_solve: bool = False, use_newton: bool = False,
                 use_lscheme: bool = False, lscheme_L: float = 0.0,
                 use_banded: bool = False):
        """
        Parameters
        ----------
        input_dir, output_dir : Path-like
            SWMS_2D.IN and SWMS_2D.OUT directories.
        use_anderson : bool, default False
            Anderson(m) acceleration of the Picard fixed-point iteration.
            *Experimental* — tested on EX.2 dry-spell, no measurable
            improvement over plain Picard.
        anderson_m : int, default 3
            Anderson buffer depth.
        refine_solve : bool, default False
            Iterative refinement of the sparse LU solve via splu factor
            reuse. *Experimental* — tested, gives bit-identical results
            to plain spsolve (confirming spsolve already at
            working-precision residual).
        use_newton : bool, default False
            Modified-Newton diagonal Jacobian correction
            F[i] * dC/dh|h_i * (h_i - h_n_i)/dt added to the Picard
            matrix diagonal. *Experimental* — tested on EX.2 dry-spell,
            no improvement (the K-derivative off-diagonal terms would
            need to be added too for full Newton, ~500 LOC more).
        use_lscheme : bool, default False
            L-scheme stabiliser L*F[i] added to diagonal (List & Radu
            2016). *Experimental and currently broken* — converges to a
            different fixed point under iterate-based convergence test;
            needs residual-based criterion to be used correctly.
        lscheme_L : float, default 0.0
            L-scheme regularisation strength.
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.use_anderson = use_anderson
        self.anderson_m = anderson_m
        self.refine_solve = refine_solve
        self.use_newton = use_newton
        self.use_lscheme = use_lscheme
        self.lscheme_L = lscheme_L
        self.use_banded = use_banded

        # Parse input
        self.cfg, self.materials, self.time, self.mesh, self.extras = \
            parse_example(self.input_dir)

        # DrainF setup — apply Vimoke-Taylor K-reduction once, store ND for
        # the runtime Kode=±5 switching in shift().
        if self.cfg.DrainF:
            from .drain import apply_vimoke_taylor
            self.drain_NDr = self.extras.get("drain_NDr", 0)
            if self.drain_NDr > 0:
                NED = self.extras["drain_NED"]
                EfDim = self.extras["drain_EfDim"]
                KElDr = self.extras["drain_KElDr"]
                DrCorr = self.extras["drain_DrCorr"]
                apply_vimoke_taylor(self.mesh, self.drain_NDr, NED, EfDim,
                                    KElDr, DrCorr)
                self.drain_ND = self.extras["drain_ND"]
            else:
                self.drain_ND = None
        else:
            self.drain_NDr = 0
            self.drain_ND = None

        # Material-derived constants
        self.thR, self.thSat, self.hSat, self.ConSat = \
            saturated_values(self.materials)

        # Log-spaced K/C/θ table (mirrors Fortran GenMat — needed for
        # bit-equal numerical match with the Fortran reference)
        self.tables = build_material_tables(
            self.materials,
            hTab1=self.extras["hTab1"], hTabN=self.extras["hTabN"],
            NTab=100,
        )

        # State arrays
        NumNP = self.mesh.NumNP
        self.ThOld = np.zeros(NumNP, np.float64)
        self.ThNew = np.zeros(NumNP, np.float64)
        self.ConO  = np.zeros(NumNP, np.float64)

        # Seepage info from SELECTOR.IN extras
        if self.cfg.SeepF:
            self.NSeep = self.extras["NSeep"]
            self.NSP = self.extras["NSP"]
            # NP from Fortran is 1-based — convert
            self.NP_seep = [[v - 1 for v in row]
                            for row in self.extras["NP"]]
        else:
            self.NSeep, self.NSP, self.NP_seep = 0, [], []

        # Sink (root uptake) — params from SELECTOR BLOCK D, normalise Beta
        self.Sink_arr: np.ndarray | None = None
        if self.cfg.SinkF:
            self.P0 = self.extras["sink_P0"]
            self.P2H = self.extras["sink_P2H"]
            self.P2L = self.extras["sink_P2L"]
            self.P3 = self.extras["sink_P3"]
            self.r2H = self.extras["sink_r2H"]
            self.r2L = self.extras["sink_r2L"]
            self.POptm = self.extras["sink_POptm"]
            normalize_beta(self.mesh, self.cfg.KAT)
        else:
            self.P3 = 0.0

        # Atmospheric — records walked by tAtm pointer
        if self.cfg.AtmInF:
            atm = self.extras["atm"]
            self.atm_records = atm["records"]   # (MaxAL, 10)
            self.atm_idx = 0
            self.GWL0L = atm["GWL0L"]
            self.Aqh = atm["Aqh"]
            self.Bqh = atm["Bqh"]
            self.hCritS = atm["hCritS"]
            self.rTop = 0.0
            self.rRoot = 0.0
            self.hCritA = -1e6
            self.tAtm = self.atm_records[0, 0]
        else:
            self.atm_records = None
            self.GWL0L = self.Aqh = self.Bqh = 0.0
            self.hCritS = 0.0
            self.rTop = 0.0
            self.rRoot = 0.0
            self.hCritA = -1e6
            self.tAtm = self.time.tMax

        # TPrint schedule
        self.TPrint = self.extras["TPrint"]
        self.PLevel = 0      # index into TPrint (0-based)

        # Time bookkeeping — preserve tInit set by parse_atmosph()
        self.time.t = self.time.tInit
        self.time.dtInit = self.time.dt
        self.time.dtOpt = self.time.dt
        self.t = self.time.tInit
        self.tOld = self.time.tInit
        self.dtOld = self.time.dt
        self.TLevel = 1
        # lMinStep: set by SetAtm when a BC value changes so TmCont
        # throttles dt back to dtInit for one step (mirrors Fortran flag).
        self.lMinStep = False

        # Solute init
        if self.cfg.lChem:
            self.ChPar = self.extras["chem_ChPar"]
            self.cBound = self.extras["chem_cBound"]
            self.KodCB = self.extras["chem_KodCB"]
            self.epsi = self.extras["chem_epsi"]
            self.tPulse = self.extras["chem_tPulse"]
            self.lUpW = self.extras["chem_lUpW"]
            self.lArtD = self.extras["chem_lArtD"]
            self.PeCr = self.extras["chem_PeCr"]
            self.cPrec = 0.0
            self.crt = 0.0
            self.cht = 0.0

        # Output writers
        heading = self.extras["heading"]
        units = self.extras["units"]
        self.h_writer  = HOutWriter (self.output_dir / "h.out",
                                     heading, units, self.cfg.KAT,
                                     atm_inf=self.cfg.AtmInF)
        self.th_writer = ThOutWriter(self.output_dir / "th.out",
                                     heading, units, self.cfg.KAT,
                                     atm_inf=self.cfg.AtmInF)
        self.c_writer  = (ConcOutWriter(self.output_dir / "conc.out",
                                        heading, units, self.cfg.KAT)
                          if self.cfg.lChem else None)

    # ----------------------------------------------------------------
    def _set_atm(self) -> None:
        """SetAtm (TIME2.FOR L26-66) — read the current atm record and apply
        it to boundary nodes; tAtm becomes the END of this record's validity.
        Sets `lMinStep` = True when any BC value changes from the previous
        record so TmCont can throttle dt back to dtInit (Fortran throttles
        to dtMax=dtInit for one step after a BC change)."""
        if self.atm_records is None:
            return
        if self.atm_idx >= self.atm_records.shape[0]:
            self.tAtm = self.time.tMax
            return
        rec = self.atm_records[self.atm_idx]
        # tAtm, Prec, cPrec, rSoil, rRoot, hCritA, rGWL, GWL, crt, cht
        rTopOld = self.rTop
        self.tAtm = rec[0]
        Prec  = abs(rec[1])
        rSoil = abs(rec[3])
        self.rRoot = abs(rec[4])
        self.hCritA = -abs(rec[5])
        rGWL = rec[6]
        GWL  = rec[7]
        hGWL = GWL + self.GWL0L
        self.rTop = rSoil - Prec
        nodes = self.mesh.nodes
        KXB = self.mesh.KXB
        Width = self.mesh.Width
        for i in range(self.mesh.NumBP):
            n = int(KXB[i])
            K = int(nodes.Kode[n])
            if K == 4 or K == -4:
                nodes.Kode[n] = -4
                nodes.Q[n] = -Width[i] * self.rTop
                continue
            if K == 3:
                if abs(nodes.hNew[n] - hGWL) > 1e-8:
                    self.lMinStep = True
                nodes.hNew[n] = hGWL
            if K == -3 and not self.cfg.qGWLF and not self.cfg.FreeD:
                if Width[i] > 0.0:
                    rGWLOld = -nodes.Q[n] / Width[i]
                    if abs(rGWLOld - rGWL) > 1e-8:
                        self.lMinStep = True
                nodes.Q[n] = -Width[i] * rGWL
        if abs(self.rTop - rTopOld) > 1e-8 and abs(self.rTop) > 0.0:
            self.lMinStep = True
        self.atm_idx += 1

    # ----------------------------------------------------------------
    def _initial_setmat(self):
        """One-shot SetMat to populate ThOld, Con, Cap before time loop."""
        # Set hTemp = hOld = hNew initially
        self.mesh.nodes.hTemp[:] = self.mesh.nodes.hNew
        Con, Cap, Th = set_mat(self.mesh, self.materials,
                               self.thR, self.thSat, self.hSat, self.ConSat,
                               Explic=False, tables=self.tables)
        self.ThOld[:] = Th
        self.ThNew[:] = Th
        self.ConO[:]  = Con
        return Con, Cap

    # ----------------------------------------------------------------
    def _adapt_dt(self, n_iter: int) -> None:
        """TmCont (TIME2.FOR) — adaptive time-step controller.

        Mirrors Fortran TmCont exactly: dtOpt is the persistent "optimum" the
        solver remembers across steps; dt is dtOpt clipped to land on the
        next fixed time (tPrint / tAtm / tMax) via the anint even-chunk snap.
        """
        tnext = self.TPrint[self.PLevel] if self.PLevel < len(self.TPrint) else self.time.tMax
        tFix = min(tnext, self.tAtm, self.time.tMax)
        # lMinStep (TIME2.FOR L7-12): one-step dtMax throttle to dtInit after
        # any boundary value change.
        if self.lMinStep:
            dtMax = min(self.time.dtMaxW, self.time.dtInit)
            self.lMinStep = False
        else:
            dtMax = self.time.dtMaxW   # no atmospheric Courant limit at Stage 1
        dtOpt = self.time.dtOpt
        # Grow / shrink dtOpt by iter count (Fortran uses <= 3 and >= 7)
        if n_iter <= 3 and (tFix - self.t) >= self.time.dMul * dtOpt:
            dtOpt = min(dtMax, self.time.dMul * dtOpt)
        if n_iter >= 7:
            dtOpt = max(self.time.dtMin, self.time.dMul2 * dtOpt)
        # Clip to remaining distance to tFix
        dt = min(dtOpt, tFix - self.t)
        # anint snap: split (tFix-t) into nearest integer N equal chunks
        rem = tFix - self.t
        if dt > 0.0 and rem > 0.0:
            N = max(1.0, round(rem / dt))
            dt = min(rem / N, dtMax)
        # If we'd land short of tFix with more than half-distance, halve it
        if abs(rem - dt) > 1e-12 and dt > rem / 2.0:
            dt = rem / 2.0
        self.time.dtOpt = dtOpt
        self.time.dt = max(dt, self.time.dtMin)

    # ----------------------------------------------------------------
    def run(self, verbose: bool = True) -> None:
        """Main time loop."""
        nodes = self.mesh.nodes

        # Apply the first atmospheric record (defines initial rTop/rRoot)
        if self.cfg.AtmInF:
            self._set_atm()

        # Initial output at t=tInit — Fortran writes the GRID.IN initial
        # h,th,conc before the time loop (SWMS_2D.FOR L137,144,145),
        # even when lWat=False; the steady-state h is written later at
        # TPrint[0] after WatFlow runs at TLevel=1.
        Con, Cap = self._initial_setmat()
        self.h_writer.write_snapshot (self.time.tInit, self.mesh, nodes.hNew.copy())
        self.th_writer.write_snapshot(self.time.tInit, self.mesh, self.ThOld.copy())

        # First time step from dtInit
        self.t = self.time.tInit + self.time.dt
        self.tOld = self.time.tInit
        self.dtOld = self.time.dt

        # Initial concentration snapshot at t=tInit (SWMS_2D.FOR L137).
        if self.c_writer:
            self.c_writer.write_snapshot(self.time.tInit, self.mesh,
                                         nodes.Conc.copy())

        while True:
            # Solve water flow for this step
            if self.TLevel != 1:
                # Linear-extrapolation predictor for non-Dirichlet nodes
                # (mirrors WATFLOW2.FOR L36-48)
                for i in range(self.mesh.NumNP):
                    nodes.hOld[i] = nodes.hNew[i]
                    self.ThOld[i] = self.ThNew[i]
                    self.ConO[i]  = Con[i]
                    if nodes.Kode[i] < 1:
                        nodes.hTemp[i] = (nodes.hNew[i]
                                           + (nodes.hNew[i] - nodes.hOld[i])
                                              * self.time.dt / self.dtOld)
                        # Note: hOld was just overwritten, so the predictor
                        # term is zero on TLevel==2; this matches Fortran
                        # exactly (it also resets hOld first).
                        nodes.hNew[i] = nodes.hTemp[i]
                    else:
                        nodes.hTemp[i] = nodes.hNew[i]

            # Compute Sink at the start of this step (Iter==0 only — Reset
            # accumulates DS once per step, matching Fortran's Iter.eq.0 guard).
            if self.cfg.SinkF:
                self.Sink_arr = set_snk(
                    self.mesh, self.materials, self.rRoot, self.mesh.rLen,
                    self.P0, self.POptm, self.P2H, self.P2L, self.P3,
                    self.r2H, self.r2L,
                )

            if self.cfg.lWat or self.TLevel == 1:
                dt_used, t_new, n_iter, converged, Con, Cap, ThNew, Q_intern = \
                    solve_water_flow(
                        self.mesh, self.cfg, self.time, self.materials,
                        self.thR, self.thSat, self.hSat, self.ConSat,
                        self.ThNew, self.ThOld, self.ConO,
                        self.NSeep, self.NSP, self.NP_seep,
                        dt=self.time.dt, dtMin=self.time.dtMin,
                        dtOld=self.dtOld, tOld=self.tOld,
                        rTop=self.rTop, hCritA=self.hCritA, hCritS=self.hCritS,
                        GWL0L=self.GWL0L, Aqh=self.Aqh, Bqh=self.Bqh,
                        Sink=self.Sink_arr, P3=self.P3,
                        tables=self.tables,
                        use_anderson=self.use_anderson,
                        anderson_m=self.anderson_m,
                        refine_solve=self.refine_solve,
                        use_newton=self.use_newton,
                        use_lscheme=self.use_lscheme,
                        lscheme_L=self.lscheme_L,
                        use_banded=self.use_banded,
                        debug_file=getattr(self, 'debug_file', None),
                        debug_TLevel=self.TLevel,
                        NDr=self.drain_NDr,
                        ND_drain=self.drain_ND,
                    )
                self.t = t_new
                self.time.dt = dt_used
                ThNew_arr = ThNew
            else:
                # lWat=False, TLevel>1: water flow is steady — reuse last state
                n_iter = 1
                ThNew_arr = self.ThNew
                self.t = self.tOld + self.time.dt

            # Solute transport step
            if self.cfg.lChem:
                nodes.Conc[:] = solute_step(
                    self.mesh, self.cfg, self.t, self.time.dt,
                    nodes.hNew, nodes.hOld,
                    Con, self.ConO, ThNew_arr, self.ThOld, self.thSat,
                    nodes.Conc,
                    self.Sink_arr if self.Sink_arr is not None
                                  else np.zeros(self.mesh.NumNP, np.float64),
                    self.ChPar, self.mesh.nodes.MatNum,
                    self.cBound, self.KodCB,
                    self.epsi, self.tPulse,
                    cPrec=self.cPrec, crt=self.crt, cht=self.cht,
                    lUpW=self.lUpW, lArtD=self.lArtD, PeCr=self.PeCr,
                )

            if verbose and (self.TLevel % 20 == 1 or self.t >= self.time.tMax):
                print(f"  T={self.t:12.4f}  dt={self.time.dt:.4g}  "
                      f"iter={n_iter}  conv={converged}  "
                      f"PLevel={self.PLevel}/{len(self.TPrint)}")

            # Print-level output. SWMS_2D.FOR L214-218 fires hOut/thOut only
            # when lWat OR (.not.lWat .and. PLevel==1, i.e. once at first match).
            if (self.PLevel < len(self.TPrint)
                and abs(self.TPrint[self.PLevel] - self.t)
                    < 0.001 * self.time.dt):
                if self.cfg.lWat or self.PLevel == 0:
                    self.h_writer.write_snapshot(self.t, self.mesh,
                                                 nodes.hNew.copy())
                    self.th_writer.write_snapshot(self.t, self.mesh,
                                                  ThNew_arr.copy())
                if self.c_writer:
                    self.c_writer.write_snapshot(self.t, self.mesh,
                                                 nodes.Conc.copy())
                self.PLevel += 1

            # Termination
            if abs(self.t - self.time.tMax) <= 0.001 * self.time.dt:
                break

            # Atmospheric BC: load next record when we hit a tAtm boundary
            if (self.cfg.AtmInF
                    and abs(self.t - self.tAtm) <= 0.001 * self.time.dt):
                self._set_atm()

            # Time-step adaptation + advance
            self.tOld = self.t
            self.dtOld = self.time.dt
            self._adapt_dt(n_iter)
            self.TLevel += 1
            self.t = self.tOld + self.time.dt

        self.h_writer.close()
        self.th_writer.close()
        if self.c_writer:
            self.c_writer.close()
        if verbose:
            print(f"\nDone. T={self.t} TLevel={self.TLevel} "
                  f"PLevel={self.PLevel}/{len(self.TPrint)}")
