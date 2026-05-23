"""SWMS_2D ↔ canonical Scenario adapter.

`load(input_dir)` reads SELECTOR.IN + GRID.IN (+ ATMOSPH.IN when
present) and produces a `Scenario`. `save(scenario, output_dir)`
writes the same ASCII files back.

Goes through the existing parsers/serialisers in `swms2d/input.py` —
this module is the boundary between physical-model land and
Fortran-format land.
"""
from __future__ import annotations
from pathlib import Path
from typing import Any

from swms2d.input import parse_example, write_selector
from swms2d.dataclasses import (
    SimulationConfig, SoilMaterial, TimeControl as SwTimeControl,
    Node, Element, Mesh,
)
from ..schema import (
    Scenario, ScenarioMeta, Units, Solver, HydraulicMaterial, TimeControl,
    FeddesRootUptake, SoluteTransport, Geometry2D, AtmosphericBC,
    AtmosphericRow, SeepageFace, SubsurfaceDrain,
)


def load(input_dir: Path | str) -> Scenario:
    """Read SWMS_2D inputs and return a canonical Scenario."""
    in_dir = Path(input_dir)
    cfg, mats, tm, mesh, extras = parse_example(in_dir)

    # --- meta + units ------------------------------------------------
    units_list = extras.get("units", []) + ["-"] * 4
    units = Units(length=units_list[0] or "cm",
                  time=units_list[1] or "day",
                  mass=units_list[2] or "-")
    meta = ScenarioMeta(
        name=extras.get("heading", in_dir.name),
        source="SWMS_2D " + str(in_dir),
    )

    # --- solver ------------------------------------------------------
    geom_kind = {0: "horizontal", 1: "axisymmetric", 2: "vertical"}.get(cfg.KAT, "vertical")
    solver = Solver(
        geometry_kind=geom_kind,
        max_picard=cfg.MaxIt,
        tol_theta=cfg.TolTh,
        tol_h=cfg.TolH,
        water_flow=cfg.lWat,
        solute_transport=cfg.lChem,
        atmospheric_bc=cfg.AtmInF,
        free_drainage=cfg.FreeD,
        seepage_face=cfg.SeepF,
        subsurface_drain=cfg.DrainF,
        short_output=cfg.ShortF,
        flux_output=cfg.FluxF,
        check_output=cfg.CheckF,
    )

    # --- materials ---------------------------------------------------
    canon_mats = [HydraulicMaterial(
        theta_r=m.thr, theta_s=m.ths,
        alpha=m.alpha, n=m.n, Ks=m.Ks,
        l=0.5,  # SWMS_2D hard-codes Mualem l=0.5
        theta_a=m.tha, theta_m=m.thm,
        theta_k=m.thk, Kk=m.Kk,
    ) for m in mats]

    # --- time --------------------------------------------------------
    tprint = list(extras.get("TPrint", []))
    time = TimeControl(
        dt=tm.dt, dt_min=tm.dtMin, dt_max=tm.dtMaxW,
        dt_mul=tm.dMul, dt_mul2=tm.dMul2,
        it_min=3, it_max=7,
        t_init=tm.tInit, t_max=tm.tMax if tm.tMax else (tprint[-1] if tprint else 0.0),
        print_times=tprint,
    )

    # --- root uptake (Block D) ---------------------------------------
    root_uptake = None
    if "sink_P0" in extras:
        root_uptake = FeddesRootUptake(
            p0=extras["sink_P0"], p2H=extras["sink_P2H"],
            p2L=extras["sink_P2L"], p3=extras["sink_P3"],
            r2H=extras["sink_r2H"], r2L=extras["sink_r2L"],
            p_optm=list(extras.get("sink_POptm", [])),
        )

    # --- solute (Block G) --------------------------------------------
    solute = None
    if cfg.lChem and "chem_ChPar" in extras:
        chpar = extras["chem_ChPar"]
        n_mat = chpar.shape[1]
        n_par = chpar.shape[0]
        chem_rows = [[float(chpar[k, m]) for k in range(n_par)]
                     for m in range(n_mat)]
        solute = SoluteTransport(
            epsi=extras.get("chem_epsi", 0.5),
            upwind=extras.get("chem_lUpW", False),
            artificial_dispersion=extras.get("chem_lArtD", False),
            pe_crit=extras.get("chem_PeCr", 0.0),
            chem_params=chem_rows,
            kod_cb=list(extras.get("chem_KodCB", [])),
            c_bound=list(extras.get("chem_cBound", [])),
            t_pulse=float(extras.get("chem_tPulse", 0.0)),
        )

    # --- geometry (2D mesh) ------------------------------------------
    n = mesh.nodes
    el = mesh.elements
    geom2d = Geometry2D(
        x=[float(v) for v in n.x],
        z=[float(v) for v in n.y],          # SWMS's "y" == canonical z
        initial_h=[float(v) for v in n.hNew],
        mat_num=[int(v) for v in n.MatNum],
        kode=[int(v) for v in n.Kode],
        axz=[float(v) for v in n.Axz],
        bxz=[float(v) for v in n.Bxz],
        dxz=[float(v) for v in n.Dxz],
        conc=[float(v) for v in n.Conc],
        q=[float(v) for v in n.Q],
        beta=[float(v) for v in n.Beta],
        elements=[[int(el.KX[e, j]) for j in range(4)] for e in range(el.KX.shape[0])],
        con_axx=[float(v) for v in el.ConAxx],
        con_azz=[float(v) for v in el.ConAzz],
        con_axz=[float(v) for v in el.ConAxz],
        layer_per_element=[int(v) for v in el.LayNum],
        boundary_nodes=[int(v) for v in mesh.KXB],
        boundary_width=[float(v) for v in mesh.Width],
        boundary_total_length=float(mesh.rLen),
    )

    # --- atmospheric (Atmosph.in) ------------------------------------
    atm = None
    if cfg.AtmInF and "atm" in extras:
        ad = extras["atm"]
        # parse_atmosph returns a 2D records array shaped (MaxAL, 10):
        # columns are (tAtm, Prec, cPrec, rSoil, rRoot, hCritA, rGWL,
        # GWL, crt, cht). Map column indices into AtmosphericRow fields.
        records = ad.get("records")
        rows: list[AtmosphericRow] = []
        if records is not None and len(records) > 0:
            for r in records:
                rows.append(AtmosphericRow(
                    t=float(r[0]),
                    precip=float(r[1]),
                    evap=float(r[3]) if len(r) > 3 else 0.0,
                    rRoot=float(r[4]) if len(r) > 4 else 0.0,
                    h_critA=float(r[5]) if len(r) > 5 else -1e30,
                    rB=float(r[6]) if len(r) > 6 else 0.0,
                    hB=float(r[7]) if len(r) > 7 else 0.0,
                    tTop=float(r[8]) if len(r) > 8 else 0.0,
                    tBot=float(r[9]) if len(r) > 9 else 0.0,
                ))
        atm = AtmosphericBC(
            sink_flag=ad.get("SinkF", False),
            qgwlf=ad.get("qGWLF", False),
            rows=rows,
        )

    # --- seepage faces -----------------------------------------------
    seep = None
    if cfg.SeepF and "NSeep" in extras:
        seep = SeepageFace(
            nodes_per_face=list(extras.get("NSP", [])),
            face_node_lists=[list(face) for face in extras.get("NP", [])],
        )

    # --- drains ------------------------------------------------------
    drain = None
    if cfg.DrainF and "drain_NDr" in extras:
        drain = SubsurfaceDrain(
            correction=float(extras.get("drain_DrCorr", 1.0)),
            nodes_per_drain=[1] * int(extras["drain_NDr"]),  # 1 node per drain
            drain_nodes=list(extras.get("drain_ND", [])),
            elements_per_drain=list(extras.get("drain_NED", [])),
            effective_dimensions=[list(row)
                for row in extras.get("drain_EfDim", [])],
            drain_elements=[list(el)
                for el in extras.get("drain_KElDr", [])],
        )

    # --- legacy carryover --------------------------------------------
    legacy = {
        "swms2d_hTab1": extras.get("hTab1"),
        "swms2d_hTabN": extras.get("hTabN"),
        "swms2d_NLay": extras.get("NLay"),
        "swms2d_NPar": extras.get("NPar"),
    }
    if "chem_remaining_lines" in extras:
        legacy["swms2d_chem_remaining_lines"] = extras["chem_remaining_lines"]

    return Scenario(
        meta=meta, units=units, solver=solver,
        materials=canon_mats, time=time,
        root_uptake=root_uptake, solute=solute,
        geometry=geom2d, atmospheric=atm,
        seepage=seep, drain=drain,
        legacy_extras={k: v for k, v in legacy.items() if v is not None},
    )


def save(scenario: Scenario, output_dir: Path | str) -> None:
    """Write canonical Scenario back as SELECTOR.IN + GRID.IN."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if scenario.dimension != "2d":
        raise ValueError(
            f"swms2d adapter expects 2D geometry, got {scenario.dimension}"
        )

    cfg, mats, tm, extras = _to_swms_legacy(scenario)

    # SELECTOR.IN via the existing serialiser
    write_selector(cfg, mats, tm, extras, out_dir / "SELECTOR.IN")

    # GRID.IN — when no on-disk GRID.IN exists (i.e. we're writing a
    # fresh scenario from a canonical .json), emit one from the
    # canonical mesh. When a GRID.IN already exists, leave it alone —
    # parameter-only edits via the GUI don't touch geometry, so the
    # known-good fixture grid stays authoritative.
    grid_path = out_dir / "GRID.IN"
    if not grid_path.exists():
        _write_grid(scenario.geometry, grid_path)
    elif _geometry_differs_from_disk(scenario, grid_path):
        _write_grid(scenario.geometry, grid_path)

    # ATMOSPH.IN — minimal stub: when AtmInF=True the case loader
    # expects this file. We carry rows in canonical.atmospheric and
    # serialise a basic table that the parser can re-read.
    if scenario.solver.atmospheric_bc and scenario.atmospheric:
        _write_atmosph(scenario.atmospheric, out_dir / "ATMOSPH.IN")


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------

def _get(d: dict, key: str, i: int, default: float = 0.0) -> float:
    v = d.get(key)
    if v is None:
        return default
    try:
        return float(v[i])
    except (IndexError, TypeError):
        return default


def _to_swms_legacy(s: Scenario):
    """Inverse of load() — build (cfg, mats, tm, extras) tuple that
    write_selector understands. Pure parameter side only; geometry
    fields stay in s.geometry and are emitted by _write_grid."""
    cfg = SimulationConfig()
    cfg.KAT = {"horizontal": 0, "axisymmetric": 1, "vertical": 2} \
        .get(s.solver.geometry_kind, 2)
    cfg.MaxIt = s.solver.max_picard
    cfg.TolTh = s.solver.tol_theta
    cfg.TolH  = s.solver.tol_h
    cfg.lWat   = s.solver.water_flow
    cfg.lChem  = s.solver.solute_transport
    cfg.CheckF = s.solver.check_output
    cfg.ShortF = s.solver.short_output
    cfg.FluxF  = s.solver.flux_output
    cfg.AtmInF = s.solver.atmospheric_bc
    cfg.SeepF  = s.solver.seepage_face
    cfg.FreeD  = s.solver.free_drainage
    cfg.DrainF = s.solver.subsurface_drain

    mats = [SoilMaterial(
        thr=m.theta_r, ths=m.theta_s,
        tha=m.vc_theta_a(), thm=m.vc_theta_m(),
        alpha=m.alpha, n=m.n,
        Ks=m.Ks, Kk=m.vc_Kk(), thk=m.vc_theta_k(),
    ) for m in s.materials]

    tm = SwTimeControl()
    tm.dt = s.time.dt
    tm.dtMin = s.time.dt_min
    tm.dtMaxW = s.time.dt_max
    tm.dMul = s.time.dt_mul
    tm.dMul2 = s.time.dt_mul2
    tm.tInit = s.time.t_init
    tm.tMax = s.time.t_max if s.time.t_max else (
        s.time.print_times[-1] if s.time.print_times else 0.0)

    import numpy as np
    extras: dict[str, Any] = {
        "heading": s.meta.name,
        "units": [s.units.length, s.units.time, s.units.mass, "-"],
        "TPrint": np.asarray(s.time.print_times, dtype=np.float64),
        "NLay": s.legacy_extras.get("swms2d_NLay", 1),
        "hTab1": s.legacy_extras.get("swms2d_hTab1", 0.001),
        "hTabN": s.legacy_extras.get("swms2d_hTabN", 200.0),
        "NPar": s.legacy_extras.get("swms2d_NPar", 9),
    }
    if s.root_uptake:
        extras.update({
            "sink_P0":  s.root_uptake.p0,
            "sink_P2H": s.root_uptake.p2H,
            "sink_P2L": s.root_uptake.p2L,
            "sink_P3":  s.root_uptake.p3,
            "sink_r2H": s.root_uptake.r2H,
            "sink_r2L": s.root_uptake.r2L,
            "sink_POptm": np.asarray(s.root_uptake.p_optm, dtype=np.float64),
        })
    if s.solute:
        chpar = np.array(s.solute.chem_params, dtype=np.float64).T \
                if s.solute.chem_params else np.zeros((9, 0))
        extras.update({
            "chem_epsi":  s.solute.epsi,
            "chem_lUpW":  s.solute.upwind,
            "chem_lArtD": s.solute.artificial_dispersion,
            "chem_PeCr":  s.solute.pe_crit,
            "chem_ChPar": chpar,
            "chem_remaining_lines":
                s.legacy_extras.get("swms2d_chem_remaining_lines", []),
        })
    if s.seepage:
        extras.update({
            "NSeep": len(s.seepage.face_node_lists),
            "NSP":   s.seepage.nodes_per_face,
            "NP":    s.seepage.face_node_lists,
        })
    if s.drain:
        extras.update({
            "drain_NDr":    len(s.drain.drain_nodes),
            "drain_DrCorr": s.drain.correction,
            "drain_ND":     np.asarray(s.drain.drain_nodes, dtype=np.int32),
            "drain_NED":    np.asarray(s.drain.elements_per_drain, dtype=np.int32),
            "drain_EfDim":  np.asarray(s.drain.effective_dimensions, dtype=np.float64),
            "drain_KElDr":  s.drain.drain_elements,
        })
    return cfg, mats, tm, extras


def _write_grid(geom: Geometry2D, path: Path) -> None:
    """Write GRID.IN from canonical Geometry2D.

    The format is the three-block layout SWMS_2D parses:
        BLOCK H — NumNP/NumEl/IJ/NumBP/NObs + node rows
        BLOCK I — element rows
        BLOCK J — boundary geometry (KXB + Width + rLen)
    The Fortran reader is whitespace-tolerant; we emit a clean, fixed-
    width layout that mirrors EX1-EX4 conventions.
    """
    n_np = len(geom.x)
    n_el = len(geom.elements)
    n_bp = len(geom.boundary_nodes)
    L: list[str] = []
    L.append("*** BLOCK H: NODAL INFORMATION " + "*" * 50)
    L.append("      NumNP     NumEl       IJ      NumBP     NObs")
    L.append(f"       {n_np}        {n_el}         2         {n_bp}     0")
    L.append("   n  Code    x      z          h       Conc      Q     M   B    Axz   Bxz   Dxz")
    for i in range(n_np):
        # B field is a width-like; SWMS uses it for axisymmetric problems.
        # When canonical doesn't track it, write 0.
        b_val = 0.0
        L.append(
            f"{i+1:4d} {geom.kode[i]:3d} {geom.x[i]:8.2f} {geom.z[i]:8.2f} "
            f"{geom.initial_h[i]:9.2f} "
            f"{geom.conc[i]:9.2e} "
            f"{geom.q[i]:9.2e} "
            f"{geom.mat_num[i]:2d} "
            f"{b_val:5.2f} "
            f"{geom.axz[i]:5.2f} {geom.bxz[i]:5.2f} {geom.dxz[i]:5.2f}"
        )

    L.append("*** BLOCK I: ELEMENT INFORMATION " + "*" * 48)
    L.append("   e   i   j   k   l   Angle  Aniz1  Aniz2  LayNum")
    for e in range(n_el):
        cell = geom.elements[e]
        # Canonical stores 0-based; SWMS expects 1-based
        i, j, k, l = (cell[0] + 1, cell[1] + 1, cell[2] + 1, cell[3] + 1)
        # The rotated tensor (ConAxx/Azz/Axz) is what we have. Recover
        # (Angle, Aniz1, Aniz2) approximately: when Axz≈0 the system is
        # diagonal → Angle=0, Aniz1=ConAxx, Aniz2=ConAzz. That's the
        # common case (EX1-4 all have Angle=0).
        angle = 0.0
        aniz1 = geom.con_axx[e] if e < len(geom.con_axx) else 1.0
        aniz2 = geom.con_azz[e] if e < len(geom.con_azz) else 1.0
        lay = geom.layer_per_element[e] if e < len(geom.layer_per_element) else 1
        L.append(
            f"{e+1:4d} {i:3d} {j:3d} {k:3d} {l:3d} "
            f"{angle:6.2f} {aniz1:6.2f} {aniz2:6.2f} {lay:3d}"
        )

    L.append("*** BLOCK J: BOUNDARY GEOMETRY INFORMATION " + "*" * 40)
    L.append("KXB(i),i=1,NumBP")
    for k in range(0, n_bp, 10):
        L.append("  " + " ".join(f"{geom.boundary_nodes[i]+1:4d}"
                                  for i in range(k, min(k+10, n_bp))))
    L.append("Width(i),i=1,NumBP")
    for k in range(0, n_bp, 10):
        L.append("  " + " ".join(f"{geom.boundary_width[i]:8.3f}"
                                  for i in range(k, min(k+10, n_bp))))
    L.append("rLen")
    L.append(f"  {geom.boundary_total_length:.4f}")
    L.append("*** END OF INPUT FILE 'GRID.IN' " + "*" * 50)
    path.write_text("\n".join(L) + "\n")


def _write_atmosph(atm: AtmosphericBC, path: Path) -> None:
    """Minimal ATMOSPH.IN writer mirroring EX2's layout."""
    rows = atm.rows
    L: list[str] = []
    L.append("*** BLOCK I: ATMOSPHERIC INFORMATION  **********************************")
    L.append("***                                   **********************************")
    L.append("************************************************************************")
    L.append("SinkF   qGWLF")
    L.append(f" {'t' if atm.sink_flag else 'f'}       {'t' if atm.qgwlf else 'f'}")
    L.append("GWL0L   Aqh     Bqh                          (if qGWLF=f then Aqh=Bqh=0)")
    L.append(" 0       0       0")
    L.append("tInit   MaxAL               (MaxAL = number of atmospheric data-records)")
    t_init = rows[0].t - 1.0 if rows else 0.0
    L.append(f" {t_init}     {len(rows)}")
    L.append("hCritS                  (max. allowed pressure head at the soil surface)")
    L.append(" 1.e30")
    L.append("  tAtm    Prec   cPrec   rSoil    rRoot  hCritA    rt    ht   crt   cht")
    for r in rows:
        L.append(f" {r.t:7.2f} {r.precip:7.4f} {0.0:7.4f} {r.evap:7.4f} "
                 f"{r.rRoot:7.4f} {r.h_critA:9.2e} {r.tTop:5.2f} {r.ht:5.2f} "
                 f"{r.tBot:5.2f} {r.hB:5.2f}")
    L.append("*** END OF INPUT FILE 'ATMOSPH.IN' *************************************")
    path.write_text("\n".join(L) + "\n")


def _geometry_differs_from_disk(scenario: Scenario, grid_path: Path) -> bool:
    # Cheap check: if we ever extend the editor to write GRID.IN we'd
    # diff the canonical mesh against the on-disk one here. For now
    # always return False (use the existing GRID.IN unchanged).
    return False
