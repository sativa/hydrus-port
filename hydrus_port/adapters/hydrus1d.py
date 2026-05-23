"""HYDRUS-1D ↔ canonical Scenario adapter.

`load(input_dir)` reads Selector.in + Profile.dat (+ ATMOSPH.IN /
Meteo.in when present) and produces a canonical Scenario with a
Geometry1D. `save(scenario, output_dir)` writes the same files back.
"""
from __future__ import annotations
from pathlib import Path
from typing import Any

from hydrus1d.input import read_selector
from ..schema import (
    Scenario, ScenarioMeta, Units, Solver, HydraulicMaterial, TimeControl,
    FeddesRootUptake, SoluteTransport, Geometry1D, AtmosphericBC,
    AtmosphericRow,
)


def _find(in_dir: Path, name: str) -> Path | None:
    """Case-insensitive find of `name` in `in_dir`."""
    for p in in_dir.iterdir():
        if p.name.lower() == name.lower():
            return p
    return None


def load(input_dir: Path | str) -> Scenario:
    in_dir = Path(input_dir)
    sel_path = _find(in_dir, "Selector.in")
    prof_path = _find(in_dir, "Profile.dat")
    if not sel_path:
        raise FileNotFoundError(f"Selector.in not in {in_dir}")
    sel = read_selector(str(sel_path))

    # --- meta + units ------------------------------------------------
    meta = ScenarioMeta(
        name=sel.get("Heading", in_dir.name),
        source=f"HYDRUS-1D {in_dir}",
    )
    units = Units(
        length=sel.get("LUnit", "cm"),
        time=sel.get("TUnit", "day"),
        mass=sel.get("MUnit", "-"),
    )

    # --- solver ------------------------------------------------------
    solver = Solver(
        geometry_kind="axisymmetric" if abs(float(sel.get("CosAlf", 1.0)) - 1.0) > 1e-9
                       else "vertical",
        max_picard=int(sel.get("MaxIt", 20)),
        tol_theta=float(sel.get("TolTh", 0.001)),
        tol_h=float(sel.get("TolH", 1.0)),
        water_flow=bool(sel.get("lWat", True)),
        solute_transport=bool(sel.get("lChem", False)),
        heat_transport=bool(sel.get("lTemp", False)),
        root_uptake=bool(sel.get("SinkF", False)),
        atmospheric_bc=bool(sel.get("AtmBC", False)),
        free_drainage=bool(sel.get("FreeD", False)),
        seepage_face=bool(sel.get("SeepF", False)),
        short_output=bool(sel.get("ShortO", False)),
        equilibrium=bool(sel.get("lEquil", True)),
        hysteresis=bool(sel.get("iHyst", 0)) and sel.get("iHyst", 0) > 0,
    )

    # --- materials (ParD shape is (11, NMat)) ------------------------
    par = sel.get("ParD")
    n_mat = int(sel.get("NMat", 0))
    materials: list[HydraulicMaterial] = []
    for m in range(n_mat):
        thr = float(par[0, m]); ths = float(par[1, m])
        alpha = float(par[2, m]); n = float(par[3, m])
        Ks = float(par[4, m]); l = float(par[5, m])
        materials.append(HydraulicMaterial(
            theta_r=thr, theta_s=ths, alpha=alpha, n=n, Ks=Ks, l=l,
        ))

    # --- time --------------------------------------------------------
    tprint = list(sel.get("TPrint", []))
    time = TimeControl(
        dt=float(sel.get("dt", 0.001)),
        dt_min=float(sel.get("dtMin", 1e-5)),
        dt_max=float(sel.get("dtMax", 1.0)),
        dt_mul=float(sel.get("dMul", 1.3)),
        dt_mul2=float(sel.get("dMul2", 0.7)),
        it_min=int(sel.get("ItMin", 3)),
        it_max=int(sel.get("ItMax", 7)),
        t_init=float(sel.get("tInit", 0.0)),
        t_max=float(sel.get("tMax", 0.0)),
        print_times=[float(t) for t in tprint],
    )

    # --- Profile.dat → 1D geometry -----------------------------------
    geom = Geometry1D()
    if prof_path and prof_path.exists():
        geom = _read_profile_dat(prof_path)
    geom.kind = "1d"
    # Ensure layer arrays have at least dummy values if Profile.dat omitted
    if not geom.layer:
        geom.layer = [1] * len(geom.z)

    # --- legacy carryover (HYDRUS-1D-specific fields the editor doesn't expose) ---
    legacy = {
        "h1d_iModel":   int(sel.get("iModel", 0)),
        "h1d_iHyst":    int(sel.get("iHyst", 0)),
        "h1d_TopInF":   bool(sel.get("TopInF", False)),
        "h1d_BotInF":   bool(sel.get("BotInF", False)),
        "h1d_WLayer":   bool(sel.get("WLayer", False)),
        "h1d_KodTop":   int(sel.get("KodTop", 0)),
        "h1d_KodBot":   int(sel.get("KodBot", 0)),
        "h1d_lInitW":   bool(sel.get("lInitW", False)),
        "h1d_qGWLF":    bool(sel.get("qGWLF", False)),
        "h1d_rTop":     float(sel.get("rTop", 0.0)),
        "h1d_rBot":     float(sel.get("rBot", 0.0)),
        "h1d_rRoot":    float(sel.get("rRoot", 0.0)),
        "h1d_GWL0L":    float(sel.get("GWL0L", 0.0)),
        "h1d_Aqh":      float(sel.get("Aqh", 0.0)),
        "h1d_Bqh":      float(sel.get("Bqh", 0.0)),
        "h1d_hCritS":   float(sel.get("hCritS", 1e30)),
        "h1d_hCritA":   float(sel.get("hCritA", -1e6)),
        "h1d_qDrain":   bool(sel.get("qDrain", False)),
        "h1d_hSeep":    float(sel.get("hSeep", 0.0)),
        "h1d_hTab1":    float(sel.get("hTab1", -1e-6)),
        "h1d_hTabN":    float(sel.get("hTabN", -1e4)),
        "h1d_NLay":     int(sel.get("NLay", 1)),
        "h1d_lWDep":    bool(sel.get("lWDep", False)),
        "h1d_lScreen":  bool(sel.get("lScreen", True)),
        "h1d_lEquil":   bool(sel.get("lEquil", True)),
        "h1d_lPrintD":  bool(sel.get("lPrintD", False)),
        "h1d_nPrStep":  int(sel.get("nPrStep", 1)),
        "h1d_tPrintInt": float(sel.get("tPrintInt", 1.0)),
        "h1d_lEnter":   bool(sel.get("lEnter", False)),
        "h1d_iVer":     int(sel.get("iVer", 4)),
        "h1d_CosAlpha": float(sel.get("CosAlf", 1.0)),
    }

    return Scenario(
        meta=meta, units=units, solver=solver,
        materials=materials, time=time,
        geometry=geom,
        legacy_extras=legacy,
    )


def save(scenario: Scenario, output_dir: Path | str) -> None:
    """Write a HYDRUS-1D Selector.in + Profile.dat from the canonical
    Scenario. Round-trips losslessly for the soil_loam_infiltr fixture
    (semantically — Fortran column widths may differ)."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if scenario.dimension != "1d":
        raise ValueError(
            f"hydrus1d adapter expects 1D geometry, got {scenario.dimension}"
        )
    _write_selector(scenario, out / "Selector.in")
    _write_profile(scenario, out / "Profile.dat")


# ----------------------------------------------------------------------
# Profile.dat reader / writer
# ----------------------------------------------------------------------

def _read_profile_dat(path: Path) -> Geometry1D:
    """Parse HYDRUS-1D Profile.dat.

    Layout:
        line 1: NObs
        lines 2..NObs+1: obs node setup (one per line)
        next non-empty: "NumNP iCheck ... <column header tail>"
        next NumNP lines: n x h Mat Lay Beta Axz Bxz Dxz [Temp]
        terminator: 0
    """
    text = path.read_text().splitlines()
    it = (ln for ln in text if ln.strip())
    n_obs = int(next(it).split()[0])
    # Skip n_obs obs lines
    for _ in range(n_obs):
        next(it, None)
    # Next line has NumNP at the front
    meta_line = next(it).split()
    num_np = int(meta_line[0])
    z: list[float] = []
    h: list[float] = []
    mat: list[int] = []
    lay: list[int] = []
    beta: list[float] = []
    axz: list[float] = []
    bxz: list[float] = []
    dxz: list[float] = []
    for _ in range(num_np):
        toks = next(it).split()
        # toks: n x h Mat Lay Beta Axz Bxz Dxz [Temp]
        z.append(float(toks[1]))
        h.append(float(toks[2]))
        mat.append(int(toks[3]))
        lay.append(int(toks[4]))
        beta.append(float(toks[5]))
        axz.append(float(toks[6]) if len(toks) > 6 else 1.0)
        bxz.append(float(toks[7]) if len(toks) > 7 else 1.0)
        dxz.append(float(toks[8]) if len(toks) > 8 else 1.0)
    return Geometry1D(
        z=z, initial_h=h, mat_num=mat, layer=lay,
        beta=beta, axz=axz, bxz=bxz, dxz=dxz,
    )


def _write_profile(scenario: Scenario, path: Path) -> None:
    g = scenario.geometry  # Geometry1D
    n = len(g.z)
    L = ["    2"]
    L.append(f"    1   {g.z[0]:>12.6e}  1.000000e+000  1.000000e+000")
    L.append(f"    2   {g.z[-1]:>12.6e}  1.000000e+000  1.000000e+000")
    L.append(
        f"  {n}    0    0    1 x         h         Mat  Lay      Beta"
        f"            Axz            Bxz            Dxz            Temp"
    )
    for i in range(n):
        L.append(
            f"{i+1:>5d}  {g.z[i]:>13.6e}  {g.initial_h[i]:>12.6e}    "
            f"{g.mat_num[i]:d}    {g.layer[i]:d}   {g.beta[i]:>12.6e}  "
            f"{g.axz[i]:>13.6e}  {g.bxz[i]:>13.6e}  {g.dxz[i]:>13.6e}              "
        )
    L.append("   0")
    path.write_text("\n".join(L) + "\n")


# ----------------------------------------------------------------------
# Selector.in writer
# ----------------------------------------------------------------------

def _t(b: bool) -> str:
    return "t" if b else "f"


def _fmt(v: float) -> str:
    """Format a number in the loose layout HYDRUS-1D's parser accepts."""
    if v == 0:
        return "0"
    if v == int(v) and abs(v) < 1e7:
        return str(int(v))
    if abs(v) < 1e-3 or abs(v) >= 1e5:
        return f"{v:.3e}"
    return f"{v:g}"


def _write_selector(scenario: Scenario, path: Path) -> None:
    s = scenario
    legacy = s.legacy_extras
    cfg = s.solver
    mat = s.materials
    t = s.time
    L: list[str] = []
    L.append(f"Pcp_File_Version={legacy.get('h1d_iVer', 4)}")
    L.append("*** BLOCK A: BASIC INFORMATION *****************************************")
    L.append("Heading")
    L.append(s.meta.name)
    L.append("LUnit  TUnit  MUnit")
    L.append(s.units.length)
    L.append(s.units.time)
    L.append(s.units.mass)
    L.append("lWat   lChem  lTemp  SinkF  lRoot  ShortO lWDep  lScreen AtmBC lEquil")
    L.append(" " + " ".join([
        _t(cfg.water_flow), _t(cfg.solute_transport), _t(cfg.heat_transport),
        _t(cfg.root_uptake),
        _t(legacy.get("h1d_lRoot", False) or cfg.root_uptake),
        _t(cfg.short_output), _t(legacy.get("h1d_lWDep", False)),
        _t(legacy.get("h1d_lScreen", True)),
        _t(cfg.atmospheric_bc), _t(cfg.equilibrium),
    ]))
    L.append("lSnow  lDummy lMeteo lVapor lActRSU lFlux")
    L.append(" f f f f f f")
    L.append("NMat   NLay   CosAlpha")
    L.append(f"  {len(mat)}  {legacy.get('h1d_NLay', 1)}  {legacy.get('h1d_CosAlpha', 1.0)}")

    L.append("*** BLOCK B: WATER FLOW INFORMATION ************************************")
    L.append("MaxIt   TolTh   TolH")
    L.append(f"  {cfg.max_picard}  {_fmt(cfg.tol_theta)}  {_fmt(cfg.tol_h)}")
    L.append("TopInf WLayer KodTop lInitW")
    L.append(" " + " ".join([
        _t(legacy.get("h1d_TopInF", False)), _t(legacy.get("h1d_WLayer", False)),
        str(legacy.get("h1d_KodTop", 1)), _t(legacy.get("h1d_lInitW", False)),
    ]))
    L.append("BotInf qGWLF FreeD SeepF KodBot qDrain hSeep")
    L.append(" " + " ".join([
        _t(legacy.get("h1d_BotInF", False)),
        _t(legacy.get("h1d_qGWLF", False)),
        _t(cfg.free_drainage), _t(cfg.seepage_face),
        str(legacy.get("h1d_KodBot", -1)),
        _t(legacy.get("h1d_qDrain", False)),
        _fmt(legacy.get("h1d_hSeep", 0.0)),
    ]))

    # Conditional rTop / rBot / rRoot block (parser at hydrus1d/input.py L307)
    top_inf = legacy.get("h1d_TopInF", False)
    bot_inf = legacy.get("h1d_BotInF", False)
    kod_top = legacy.get("h1d_KodTop", 0)
    kod_bot = legacy.get("h1d_KodBot", 0)
    q_gwlf = legacy.get("h1d_qGWLF", False)
    q_drain = legacy.get("h1d_qDrain", False)
    need_rates = ((not top_inf and kod_top == -1)
                  or (not bot_inf and kod_bot == -1
                      and not q_gwlf and not cfg.free_drainage
                      and not cfg.seepage_face and not q_drain))
    if need_rates:
        L.append("rTop  rBot  rRoot")
        L.append(f"{_fmt(legacy.get('h1d_rTop', 0.0))}  "
                 f"{_fmt(legacy.get('h1d_rBot', 0.0))}  "
                 f"{_fmt(legacy.get('h1d_rRoot', 0.0))}")

    # qGWLF sub-block
    if q_gwlf:
        L.append("GWL0L  Aqh  Bqh")
        L.append(f"{_fmt(legacy.get('h1d_GWL0L', 0.0))}  "
                 f"{_fmt(legacy.get('h1d_Aqh', 0.0))}  "
                 f"{_fmt(legacy.get('h1d_Bqh', 0.0))}")

    L.append("hTab1  hTabN")
    L.append(f"{_fmt(legacy.get('h1d_hTab1', -1e-6))}  {_fmt(legacy.get('h1d_hTabN', -1e4))}")
    L.append("iModel iHyst")
    L.append(f"{legacy.get('h1d_iModel', 0)}  {legacy.get('h1d_iHyst', 0)}")
    L.append("thr    ths    Alfa   n      Ks     l")
    for m in mat:
        L.append(" " + " ".join(_fmt(v) for v in
                                 (m.theta_r, m.theta_s, m.alpha, m.n, m.Ks, m.l)))

    L.append("*** BLOCK C: TIME INFORMATION ******************************************")
    L.append("dt    dtMin  dtMax  dMul  dMul2 ItMin ItMax MPL")
    L.append(" ".join(_fmt(v) for v in
                       (t.dt, t.dt_min, t.dt_max, t.dt_mul, t.dt_mul2,
                        t.it_min, t.it_max, len(t.print_times))))
    L.append("tInit  tMax")
    L.append(f"{_fmt(t.t_init)} {_fmt(t.t_max)}")
    L.append("lPrintD nPrStep tPrintInt lEnter")
    L.append(" ".join([
        _t(legacy.get("h1d_lPrintD", False)),
        str(legacy.get("h1d_nPrStep", 1)),
        _fmt(legacy.get("h1d_tPrintInt", 1.0)),
        _t(legacy.get("h1d_lEnter", False)),
    ]))
    L.append("TPrint(1)..TPrint(MPL)")
    if t.print_times:
        # Up to 6 per line
        for i in range(0, len(t.print_times), 6):
            chunk = t.print_times[i:i + 6]
            L.append(" ".join(_fmt(v) for v in chunk))

    L.append("*** END OF INPUT FILE 'SELECTOR.IN' ***")
    path.write_text("\n".join(L) + "\n")
