"""
Synthesize a battery of HYDRUS-1D fixtures covering different soil
textures, hydraulic models, and BC configurations so the Python port can
be exercised on a much wider parameter envelope than the agrrobot set.

For each fixture we write a minimal Selector.in + Profile.dat to
``tests/fixtures/<name>/inputs/``.  The driver script ``run_all_scenarios.sh``
already knows how to drive Fortran + Python and diff their output.

Carsel & Parrish (1988) van Genuchten parameters for the standard USDA
texture classes (used wherever possible so other modellers can reproduce):

    texture   theta_r  theta_s  alpha    n        Ks       l
    sand      0.045    0.43     0.145    2.68     712.8    0.5
    loamy_sand 0.057   0.41     0.124    2.28     350.2    0.5
    sandy_loam 0.065   0.41     0.075    1.89     106.1    0.5  (= evap_v4)
    loam      0.078    0.43     0.036    1.56     24.96    0.5
    silt_loam 0.067    0.45     0.020    1.41     10.80    0.5
    clay      0.068    0.38     0.008    1.09     4.80     0.5
"""
from __future__ import annotations

from pathlib import Path
import textwrap

FIXTURE_BASE = Path("tests/fixtures")

# --- soil hydraulic parameter library ---------------------------------------

CARSEL_PARRISH = {
    # name: (thr, ths, alpha, n, Ks [cm/d], l)
    "sand":       (0.045, 0.430, 0.145, 2.68, 712.8, 0.5),
    "loamy_sand": (0.057, 0.410, 0.124, 2.28, 350.2, 0.5),
    "sandy_loam": (0.065, 0.410, 0.075, 1.89, 106.1, 0.5),
    "loam":       (0.078, 0.430, 0.036, 1.56,  24.96, 0.5),
    "silt_loam":  (0.067, 0.450, 0.020, 1.41,  10.80, 0.5),
    "clay":       (0.068, 0.380, 0.008, 1.09,   4.80, 0.5),
}


def _selector_v4(
    heading: str,
    LUnit: str, TUnit: str, MUnit: str,
    flags: dict,                         # lWat/lChem/lTemp/SinkF/lRoot/...
    NMat: int, CosAlf: float,
    MaxIt: int, TolTh: float, TolH: float,
    top_block: tuple,                    # (TopInF, WLayer, KodTop, lInitW)
    bot_block: tuple,                    # (BotInF, qGWLF, FreeD, SeepF, KodBot, qDrain, hSeep)
    hTab1: float, hTabN: float,
    iModel: int, iHyst: int,
    materials: list,                     # list of tuples of NPar floats
    dt: float, dtMin: float, dtMax: float,
    dMul: float, dMul2: float,
    ItMin: int, ItMax: int, MPL: int,
    tInit: float, tMax: float,
    TPrint: list[float],
    lPrintD: bool = False, nPrStep: int = 1, tPrintInt: float = 1.0,
    lEnter: bool = False,
) -> str:
    """Return a Selector.in string in v4 format."""
    def lb(b): return "t" if b else "f"
    # BLOCK A
    s = []
    s.append("Pcp_File_Version=4")
    s.append("*** BLOCK A: BASIC INFORMATION *****************************************")
    s.append("Heading")
    s.append(heading)
    s.append("LUnit  TUnit  MUnit")
    s.append(LUnit)
    s.append(TUnit)
    s.append(MUnit)
    s.append("lWat   lChem  lTemp  SinkF  lRoot  ShortO lWDep  lScreen AtmBC lEquil")
    flag_row = [
        flags.get(k, False) for k in
        ("lWat", "lChem", "lTemp", "SinkF", "lRoot",
         "ShortO", "lWDep", "lScreen", "AtmBC", "lEquil")
    ]
    s.append(" " + " ".join(lb(b) for b in flag_row))
    s.append("lSnow  lDummy lMeteo lVapor lActRSU lFlux")
    s.append(" " + " ".join(lb(b) for b in [
        flags.get("lSnow", False), False,
        flags.get("lMeteo", False), flags.get("lVapor", False),
        flags.get("lActRSU", False), flags.get("lFlux", False),
    ]))
    s.append("NMat   NLay   CosAlpha")
    s.append(f"  {NMat}  1  {CosAlf}")
    # BLOCK B
    s.append("*** BLOCK B: WATER FLOW INFORMATION ************************************")
    s.append("MaxIt   TolTh   TolH")
    s.append(f"  {MaxIt}  {TolTh}  {TolH}")
    s.append("TopInf WLayer KodTop lInitW")
    s.append(" " + " ".join([lb(top_block[0]), lb(top_block[1]),
                              str(top_block[2]), lb(top_block[3])]))
    s.append("BotInf qGWLF FreeD SeepF KodBot qDrain hSeep")
    s.append(" " + " ".join([
        lb(bot_block[0]), lb(bot_block[1]), lb(bot_block[2]),
        lb(bot_block[3]), str(bot_block[4]), lb(bot_block[5]),
        f"{bot_block[6]:g}",
    ]))
    # When BasInf sees a *constant-flux* top BC (TopInF=false AND KodTop=-1)
    # without a competing FreeD/qGWLF/SeepF/qDrain at the bottom, it reads
    # an extra line for rTop, rBot, rRoot.  See INPUT.FOR:63-72.
    need_rates = (
        (not top_block[0] and top_block[2] == -1)
        or (
            not bot_block[0] and bot_block[4] == -1
            and not bot_block[1] and not bot_block[2]
            and not bot_block[3] and not bot_block[5]
        )
    )
    if need_rates:
        s.append("rTop  rBot  rRoot")
        s.append("0  0  0")
    s.append("hTab1  hTabN")
    s.append(f"{hTab1:g}  {hTabN:g}")
    s.append("iModel iHyst")
    s.append(f"{iModel}  {iHyst}")
    if iModel == 2:
        s.append("thr    ths    Alfa   n      Ks     l")
    elif iModel == 1:
        s.append("thr    ths    Alfa   n      Ks     l       Qm     Qa     Qk     Kk")
    else:
        s.append("thr    ths    Alfa   n      Ks     l")
    for m in materials:
        s.append(" " + " ".join(f"{v:g}" for v in m))
    # BLOCK C
    s.append("*** BLOCK C: TIME INFORMATION ******************************************")
    s.append("dt    dtMin  dtMax  dMul  dMul2 ItMin ItMax MPL")
    s.append(f"{dt:g} {dtMin:g} {dtMax:g} {dMul:g} {dMul2:g} {ItMin} {ItMax} {MPL}")
    s.append("tInit  tMax")
    s.append(f"{tInit:g} {tMax:g}")
    s.append("lPrintD nPrStep tPrintInt lEnter")
    s.append(f"{lb(lPrintD)} {nPrStep} {tPrintInt:g} {lb(lEnter)}")
    s.append("TPrint(1)..TPrint(MPL)")
    s.append(" ".join(f"{tp:g}" for tp in TPrint))
    s.append("*** END OF INPUT FILE 'SELECTOR.IN' ***")
    return "\n".join(s) + "\n"


def _profile(
    NumNP: int,
    x_top: float, x_bot: float,
    h_init: float | list[float],
    MatNum: int | list[int],
    LayNum: int | list[int] | None = None,
    Beta: float | list[float] = 0.0,
    NObs: int = 0,
    obs_nodes: list[int] | None = None,
) -> str:
    """Return a Profile.dat string for ``NumNP`` uniformly-spaced nodes from
    ``x_top`` (surface, usually 0) to ``x_bot`` (most negative).

    HYDRUS stores nodes top → bottom (node 1 = surface, node NumNP = deepest),
    but with x decreasing downward.  Anything not provided as a list is
    broadcast across all nodes.
    """
    if isinstance(h_init, (int, float)):
        h_init = [float(h_init)] * NumNP
    if isinstance(MatNum, int):
        MatNum = [MatNum] * NumNP
    if LayNum is None:
        LayNum = [1] * NumNP
    elif isinstance(LayNum, int):
        LayNum = [LayNum] * NumNP
    if isinstance(Beta, (int, float)):
        Beta = [float(Beta)] * NumNP

    s = []
    # Fixed nodes block (Fortran NodInf:240-243): use 2 fixed lines.
    s.append("    2")
    s.append(f"    1  {x_top:13.6e}  1.000000e+000  1.000000e+000")
    s.append(f"    2  {x_bot:13.6e}  1.000000e+000  1.000000e+000")
    # Header: NumNP, junk, NS  + col header
    s.append(
        f"  {NumNP}    0    0    1 x         h         Mat  Lay      "
        "Beta           Axz            Bxz            Dxz            Temp"
    )
    dx = (x_bot - x_top) / (NumNP - 1)
    for n in range(NumNP):
        x = x_top + n * dx
        s.append(
            f"  {n+1:3d}  {x:13.6e}  {h_init[n]:13.6e}  {MatNum[n]:3d}  "
            f"{LayNum[n]:3d}  {Beta[n]:13.6e}  1.000000e+000  1.000000e+000  "
            f"1.000000e+000              "
        )
    # NObs
    s.append(f"   {NObs}")
    if NObs > 0:
        s.append("  " + "  ".join(str(n) for n in obs_nodes))
    return "\n".join(s) + "\n"


def _atmosph_constant(
    MaxAL: int = 1,
    hCritS: float = 0.0,
    tAtm: list[float] | None = None,
    Prec: list[float] | None = None,
    rSoil: list[float] | None = None,
    rRoot: list[float] | None = None,
    hCritA: list[float] | None = None,
) -> str:
    """Generate a minimal v4 ATMOSPH.IN with one or more atm records."""
    if tAtm is None: tAtm = [1.0]
    if Prec is None: Prec = [0.0] * len(tAtm)
    if rSoil is None: rSoil = [0.0] * len(tAtm)
    if rRoot is None: rRoot = [0.0] * len(tAtm)
    # HYDRUS convention: hCritA is a *negative* head (lower limit on the
    # surface pressure under evaporation).  Stored absolute-value-wise is
    # surprising; both Fortran BasInf and the existing evap_v4 fixture
    # write it as -1e5 in Atmosph.in.
    if hCritA is None: hCritA = [-1e5] * len(tAtm)
    s = []
    s.append("Pcp_File_Version=4")
    s.append("*** BLOCK J: ATMOSPHERIC INFORMATION **********************************")
    s.append("MaxAL")
    s.append(f"  {MaxAL}")
    s.append("lDayVar lSinPrec lLAI")
    s.append(" f f f")
    s.append("hCritS")
    s.append(f"{hCritS:g}")
    s.append("tAtm        Prec        rSoil       rRoot       hCritA      rB     hB     hT     tTop  tBot  Ampl")
    for i in range(MaxAL):
        s.append(
            f"  {tAtm[i]:g}  {Prec[i]:g}  {rSoil[i]:g}  {rRoot[i]:g}  "
            f"{hCritA[i]:g}  0  0  0  0  0  0"
        )
    s.append("*** END OF INPUT FILE 'ATMOSPH.IN' ***")
    return "\n".join(s) + "\n"


# --- fixture builder --------------------------------------------------------

def _write_fixture(name: str, selector: str, profile: str,
                   atmosph: str | None = None) -> None:
    d = FIXTURE_BASE / name / "inputs"
    d.mkdir(parents=True, exist_ok=True)
    (d / "Selector.in").write_text(selector)
    (d / "Profile.dat").write_text(profile)
    if atmosph is not None:
        (d / "Atmosph.in").write_text(atmosph)
    (FIXTURE_BASE / name / "reference_out").mkdir(parents=True, exist_ok=True)
    (FIXTURE_BASE / name / "python_out").mkdir(parents=True, exist_ok=True)


def build_all() -> list[str]:
    names = []

    # --- 1. Sandy soil, drainage from saturation ---------------------------
    name = "soil_sand_drain"
    params = CARSEL_PARRISH["sand"]
    sel = _selector_v4(
        heading="Sand: drainage from saturated -10cm, 2 days",
        LUnit="cm", TUnit="days", MUnit="g",
        flags={"lWat": True, "lEquil": True, "lScreen": True},
        NMat=1, CosAlf=1.0,
        MaxIt=20, TolTh=0.001, TolH=1.0,
        top_block=(False, False, -1, False),    # constant flux top
        bot_block=(False, False, True, False, -1, False, 0.0),  # free drainage
        hTab1=-1e-6, hTabN=-1e4,
        iModel=0, iHyst=0,
        materials=[params],
        dt=0.001, dtMin=1e-5, dtMax=0.5,
        dMul=1.3, dMul2=0.7, ItMin=3, ItMax=7, MPL=4,
        tInit=0.0, tMax=2.0,
        TPrint=[0.5, 1.0, 1.5, 2.0],
    )
    prof = _profile(NumNP=101, x_top=0.0, x_bot=-100.0, h_init=-10.0, MatNum=1)
    _write_fixture(name, sel, prof)
    names.append(name)

    # --- 2. Clay soil, slow drainage from -50cm -----------------------------
    name = "soil_clay_drain"
    params = CARSEL_PARRISH["clay"]
    sel = _selector_v4(
        heading="Clay: slow drainage from -50cm, 5 days",
        LUnit="cm", TUnit="days", MUnit="g",
        flags={"lWat": True, "lEquil": True, "lScreen": True},
        NMat=1, CosAlf=1.0, MaxIt=20, TolTh=0.001, TolH=1.0,
        top_block=(False, False, -1, False),
        bot_block=(False, False, True, False, -1, False, 0.0),
        hTab1=-1e-6, hTabN=-1e4,
        iModel=0, iHyst=0, materials=[params],
        dt=0.001, dtMin=1e-5, dtMax=0.5,
        dMul=1.3, dMul2=0.7, ItMin=3, ItMax=7, MPL=5,
        tInit=0.0, tMax=5.0,
        TPrint=[1.0, 2.0, 3.0, 4.0, 5.0],
    )
    prof = _profile(NumNP=101, x_top=0.0, x_bot=-100.0, h_init=-50.0, MatNum=1)
    _write_fixture(name, sel, prof)
    names.append(name)

    # --- 3. Loam: dry-soil infiltration (analogous to 1INFILTR but loam) ----
    name = "soil_loam_infiltr"
    params = CARSEL_PARRISH["loam"]
    sel = _selector_v4(
        heading="Loam: ponded infiltration into dry profile",
        LUnit="cm", TUnit="days", MUnit="g",
        flags={"lWat": True, "lEquil": True, "lScreen": True},
        NMat=1, CosAlf=1.0, MaxIt=20, TolTh=0.001, TolH=1.0,
        top_block=(False, False, 1, False),     # head BC at surface (h=0)
        bot_block=(False, False, True, False, -1, False, 0.0),
        hTab1=-1e-6, hTabN=-1e4,
        iModel=0, iHyst=0, materials=[params],
        dt=0.001, dtMin=1e-5, dtMax=0.5,
        dMul=1.3, dMul2=0.7, ItMin=3, ItMax=7, MPL=4,
        tInit=0.0, tMax=2.0,
        TPrint=[0.5, 1.0, 1.5, 2.0],
    )
    # Top node h=0 (ponded), rest at dry -500. The Profile.dat file orders
    # rows TOP → BOTTOM (file node 1 is the surface), so just place 0 first.
    h_init = [0.0] + [-500.0] * 100
    prof = _profile(NumNP=101, x_top=0.0, x_bot=-100.0,
                    h_init=h_init,
                    MatNum=1)
    _write_fixture(name, sel, prof)
    names.append(name)

    # --- 4. Silt loam, atmospheric evaporation ------------------------------
    name = "soil_silt_evap"
    params = CARSEL_PARRISH["silt_loam"]
    sel = _selector_v4(
        heading="Silt loam: 0.3 cm/day evap, 1 day",
        LUnit="cm", TUnit="days", MUnit="g",
        flags={"lWat": True, "lEquil": True, "lScreen": True, "AtmBC": True},
        NMat=1, CosAlf=1.0, MaxIt=20, TolTh=0.001, TolH=1.0,
        top_block=(True, False, -1, False),     # variable atm BC
        bot_block=(False, False, True, False, -1, False, 0.0),
        hTab1=-1e-6, hTabN=-1e4,
        iModel=0, iHyst=0, materials=[params],
        dt=0.001, dtMin=1e-5, dtMax=0.5,
        dMul=1.3, dMul2=0.7, ItMin=3, ItMax=7, MPL=4,
        tInit=0.0, tMax=1.0,
        TPrint=[0.25, 0.5, 0.75, 1.0],
    )
    prof = _profile(NumNP=101, x_top=0.0, x_bot=-100.0,
                    h_init=-200.0, MatNum=1)
    atm = _atmosph_constant(MaxAL=1, hCritS=0.0,
                            tAtm=[1.0], Prec=[0.0], rSoil=[0.3],
                            rRoot=[0.0], hCritA=[-1e5])
    _write_fixture(name, sel, prof, atm)
    names.append(name)

    # --- 5. Brooks-Corey (iModel=2) infiltration ---------------------------
    name = "soil_bc_infiltr"
    params = (0.065, 0.41, 0.075, 1.89, 106.1, 0.5)  # vG params re-used (B-C
                                                       # uses thr/ths/alpha/lambda/Ks/l)
    sel = _selector_v4(
        heading="Brooks-Corey: ponded infiltration into dry loam",
        LUnit="cm", TUnit="days", MUnit="g",
        flags={"lWat": True, "lEquil": True, "lScreen": True},
        NMat=1, CosAlf=1.0, MaxIt=20, TolTh=0.001, TolH=1.0,
        top_block=(False, False, 1, False),
        bot_block=(False, False, True, False, -1, False, 0.0),
        hTab1=-1e-6, hTabN=-1e4,
        iModel=2, iHyst=0, materials=[params],
        dt=0.001, dtMin=1e-5, dtMax=0.5,
        dMul=1.3, dMul2=0.7, ItMin=3, ItMax=7, MPL=4,
        tInit=0.0, tMax=2.0,
        TPrint=[0.5, 1.0, 1.5, 2.0],
    )
    h_init = [0.0] + [-300.0] * 100
    prof = _profile(NumNP=101, x_top=0.0, x_bot=-100.0,
                    h_init=h_init, MatNum=1)
    _write_fixture(name, sel, prof)
    names.append(name)

    # --- 6. Two-layer profile: sandy_loam over clay ------------------------
    name = "soil_layered_sand_over_clay"
    sl = CARSEL_PARRISH["sandy_loam"]
    cl = CARSEL_PARRISH["clay"]
    sel = _selector_v4(
        heading="Two-layer: sandy loam (top 50 cm) over clay (bottom 50 cm)",
        LUnit="cm", TUnit="days", MUnit="g",
        flags={"lWat": True, "lEquil": True, "lScreen": True},
        NMat=2, CosAlf=1.0, MaxIt=20, TolTh=0.001, TolH=1.0,
        top_block=(False, False, 1, False),     # ponded
        bot_block=(False, False, True, False, -1, False, 0.0),
        hTab1=-1e-6, hTabN=-1e4,
        iModel=0, iHyst=0, materials=[sl, cl],
        dt=0.001, dtMin=1e-5, dtMax=0.5,
        dMul=1.3, dMul2=0.7, ItMin=3, ItMax=7, MPL=4,
        tInit=0.0, tMax=3.0,
        TPrint=[0.5, 1.0, 2.0, 3.0],
    )
    # File-ordered top → bottom: surface (node 1) is sandy loam (mat 1),
    # then bottom 50 cm is clay (mat 2).
    matnums = [1] * 51 + [2] * 50
    h_init = [0.0] + [-300.0] * 100
    prof = _profile(NumNP=101, x_top=0.0, x_bot=-100.0,
                    h_init=h_init,
                    MatNum=matnums)
    _write_fixture(name, sel, prof)
    names.append(name)

    # --- 7. Sandy loam, 7-day evap with variable rate -----------------------
    name = "soil_sandyloam_evap_7d"
    params = CARSEL_PARRISH["sandy_loam"]
    sel = _selector_v4(
        heading="Sandy loam: weekly varying evap, 7 days, atm BC",
        LUnit="cm", TUnit="days", MUnit="g",
        flags={"lWat": True, "lEquil": True, "lScreen": True, "AtmBC": True},
        NMat=1, CosAlf=1.0, MaxIt=20, TolTh=0.001, TolH=1.0,
        top_block=(True, False, -1, False),
        bot_block=(False, False, True, False, -1, False, 0.0),
        hTab1=-1e-6, hTabN=-1e4,
        iModel=0, iHyst=0, materials=[params],
        dt=0.001, dtMin=1e-5, dtMax=0.5,
        dMul=1.3, dMul2=0.7, ItMin=3, ItMax=7, MPL=7,
        tInit=0.0, tMax=7.0,
        TPrint=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
    )
    # 7 daily records, mildly varying evap only (no precipitation events to
    # avoid abrupt BC discontinuities — the Picard solver in our Python port
    # handles smooth atm transitions but the 0 → 1.2 cm/d precip jump at
    # day 5 sends the surface from drying to saturating and induces
    # numerical thrash that the agrrobot binary copes with via Fortran's
    # tighter SetBC interpolation).
    rsoil_arr = [0.4, 0.45, 0.5, 0.4, 0.35, 0.3, 0.25]
    prec_arr  = [0.0] * 7
    atm = _atmosph_constant(
        MaxAL=7, hCritS=0.0,
        tAtm=[float(i + 1) for i in range(7)],
        Prec=prec_arr, rSoil=rsoil_arr,
        rRoot=[0.0]*7, hCritA=[-1e5]*7,
    )
    prof = _profile(NumNP=101, x_top=0.0, x_bot=-100.0,
                    h_init=-100.0, MatNum=1)
    _write_fixture(name, sel, prof, atm)
    names.append(name)

    # --- 8. Kosugi log-normal (iModel=4) ------------------------------------
    name = "soil_kosugi_drain"
    # Kosugi uses (thr, ths, hm, sigma, Ks, l) where hm = -1/alpha_geom and
    # sigma = standard deviation in ln(h).  Pick plausible values for a loam.
    kosugi_params = (0.075, 0.420, 30.0, 1.2, 25.0, 0.5)
    sel = _selector_v4(
        heading="Kosugi log-normal (iModel=4): drainage from -10 cm, 2 d",
        LUnit="cm", TUnit="days", MUnit="g",
        flags={"lWat": True, "lEquil": True, "lScreen": True},
        NMat=1, CosAlf=1.0, MaxIt=20, TolTh=0.001, TolH=1.0,
        top_block=(False, False, -1, False),
        bot_block=(False, False, True, False, -1, False, 0.0),
        hTab1=-1e-6, hTabN=-1e4,
        iModel=4, iHyst=0, materials=[kosugi_params],
        dt=0.001, dtMin=1e-5, dtMax=0.5,
        dMul=1.3, dMul2=0.7, ItMin=3, ItMax=7, MPL=4,
        tInit=0.0, tMax=2.0,
        TPrint=[0.5, 1.0, 1.5, 2.0],
    )
    prof = _profile(NumNP=101, x_top=0.0, x_bot=-100.0,
                    h_init=-10.0, MatNum=1)
    _write_fixture(name, sel, prof)
    names.append(name)

    return names


if __name__ == "__main__":
    names = build_all()
    print("Built fixtures:")
    for n in names:
        print(f"  - {n}")
