"""
Input file parsing for HYDRUS-1D Python port.
=============================================

Direct port of INPUT.FOR. Implements `Pcp_File_Version` detection plus all
16 input subroutines: BasInf, Conversion, NodInf, InitW, InitDualPor, MatIn,
HysterIn, GenMat (numerical table, not file IO), TmIn, MeteoIn, SinkIn,
RootIn, TempIn, ChemIn, OpenSoluteFiles, Init.

Public entry points used by hydrus.Hydrus1DSimulation:
    read_selector(path)        -> dict
    read_profile(path, sel)    -> dict
    read_atmospheric(path,sel) -> dict
    read_meteorological(path)  -> dict
    iGetFileVersion(path)      -> int
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import numpy as np
from numpy.typing import NDArray


# ============================================================================
# Fortran free-format reader
# ============================================================================

class FortranReader:
    """
    Minimal `read(unit, *)` emulator.

    Fortran list-directed read consumes whitespace/comma-separated tokens that
    may span lines. Critically, each `read` statement *starts a new record*:
    any unread tokens on the last touched line are discarded once a read
    completes. `skip()` mimics a bare `read(unit, *)` with no I/O list.
    """

    def __init__(self, path: str):
        self.path = path
        self.f = open(path, "r")
        self._buf: List[str] = []
        self._lineno = 0

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def close(self):
        if not self.f.closed:
            self.f.close()

    # ---- low-level ---------------------------------------------------------

    def _refill_line(self) -> None:
        line = self.f.readline()
        if not line:
            raise EOFError(
                f"unexpected EOF at line {self._lineno + 1} in {self.path}"
            )
        self._lineno += 1
        self._buf = _tokenize(line)

    def _take_n(self, n: int, conv) -> List[Any]:
        out: List[Any] = []
        while len(out) < n:
            if not self._buf:
                self._refill_line()
            while self._buf and len(out) < n:
                out.append(conv(self._buf.pop(0)))
        # New record on next call:
        self._buf = []
        return out

    # ---- public API --------------------------------------------------------

    def skip(self, n: int = 1) -> None:
        """Skip n records (header / comment lines)."""
        for _ in range(n):
            self._buf = []
            line = self.f.readline()
            if line == "":
                # Allow trailing skips at EOF without raising.
                return
            self._lineno += 1

    def read_int(self, n: int = 1):
        out = self._take_n(n, _to_int)
        return out[0] if n == 1 else out

    def read_float(self, n: int = 1):
        out = self._take_n(n, _to_float)
        return out[0] if n == 1 else out

    def read_bool(self, n: int = 1):
        out = self._take_n(n, _to_bool)
        return out[0] if n == 1 else out

    def read_record(self) -> List[str]:
        """Consume the next non-empty record and return its raw tokens."""
        self._buf = []
        while True:
            line = self.f.readline()
            if line == "":
                raise EOFError(f"EOF in {self.path}")
            self._lineno += 1
            toks = _tokenize(line)
            if toks:
                return toks

    def read_line(self) -> str:
        """Read one full line as a stripped string."""
        self._buf = []
        line = self.f.readline()
        if line == "":
            raise EOFError(f"EOF in {self.path}")
        self._lineno += 1
        return line.rstrip("\n").rstrip("\r").strip()

    def detect_version(self) -> int:
        """
        Direct port of iGetFileVersion: if line 1 starts with
        ``Pcp_File_Version=N`` consume it and return N; otherwise rewind and
        return 0.
        """
        pos = self.f.tell()
        line = self.f.readline()
        if not line:
            self.f.seek(pos)
            return 0
        s = line.lstrip()
        if s.startswith("Pcp_File_Version="):
            try:
                ver = int(s.split("=", 1)[1].strip())
            except ValueError:
                ver = 0
            self._lineno += 1
            self._buf = []
            return ver
        self.f.seek(pos)
        return 0


def _tokenize(line: str) -> List[str]:
    # Strip CRLF, normalize commas to spaces, split on whitespace.
    return line.rstrip("\n").rstrip("\r").replace(",", " ").split()


def _to_int(t: str) -> int:
    s = t.strip()
    # Fortran free-format integer also accepts +/- prefix
    try:
        return int(s)
    except ValueError:
        # Some Fortran writers emit 1.0 for an INTEGER slot (rare but seen)
        return int(float(s.lower().replace("d", "e")))


def _to_float(t: str) -> float:
    s = t.strip().lower()
    # Fortran D-exponent (1.5d-3) → Python e-exponent (1.5e-3)
    if "d" in s:
        s = s.replace("d", "e")
    return float(s)


def _to_bool(t: str) -> bool:
    s = t.strip().lstrip(".").lower()
    if not s:
        return False
    return s[0] in ("t", "1", "y")


# ============================================================================
# Version detection (public helper)
# ============================================================================

def iGetFileVersion(filename: str) -> int:
    """Probe `Pcp_File_Version=N` in the first line; return 0 if absent."""
    if not os.path.exists(filename):
        return 0
    try:
        with open(filename, "r") as f:
            line = f.readline()
    except OSError:
        return 0
    if line.lstrip().startswith("Pcp_File_Version="):
        try:
            return int(line.split("=", 1)[1].strip())
        except ValueError:
            return 0
    return 0


# ============================================================================
# Block A : BasInf — basic simulation setup (subroutine BasInf, line 3 of INPUT.FOR)
# ============================================================================

def _conversion(LUnit: str, TUnit: str) -> tuple[float, float]:
    """Direct port of Conversion subroutine (INPUT.FOR:182)."""
    xConv = 1.0
    tConv = 1.0
    u = LUnit.strip().lower()
    if u.startswith("cm"):
        xConv = 100.0
    elif u.startswith("mm"):
        xConv = 1000.0
    u = TUnit.strip().lower()
    if u.startswith("min"):
        tConv = 1.0 / 60.0
    elif u.startswith("hours") or u == "hour" or u == "hours":
        tConv = 1.0 / 3600.0
    elif u.startswith("days") or u == "day":
        tConv = 1.0 / 86400.0
    elif u.startswith("years") or u == "year":
        tConv = 1.0 / (86400.0 * 365.0)
    return xConv, tConv


def _read_bas_inf(r: FortranReader, ver: int) -> Dict[str, Any]:
    """
    Block A: basic information. Port of BasInf (INPUT.FOR:3-178).

    Fills `Heading`, units, all top/bottom BC flags, the iVer-dependent
    extended flag row, KodTop/KodBot/qDrain sub-blocks, hCritA, etc.
    """
    out: Dict[str, Any] = {"iVer": ver}

    r.skip()                          # *** BLOCK A *** banner
    r.skip()                          # "Heading"
    out["Heading"] = r.read_line()
    r.skip()                          # "LUnit  TUnit  MUnit"
    LUnit = r.read_line()
    TUnit = r.read_line()
    MUnit = r.read_line()
    out["LUnit"] = LUnit
    out["TUnit"] = TUnit
    out["MUnit"] = MUnit
    xConv, tConv = _conversion(LUnit, TUnit)
    out["xConv"] = xConv
    out["tConv"] = tConv

    r.skip()                          # lWat lChem lTemp ... lEquil header
    (
        out["lWat"], out["lChem"], out["lTemp"], out["SinkF"],
        out["lRoot"], out["ShortO"], out["lWDep"], out["lScreen"],
        out["AtmBC"], out["lEquil"],
    ) = r.read_bool(10)

    # Defaults for the version-specific extended row.
    out["lSnow"] = False
    out["lMeteo"] = False
    out["lVapor"] = False
    out["lActRSU"] = False
    out["lFlux"] = False
    if ver == 3:
        r.skip()
        flags = r.read_bool(4)
        out["lSnow"] = flags[0]
    elif ver == 4:
        r.skip()
        flags = r.read_bool(6)
        out["lSnow"], _dummy, out["lMeteo"], out["lVapor"], out["lActRSU"], out["lFlux"] = flags
    if out["lSnow"] and not out["lTemp"]:
        out["lSnow"] = False

    r.skip()                          # NMat NLay CosAlf header
    out["NMat"], out["NLay"], out["CosAlf"] = _read_record_typed(
        r, [int, int, float]
    )

    r.skip()                          # *** BLOCK B banner
    r.skip()                          # "MaxIt TolTh TolH" header
    out["MaxIt"], out["TolTh"], out["TolH"] = _read_record_typed(r, [int, float, float])

    r.skip()                          # TopInF WLayer KodTop lInitW header
    out["TopInF"], out["WLayer"], KodTop, out["lInitW"] = _read_record_typed(
        r, [bool, bool, int, bool]
    )

    r.skip()                          # BotInF qGWLF FreeD SeepF KodBot qDrain (hSeep) header
    if ver <= 3:
        # Try reading 6 fields; if qDrain is missing assume False (Fortran err=904 path)
        toks = r.read_record()
        # Pad to 6 tokens if needed
        while len(toks) < 6:
            toks.append("f")
        BotInF = _to_bool(toks[0])
        qGWLF = _to_bool(toks[1])
        FreeD = _to_bool(toks[2])
        SeepF = _to_bool(toks[3])
        KodBot = _to_int(toks[4])
        qDrain = _to_bool(toks[5])
        hSeep = 0.0
    else:
        BotInF, qGWLF, FreeD, SeepF, KodBot, qDrain, hSeep = _read_record_typed(
            r, [bool, bool, bool, bool, int, bool, float]
        )
    out.update(BotInF=BotInF, qGWLF=qGWLF, FreeD=FreeD, SeepF=SeepF,
               qDrain=qDrain, hSeep=hSeep)

    # Conditional rTop/rBot/rRoot block
    need_rates = (
        (not out["TopInF"] and KodTop == -1)
        or (not BotInF and KodBot == -1
            and not qGWLF and not FreeD and not SeepF and not qDrain)
    )
    if need_rates:
        r.skip()
        out["rTop"], out["rBot"], out["rRoot"] = _read_record_typed(
            r, [float, float, float]
        )
    else:
        out["rTop"], out["rBot"], out["rRoot"] = 0.0, 0.0, 0.0

    # qGWLF sub-block
    if qGWLF:
        r.skip()
        out["GWL0L"], out["Aqh"], out["Bqh"] = _read_record_typed(
            r, [float, float, float]
        )
    else:
        out["GWL0L"], out["Aqh"], out["Bqh"] = 0.0, 0.0, 0.0

    # qDrain sub-block
    if qDrain:
        r.skip()
        out["iPosDr"] = r.read_int()
        r.skip()
        out["zBotDr"], out["rSpacing"], out["Entres"] = _read_record_typed(
            r, [float, float, float]
        )
        out["zBotDr"] = -abs(out["zBotDr"])
        r.skip()
        pos = out["iPosDr"]
        rec = r.read_record()
        rec_f = [_to_float(t) for t in rec]
        if pos == 1:
            out["rKhTop"] = rec_f[0]
        elif pos == 2:
            out["BaseGW"], out["rKhTop"], out["WetPer"] = rec_f[:3]
        elif pos == 3:
            out["BaseGW"], out["rKhTop"], out["rKhBot"], out["WetPer"] = rec_f[:4]
        elif pos == 4:
            (out["BaseGW"], out["rKvTop"], out["rKvBot"], out["rKhBot"],
             out["WetPer"], out["zInTF"]) = rec_f[:6]
        elif pos == 5:
            (out["BaseGW"], out["rKhTop"], out["rKvTop"], out["rKhBot"],
             out["WetPer"], out["zInTF"], out["GeoFac"]) = rec_f[:7]
        out["BaseGW"] = -abs(out.get("BaseGW", 0.0))
        out["zInTF"] = -abs(out.get("zInTF", 0.0))

    # Post-processing identical to BasInf lines 100-115
    out["rRoot"] = abs(out["rRoot"])
    out["hCritA"] = -abs(1.0e10)
    if out["TopInF"]:
        KodTop = _isign(3, KodTop)
    if BotInF:
        KodBot = _isign(3, KodBot)
    out["hCritS"] = 0.0
    if out["AtmBC"] and KodTop < 0:
        out["hCritS"] = 0.0
        KodTop = -4
    if out["WLayer"]:
        KodTop = -abs(KodTop)
    if qGWLF:
        KodBot = -7
    if FreeD:
        KodBot = -5
    if SeepF:
        KodBot = -2
    out["KodTop"] = KodTop
    out["KodBot"] = KodBot
    out["kTOld"] = KodTop
    out["kBOld"] = KodBot
    return out


def _isign(a: int, b: int) -> int:
    """Fortran ISIGN(a, b) - sign of b applied to magnitude of a."""
    return abs(a) if b >= 0 else -abs(a)


def _read_record_typed(r: FortranReader, types: List[type]) -> List[Any]:
    """Read one record and cast tokens by the given type list."""
    rec = r.read_record()
    if len(rec) < len(types):
        raise ValueError(
            f"{r.path}: need {len(types)} tokens, got {len(rec)} at line {r._lineno}"
        )
    out = []
    for tok, ty in zip(rec, types):
        if ty is bool:
            out.append(_to_bool(tok))
        elif ty is int:
            out.append(_to_int(tok))
        elif ty is float:
            out.append(_to_float(tok))
        else:
            out.append(tok)
    return out


# ============================================================================
# Block B : MatIn — soil hydraulic parameters (INPUT.FOR:536-728)
# ============================================================================

# nTabMod is set by Init() to 10 (the canonical 'sentinel' for table input).
NTAB_MOD = 10
# Default table length (Fortran HYDRUS.FOR Init: NTab(1) = 100).
NTAB_DEFAULT = 100


def gen_mat_tables(sel: Dict[str, Any]) -> None:
    """Direct port of INPUT.FOR:762-862 GenMat.

    Builds log-spaced ``hTab(NTab, NMat)`` and the corresponding
    ``ConTab/CapTab/TheTab`` tables.  Fortran SetMat then *linearly*
    interpolates these in physical-h space (not log-h) — replicating that
    interpolation exactly is the only way to match the surface boundary
    flux of the Fortran binary at the 0.3 % level.
    """
    from .material import FK, FC, FQ, FH

    iModel = sel.get("iModel", 0)
    if iModel >= NTAB_MOD:
        return                            # table input cases not handled here
    if not sel.get("lTable", True):
        return                            # skip if user disabled tables

    NMat = sel["NMat"]
    ParD = sel["ParD"]
    NTab = NTAB_DEFAULT

    hTab1 = float(sel["hTab1"])
    hTabN = float(sel["hTabN"])
    if hTab1 == 0.0 or hTabN == 0.0 or hTab1 == hTabN:
        return

    import math as _math
    alh1 = _math.log10(-hTab1)
    alhN = _math.log10(-hTabN)
    dlh = (alhN - alh1) / (NTab - 1)

    hTab = np.zeros((NTab, NMat), dtype=np.float64)
    ConTab = np.zeros((NTab, NMat), dtype=np.float64)
    CapTab = np.zeros((NTab, NMat), dtype=np.float64)
    TheTab = np.zeros((NTab, NMat), dtype=np.float64)
    hSat = np.zeros(NMat, dtype=np.float64)
    ConSat = np.zeros(NMat, dtype=np.float64)

    for M in range(NMat):
        hSat[M] = FH(iModel, 1.0, ParD[:, M])
        ConSat[M] = ParD[4, M]
        for i in range(NTab):
            alh = alh1 + i * dlh
            h_val = -10.0 ** alh
            hTab[i, M] = h_val
            ConTab[i, M] = FK(iModel, h_val, ParD[:, M])
            CapTab[i, M] = FC(iModel, h_val, ParD[:, M])
            TheTab[i, M] = FQ(iModel, h_val, ParD[:, M])

    sel["NTab"] = NTab
    sel["hTab"] = hTab
    sel["ConTab"] = ConTab
    sel["CapTab"] = CapTab
    sel["TheTab"] = TheTab
    sel["hSat_M"] = hSat
    sel["ConSat"] = ConSat
    sel["alh1"] = alh1
    sel["dlh"] = dlh


def interp_K(h: float, M: int, sel: Dict[str, Any]) -> float:
    """Fortran-style linear-in-h interpolation of K from the table.

    Falls back to analytical FK when h falls outside the tabulated range.
    """
    from .material import FK
    if "hTab" not in sel:
        return FK(sel.get("iModel", 0), h, sel["ParD"][:, M])
    hTab = sel["hTab"]; ConTab = sel["ConTab"]
    NTab = sel["NTab"]; alh1 = sel["alh1"]; dlh = sel["dlh"]
    hSat = sel["hSat_M"][M]
    if h >= hSat:
        return float(sel["ConSat"][M])
    import math as _math
    if not (hTab[NTab - 1, M] <= h <= hTab[0, M]):
        return FK(sel.get("iModel", 0), h, sel["ParD"][:, M])
    iT = int((_math.log10(-h) - alh1) / dlh)
    iT = max(0, min(iT, NTab - 2))
    dh = (h - hTab[iT, M]) / (hTab[iT + 1, M] - hTab[iT, M])
    return float(ConTab[iT, M] + (ConTab[iT + 1, M] - ConTab[iT, M]) * dh)


def interp_theta_cap(h: float, M: int, sel: Dict[str, Any]) -> tuple[float, float]:
    """Linear-in-h interpolation of (theta, Cap) from the table.

    Falls back to analytical FQ/FC outside the range.  Returns
    ``(theta, Cap)``.
    """
    from .material import FQ, FC
    iModel = sel.get("iModel", 0)
    if "hTab" not in sel:
        return (float(FQ(iModel, h, sel["ParD"][:, M])),
                float(FC(iModel, h, sel["ParD"][:, M])))
    hTab = sel["hTab"]; TheTab = sel["TheTab"]; CapTab = sel["CapTab"]
    NTab = sel["NTab"]; alh1 = sel["alh1"]; dlh = sel["dlh"]
    hSat = sel["hSat_M"][M]
    if h >= hSat:
        # Saturated
        return float(sel["ParD"][1, M]), 0.0
    import math as _math
    if not (hTab[NTab - 1, M] <= h <= hTab[0, M]):
        return (float(FQ(iModel, h, sel["ParD"][:, M])),
                float(FC(iModel, h, sel["ParD"][:, M])))
    iT = int((_math.log10(-h) - alh1) / dlh)
    iT = max(0, min(iT, NTab - 2))
    dh = (h - hTab[iT, M]) / (hTab[iT + 1, M] - hTab[iT, M])
    theta = float(TheTab[iT, M] + (TheTab[iT + 1, M] - TheTab[iT, M]) * dh)
    cap = float(CapTab[iT, M] + (CapTab[iT + 1, M] - CapTab[iT, M]) * dh)
    return theta, cap


def _read_mat_in(r: FortranReader, sel: Dict[str, Any]) -> None:
    """
    Read Block B material parameters into ParD/ParW (11 x NMat).

    Mutates `sel` in-place by adding: hTab1, hTabN, iModel, iHyst,
    IKappa, lTable, ParD, ParW.
    """
    NMat = sel["NMat"]
    xConv = sel["xConv"]

    r.skip()                          # "hTab1 hTabN" header
    hTab1, hTabN = _read_record_typed(r, [float, float])
    r.skip()                          # "iModel iHyst" header
    iModel, iHyst = _read_record_typed(r, [int, int])

    if iModel == 8:
        raise NotImplementedError(
            "iModel=8 (dual-permeability) is handled by HYDRUS-1D-DualPerm,"
            " not the standard distribution."
        )

    iDualPor = sel.get("iDualPor", 0)
    if iModel < NTAB_MOD:
        hTab1 = -min(abs(hTab1), abs(hTabN))
        hTabN = -max(abs(hTab1), abs(hTabN))
        lTable = True
        if (hTab1 > -1e-5 and hTabN > -1e-5) or hTab1 == hTabN:
            lTable = False
            hTab1 = -1.0e-4 * xConv
            hTabN = -100.0 * xConv
    else:
        lTable = True

    if iHyst > 0:
        r.skip()
        IKappa = r.read_int()
    else:
        IKappa = -1

    # NPar per iModel — mirror MatIn:611-629
    if iModel in (0, 2, 3, 4):
        NPar = 6
    elif iModel == 1:
        NPar = 10
    elif iModel == 5:
        NPar = 9
    elif iModel == 6:
        NPar = 9
        iModel = 0
        iDualPor = 1
    elif iModel == 7:
        NPar = 11
        iModel = 0
        iDualPor = 2
    elif iModel == NTAB_MOD:
        NPar = 3
    else:
        NPar = 6

    ParD = np.zeros((11, NMat), dtype=np.float64)
    ParW = np.zeros((11, NMat), dtype=np.float64)

    r.skip()                          # parameter table header
    rHEntry = 0.02 * xConv
    for M in range(NMat):
        if iHyst == 0:
            vals = r.read_float(NPar)
            for i, v in enumerate(vals):
                ParD[i, M] = v
            if iModel == 1:
                ParD[6, M] = max(ParD[6, M], ParD[1, M])
                ParD[7, M] = min(ParD[7, M], ParD[0, M])
                ParD[8, M] = min(ParD[8, M], ParD[1, M])
                ParD[9, M] = min(ParD[9, M], ParD[4, M])
            elif iModel == 3:
                # ParD(7) = qa derived from van Genuchten formula
                qr, qs, alpha, n = ParD[0, M], ParD[1, M], ParD[2, M], ParD[3, M]
                ParD[6, M] = qr + (qs - qr) * (1.0 + (alpha * rHEntry) ** n) ** (1.0 - 1.0 / n)
        else:
            # Hysteresis: read 7 dry-curve params + 3 wet-curve params
            vals = r.read_float(10)
            for i in range(7):
                ParD[i, M] = vals[i]
            ParW[1, M] = vals[7]
            ParW[2, M] = vals[8]
            ParW[4, M] = vals[9]
            ParD[6, M] = max(ParD[6, M], ParD[1, M])
            ParW[0, M] = ParD[0, M]
            ParW[3, M] = ParD[3, M]
            ParW[5, M] = ParD[5, M]
            ParW[6, M] = ParW[0, M] + ((ParW[1, M] - ParW[0, M]) / (ParD[1, M] - ParD[0, M])) * (ParD[6, M] - ParD[0, M])
            ParD[7, M] = ParD[0, M]
            ParD[8, M] = ParD[1, M]
            ParD[9, M] = ParD[4, M]
            ParW[7, M] = ParW[0, M]
            ParW[8, M] = ParW[1, M]
            ParW[9, M] = ParW[4, M]

    # When there is no hysteresis Fortran uses ParD wherever ParW would be
    # referenced (see HYSTER.FOR / WATFLOW.FOR). Mirror that here by copying
    # the dry curve into ParW so downstream code can index ParW[1,M],
    # ParW[10,M], … unconditionally.
    if iHyst == 0:
        ParW[:] = ParD

    sel.update(
        hTab1=hTab1, hTabN=hTabN, iModel=iModel, iHyst=iHyst,
        IKappa=IKappa, lTable=lTable, ParD=ParD, ParW=ParW,
        iDualPor=iDualPor, NPar=NPar,
    )


# ============================================================================
# Block C : TmIn — time control + atmospheric header (INPUT.FOR:866-921)
# ============================================================================

def _read_tm_in(r: FortranReader, sel: Dict[str, Any]) -> None:
    """Read the time block from Selector.in. Atmospheric header is handled
    separately in read_atmospheric."""
    ver = sel["iVer"]

    r.skip()                          # *** BLOCK C banner (also functions as the leading skip)
    r.skip()                          # column header for dt/dtMin/...
    rec = r.read_record()
    dt = _to_float(rec[0]); dtMin = _to_float(rec[1]); dtMax = _to_float(rec[2])
    dMul = _to_float(rec[3]); dMul2 = _to_float(rec[4])
    ItMin = _to_int(rec[5]); ItMax = _to_int(rec[6]); MPL = _to_int(rec[7])

    r.skip()                          # "tInit tMax"
    tInit, tMax = _read_record_typed(r, [float, float])

    lPrintD = False
    nPrStep = 1
    tPrintInt = 86400.0
    lEnter = True
    if ver > 2:
        r.skip()                      # "lPrintD nPrStep tPrintInt lEnter" header
        lPrintD, nPrStep, tPrintInt, lEnter = _read_record_typed(
            r, [bool, int, float, bool]
        )

    r.skip()                          # "TPrint(1)..TPrint(MPL)"
    TPrint = np.array(r.read_float(MPL), dtype=np.float64)

    sel.update(
        dt=dt, dtMin=dtMin, dtMax=dtMax, dMul=dMul, dMul2=dMul2,
        ItMin=ItMin, ItMax=ItMax, MPL=MPL, tInit=tInit, tMax=tMax,
        lPrintD=lPrintD, nPrStep=nPrStep, tPrintInt=tPrintInt, lEnter=lEnter,
        TPrint=TPrint,
    )


# ============================================================================
# RootIn — root growth info (INPUT.FOR:1130-1192)
# ============================================================================

def _read_root_in(r: FortranReader, sel: Dict[str, Any]) -> None:
    ver = sel["iVer"]
    tRPeriod = 1.0e30
    iRootIn = 2
    nGrowth = 0
    rGrowth: Optional[NDArray[np.float64]] = None

    r.skip()                          # banner / header
    if ver == 4:
        r.skip()
        iRootIn = r.read_int()
        if iRootIn == 1:
            r.skip()
            nGrowth = r.read_int()
            if nGrowth > 1000:
                raise ValueError("Number of crop growth data > 1000")
            r.skip()
            rGrowth = np.zeros((nGrowth, 5), dtype=np.float64)
            for i in range(nGrowth):
                t_i, x_i = _read_record_typed(r, [float, float])
                rGrowth[i, 0] = t_i
                rGrowth[i, 4] = x_i

    tRMin = tRHarv = xRMin = xRMax = RGR = 0.0
    if ver < 4 or iRootIn == 2:
        iRootIn = 2
        r.skip()
        if ver < 4:
            rec = r.read_record()
            iRFak = _to_int(rec[0]); tRMin = _to_float(rec[1]); tRMed = _to_float(rec[2])
            tRHarv = _to_float(rec[3]); xRMin = _to_float(rec[4]); xRMed = _to_float(rec[5])
            xRMax = _to_float(rec[6])
        else:
            rec = r.read_record()
            iRFak = _to_int(rec[0]); tRMin = _to_float(rec[1]); tRMed = _to_float(rec[2])
            tRHarv = _to_float(rec[3]); xRMin = _to_float(rec[4]); xRMed = _to_float(rec[5])
            xRMax = _to_float(rec[6]); tRPeriod = _to_float(rec[7])
        if iRFak == 1:
            tRMed = (tRHarv + tRMin) / 2.0
            xRMed = (xRMax + xRMin) / 2.0
        rtm = tRMed - tRMin
        if rtm < 1e-20 or xRMed < 1e-10:
            raise ValueError("RootIn: tRMed - tRMin too small or xRMed too small")
        import math
        RGR = -(1.0 / rtm) * math.log(
            max(1e-4, (xRMin * (xRMax - xRMed))) / (xRMed * (xRMax - xRMin))
        )

    sel.update(
        iRootIn=iRootIn, nGrowth=nGrowth, rGrowth=rGrowth,
        tRMin=tRMin, tRHarv=tRHarv, xRMin=xRMin, xRMax=xRMax, RGR=RGR,
        tRPeriod=tRPeriod,
    )


# ============================================================================
# TempIn — heat transport setup (INPUT.FOR:1196-1237)
# ============================================================================

def _read_temp_in(r: FortranReader, sel: Dict[str, Any]) -> None:
    NMat = sel["NMat"]
    ver = sel["iVer"]

    r.skip()                          # banner
    r.skip()                          # column header
    TPar = np.zeros((10, NMat), dtype=np.float64)
    for i in range(NMat):
        vals = r.read_float(9)
        for j, v in enumerate(vals):
            TPar[j, i] = v

    r.skip()
    if ver <= 2:
        Ampl, tPeriod = _read_record_typed(r, [float, float])
        iCampbell = 0
        SnowMF = 0.0
    else:
        rec = r.read_record()
        Ampl = _to_float(rec[0]); tPeriod = _to_float(rec[1])
        iCampbell = _to_int(rec[2])
        SnowMF = _to_float(rec[3]) if len(rec) > 3 else 0.0

    r.skip()
    rec = r.read_record()
    kTopT = _to_int(rec[0]); tTop_T = _to_float(rec[1])
    kBotT = _to_int(rec[2]); tBot_T = _to_float(rec[3])
    # Only override tTop/tBot when respective top/bot is *not* time-variable
    if not sel.get("TopInF", False):
        sel["TTop"] = tTop_T
    if not sel.get("BotInF", False):
        sel["TBot"] = tBot_T

    sel.update(
        TPar=TPar, Ampl=Ampl, tPeriod=tPeriod, iCampbell=iCampbell,
        SnowMF=SnowMF, KodTopT=kTopT, KodBotT=kBotT,
    )


# ============================================================================
# ChemIn — solute transport setup (INPUT.FOR:1274-1445)
# ============================================================================

def _read_chem_in(r: FortranReader, sel: Dict[str, Any]) -> None:
    NMat = sel["NMat"]
    ver = sel["iVer"]
    Par = sel["ParD"]

    r.skip()                          # banner
    r.skip()                          # column header

    if ver <= 2:
        rec = r.read_record()
        epsi = _to_float(rec[0])
        lUpW = _to_bool(rec[1]); lArtD = _to_bool(rec[2]); lTDep = _to_bool(rec[3])
        cTolA = _to_float(rec[4]); cTolR = _to_float(rec[5])
        MaxItC = _to_int(rec[6])
        PeCr = _to_float(rec[7]); NS = _to_int(rec[8])
        lTort = _to_bool(rec[9])
        iBact = 0; lFiltr = False
    else:
        rec = r.read_record()
        epsi = _to_float(rec[0])
        lUpW = _to_bool(rec[1]); lArtD = _to_bool(rec[2]); lTDep = _to_bool(rec[3])
        cTolA = _to_float(rec[4]); cTolR = _to_float(rec[5])
        MaxItC = _to_int(rec[6])
        PeCr = _to_float(rec[7]); NS = _to_int(rec[8])
        lTort = _to_bool(rec[9])
        iBact = _to_int(rec[10]) if len(rec) > 10 else 0
        lFiltr = _to_bool(rec[11]) if len(rec) > 11 else False
    lBact = (iBact == 1)

    iMoistDep = 0
    lMoistDep = False
    lDualNEq = False
    lMassIni = False
    lEqInit = False
    iTort = 0
    if ver == 4:
        r.skip()
        rec = r.read_record()
        iNonEqul = _to_int(rec[0])
        lMoistDep = _to_bool(rec[1])
        lDualNEq = _to_bool(rec[2])
        lMassIni = _to_bool(rec[3])
        lEqInit = _to_bool(rec[4])
        lVar = _to_bool(rec[5]) if len(rec) > 5 else False
        if lMoistDep:
            iMoistDep = 1
        if lVar:
            iTort = 1

    PeCr = max(PeCr, 0.1)

    r.skip()                          # Material density / dispersion header
    ChPar = np.zeros((NS * 16 + 4, NMat), dtype=np.float64)
    lEquil = True
    lMobIm = np.zeros(NMat, dtype=bool)
    for M in range(NMat):
        vals = r.read_float(4)
        for j in range(4):
            ChPar[j, M] = vals[j]
        if ChPar[2, M] < 1.0 or ChPar[3, M] > 0.0 or lBact:
            lEquil = False
        if not lBact and ChPar[3, M] > 0.0:
            lMobIm[M] = True
        if not lEquil and ChPar[0, M] == 0.0:
            raise ValueError(f"ChemIn: bulk density zero for material {M+1}")

    lLinear = np.ones(NS, dtype=bool)
    for jj in range(NS):
        jjj = jj * 16
        r.skip()                      # solute header
        Dw, Dg = r.read_float(2)
        ChPar[jjj + 4, 0] = Dw
        ChPar[jjj + 5, 0] = Dg
        r.skip()                      # column header
        for M in range(NMat):
            ChPar[jjj + 4, M] = Dw
            ChPar[jjj + 5, M] = Dg
            vals = r.read_float(14)
            for k, v in enumerate(vals):
                ChPar[jjj + 6 + k, M] = v
            if abs(ChPar[jjj + 7, M]) > 1e-12 or abs(ChPar[jjj + 8, M] - 1.0) > 1e-3:
                lLinear[jj] = False
            if lBact and (ChPar[jjj + 17, M] > 0.0 or ChPar[jjj + 14, M] > 0.0):
                lLinear[jj] = False

    TDep = np.zeros(NS * 16 + 4, dtype=np.float64)
    WDep = np.ones((2 + NMat, NS * 9), dtype=np.float64)
    WDep[1, :] = 0.0
    CumCh = np.zeros((10, NS), dtype=np.float64)

    if lTDep:
        for jj in range(NS):
            jjj = jj * 16
            if jj == 0:
                r.skip()
            r.skip()
            v5, v6 = r.read_float(2)
            TDep[jjj + 4] = v5
            TDep[jjj + 5] = v6
            r.skip()
            vals = r.read_float(14)
            for k, v in enumerate(vals):
                TDep[jjj + 6 + k] = v

    if iMoistDep == 1:
        for jj in range(NS):
            if jj == 0:
                r.skip()
            r.skip()
            nPar2 = r.read_int()      # unused
            jjj = jj * 9
            r.skip()
            row1 = r.read_float(9)
            row2 = r.read_float(9)
            for j in range(9):
                WDep[0, jjj + j] = row1[j]
                WDep[1, jjj + j] = row2[j]
            # Optional: WDep(2+M,jjj+j) = FQ(...) — deferred to runtime in
            # solute.compute_coefficients to avoid circular imports.

    r.skip()                          # "kTopCh cTop(1..NS) kBotCh cBot(1..NS)"
    rec = r.read_record()
    kTopCh = _to_int(rec[0])
    cTop = np.array([_to_float(t) for t in rec[1:1 + NS]], dtype=np.float64)
    kBotCh = _to_int(rec[1 + NS])
    cBot = np.array([_to_float(t) for t in rec[2 + NS:2 + 2 * NS]], dtype=np.float64)

    dSurf = 0.0
    cAtm = 0.0
    if kTopCh == -2:
        r.skip()
        dSurf, cAtm = _read_record_typed(r, [float, float])

    r.skip()                          # "tPulse" header
    tPulse = r.read_float()

    sel.update(
        epsi=epsi, lUpW=lUpW, lArtD=lArtD, lTDep=lTDep,
        cTolA=cTolA, cTolR=cTolR, MaxItC=MaxItC, PeCr=PeCr,
        NS=NS, lTort=lTort, iBact=iBact, lFiltr=lFiltr, lBact=lBact,
        iMoistDep=iMoistDep, lDualNEq=lDualNEq, lMassIni=lMassIni,
        lEqInit=lEqInit, iTort=iTort, lEquil=lEquil and sel.get("lEquil", True),
        ChPar=ChPar, TDep=TDep, WDep=WDep, CumCh=CumCh,
        lLinear=lLinear, lMobIm=lMobIm,
        kTopCh=kTopCh, cTop=cTop, kBotCh=kBotCh, cBot=cBot,
        dSurf=dSurf, cAtm=cAtm, tPulse=tPulse,
    )


# ============================================================================
# SinkIn — root water uptake (INPUT.FOR:1061-1126)
# ============================================================================

def _read_sink_in(r: FortranReader, sel: Dict[str, Any]) -> None:
    NMat = sel["NMat"]
    ver = sel["iVer"]
    NS = sel.get("NS", 0)
    lChem = sel.get("lChem", False)

    r.skip()                          # banner
    r.skip()                          # "iMoSink cRootMax(1..NS) [OmegaC]" header
    rec = r.read_record()
    iMoSink = _to_int(rec[0])
    cRootMax = np.array([_to_float(rec[1 + i]) for i in range(NS)], dtype=np.float64) \
        if NS > 0 else np.zeros(max(NS, 1), dtype=np.float64)
    if ver <= 2:
        OmegaC = 1.0
    else:
        OmegaC = _to_float(rec[1 + NS]) if len(rec) > 1 + NS else 1.0

    lMoSink = (iMoSink == 0)

    P0 = P2H = P2L = P3 = r2H = r2L = 0.0
    POptm = np.zeros(NMat, dtype=np.float64)
    if lMoSink:
        r.skip()
        P0, P2H, P2L, P3, r2H, r2L = _read_record_typed(
            r, [float, float, float, float, float, float]
        )
        r.skip()
        POptm = np.array(r.read_float(NMat), dtype=np.float64)
        P0 = -abs(P0); P2L = -abs(P2L); P2H = -abs(P2H); P3 = -abs(P3)
    else:
        r.skip()
        P0, P3 = _read_record_typed(r, [float, float])

    lSolRed = False
    lSolAdd = False
    lMsSink = False
    aOsm = np.zeros(max(NS, 1), dtype=np.float64)
    c50 = 0.0; P3c = 0.0
    OmegaS = 1.0; SPot = 0.0; rKM = 0.0; cMin = 0.0; lOmegaW = False
    if lChem:
        r.skip()
        lSolRed = r.read_bool()
        if lSolRed:
            r.skip()
            lSolAdd = r.read_bool()
            r.skip()
            if lSolAdd:
                aOsm = np.array(r.read_float(NS), dtype=np.float64)
            else:
                rec = r.read_record()
                c50 = _to_float(rec[0]); P3c = _to_float(rec[1])
                aOsm = np.array([_to_float(rec[2 + i]) for i in range(NS)], dtype=np.float64)
                iMsSink = _to_int(rec[2 + NS]) if len(rec) > 2 + NS else 0
                lMsSink = (iMsSink != 0)
        lActRSU = sel.get("lActRSU", False)
        if NS > 1:
            lActRSU = False
        if lActRSU and NS == 1:
            r.skip()
            rec = r.read_record()
            OmegaS = _to_float(rec[0]); SPot = _to_float(rec[1])
            rKM = _to_float(rec[2]); cMin = _to_float(rec[3])
            lOmegaW = _to_bool(rec[4])
        sel["lActRSU"] = lActRSU

    sel.update(
        iMoSink=iMoSink, lMoSink=lMoSink, OmegaC=OmegaC,
        P0=P0, P2H=P2H, P2L=P2L, P3=P3, r2H=r2H, r2L=r2L,
        POptm=POptm, cRootMax=cRootMax,
        lSolRed=lSolRed, lSolAdd=lSolAdd, lMsSink=lMsSink, aOsm=aOsm,
        c50=c50, P3c=P3c, OmegaS=OmegaS, SPot=SPot, rKM=rKM, cMin=cMin,
        lOmegaW=lOmegaW,
    )


# ============================================================================
# Top-level read_selector
# ============================================================================

def read_selector(path: str) -> Dict[str, Any]:
    """Parse Selector.in and return a flat dict of every field."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Selector.in not found: {path}")
    with FortranReader(path) as r:
        ver = r.detect_version()
        sel = _read_bas_inf(r, ver)
        # MatIn always follows BasInf in the file
        _read_mat_in(r, sel)
        _read_tm_in(r, sel)
        # The order in Selector.in for v4 is: BLOCK D (RootIn) when lRoot,
        # then BLOCK E (TempIn) when lTemp, then BLOCK F (ChemIn) when lChem,
        # then BLOCK G (SinkIn) when SinkF. Old versions used the same order.
        if sel.get("lRoot", False):
            _read_root_in(r, sel)
        if sel.get("lTemp", False):
            _read_temp_in(r, sel)
        if sel.get("lChem", False):
            _read_chem_in(r, sel)
        if sel.get("SinkF", False):
            _read_sink_in(r, sel)
    _populate_aliases(sel)
    # Build K / theta / Cap lookup tables (Fortran GenMat).  This MUST run
    # after MatIn so that ParD / hTab1 / hTabN / lTable are populated.
    gen_mat_tables(sel)
    return sel


def _populate_aliases(sel: Dict[str, Any]) -> None:
    """Add keys that hydrus._populate_state looks up via .get()."""
    # hydrus.py expects 'N', 'NSD', 'lSolute', 'lFreeDrain', etc.
    sel.setdefault("NSD", max(sel.get("NS", 0), 1))
    sel.setdefault("lSolute", sel.get("lChem", False))
    sel.setdefault("lHyst", sel.get("iHyst", 0) > 0)
    sel.setdefault("lSink", sel.get("SinkF", False))
    sel.setdefault("lFreeDrain", sel.get("FreeD", False))
    sel.setdefault("lGeom", False)
    sel.setdefault("lAdapt", True)
    sel.setdefault("lCumFlux", True)
    sel.setdefault("lMassBal", True)
    sel.setdefault("dtInit", sel.get("dt", 0.001))
    sel.setdefault("dtFact", sel.get("dMul", 1.3))
    sel.setdefault("IterMax", sel.get("MaxIt", 20))
    sel.setdefault("dtMaxC", 1.0e30)
    sel.setdefault("dtMaxT", 1.0e30)
    sel.setdefault("dtFactC", 1.0)
    sel.setdefault("dtFactT", 1.0)
    sel.setdefault("IterMaxC", sel.get("MaxItC", 10))
    sel.setdefault("IterMaxT", 10)
    sel.setdefault("tOut", 1.0)
    sel.setdefault("tOutC", 1.0)
    sel.setdefault("tOutT", 1.0)
    sel.setdefault("t0", sel.get("tInit", 0.0))
    sel.setdefault("tEnd", sel.get("tMax", 1.0))
    sel.setdefault("mConv", 1.0)
    sel.setdefault("rMin", 1.0e-37)
    sel.setdefault("rMax", 0.01)
    sel.setdefault("rMaxC", 0.01)
    sel.setdefault("rMaxT", 0.01)
    sel.setdefault("hTop", 0.0)
    sel.setdefault("hBot", 0.0)
    sel.setdefault("TTop", 20.0)
    sel.setdefault("TBot", 20.0)
    sel.setdefault("KodTopT", 1)
    sel.setdefault("KodBotT", 1)
    sel.setdefault("iOut", 0)
    sel.setdefault("iOutC", 0)
    sel.setdefault("iOutT", 0)
    sel.setdefault("dSurfT", 0.0)


# ============================================================================
# Profile.dat parsing (NodInf + InitW + InitDualPor + Profil)
# ============================================================================

def read_profile(path: str, sel: Dict[str, Any]) -> Dict[str, Any]:
    """
    Parse Profile.dat. Direct port of NodInf + InitW + InitDualPor.

    The Fortran file uses 1-based node indices counted from the surface
    downward (n=1 is top). HYDRUS internally re-numbers so that node 1 is the
    *bottom* of the profile (Fortran NodInf line 253: `n = NumNP-n+1`). We
    keep that convention here.

    Returns dict with keys: x, hOld, hNew, hTemp, thOld, thNew, MatNum,
    LayNum, Beta, Ah, AK, ATh, Conc, Sorb, TempO, TempN, NObs, Node,
    xSurf, hBot, hTop.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Profile.dat not found: {path}")

    NS = sel.get("NS", 0)
    lChem = sel.get("lChem", False)
    lTemp = sel.get("lTemp", False)
    lEquil = sel.get("lEquil", True)
    lBact = sel.get("lBact", False)
    lDualNEq = sel.get("lDualNEq", False)

    with FortranReader(path) as r:
        _ver = r.detect_version()       # consumed if present; unused after
        n_fixed_lines = r.read_int()    # number of fixed nodes (e.g. 2)
        for _ in range(n_fixed_lines):
            r.read_record()             # discard the fixed-node block
        rec = r.read_record()
        NumNP = _to_int(rec[0])
        # rec[1] is reserved; rec[2] is NS-from-file
        NS_file = _to_int(rec[2]) if len(rec) > 2 else NS

        x = np.zeros(NumNP, dtype=np.float64)
        hOld = np.zeros(NumNP, dtype=np.float64)
        MatNum = np.ones(NumNP, dtype=np.int64)
        LayNum = np.ones(NumNP, dtype=np.int64)
        Beta = np.zeros(NumNP, dtype=np.float64)
        Ah = np.ones(NumNP, dtype=np.float64)
        AK = np.ones(NumNP, dtype=np.float64)
        ATh = np.ones(NumNP, dtype=np.float64)
        TempO = np.full(NumNP, 20.0, dtype=np.float64)
        Conc = np.zeros((max(NS, 1), NumNP), dtype=np.float64)
        Sorb = np.zeros((max(NS, 1), NumNP), dtype=np.float64)
        Sorb2 = np.zeros((max(NS, 1), NumNP), dtype=np.float64)

        # NodInf reads nodes from the TOP downward but stores them
        # bottom-up: n_stored = NumNP - n_read + 1 (1-based).
        # Sparse-node format: only some node indices are given; in between,
        # values are linearly interpolated from neighbouring given nodes.
        nOld_stored = -1
        j_stored = NumNP    # 1-based countdown of remaining slots
        while j_stored > 0:
            rec = r.read_record()
            tokens = [_to_float(t) for t in rec]
            # Layout: n, x, h, M, L, B, Ax, Bx, Dx [, Te [, Conc(1..NS) [, Sorb(1..NS)]]]
            n_read = int(round(tokens[0]))
            x_v = tokens[1]; h_v = tokens[2]
            M_v = int(round(tokens[3])); L_v = int(round(tokens[4]))
            B_v = tokens[5]; Ax_v = tokens[6]; Bx_v = tokens[7]; Dx_v = tokens[8]
            base = 9
            Te_v = 20.0
            if lChem or lTemp:
                Te_v = tokens[base]; base += 1
            C_vals = [0.0] * max(NS, 1)
            S_vals = [0.0] * max(NS, 1)
            if lChem and NS > 0:
                for k in range(NS):
                    C_vals[k] = tokens[base + k]
                base += NS
                if not lEquil:
                    for k in range(NS):
                        S_vals[k] = tokens[base + k]
                    base += NS

            # store at index (NumNP-n_read) (0-based)
            n_stored = NumNP - n_read + 1   # 1-based, top of stored array
            ii = n_stored - 1               # 0-based
            x[ii] = x_v
            hOld[ii] = h_v
            MatNum[ii] = M_v
            LayNum[ii] = L_v
            Beta[ii] = B_v
            Ah[ii] = Ax_v
            AK[ii] = Bx_v
            ATh[ii] = Dx_v
            TempO[ii] = Te_v
            for k in range(min(NS, len(C_vals))):
                Conc[k, ii] = C_vals[k]
                if not lEquil:
                    Sorb[k, ii] = S_vals[k]

            if nOld_stored > 0 and (nOld_stored - n_stored) > 1:
                # Interpolate sparse gap between previous record and this one.
                old = nOld_stored - 1
                new = ii
                dx = x[old] - x[new]
                if dx == 0.0:
                    dx = 1.0
                ShOld = (hOld[old] - hOld[new]) / dx
                SBeta = (Beta[old] - Beta[new]) / dx
                SAh = (Ah[old] - Ah[new]) / dx
                SAK = (AK[old] - AK[new]) / dx
                SATh = (ATh[old] - ATh[new]) / dx
                STemp = (TempO[old] - TempO[new]) / dx
                for i in range(old - 1, new, -1):
                    ddx = x[old] - x[i] if x[i] != 0.0 else (x[old] - (x[old] - (old - i)))
                    # Use linear interpolation in distance just like Fortran
                    # but compute x[i] first via linear distance:
                    x[i] = x[old] - (x[old] - x[new]) * (old - i) / (old - new)
                    ddx = x[old] - x[i]
                    hOld[i] = hOld[old] - ShOld * ddx
                    Beta[i] = Beta[old] - SBeta * ddx
                    Ah[i] = Ah[old] - SAh * ddx
                    AK[i] = AK[old] - SAK * ddx
                    ATh[i] = ATh[old] - SATh * ddx
                    TempO[i] = TempO[old] - STemp * ddx
                    if lChem and NS > 0:
                        SC = (Conc[:, old] - Conc[:, new]) / (x[old] - x[new])
                        Conc[:, i] = Conc[:, old] - SC * ddx
                        if not lEquil:
                            SS = (Sorb[:, old] - Sorb[:, new]) / (x[old] - x[new])
                            Sorb[:, i] = Sorb[:, old] - SS * ddx
                    MatNum[i] = MatNum[i + 1]
                    LayNum[i] = LayNum[i + 1]

            nOld_stored = n_stored
            j_stored = n_stored - 1

        # Normalize Beta so that ∫Beta dz over the profile = 1
        SBeta = 0.0
        if Beta[-1] > 0.0:
            SBeta = Beta[-1] * (x[-1] - x[-2]) / 2.0
        for i in range(1, NumNP - 1):
            if Beta[i] > 0.0:
                SBeta += Beta[i] * (x[i + 1] - x[i - 1]) / 2.0
        if SBeta > 0.0:
            for i in range(1, NumNP):
                Beta[i] = Beta[i] / SBeta if Beta[i] > 0.0 else 0.0
        else:
            Beta[1:] = 0.0

        # NObs block
        NObs = 0
        Node = np.zeros(0, dtype=np.int64)
        try:
            NObs = r.read_int()
            if NObs > 0:
                node_raw = r.read_int(NObs) if NObs > 1 else [r.read_int()]
                Node = np.array(
                    [NumNP - int(v) + 1 for v in node_raw], dtype=np.int64
                )
        except EOFError:
            NObs = 0

    hNew = hOld.copy()
    hTemp = hOld.copy()
    TempN = TempO.copy()

    # InitW: if lInitW, treat hOld as initial saturation, invert to head.
    # Implemented when lInitW=True in caller after material params are present.

    xSurf = x[-1]
    hBot = hNew[0]
    hTop = hNew[-1]

    prof = {
        "NumNP": NumNP, "x": x, "hOld": hOld, "hNew": hNew, "hTemp": hTemp,
        "MatNum": MatNum, "LayNum": LayNum, "Beta": Beta,
        "Ah": Ah, "AK": AK, "ATh": ATh,
        "TempO": TempO, "TempN": TempN,
        "Conc": Conc, "Sorb": Sorb, "Sorb2": Sorb2,
        "NObs": NObs, "Node": Node,
        "xSurf": xSurf, "hBot": hBot, "hTop": hTop,
    }

    # Convert hNew/hOld → theta via the soil hydraulic model.
    _compute_theta_from_h(prof, sel)

    # Apply InitW inversion if requested.
    if sel.get("lInitW", False):
        _apply_init_w(prof, sel)

    # InitDualPor (no-op when iDualPor==0)
    _apply_init_dualpor(prof, sel)

    # Add 'N' alias for hydrus.py
    prof["N"] = NumNP
    return prof


def _compute_theta_from_h(prof: Dict[str, Any], sel: Dict[str, Any]) -> None:
    """Populate thNew/thOld from hNew via the same table interpolation that
    Fortran's SetMat uses so the initial water-volume matches the Fortran
    binary at the 1e-4 level rather than via analytical FQ."""
    NumNP = prof["NumNP"]
    hNew = prof["hNew"]
    th = np.zeros(NumNP, dtype=np.float64)
    for i in range(NumNP):
        M = int(prof["MatNum"][i]) - 1
        theta_i, _ = interp_theta_cap(float(hNew[i]), M, sel)
        th[i] = theta_i
    prof["thNew"] = th
    prof["thOld"] = th.copy()


def _apply_init_w(prof: Dict[str, Any], sel: Dict[str, Any]) -> None:
    """Port of InitW: when lInitW=true, hOld actually holds initial theta —
    invert to pressure head via FH()."""
    from .material import FH
    NumNP = prof["NumNP"]
    iModel = sel.get("iModel", 0)
    ParD = sel["ParD"]
    hNew = prof["hNew"]
    for i in range(NumNP):
        M = int(prof["MatNum"][i]) - 1
        ThTotal = hNew[i]
        Qe = min((ThTotal - ParD[0, M]) / (ParD[1, M] - ParD[0, M]), 1.0)
        if Qe < 0.0:
            raise ValueError(f"InitW: Qe < 0 at node {i}: theta_init={ThTotal} below thr")
        hNew[i] = FH(iModel, Qe, ParD[:, M])
    prof["hNew"] = hNew
    prof["hOld"] = hNew.copy()
    prof["hTemp"] = hNew.copy()
    prof["hBot"] = hNew[0]
    prof["hTop"] = hNew[-1]


def _apply_init_dualpor(prof: Dict[str, Any], sel: Dict[str, Any]) -> None:
    iDualPor = sel.get("iDualPor", 0)
    NumNP = prof["NumNP"]
    ThNewIm = np.zeros(NumNP, dtype=np.float64)
    if iDualPor != 0:
        from .material import FQ
        ParD = sel["ParD"]
        iModel = sel.get("iModel", 0)
        for i in range(NumNP):
            M = int(prof["MatNum"][i]) - 1
            if iDualPor == 1:
                Se = (prof["thNew"][i] - ParD[0, M]) / (ParD[1, M] - ParD[0, M])
                ThNewIm[i] = ParD[6, M] + Se * (ParD[7, M] - ParD[6, M])
            else:
                ThNewIm[i] = FQ(0, prof["hNew"][i], ParD[6:, M])
    prof["ThNewIm"] = ThNewIm
    prof["ThOldIm"] = ThNewIm.copy()


# ============================================================================
# ATMOSPH.IN parsing
# ============================================================================

def read_atmospheric(path: str, sel: Dict[str, Any]) -> Dict[str, Any]:
    """Parse ATMOSPH.IN. Header block (MaxAL, hCritS, lDayVar...) + time
    series of MaxAL records. Returns a dict with arrays of length MaxAL."""
    if not os.path.exists(path):
        # ATMOSPH.IN is optional unless TopInF / BotInF / AtmBC set.
        return {"MaxAL": 0}

    ver = sel.get("iVer", 0)
    NS = sel.get("NS", 0)

    with FortranReader(path) as r:
        _verA = r.detect_version()
        r.skip()                      # banner
        r.skip()                      # "MaxAL" header
        MaxAL = r.read_int()
        lDayVar = False
        lSinPrec = False
        lLAI = False
        rExtinct = 0.39
        if _verA == 4:
            r.skip()
            rec = r.read_record()
            lDayVar = _to_bool(rec[0])
            lSinPrec = _to_bool(rec[1])
            lLAI = _to_bool(rec[2])
            if lLAI:
                r.skip()
                rExtinct = r.read_float()
        r.skip()
        hCritS = r.read_float()
        r.skip()                      # column header for time series

        cols_per_row = 11             # tAtm, Prec, rSoil, rRoot, hCritA, rB, hB, hT, tTop, tBot, Ampl
        # Plus optional cTop(NS), cBot(NS), cTop_T (1 column for SnowMF/etc in some versions)
        # We'll just read tokens flexibly.
        tAtm = np.zeros(MaxAL, dtype=np.float64)
        Prec = np.zeros(MaxAL, dtype=np.float64)
        rSoil = np.zeros(MaxAL, dtype=np.float64)
        rRoot = np.zeros(MaxAL, dtype=np.float64)
        hCritA_arr = np.zeros(MaxAL, dtype=np.float64)
        rB = np.zeros(MaxAL, dtype=np.float64)
        hB = np.zeros(MaxAL, dtype=np.float64)
        hT = np.zeros(MaxAL, dtype=np.float64)
        tTop_arr = np.zeros(MaxAL, dtype=np.float64)
        tBot_arr = np.zeros(MaxAL, dtype=np.float64)
        Ampl_arr = np.zeros(MaxAL, dtype=np.float64)
        cTopA = np.zeros((max(NS, 1), MaxAL), dtype=np.float64)
        cBotA = np.zeros((max(NS, 1), MaxAL), dtype=np.float64)

        for i in range(MaxAL):
            rec = r.read_record()
            if not rec or rec[0].lower().startswith("end"):
                MaxAL = i
                break
            vals = [_to_float(t) for t in rec]
            tAtm[i] = vals[0]
            Prec[i] = vals[1] if len(vals) > 1 else 0.0
            rSoil[i] = vals[2] if len(vals) > 2 else 0.0
            rRoot[i] = vals[3] if len(vals) > 3 else 0.0
            hCritA_arr[i] = vals[4] if len(vals) > 4 else 0.0
            rB[i] = vals[5] if len(vals) > 5 else 0.0
            hB[i] = vals[6] if len(vals) > 6 else 0.0
            hT[i] = vals[7] if len(vals) > 7 else 0.0
            tTop_arr[i] = vals[8] if len(vals) > 8 else 0.0
            tBot_arr[i] = vals[9] if len(vals) > 9 else 0.0
            Ampl_arr[i] = vals[10] if len(vals) > 10 else 0.0
            for k in range(NS):
                idx = 11 + k
                if len(vals) > idx:
                    cTopA[k, i] = vals[idx]
                idx = 11 + NS + k
                if len(vals) > idx:
                    cBotA[k, i] = vals[idx]

    return {
        "MaxAL": MaxAL,
        "hCritS": hCritS,
        "lDayVar": lDayVar,
        "lSinPrec": lSinPrec,
        "lLAI": lLAI,
        "rExtinct": rExtinct,
        "tAtm": tAtm[:MaxAL],
        "Prec": Prec[:MaxAL],
        "rSoil": rSoil[:MaxAL],
        "rRoot": rRoot[:MaxAL],
        "hCritA": hCritA_arr[:MaxAL],
        "rB": rB[:MaxAL],
        "hB": hB[:MaxAL],
        "hT": hT[:MaxAL],
        "tTop": tTop_arr[:MaxAL],
        "tBot": tBot_arr[:MaxAL],
        "Ampl": Ampl_arr[:MaxAL],
        "cTopA": cTopA[:, :MaxAL],
        "cBotA": cBotA[:, :MaxAL],
    }


# ============================================================================
# Meteo.in parsing
# ============================================================================

def read_meteorological(path: str) -> Dict[str, Any]:
    """Parse Meteo.in. Direct port of MeteoIn. Optional file; returns empty
    dict when missing."""
    if not os.path.exists(path):
        return {"MaxALMet": 0}

    with FortranReader(path) as r:
        _verM = r.detect_version()
        r.skip()
        r.skip()
        MaxALMet, iRadiation, lHargr = _read_record_typed(r, [int, int, bool])
        lEnBal = False
        lMetDaily = False
        if _verM >= 4:
            r.skip()
            lEnBal, lMetDaily = _read_record_typed(r, [bool, bool])

        Latitude = 0.0; Altitude = 0.0
        ShortWaveRadA = 0.0; ShortWaveRadB = 0.0
        LongWaveRadA = 0.0; LongWaveRadB = 0.0
        LongWaveRadA1 = 0.0; LongWaveRadB1 = 0.0
        if iRadiation != 2:
            r.skip(); Latitude, Altitude = _read_record_typed(r, [float, float])
            r.skip(); ShortWaveRadA, ShortWaveRadB = _read_record_typed(r, [float, float])
            r.skip(); LongWaveRadA, LongWaveRadB = _read_record_typed(r, [float, float])
            r.skip(); LongWaveRadA1, LongWaveRadB1 = _read_record_typed(r, [float, float])
        r.skip()
        WindHeight, TempHeight = _read_record_typed(r, [float, float])
        r.skip()
        iCrop, iSunSh, iRelHum = _read_record_typed(r, [int, int, int])
        CloudF_Ac = 0.0; CloudF_Bc = 0.0
        if iRadiation == 1 and iSunSh == 3:
            r.skip()
            CloudF_Ac, CloudF_Bc = _read_record_typed(r, [float, float])

        iLAI = 0; rExtinct = 0.39; iInterc = 0; aInterc = 0.0
        CropHeight = 0.0; Albedo = 0.0; LAI = 0.0; xRoot = 0.0
        iGrowth = 0
        rGrowth: Optional[NDArray[np.float64]] = None
        if iCrop >= 1:
            r.skip(); iLAI, rExtinct = _read_record_typed(r, [int, float])
            r.skip(); iInterc = r.read_int()
            if iCrop == 1:
                r.skip()
                CropHeight, Albedo, LAI, xRoot = _read_record_typed(
                    r, [float, float, float, float]
                )
            elif iCrop == 2:
                r.skip(); iGrowth = r.read_int()
                if iGrowth > 1000:
                    raise ValueError("Meteo iGrowth > 1000")
                r.skip()
                rGrowth = np.zeros((iGrowth, 5), dtype=np.float64)
                for i in range(iGrowth):
                    rGrowth[i, :] = r.read_float(5)
            if iInterc == 1:
                r.skip(); aInterc = r.read_float()
        else:
            r.skip(); Albedo = r.read_float()

        # Three skip lines before the daily table (per MeteoIn:1012-1014).
        r.skip(); r.skip(); r.skip()

        # Daily data table. We read until "end" sentinel or EOF.
        rows: List[List[float]] = []
        while True:
            try:
                rec = r.read_record()
            except EOFError:
                break
            if not rec or rec[0].lower().startswith("end"):
                break
            rows.append([_to_float(t) for t in rec])

    return {
        "MaxALMet": MaxALMet, "iRadiation": iRadiation, "lHargr": lHargr,
        "lEnBal": lEnBal, "lMetDaily": lMetDaily,
        "Latitude": Latitude, "Altitude": Altitude,
        "ShortWaveRadA": ShortWaveRadA, "ShortWaveRadB": ShortWaveRadB,
        "LongWaveRadA": LongWaveRadA, "LongWaveRadB": LongWaveRadB,
        "LongWaveRadA1": LongWaveRadA1, "LongWaveRadB1": LongWaveRadB1,
        "WindHeight": WindHeight, "TempHeight": TempHeight,
        "iCrop": iCrop, "iSunSh": iSunSh, "iRelHum": iRelHum,
        "CloudF_Ac": CloudF_Ac, "CloudF_Bc": CloudF_Bc,
        "iLAI": iLAI, "rExtinct": rExtinct, "iInterc": iInterc,
        "aInterc": aInterc, "CropHeight": CropHeight, "Albedo": Albedo,
        "LAI": LAI, "xRoot": xRoot,
        "iGrowth": iGrowth, "rGrowth": rGrowth,
        "MeteoData": np.array(rows, dtype=np.float64) if rows else np.zeros((0, 0)),
    }


# ============================================================================
# Hysteresis.in (optional, called only when iHyst > 0 and a hysteresis
# restart file is present). Pure-Python equivalent of HysterIn.
# ============================================================================

def read_hysteresis(path: str, NumNP: int) -> Optional[Dict[str, Any]]:
    if not os.path.exists(path):
        return None
    with FortranReader(path) as r:
        r.skip(); r.skip()
        ThOld = np.zeros(NumNP, dtype=np.float64)
        KappaO = np.zeros(NumNP, dtype=np.int64)
        for i in range(NumNP):
            rec = r.read_record()
            ThOld[i] = _to_float(rec[1])
            KappaO[i] = _to_int(rec[2])
    return {"ThOld": ThOld, "KappaO": KappaO}
