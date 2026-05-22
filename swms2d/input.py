"""
Input file parsers for SWMS_2D Python port.
===========================================

Parses the two mandatory SWMS_2D input files:
    SWMS_2D.IN/SELECTOR.IN   global config, materials, time control, BC
    SWMS_2D.IN/GRID.IN        FE mesh (nodes + elements + boundary)

Optional:
    SWMS_2D.IN/Atmosph.in     time-variable atmospheric BC (when AtmInF=True)

Direct port of INPUT2.FOR. The Fortran reader is intermixed with the main
program; here we split it into discrete parse_* functions.

Conventions:
    - Fortran 1-based node/element ids → Python 0-based at parse time
    - Fortran `t`/`f` logical → Python True/False
    - Anisotropy (Angle, Aniz1, Aniz2) rotation applied at parse time
      via mesh.rotate_anisotropy()
"""

from __future__ import annotations
from pathlib import Path
import re
from dataclasses import replace
import numpy as np

from .dataclasses import (
    SimulationConfig, SoilMaterial, TimeControl, BoundaryConditions,
    Node, Element, Mesh,
)
from .mesh import rotate_anisotropy, build_listne


# ============================================================================
# Low-level token helpers
# ============================================================================

def _is_header(line: str) -> bool:
    """SWMS_2D headers and column labels start with '*' or contain only
    non-numeric text. We skip them by detecting absence of digits or a
    leading '*'."""
    s = line.strip()
    if not s or s.startswith("*"):
        return True
    # Lines that look like column-label rows (no signed number tokens)
    # are skipped by callers using explicit line offsets, not here.
    return False


def _tokens(line: str) -> list[str]:
    return line.split()


def _parse_bool(tok: str) -> bool:
    return tok.lower().startswith("t")


# ============================================================================
# SELECTOR.IN
# ============================================================================

def parse_selector(path: Path) -> tuple[
    SimulationConfig, list[SoilMaterial], TimeControl, dict
]:
    """
    Read SELECTOR.IN. Returns:
        config       SimulationConfig
        materials    list[SoilMaterial]
        time         TimeControl (with tInit, tMax derived from MPL+TPrint)
        extras       dict with: heading, units, TPrint (np.ndarray),
                                seepage info if SeepF
    """
    lines = path.read_text().splitlines()

    # The Fortran reader does NOT consume header/comment lines as data —
    # they're present in the file as visual anchors. We strip them out and
    # parse the remaining "data lines" in fixed order.
    def _is_data_line(s: str) -> bool:
        s = s.lstrip()
        if not s or s.startswith("*"):
            return False
        first = s.split()[0]
        if first[0] in ("'", '"'):       # quoted heading or units row
            return True
        try:
            float(first)                  # numeric row
            return True
        except ValueError:
            pass
        if first.lower() in ("t", "f", ".true.", ".false."):  # logical row
            return True
        return False                       # label / column-name row

    data_lines = [ln for ln in lines if _is_data_line(ln)]

    it = iter(data_lines)
    cfg = SimulationConfig()
    extras: dict = {}

    # 1: heading
    extras["heading"] = next(it).strip("'\" ")

    # 2: units
    units_line = next(it)
    units = [u.strip("'\" ") for u in re.findall(r"'[^']*'", units_line)]
    extras["units"] = units  # [LUnit, TUnit, MUnit, BUnit]

    # 3: Kat
    cfg.KAT = int(next(it))

    # 4: MaxIt, TolTh, TolH
    toks = _tokens(next(it))
    cfg.MaxIt = int(toks[0])
    cfg.TolTh = float(toks[1])
    cfg.TolH = float(toks[2])

    # 5: 9 logical flags: lWat lChem CheckF ShortF FluxF AtmInF SeepF FreeD DrainF
    toks = _tokens(next(it))
    cfg.lWat   = _parse_bool(toks[0])
    cfg.lChem  = _parse_bool(toks[1])
    cfg.CheckF = _parse_bool(toks[2])
    cfg.ShortF = _parse_bool(toks[3])
    cfg.FluxF  = _parse_bool(toks[4])
    cfg.AtmInF = _parse_bool(toks[5])
    cfg.SeepF  = _parse_bool(toks[6])
    cfg.FreeD  = _parse_bool(toks[7])
    cfg.DrainF = _parse_bool(toks[8])

    # 6: BLOCK B counts: NMat NLay hTab1 hTabN NPar
    toks = _tokens(next(it))
    NMat = int(toks[0])
    extras["NLay"] = int(toks[1])
    extras["hTab1"] = float(toks[2])
    extras["hTabN"] = float(toks[3])
    extras["NPar"] = int(toks[4])

    # 7: NMat material rows: thr ths tha thm Alfa n Ks Kk thk
    materials: list[SoilMaterial] = []
    for _ in range(NMat):
        toks = _tokens(next(it))
        materials.append(SoilMaterial(
            thr=float(toks[0]), ths=float(toks[1]),
            tha=float(toks[2]), thm=float(toks[3]),
            alpha=float(toks[4]), n=float(toks[5]),
            Ks=float(toks[6]), Kk=float(toks[7]), thk=float(toks[8]),
        ))

    # 8: BLOCK C time: dt dtMin dtMax DMul DMul2 MPL
    toks = _tokens(next(it))
    time = TimeControl(
        dt=float(toks[0]), dtMin=float(toks[1]), dtMaxW=float(toks[2]),
        dMul=float(toks[3]), dMul2=float(toks[4]),
    )
    MPL = int(toks[5])

    # 9: TPrint(1..MPL) — may span multiple data lines
    tprint: list[float] = []
    while len(tprint) < MPL:
        toks = _tokens(next(it))
        tprint.extend(float(t) for t in toks)
    tprint = tprint[:MPL]
    extras["TPrint"] = np.array(tprint, np.float64)
    if MPL > 0:
        time.tMax = tprint[-1]

    # Remaining BLOCKs (D-sink, E-seep, F-solute) come in physical file order.
    # We scan for known marker lines instead of relying on flag combinations,
    # since SinkF is decided by ATMOSPH.IN later in the Fortran flow.
    remaining = [ln for ln in lines if _is_data_line(ln) or "BLOCK" in ln]
    # Skip lines we've already consumed (matched in `data_lines`)
    consumed = sum(1 for ln in lines if _is_data_line(ln)) - sum(1 for _ in it)
    # Simpler: search original `lines` for each BLOCK header.
    block_positions: dict[str, int] = {}
    for idx, ln in enumerate(lines):
        s = ln.lstrip()
        for tag in ("BLOCK D", "BLOCK E", "BLOCK F"):
            if s.startswith(f"*** {tag}"):
                block_positions[tag] = idx

    def _block_data_lines(tag: str, next_tag: str | None) -> list[str]:
        if tag not in block_positions:
            return []
        start = block_positions[tag] + 1
        if next_tag and next_tag in block_positions:
            end = block_positions[next_tag]
        else:
            end = len(lines)
        return [ln for ln in lines[start:end] if _is_data_line(ln)]

    # ---- BLOCK D: sink parameters ----
    d_data = _block_data_lines("BLOCK D", "BLOCK E")
    if d_data:
        toks = _tokens(d_data[0])
        extras["sink_P0"]  = -abs(float(toks[0]))
        extras["sink_P2H"] = -abs(float(toks[1]))
        extras["sink_P2L"] = -abs(float(toks[2]))
        extras["sink_P3"]  = -abs(float(toks[3]))
        extras["sink_r2H"] =  float(toks[4])
        extras["sink_r2L"] =  float(toks[5])
        # POptm — may span multiple lines until we have NMat values
        poptm: list[float] = []
        for ln in d_data[1:]:
            poptm.extend(float(t) for t in _tokens(ln))
            if len(poptm) >= NMat:
                break
        extras["sink_POptm"] = np.array(poptm[:NMat], np.float64)

    # ---- BLOCK E: seepage faces ----
    e_data = _block_data_lines("BLOCK E", "BLOCK F")
    if cfg.SeepF and e_data:
        extras["NSeep"] = int(_tokens(e_data[0])[0])
        nseep = extras["NSeep"]
        extras["NSP"] = [int(x) for x in _tokens(e_data[1])][:nseep]
        np_seep = []
        for k in range(nseep):
            np_seep.append([int(x) for x in _tokens(e_data[2 + k])])
        extras["NP"] = np_seep

    # ---- BLOCK F: solute transport ----
    f_data = _block_data_lines("BLOCK F", None)
    if cfg.lChem and f_data:
        # Line 1: epsi, lUpW, lArtD, PeCr
        toks = _tokens(f_data[0])
        extras["chem_epsi"]  = float(toks[0])
        extras["chem_lUpW"]  = _parse_bool(toks[1])
        extras["chem_lArtD"] = _parse_bool(toks[2])
        extras["chem_PeCr"]  = max(float(toks[3]), 0.01)
        # Next NMat lines: 9 ChPar values per material
        chpar = np.zeros((9, NMat), np.float64)
        for M in range(NMat):
            toks = _tokens(f_data[1 + M])
            for j in range(9):
                chpar[j, M] = float(toks[j])
        extras["chem_ChPar"] = chpar
        # Remaining solute lines: see parse_chem_extras helper if needed.
        extras["chem_remaining_lines"] = f_data[1 + NMat:]

    return cfg, materials, time, extras


# ============================================================================
# ATMOSPH.IN
# ============================================================================

def parse_atmosph(path: Path) -> dict:
    """Read ATMOSPH.IN. Returns dict with:
        SinkF, qGWLF, GWL0L, Aqh, Bqh, tInit, MaxAL, hCritS,
        records: structured array of (tAtm, Prec, cPrec, rSoil, rRoot,
                                       hCritA, rGWL, GWL, crt, cht)
        tMax
    """
    lines = path.read_text().splitlines()
    # Strip headers/comments
    def _is_data(s: str) -> bool:
        s = s.lstrip()
        if not s or s.startswith("*"):
            return False
        first = s.split()[0]
        try:
            float(first); return True
        except ValueError:
            pass
        if first.lower() in ("t", "f"):
            return True
        return False
    data = [ln for ln in lines if _is_data(ln)]
    it = iter(data)
    out: dict = {}
    toks = _tokens(next(it))
    out["SinkF"] = _parse_bool(toks[0])
    out["qGWLF"] = _parse_bool(toks[1])
    toks = _tokens(next(it))
    out["GWL0L"] = float(toks[0])
    out["Aqh"]   = float(toks[1])
    out["Bqh"]   = float(toks[2])
    toks = _tokens(next(it))
    out["tInit"] = float(toks[0])
    out["MaxAL"] = int(toks[1])
    out["hCritS"] = float(_tokens(next(it))[0])
    # MaxAL records
    records = []
    for _ in range(out["MaxAL"]):
        toks = _tokens(next(it))
        records.append([float(t) for t in toks[:10]])
    out["records"] = np.array(records, np.float64)
    out["tMax"] = out["records"][-1, 0]
    return out


# ============================================================================
# GRID.IN
# ============================================================================

def parse_grid(path: Path) -> Mesh:
    """
    Read GRID.IN: 3 blocks (Nodes / Elements / Boundary geometry).
    """
    lines = path.read_text().splitlines()
    # Find block boundaries
    block_starts: dict[str, int] = {}
    for i, ln in enumerate(lines):
        s = ln.lstrip()
        if s.startswith("*** BLOCK H"):
            block_starts["H"] = i
        elif s.startswith("*** BLOCK I"):
            block_starts["I"] = i
        elif s.startswith("*** BLOCK J"):
            block_starts["J"] = i
        elif s.startswith("*** END"):
            block_starts["END"] = i

    # ---- Block H: Nodal Information ----
    # Layout:
    #   BLOCK H header
    #   "      NumNP     NumEl       IJ      NumBP     NObs"
    #   counts
    #   "   n  Code    x    z   h ..."  (column labels)
    #   NumNP node rows
    h_lines = lines[block_starts["H"] + 1 : block_starts["I"]]
    # Skip the column-name line(s); find the first line whose first token is an int
    body = []
    for ln in h_lines:
        toks = ln.split()
        if not toks: continue
        try:
            int(toks[0])
            body.append(ln)
        except ValueError:
            continue
    # body[0] = counts, body[1:] = node rows
    counts = body[0].split()
    NumNP = int(counts[0])
    NumEl = int(counts[1])
    IJ    = int(counts[2])
    NumBP = int(counts[3])
    NObs  = int(counts[4])

    node = Node(
        Kode  = np.zeros(NumNP, np.int32),
        x     = np.zeros(NumNP, np.float64),
        y     = np.zeros(NumNP, np.float64),
        hNew  = np.zeros(NumNP, np.float64),
        hOld  = np.zeros(NumNP, np.float64),
        hTemp = np.zeros(NumNP, np.float64),
        Q     = np.zeros(NumNP, np.float64),
        Conc  = np.zeros(NumNP, np.float64),
        MatNum= np.zeros(NumNP, np.int32),
        Beta  = np.zeros(NumNP, np.float64),
        Axz   = np.zeros(NumNP, np.float64),
        Bxz   = np.zeros(NumNP, np.float64),
        Dxz   = np.zeros(NumNP, np.float64),
    )

    # SWMS_2D supports skipping nodes and interpolating — we implement
    # the explicit case (every node listed) here, which covers EXAMPLE.1-4.
    # See INPUT2.FOR L295-320 for interpolation logic if a non-contiguous
    # example is encountered.
    for row in body[1 : 1 + NumNP]:
        toks = row.split()
        # n Code x z h Conc Q M B Axz Bxz Dxz
        n = int(toks[0]) - 1   # 1-based → 0-based
        node.Kode[n]  = int(toks[1])
        node.x[n]     = float(toks[2])
        node.y[n]     = float(toks[3])
        node.hOld[n]  = float(toks[4])
        node.hNew[n]  = float(toks[4])
        node.hTemp[n] = float(toks[4])
        node.Conc[n]  = float(toks[5])
        node.Q[n]     = float(toks[6])
        node.MatNum[n]= int(toks[7])
        node.Beta[n]  = float(toks[8])
        node.Axz[n]   = float(toks[9])
        node.Bxz[n]   = float(toks[10])
        node.Dxz[n]   = float(toks[11])

    # ---- Block I: Element Information ----
    i_lines = lines[block_starts["I"] + 1 : block_starts["J"]]
    body = []
    for ln in i_lines:
        toks = ln.split()
        if not toks: continue
        try:
            int(toks[0])
            body.append(ln)
        except ValueError:
            continue

    elem = Element(
        KX     = np.zeros((NumEl, 4), np.int32),
        ConAxx = np.zeros(NumEl, np.float64),
        ConAzz = np.zeros(NumEl, np.float64),
        ConAxz = np.zeros(NumEl, np.float64),
        LayNum = np.zeros(NumEl, np.int32),
    )

    for row in body[:NumEl]:
        toks = row.split()
        # e i j k l Angle Aniz1 Aniz2 LayNum
        e = int(toks[0]) - 1
        for k in range(4):
            v = int(toks[1 + k])
            if v == 0 and k == 3:
                v = int(toks[1 + 2])  # degenerate quad → triangle
            elem.KX[e, k] = v - 1   # 1-based → 0-based
        angle_deg = float(toks[5])
        aniz1     = float(toks[6])
        aniz2     = float(toks[7])
        Axx, Azz, Axz = rotate_anisotropy(angle_deg, aniz1, aniz2)
        elem.ConAxx[e] = Axx
        elem.ConAzz[e] = Azz
        elem.ConAxz[e] = Axz
        elem.LayNum[e] = int(toks[8])

    # ---- Block J: Boundary Geometry ----
    j_lines = lines[block_starts["J"] + 1 : block_starts.get("END", len(lines))]
    # Find: "Node number array:" then numbers, "Width array:" then numbers, "Length:" then number
    KXB    = np.zeros(NumBP, np.int32)
    Width  = np.zeros(NumBP, np.float64)
    rLen   = 0.0
    section = None
    nums_buf: list[float] = []
    def flush_into():
        nonlocal nums_buf
        if section == "node":
            KXB[:NumBP] = [int(v) - 1 for v in nums_buf[:NumBP]]
        elif section == "width":
            Width[:NumBP] = nums_buf[:NumBP]
        nums_buf = []

    for ln in j_lines:
        s = ln.strip()
        if not s: continue
        low = s.lower()
        if "node number" in low and "obs" not in low and "(" not in low:
            flush_into(); section = "node"; continue
        if "width array" in low:
            flush_into(); section = "width"; continue
        if low.startswith("length"):
            flush_into(); section = "length"; continue
        if "node(" in low or "nobs" in low or "obs" in low:
            # Switch to a sink section so further numbers are ignored
            flush_into(); section = "obs"; continue
        toks = s.split()
        try:
            nums = [float(t) for t in toks]
        except ValueError:
            continue
        if section == "length":
            rLen = nums[0]
            section = "obs"   # consume only one value
        elif section == "obs":
            pass
        else:
            nums_buf.extend(nums)
    flush_into()

    mesh = Mesh(
        nodes=node, elements=elem,
        KXB=KXB, Width=Width, rLen=rLen,
        NumNP=NumNP, NumEl=NumEl, NumBP=NumBP, IJ=IJ, NObs=NObs,
    )
    build_listne(mesh)
    return mesh


# ============================================================================
# Convenience: read everything for one EXAMPLE directory
# ============================================================================

def parse_example(in_dir: Path
                  ) -> tuple[SimulationConfig, list[SoilMaterial],
                             TimeControl, Mesh, dict]:
    """Convenience: parse SELECTOR.IN + GRID.IN (+ ATMOSPH.IN if AtmInF) from
    a SWMS_2D.IN/ directory."""
    in_dir = Path(in_dir)
    sel_path = next(p for p in in_dir.iterdir() if p.name.lower() == "selector.in")
    grd_path = next(p for p in in_dir.iterdir() if p.name.lower() == "grid.in")
    cfg, mats, time, extras = parse_selector(sel_path)
    mesh = parse_grid(grd_path)
    if cfg.AtmInF:
        atm_path = next(
            (p for p in in_dir.iterdir() if p.name.lower() == "atmosph.in"),
            None,
        )
        if atm_path is None:
            raise FileNotFoundError(
                f"AtmInF=True but no ATMOSPH.IN found in {in_dir}"
            )
        atm = parse_atmosph(atm_path)
        extras["atm"] = atm
        cfg.SinkF = atm["SinkF"]
        cfg.qGWLF = atm["qGWLF"]
        # tInit & tMax from atmosphere override any earlier values
        time.tInit = atm["tInit"]
        time.tMax = max(time.tMax, atm["tMax"])
    return cfg, mats, time, mesh, extras
