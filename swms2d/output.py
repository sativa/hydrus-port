"""
Output file writers for SWMS_2D Python port.
============================================

Mirrors OUTPUT2.FOR formats exactly so numeric diff against the Fortran
golden master (compare_outputs.py) doesn't suffer from layout drift.

Stage 1 implements only the writers needed for EXAMPLE.1 (water-only):
    h.out   pressure head field at print times
    th.out  water content field at print times

Other writers (Q.out, vx.out, vz.out, Cum_Q.out, Balance.out, ...)
follow the same pattern and can be added as needed.
"""

from __future__ import annotations
from pathlib import Path
from typing import TextIO, Optional
import numpy as np
from numpy.typing import NDArray

from .dataclasses import Mesh


# ---------------------------------------------------------------------------
# Header (written once at file open)
# ---------------------------------------------------------------------------

def _fortran_e(x: float, width: int = 11, digits: int = 3) -> str:
    """Format like Fortran's e<width>.<digits>: leading-zero mantissa.

    Fortran's `e11.3` writes  `0.XXXE+NN` (3 fractional digits in mantissa,
    sign and E+NN take 4 chars, plus the leading space, total 11). Python's
    standard `.3e` is `X.XXXe+NN` (one digit before the decimal). This
    function converts to the leading-zero form by shifting the mantissa
    one position right and bumping the exponent by 1.
    """
    if x == 0.0:
        s = " 0." + "0" * digits + "E+00"
        return s.rjust(width)
    # Round to (digits) sig figs in Fortran's "0.XXX" sense — that's the
    # same as Python's `(digits-1)e` because Python keeps one digit before
    # the decimal. Example: 0.5557 with digits=3 -> Python "5.56e-01" ->
    # Fortran "0.556E+00".
    s = f"{x:.{digits-1}e}"
    if 'e' in s:
        mant, exp = s.split('e')
    else:
        mant, exp = s, '+00'
    sign = '-' if mant.startswith('-') else ''
    if sign:
        mant = mant[1:]
    # Mant is "X.YY..." with (digits-1) chars after dot
    head, _, tail = mant.partition('.')
    body = head + tail   # e.g. "556"
    new_mant = f"{sign}0.{body[:digits]}"
    new_exp_val = int(exp) + 1
    sign_exp = '+' if new_exp_val >= 0 else '-'
    new_exp = f"E{sign_exp}{abs(new_exp_val):02d}"
    return f"{new_mant}{new_exp}".rjust(width)


def _write_common_header(f: TextIO, heading: str, units: list[str],
                         kat: int, atm_inf: bool = False) -> None:
    """Header block shared by h.out / th.out / Q.out etc."""
    f.write(f" {heading:<72}\n")
    f.write("\n")
    f.write(" Program SWMS_2D\n")
    if atm_inf:
        f.write(" Time dependent boundary conditions\n")
    else:
        f.write(" Time independent boundary conditions\n")
    if kat == 0:
        f.write(" Horizontal plane flow, V = L*L\n")
    elif kat == 1:
        f.write(" Axisymmetric flow, V = L*L*L\n")
    else:
        f.write(" Vertical plane flow, V = L*L\n")
    L, T, M = (units + ["-", "-", "-"])[:3]
    f.write(f" Units: L = {L:<5}, T = {T:<5}, M = {M:<5}\n")


# ---------------------------------------------------------------------------
# h.out — pressure head field
# ---------------------------------------------------------------------------

class HOutWriter:
    """Wraps file handle + state needed across multiple h.out writes."""

    def __init__(self, path: Path, heading: str, units: list[str], kat: int,
                 atm_inf: bool = False):
        self.f = open(path, "w")
        _write_common_header(self.f, heading, units, kat, atm_inf)

    def write_snapshot(self, t: float, mesh: Mesh, h: NDArray[np.float64]) -> None:
        """Append one time-block to h.out matching OUTPUT2.FOR L406-426."""
        f = self.f
        # Header for this time block: blank line + Time + blank + column labels + blank
        f.write("\n\n")
        f.write(f" Time  ***{t:12.4f} ***\n")
        f.write("\n")
        f.write("    n    x(n)   z(n)       h(n)      h(n+1) ...\n")
        f.write("\n")
        IJ = mesh.IJ if mesh.IJ > 0 else 1
        NumNP = mesh.NumNP
        L1 = (IJ - 1) // 10 + 1
        # Fortran 1-based outer step: do n=1,NumNP,IJ
        for n in range(0, NumNP, IJ):
            for L in range(L1):
                m = n + L * 10
                k = m + 9
                if L == L1 - 1:
                    k = n + IJ - 1
                # node, x, y, then up to 10 h values
                row = f"{m+1:5d}{mesh.nodes.x[m]:8.1f}{mesh.nodes.y[m]:8.1f}"
                for j in range(m, k + 1):
                    if j < NumNP:
                        row += f"{h[j]:10.1f}"
                f.write(row + "\n")

    def close(self) -> None:
        self.f.close()


# ---------------------------------------------------------------------------
# th.out — water content field
# ---------------------------------------------------------------------------

class ConcOutWriter:
    """conc.out — solute concentration field. Matches Fortran cOut format
    (OUTPUT2.FOR L502-521): no shared header, 10e11.3 numeric format."""

    def __init__(self, path: Path, heading: str, units: list[str], kat: int):
        self.f = open(path, "w")

    def write_snapshot(self, t: float, mesh: Mesh,
                       conc: NDArray[np.float64]) -> None:
        f = self.f
        f.write("\n\n")
        f.write(f" Time  ***{t:12.4f} ***\n")
        f.write("\n")
        f.write("    n    x(n)   z(n)      Conc(n)   Conc(n+1)  ...\n")
        f.write("\n")
        IJ = mesh.IJ if mesh.IJ > 0 else 1
        NumNP = mesh.NumNP
        L1 = (IJ - 1) // 10 + 1
        for n in range(0, NumNP, IJ):
            for L in range(L1):
                m = n + L * 10
                k = m + 9
                if L == L1 - 1:
                    k = n + IJ - 1
                row = f"{m+1:5d}{mesh.nodes.x[m]:8.1f}{mesh.nodes.y[m]:8.1f}"
                for j in range(m, k + 1):
                    if j < NumNP:
                        row += _fortran_e(float(conc[j]), width=11, digits=3)
                f.write(row + "\n")

    def close(self) -> None:
        self.f.close()


class ThOutWriter:
    def __init__(self, path: Path, heading: str, units: list[str], kat: int,
                 atm_inf: bool = False):
        self.f = open(path, "w")
        _write_common_header(self.f, heading, units, kat, atm_inf)

    def write_snapshot(self, t: float, mesh: Mesh,
                       theta: NDArray[np.float64]) -> None:
        f = self.f
        f.write("\n\n")
        f.write(f" Time  ***{t:12.4f} ***\n")
        f.write("\n")
        f.write("    n    x(n)   z(n)      th(n)     th(n+1) ...\n")
        f.write("\n")
        IJ = mesh.IJ if mesh.IJ > 0 else 1
        NumNP = mesh.NumNP
        # th.out uses 16 values per row instead of 10
        L1 = (IJ - 1) // 16 + 1
        for n in range(0, NumNP, IJ):
            for L in range(L1):
                m = n + L * 16
                k = m + 15
                if L == L1 - 1:
                    k = n + IJ - 1
                row = f"{m+1:5d}{mesh.nodes.x[m]:8.1f}{mesh.nodes.y[m]:8.1f}"
                for j in range(m, k + 1):
                    if j < NumNP:
                        row += f"{theta[j]:6.3f}"
                f.write(row + "\n")

    def close(self) -> None:
        self.f.close()


# ---------------------------------------------------------------------------
# Run_Inf.out — per-timestep convergence info
# ---------------------------------------------------------------------------

class RunInfWriter:
    """Run_Inf.out — one line per WatFlow call. Matches OUTPUT2.FOR
    TLInf 110 format: `i5, 2e12.3, i5, i6`."""

    def __init__(self, path: Path, lChem: bool = False, lWat: bool = True):
        self.f = open(path, "w")
        if not lChem:
            self.f.write("\n\n TLevel   Time         dt      Iter  ItCum\n\n")
        elif lWat:
            self.f.write("\n\n TLevel   Time         dt      Iter  ItCum  "
                         "Peclet   Courant\n\n")
        else:
            self.f.write("\n\n TLevel   Time         dt        "
                         "Peclet   Courant\n\n")
        self.lChem = lChem
        self.lWat = lWat

    def write_line(self, TLevel: int, t: float, dt: float,
                   Iter: int, ItCum: int,
                   Peclet: float = 0.0, Courant: float = 0.0) -> None:
        if not self.lChem:
            self.f.write(f"{TLevel:5d}{_fortran_e(t, 12, 3)}"
                         f"{_fortran_e(dt, 12, 3)}{Iter:5d}{ItCum:6d}\n")
        elif self.lWat:
            self.f.write(f"{TLevel:5d}{_fortran_e(t, 12, 3)}"
                         f"{_fortran_e(dt, 12, 3)}{Iter:5d}{ItCum:6d}"
                         f"{Peclet:10.3f}{Courant:10.3f}\n")
        else:
            self.f.write(f"{TLevel:5d}{_fortran_e(t, 12, 3)}"
                         f"{_fortran_e(dt, 12, 3)}"
                         f"{Peclet:10.3f}{Courant:10.3f}\n")

    def close(self) -> None:
        self.f.close()


# ---------------------------------------------------------------------------
# ObsNod.out — observation-node time series
# ---------------------------------------------------------------------------

class ObsNodWriter:
    """ObsNod.out — appended per output event. Mirrors OUTPUT2.FOR ObsNod
    L393-402 (`f11.3, 5(f11.3, f9.4, e11.3)`)."""

    def __init__(self, path: Path, obs_nodes: list[int]):
        self.f = open(path, "w")
        if obs_nodes:
            self.f.write("    Time   ")
            for n in obs_nodes:
                self.f.write(f"     h({n:3d})    q({n:3d})  Conc({n:3d})")
            self.f.write("\n\n")
        self.obs_nodes = obs_nodes

    def write_line(self, t: float, hNew: NDArray[np.float64],
                   ThNew: NDArray[np.float64],
                   Conc: NDArray[np.float64]) -> None:
        row = f"{t:11.3f}"
        for n in self.obs_nodes:
            idx = n - 1   # Fortran 1-based -> 0-based
            row += f"{hNew[idx]:11.3f}{ThNew[idx]:9.4f}"
            row += _fortran_e(float(Conc[idx]), 11, 3)
        self.f.write(row + "\n")

    def close(self) -> None:
        self.f.close()


# ---------------------------------------------------------------------------
# vx.out / vz.out — Darcy velocity field at print times
# ---------------------------------------------------------------------------

class FluxOutWriter:
    """Pair of vx.out + vz.out writers (FlxOut, OUTPUT2.FOR L475-498)."""

    def __init__(self, vx_path: Path, vz_path: Path):
        self.fx = open(vx_path, "w")
        self.fz = open(vz_path, "w")

    def write_snapshot(self, t: float, mesh: Mesh,
                       Vx: NDArray[np.float64],
                       Vz: NDArray[np.float64]) -> None:
        IJ = mesh.IJ if mesh.IJ > 0 else 1
        NumNP = mesh.NumNP
        L1 = (IJ - 1) // 10 + 1
        self.fz.write(f"\n\n Time  ***{t:12.4f} ***\n\n"
                      "    n    x(n)  z(n)     vz(n)     vz(n+1) ...\n\n")
        self.fx.write(f"\n\n Time  ***{t:12.4f} ***\n\n"
                      "    n    x(n)  z(n)     vx(n)     vx(n+1) ...\n\n")
        for n in range(0, NumNP, IJ):
            for L in range(L1):
                m = n + L * 10
                k = m + 9
                if L == L1 - 1:
                    k = n + IJ - 1
                row_head = (f"{m+1:5d}{mesh.nodes.x[m]:8.1f}"
                            f"{mesh.nodes.y[m]:8.1f}")
                row_x = row_head
                row_z = row_head
                for j in range(m, k + 1):
                    if j < NumNP:
                        row_x += _fortran_e(float(Vx[j]), 10, 2)
                        row_z += _fortran_e(float(Vz[j]), 10, 2)
                self.fx.write(row_x + "\n")
                self.fz.write(row_z + "\n")

    def close(self) -> None:
        self.fx.close()
        self.fz.close()


# ---------------------------------------------------------------------------
# Q.out — nodal fluxes at print times (same layout as h.out, e11.3 format)
# ---------------------------------------------------------------------------

class QOutWriter:
    """Q.out — boundary/internal Q per node at print times (QOut, L429-447)."""

    def __init__(self, path: Path):
        self.f = open(path, "w")

    def write_snapshot(self, t: float, mesh: Mesh,
                       Q: NDArray[np.float64]) -> None:
        IJ = mesh.IJ if mesh.IJ > 0 else 1
        NumNP = mesh.NumNP
        L1 = (IJ - 1) // 10 + 1
        self.f.write(f"\n\n Time  ***{t:12.4f} ***\n\n"
                     "    n    x(n)   z(n)       Q(n)      Q(n+1) ...\n\n")
        for n in range(0, NumNP, IJ):
            for L in range(L1):
                m = n + L * 10
                k = m + 9
                if L == L1 - 1:
                    k = n + IJ - 1
                row = (f"{m+1:5d}{mesh.nodes.x[m]:8.1f}"
                       f"{mesh.nodes.y[m]:8.1f}")
                for j in range(m, k + 1):
                    if j < NumNP:
                        row += _fortran_e(float(Q[j]), 11, 3)
                self.f.write(row + "\n")

    def close(self) -> None:
        self.f.close()


# ---------------------------------------------------------------------------
# Balance.out — sub-region mass balance per print level
# ---------------------------------------------------------------------------

class BalanceWriter:
    """Balance.out — per-PLevel sub-region water balance summary (SubReg,
    OUTPUT2.FOR L159-308). Cumulative water-balance error WatBalT/WatBalR.

    Stateful: stashes the initial volume (PLevel==0) and per-element water
    storage so DeltW can be computed at subsequent calls.
    """

    def __init__(self, path: Path, heading: str, units: list[str], kat: int,
                 atm_inf: bool = False):
        self.f = open(path, "w")
        _write_common_header(self.f, heading, units, kat, atm_inf)
        self.kat = kat
        self.wVolI: float | None = None
        self.WatIn = None

    def write_snapshot(self, t: float, mesh: Mesh,
                       hNew: NDArray[np.float64],
                       ThOld: NDArray[np.float64],
                       ThNew: NDArray[np.float64],
                       dt: float, PLevel: int,
                       lWat: bool, wCumA: float = 0.0,
                       wCumT: float = 0.0) -> None:
        KX = mesh.elements.KX
        x = mesh.nodes.x
        y = mesh.nodes.y
        LayNum = mesh.elements.LayNum
        NumEl = mesh.NumEl
        NLay = int(LayNum.max()) if NumEl > 0 else 1
        if NLay < 1:
            NLay = 1
        Area = np.zeros(NLay + 1)
        SubVol = np.zeros(NLay + 1)
        SubCha = np.zeros(NLay + 1)
        hMeanL = np.zeros(NLay + 1)
        Volume = 0.0; Change = 0.0; hTot = 0.0
        DeltW = 0.0
        if self.WatIn is None:
            self.WatIn = np.zeros(NumEl)
        for e in range(NumEl):
            Lay = int(LayNum[e])
            wEl = 0.0
            NUS = 3 if KX[e, 2] == KX[e, 3] else 4
            for k in range(NUS - 2):
                i = KX[e, 0]; j = KX[e, k + 1]; l = KX[e, k + 2]
                Cj = x[i] - x[l]; Ck = x[j] - x[i]
                Bj = y[l] - y[i]; Bk = y[i] - y[j]
                xMul = 1.0
                if self.kat == 1:
                    xMul = 2.0 * 3.1416 * (x[i] + x[j] + x[l]) / 3.0
                AE = xMul * (Ck * Bj - Cj * Bk) / 2.0
                Area[Lay] += AE
                hE = (hNew[i] + hNew[j] + hNew[l]) / 3.0
                VNewE = AE * (ThNew[i] + ThNew[j] + ThNew[l]) / 3.0
                VOldE = AE * (ThOld[i] + ThOld[j] + ThOld[l]) / 3.0
                Volume += VNewE; wEl += VNewE
                Change += (VNewE - VOldE) / dt
                SubVol[Lay] += VNewE
                SubCha[Lay] += (VNewE - VOldE) / dt
                hTot += hE * AE
                hMeanL[Lay] += hE * AE
                if k == NUS - 3:
                    if PLevel == 0:
                        self.WatIn[e] = wEl
                    else:
                        DeltW += abs(self.WatIn[e] - wEl)
        ATot = float(Area[1:NLay + 1].sum())
        for Lay in range(1, NLay + 1):
            if Area[Lay] > 0:
                hMeanL[Lay] /= Area[Lay]
        if ATot > 0:
            hTot /= ATot
        f = self.f
        if PLevel == 0:
            f.write("\n Time [T]             Total     "
                    "Sub-region number ...\n")
        f.write(f"\n{t:12.4f}{'':16s}")
        for i in range(1, NLay + 1):
            f.write(f"{i:7d}    ")
        f.write("\n")
        f.write(f" Area    [V]       {ATot:11.3e}")
        for i in range(1, NLay + 1):
            f.write(f"{Area[i]:11.3e}")
        f.write("\n")
        f.write(f" Volume  [V]       {Volume:11.3e}")
        for i in range(1, NLay + 1):
            f.write(f"{SubVol[i]:11.3e}")
        f.write("\n")
        f.write(f" InFlow  [V/T]     {Change:11.3e}")
        for i in range(1, NLay + 1):
            f.write(f"{SubCha[i]:11.3e}")
        f.write("\n")
        f.write(f" hMean   [L]       {hTot:11.3e}")
        for i in range(1, NLay + 1):
            f.write(f"{hMeanL[i]:11.1f}")
        f.write("\n")
        if PLevel == 0:
            self.wVolI = Volume
        elif lWat and self.wVolI is not None:
            wBalT = Volume - self.wVolI + wCumT
            f.write(f" WatBalT [V]       {wBalT:11.3e}\n")
            ww = max(DeltW, wCumA)
            if ww >= 1e-25:
                wBalR = abs(wBalT) / ww * 100.0
                f.write(f" WatBalR [%]       {wBalR:11.3f}\n")

    def close(self) -> None:
        self.f.close()


# ---------------------------------------------------------------------------
# Cum_Q.out — per-timestep cumulative boundary fluxes
# ---------------------------------------------------------------------------

class CumQWriter:
    """Cum_Q.out — per-step running totals of boundary fluxes. Matches
    OUTPUT2.FOR TLInf 190 format. Columns: t, CumQAP, CumQRP, CumQA, CumQR,
    CumQ3, CumQ1, CumQS, CumQ5, CumQ6."""

    def __init__(self, path: Path, lWat: bool = True,
                 heading: str = "", units: list[str] | None = None,
                 kat: int = 2, atm_inf: bool = True):
        self.f = open(path, "w")
        if heading:
            _write_common_header(self.f, heading, units or ["-", "-", "-"],
                                 kat, atm_inf)
        if lWat:
            self.f.write(
                "\n All cumulative fluxes (CumQ) are positive out of the region\n\n"
                "     Time     CumQAP      CumQRP     CumQA      CumQR     "
                "CumQ3       CumQ1      CumQS      CumQ5       CumQ6 ....\n"
                "      [T]       [V]         [V]       [V]        [V]       "
                "[V]         [V]        [V]        [V]         [V]\n\n"
            )

    def write_line(self, t: float, CumQrT: float, CumQrR: float,
                   CumQ4: float, CumQvR: float, CumQ3: float,
                   CumQ1: float, CumQ2: float,
                   extra: list[float] | None = None) -> None:
        extra = extra or []
        row = f"{t:12.4f}"
        for v in [CumQrT, CumQrR, CumQ4, CumQvR, CumQ3, CumQ1, CumQ2, *extra]:
            row += _fortran_e(float(v), 11, 3)
        self.f.write(row + "\n")

    def close(self) -> None:
        self.f.close()


# ---------------------------------------------------------------------------
# A_Level.out — atmospheric-level summary (one row per atm record)
# ---------------------------------------------------------------------------

class ALevelWriter:
    """A_Level.out — once per atm-record advancement (ALInf, OUTPUT2.FOR
    L137-155). Columns: t, CumQAP, CumQRP, CumQA, CumQR, CumQ3, hAtm,
    hRoot, hKode3, ALevel."""

    def __init__(self, path: Path,
                 heading: str = "", units: list[str] | None = None,
                 kat: int = 2, atm_inf: bool = True):
        self.f = open(path, "w")
        if heading:
            _write_common_header(self.f, heading, units or ["-", "-", "-"],
                                 kat, atm_inf)
        self.f.write(
            "\n All cumulative fluxes (CumQ) are positive out of the region\n\n"
            "      Time      CumQAP     CumQRP     CumQA      CumQR     "
            "CumQ3        hAtm       hRoot     hKode3    A-level\n"
            "      [T]         [V]        [V]       [V]        [V]       "
            "[V]          [L]        [L]        [L]\n\n"
        )

    def write_line(self, t: float, CumQrT: float, CumQrR: float,
                   CumQ4: float, CumQvR: float, CumQ3: float,
                   hMeanT: float, hMeanR: float, hMeanG: float,
                   ALevel: int) -> None:
        row = f"{t:12.4f}"
        for v in [CumQrT, CumQrR, CumQ4, CumQvR, CumQ3]:
            row += _fortran_e(float(v), 11, 3)
        row += f"{hMeanT:11.1f}{hMeanR:11.1f}{hMeanG:11.1f}{ALevel:8d}"
        self.f.write(row + "\n")

    def close(self) -> None:
        self.f.close()


# ---------------------------------------------------------------------------
# Boundary.out — per-print-event boundary node table
# ---------------------------------------------------------------------------

class BouOutWriter:
    """Boundary.out — boundary-node table per print time (BouOut, L312-348).
    Columns: n, x, z, Code, Q, h, theta, conc."""

    def __init__(self, path: Path):
        self.f = open(path, "w")

    def write_snapshot(self, t: float, mesh: Mesh,
                       hNew: NDArray[np.float64],
                       theta: NDArray[np.float64],
                       Q: NDArray[np.float64],
                       Conc: Optional[NDArray[np.float64]] = None) -> None:
        f = self.f
        f.write(f"\n Time  ***{t:12.4f} ***\n\n")
        f.write("    n    x(n)   z(n) Code        Q          h          "
                "theta       conc\n\n")
        KXB = mesh.KXB
        Kode = mesh.nodes.Kode
        x = mesh.nodes.x; y = mesh.nodes.y
        if Conc is None:
            Conc_arr = np.zeros(mesh.NumNP)
        else:
            Conc_arr = Conc
        for i in range(mesh.NumBP):
            n = int(KXB[i])
            row = (f"{n+1:5d}{x[n]:8.1f}{y[n]:8.1f}{int(Kode[n]):5d}"
                   f"{_fortran_e(float(Q[n]), 11, 3)}"
                   f"{hNew[n]:11.3f}{theta[n]:11.4f}"
                   f"{_fortran_e(float(Conc_arr[n]), 11, 3)}")
            f.write(row + "\n")

    def close(self) -> None:
        self.f.close()

