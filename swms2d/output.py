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
from typing import TextIO
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
