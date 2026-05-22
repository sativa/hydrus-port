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

def _write_common_header(f: TextIO, heading: str, units: list[str],
                         kat: int) -> None:
    """Header block shared by h.out / th.out / Q.out etc."""
    # Original Fortran writes heading padded to 72 chars + trailing newline.
    f.write(f" {heading:<72}\n")
    f.write("\n")
    f.write(" Program SWMS_2D\n")
    f.write(" Time independent boundary conditions\n")
    if kat == 0:
        f.write(" Horizontal plane flow, V = L*L\n")
    elif kat == 1:
        f.write(" Axisymmetric flow, V = L*L*L\n")
    else:
        f.write(" Vertical plane flow, V = L*L\n")
    # Units line: '%-5s, T = %-5s, M = %-5s'  -- match Fortran width
    L, T, M = (units + ["-", "-", "-"])[:3]
    f.write(f" Units: L = {L:<5}, T = {T:<5}, M = {M:<5}\n")


# ---------------------------------------------------------------------------
# h.out — pressure head field
# ---------------------------------------------------------------------------

class HOutWriter:
    """Wraps file handle + state needed across multiple h.out writes."""

    def __init__(self, path: Path, heading: str, units: list[str], kat: int):
        self.f = open(path, "w")
        _write_common_header(self.f, heading, units, kat)

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

class ThOutWriter:
    def __init__(self, path: Path, heading: str, units: list[str], kat: int):
        self.f = open(path, "w")
        _write_common_header(self.f, heading, units, kat)

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
