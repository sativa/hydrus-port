"""
Subsurface drainage for SWMS_2D Python port.
============================================

Port of WATFLOW2.FOR Shift's DrainF branch (L356-371) + INPUT2.FOR
DrainIn (L204-251). Drainage is implemented as a discrete-node boundary
condition with effective-K reduction in the surrounding elements via
the Vimoke-Taylor (1962) correction factor.

State machine per drain node (analogous to Seepage face):
    Kode = -5  →  drain inactive, flux = 0 (no water collected)
    Kode = +5  →  drain active, h = 0 (water flowing INTO drain at
                  atmospheric pressure)

Switch logic in `shift_drain`:
    if Kode == -5: if hNew >= 0 → switch to +5, hNew := 0
    else:          if    Q >= 0 → switch to -5, Q := 0

Effective-K reduction (Vimoke & Taylor 1962):
    ρ    = EfDim[1] / EfDim[0]      (long axis / short axis of drain
                                     equivalent ellipse)
    A    = (1 + 0.405 ρ⁻⁴) / (1 - 0.405 ρ⁻⁴)
    B    = (1 + 0.163 ρ⁻⁸) / (1 - 0.163 ρ⁻⁸)
    C    = (1 + 0.067 ρ⁻¹²) / (1 - 0.067 ρ⁻¹²)
    Red  = 376.7 / (138 log₁₀ ρ + 6.48 - 2.34 A - 0.48 B - 0.12 C)
           / DrCorr
The reduction is applied once at simulation setup to ConAxx/ConAxz/ConAzz
of every element flagged as a drain element.

The original SWMS_2D 1.22 input file BLOCK F (drainage) is rare; this
module mirrors the Fortran reader exactly for compatibility.
"""

from __future__ import annotations
import numpy as np
from numpy.typing import NDArray

from .dataclasses import Mesh


def apply_vimoke_taylor(mesh: Mesh,
                        NDr: int,
                        NED: NDArray[np.int32],
                        EfDim: NDArray[np.float64],
                        KElDr: list[list[int]],
                        DrCorr: float = 1.0,
                        ) -> None:
    """Apply the Vimoke-Taylor (1962) K-reduction in-place to mesh.elements
    for every drain element. Mirrors INPUT2.FOR DrainIn L237-249."""
    for i in range(NDr):
        rho = EfDim[i, 1] / EfDim[i, 0]
        A = (1.0 + 0.405 * rho ** (-4))  / (1.0 - 0.405 * rho ** (-4))
        B = (1.0 + 0.163 * rho ** (-8))  / (1.0 - 0.163 * rho ** (-8))
        C = (1.0 + 0.067 * rho ** (-12)) / (1.0 - 0.067 * rho ** (-12))
        Red = (376.7 /
               (138.0 * np.log10(rho) + 6.48 - 2.34 * A - 0.48 * B - 0.12 * C)
               / DrCorr)
        # Convert 1-based element indices to 0-based and apply
        for e_1based in KElDr[i][:NED[i]]:
            e = e_1based - 1
            mesh.elements.ConAxx[e] *= Red
            mesh.elements.ConAxz[e] *= Red
            mesh.elements.ConAzz[e] *= Red


def shift_drain(mesh: Mesh,
                NDr: int,
                ND: NDArray[np.int32],
                ) -> None:
    """Drainage switching per drain node (Kode=-5 ↔ +5).

    Mirrors WATFLOW2.FOR L356-371. Called every Picard iteration when
    DrainF is True; mutates mesh.nodes.Kode, .hNew, .Q.
    """
    Kode = mesh.nodes.Kode
    hNew = mesh.nodes.hNew
    Q = mesh.nodes.Q
    for i in range(NDr):
        n = int(ND[i]) - 1   # 1-based → 0-based
        if Kode[n] == -5:
            if hNew[n] >= 0.0:
                Kode[n] = 5
                hNew[n] = 0.0
        else:
            if Q[n] >= 0.0:
                Kode[n] = -5
                Q[n] = 0.0
