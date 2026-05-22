"""
SWMS_2D material property adapter.
==================================

Wraps hydrus1d.material's FK/FC/FQ functions with SWMS_2D's parameter
ordering, since the two packages use different Par[] layouts.

SWMS_2D SoilMaterial → H1D Par[11] mapping:
    H1D Par[0]  = SWMS thr     (theta_r)
    H1D Par[1]  = SWMS ths     (theta_s)
    H1D Par[2]  = SWMS alpha
    H1D Par[3]  = SWMS n
    H1D Par[4]  = SWMS Ks
    H1D Par[5]  = 0.5          (beta — Mualem default, SWMS_2D hardcodes this)
    H1D Par[6]  = SWMS thm
    H1D Par[7]  = SWMS tha
    H1D Par[8]  = SWMS thk
    H1D Par[9]  = SWMS Kk
    H1D Par[10] = 0.0          (h_sat)

Use iModel=0 (van Genuchten-Mualem). SWMS_2D v1.22 supports only the
standard VG model with optional Vogel-Cislerova modification (when
thm != ths), which corresponds to H1D iModel=1.
"""

from __future__ import annotations
import numpy as np
from numpy.typing import NDArray

# Import H1D functions — they need to be on sys.path; from a parent
# package import this works when running with H1D_Src/ as CWD.
from hydrus1d.material import FK as _FK, FC as _FC, FQ as _FQ, FS as _FS

from .dataclasses import SoilMaterial


def to_h1d_par(mat: SoilMaterial) -> NDArray[np.float64]:
    """Convert SoilMaterial dataclass → 11-element H1D-format Par array."""
    return np.array([
        mat.thr, mat.ths, mat.alpha, mat.n, mat.Ks,
        0.5,                       # beta (Mualem)
        mat.thm, mat.tha,
        mat.thk, mat.Kk,
        0.0,                       # h_sat
    ], dtype=np.float64)


def select_imodel(mat: SoilMaterial) -> int:
    """Pick H1D iModel that matches SWMS_2D's behaviour.

    SWMS_2D's FK in MATERIA2.FOR always uses the full Vogel-Cislerova
    K(h) formula with thm/tha/thk/Kk. It "collapses" to standard VG only
    when thm == ths, tha == thr, thk == ths, AND Kk == Ks — all four must
    hold, because thk/Kk control a linear-K segment between Hk and Hs even
    when thm/tha do not.
    """
    if (abs(mat.thm - mat.ths) < 1e-12
        and abs(mat.tha - mat.thr) < 1e-12
        and abs(mat.thk - mat.ths) < 1e-12
        and abs(mat.Kk  - mat.Ks ) < 1e-12):
        return 0  # standard VG-Mualem
    return 1      # modified VG (Vogel-Cislerova)


def FK(mat: SoilMaterial, h: float) -> float:
    return _FK(select_imodel(mat), float(h), to_h1d_par(mat))


def FC(mat: SoilMaterial, h: float) -> float:
    return _FC(select_imodel(mat), float(h), to_h1d_par(mat))


def FQ(mat: SoilMaterial, h: float) -> float:
    return _FQ(select_imodel(mat), float(h), to_h1d_par(mat))


def FS(mat: SoilMaterial, h: float) -> float:
    return _FS(select_imodel(mat), float(h), to_h1d_par(mat))


# ----------------------------------------------------------------------------
# Numerical derivatives for the Newton-Raphson Jacobian
# ----------------------------------------------------------------------------

def dC_dh_numeric(iModel: int, h: float, Par: NDArray[np.float64],
                  hSat: float = 0.0) -> float:
    """∂C/∂h via central finite difference, used by Newton in reset().

    Stays away from h = 0 (saturation kink) and clamps very dry h where
    C is essentially zero.
    """
    if h >= hSat - 1e-6 or h <= -1e6:
        return 0.0
    eps = max(1e-3, 1e-4 * abs(h))
    hi = min(h + eps, hSat - 1e-6)
    lo = h - eps
    return (_FC(iModel, hi, Par) - _FC(iModel, lo, Par)) / (hi - lo)


def dK_dh_numeric(iModel: int, h: float, Par: NDArray[np.float64],
                  hSat: float = 0.0) -> float:
    """∂K/∂h via central finite difference."""
    if h >= hSat - 1e-6 or h <= -1e6:
        return 0.0
    eps = max(1e-3, 1e-4 * abs(h))
    hi = min(h + eps, hSat - 1e-6)
    lo = h - eps
    return (_FK(iModel, hi, Par) - _FK(iModel, lo, Par)) / (hi - lo)


# ----------------------------------------------------------------------------
# Vectorised batch helpers (avoid per-node Python overhead in assembly loops)
# ----------------------------------------------------------------------------

def evaluate_at_nodes(materials: list[SoilMaterial], MatNum: NDArray[np.int32],
                      h: NDArray[np.float64]
                      ) -> tuple[NDArray[np.float64], NDArray[np.float64],
                                 NDArray[np.float64]]:
    """
    For every node, evaluate Con(h), Cap(h), Theta(h) from its material.

    Returns
    -------
    Con  : K(h) per node (cm/T)
    Cap  : C(h) = dθ/dh per node (1/cm)
    Theta: θ(h) per node (cm³/cm³)
    """
    n = h.shape[0]
    Con = np.empty(n, np.float64)
    Cap = np.empty(n, np.float64)
    Th  = np.empty(n, np.float64)
    # Cache (par, iModel) per material since materials list is small (≤ NMatD=20)
    pars = [(select_imodel(m), to_h1d_par(m)) for m in materials]
    for i in range(n):
        m = MatNum[i] - 1                     # 1-based → 0-based
        iModel, Par = pars[m]
        Con[i] = _FK(iModel, float(h[i]), Par)
        Cap[i] = _FC(iModel, float(h[i]), Par)
        Th[i]  = _FQ(iModel, float(h[i]), Par)
    return Con, Cap, Th


def saturated_values(materials: list[SoilMaterial]
                     ) -> tuple[NDArray[np.float64], NDArray[np.float64],
                                NDArray[np.float64], NDArray[np.float64]]:
    """
    Per-material θ_sat, θ_r, h_sat, K_sat — needed for the convergence
    test in WatFlow (see WATFLOW2.FOR L99-104).

    h_sat is the pressure head at which θ = θ_sat. For standard VG this
    is 0; for modified VG (Vogel-Cislerova) it's where θ_m saturates.
    """
    nm = len(materials)
    thSat = np.empty(nm, np.float64)
    thR   = np.empty(nm, np.float64)
    hSat  = np.zeros(nm, np.float64)   # 0 for VG; SWMS_2D allows non-zero
    ConSat = np.empty(nm, np.float64)
    for i, m in enumerate(materials):
        thSat[i] = m.ths
        thR[i]   = m.thr
        ConSat[i] = m.Ks
    return thR, thSat, hSat, ConSat
