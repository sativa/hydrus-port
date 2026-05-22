"""
Soil water retention hysteresis for SWMS_2D Python port.
========================================================

The original SWMS_2D 1.22 source has **no** hysteresis. This module
adds it as a parallel water-retention path that the caller can opt
into per simulation. HYDRUS-1D supports hysteresis via the
Scott et al. (1983) and Mualem (1974) scaling models — here we
implement the simpler **Scott linear scaling** which is sufficient
for most agricultural applications:

    During scanning, theta(h) interpolates linearly between the main
    drying curve (MDC) and main wetting curve (MWC), scaled so that
    the scanning curve passes through the reversal point.

State per node, in addition to standard h/theta:
    iHyst[i] in {0, 1, 2, 3}
        0 = initial / fully wet (on MWC or above)
        1 = on main wetting curve (MWC)
        2 = on main drying curve  (MDC)
        3 = on internal scanning curve from a reversal point
    h_rev[i], th_rev[i]   reversal point (only meaningful when iHyst==3)
    iHyst_anchor[i]        which main curve the scanning is anchored to

Two VG parameter sets per material:
    Drying  : (theta_r, theta_s, alpha_d, n_d, K_s, l, ...)
    Wetting : (theta_r, theta_s, alpha_w, n_w, K_s, l, ...)

Hysteresis is detected by sign of (h_new - h_prev): wetting if positive,
drying if negative. Reversal triggers a state-machine transition.
"""

from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from numpy.typing import NDArray

from .dataclasses import SoilMaterial
from hydrus1d.material import FQ as _FQ_h1d, FC as _FC_h1d
from . import material as _mat


# State codes
IHYST_INITIAL  = 0
IHYST_WETTING  = 1   # on main wetting curve
IHYST_DRYING   = 2   # on main drying curve
IHYST_SCANNING = 3   # on internal scanning curve


@dataclass
class HysteresisState:
    """Per-node hysteresis state — created once per simulation."""
    iHyst:    NDArray[np.int32]
    h_rev:    NDArray[np.float64]
    th_rev:   NDArray[np.float64]
    iHyst_anchor: NDArray[np.int32]  # which main curve the scanning anchors to


def init_state(NumNP: int, default_branch: int = IHYST_DRYING
               ) -> HysteresisState:
    """Initialise all nodes to the same main curve (typically MDC)."""
    return HysteresisState(
        iHyst=np.full(NumNP, default_branch, np.int32),
        h_rev=np.zeros(NumNP, np.float64),
        th_rev=np.zeros(NumNP, np.float64),
        iHyst_anchor=np.full(NumNP, default_branch, np.int32),
    )


# ============================================================================
# Main wetting / drying curves
# ============================================================================

@dataclass
class HysteresisMaterial:
    """Two VG curve parameter sets for one material — drying and wetting."""
    drying: SoilMaterial
    wetting: SoilMaterial

    def theta_drying(self, h: float) -> float:
        if h >= 0.0:
            return self.drying.ths
        iModel = _mat.select_imodel(self.drying)
        return _FQ_h1d(iModel, h, _mat.to_h1d_par(self.drying))

    def theta_wetting(self, h: float) -> float:
        if h >= 0.0:
            return self.wetting.ths
        iModel = _mat.select_imodel(self.wetting)
        return _FQ_h1d(iModel, h, _mat.to_h1d_par(self.wetting))

    def cap_drying(self, h: float) -> float:
        if h >= 0.0:
            return 0.0
        iModel = _mat.select_imodel(self.drying)
        return _FC_h1d(iModel, h, _mat.to_h1d_par(self.drying))

    def cap_wetting(self, h: float) -> float:
        if h >= 0.0:
            return 0.0
        iModel = _mat.select_imodel(self.wetting)
        return _FC_h1d(iModel, h, _mat.to_h1d_par(self.wetting))


# ============================================================================
# Scanning curve formulae — Scott et al. (1983) linear scaling
# ============================================================================

def _scanning_theta(h: float, h_rev: float, th_rev: float,
                    mat: HysteresisMaterial,
                    direction: str) -> float:
    """theta(h) on a scanning curve anchored at (h_rev, th_rev).

    Scott (1983) linear shift: scanning curve runs parallel to the target
    main curve, shifted to pass through the reversal point.
    """
    if direction == 'wetting':
        th_target = mat.theta_wetting(h)
        th_target_rev = mat.theta_wetting(h_rev)
        return th_rev + (th_target - th_target_rev)
    else:  # drying
        th_target = mat.theta_drying(h)
        th_target_rev = mat.theta_drying(h_rev)
        return th_rev + (th_target - th_target_rev)


def _scanning_cap(h: float, mat: HysteresisMaterial,
                  direction: str) -> float:
    """dtheta/dh on scanning curve = dtheta/dh of target main curve
    (constant shift doesn't change the slope)."""
    if direction == 'wetting':
        return mat.cap_wetting(h)
    return mat.cap_drying(h)


# ============================================================================
# State update + theta/C evaluation (called per Picard iter)
# ============================================================================

def _theta_on_branch(state: HysteresisState, i: int, h: float,
                     mat: HysteresisMaterial) -> float:
    """Evaluate theta at h on the node's current branch."""
    if state.iHyst[i] == IHYST_WETTING:
        return mat.theta_wetting(h)
    if state.iHyst[i] == IHYST_DRYING:
        return mat.theta_drying(h)
    # Scanning
    scan_dir = ('drying' if state.iHyst_anchor[i] == IHYST_WETTING
                          else 'wetting')
    return _scanning_theta(h, state.h_rev[i], state.th_rev[i], mat, scan_dir)


def step_state(state: HysteresisState,
               h_new: NDArray[np.float64],
               h_old: NDArray[np.float64],
               MatNum: NDArray[np.int32],
               materials: list[HysteresisMaterial],
               rev_eps: float = 1e-4,
               ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Update each node's hysteresis state from h_old to h_new, then return
    per-node theta(h_new) and C(h_new) = dtheta/dh on the new branch.

    Direction-reversal detection uses sign(h_new - h_old) against the
    branch the node was on. `rev_eps` filters out numerical wiggles.
    """
    n = h_new.shape[0]
    theta = np.zeros(n, np.float64)
    cap = np.zeros(n, np.float64)
    for i in range(n):
        M = int(MatNum[i]) - 1
        mat = materials[M]
        h = float(h_new[i])
        h_o = float(h_old[i])
        dh = h - h_o

        # Direction: 'wetting' if going wetter (h up), 'drying' if down.
        direction: str | None = None
        if dh > rev_eps:
            direction = 'wetting'
        elif dh < -rev_eps:
            direction = 'drying'

        # Theta at h_old on the CURRENT branch (before transition)
        th_old_on_branch = _theta_on_branch(state, i, h_o, mat)

        # State transitions
        if state.iHyst[i] == IHYST_INITIAL:
            state.iHyst[i] = (IHYST_WETTING if direction == 'wetting'
                              else IHYST_DRYING)
        elif state.iHyst[i] == IHYST_WETTING and direction == 'drying':
            # Leaving MWC by drying — start a drying scan from (h_o, MWC(h_o))
            state.h_rev[i] = h_o
            state.th_rev[i] = th_old_on_branch
            state.iHyst[i] = IHYST_SCANNING
            state.iHyst_anchor[i] = IHYST_WETTING
        elif state.iHyst[i] == IHYST_DRYING and direction == 'wetting':
            state.h_rev[i] = h_o
            state.th_rev[i] = th_old_on_branch
            state.iHyst[i] = IHYST_SCANNING
            state.iHyst_anchor[i] = IHYST_DRYING
        elif state.iHyst[i] == IHYST_SCANNING:
            scan_dir = ('drying' if state.iHyst_anchor[i] == IHYST_WETTING
                                  else 'wetting')
            if direction == 'wetting' and scan_dir == 'drying':
                # Reversal within scanning — new scan from (h_o, th_old)
                state.h_rev[i] = h_o
                state.th_rev[i] = th_old_on_branch
                state.iHyst_anchor[i] = IHYST_WETTING
            elif direction == 'drying' and scan_dir == 'wetting':
                state.h_rev[i] = h_o
                state.th_rev[i] = th_old_on_branch
                state.iHyst_anchor[i] = IHYST_DRYING

        # Evaluate theta and C on the (possibly transitioned) branch
        if state.iHyst[i] == IHYST_WETTING:
            theta[i] = mat.theta_wetting(h)
            cap[i]   = mat.cap_wetting(h)
        elif state.iHyst[i] == IHYST_DRYING:
            theta[i] = mat.theta_drying(h)
            cap[i]   = mat.cap_drying(h)
        else:
            scan_dir = ('drying' if state.iHyst_anchor[i] == IHYST_WETTING
                                  else 'wetting')
            th_scan = _scanning_theta(h, state.h_rev[i], state.th_rev[i],
                                       mat, scan_dir)
            # Clamp to physical range; if saturated, snap to MWC; if at
            # residual, snap to MDC.
            ths = mat.drying.ths
            thr = mat.drying.thr
            if th_scan >= ths - 1e-6:
                state.iHyst[i] = IHYST_WETTING
                theta[i] = mat.theta_wetting(h)
                cap[i]   = mat.cap_wetting(h)
            elif th_scan <= thr + 1e-6:
                state.iHyst[i] = IHYST_DRYING
                theta[i] = mat.theta_drying(h)
                cap[i]   = mat.cap_drying(h)
            else:
                theta[i] = th_scan
                cap[i]   = _scanning_cap(h, mat, scan_dir)

    return theta, cap
