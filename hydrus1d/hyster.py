"""
Hysteresis module for HYDRUS-1D.
================================

Hysteretic K-S-P relationships:
- Main drying curve
- Secondary wetting/drying curves
- Scanning curve tracking
- Reversal point management
- Lenhard model for gas relative permeability

Direct port of HYSTER.FOR.
"""

from __future__ import annotations
import numpy as np
from numpy.typing import NDArray
from typing import Tuple

from .material import FK, FC, FQ, FH, FS


# ============================================================================
# Hysteresis state management
# ============================================================================

def update_hysteresis(
    N: int,
    hNew: NDArray[np.float64],
    hOld: NDArray[np.float64],
    hTemp: NDArray[np.float64],
    Kappa: NDArray[np.int64],
    KappaO: NDArray[np.int64],
    Ah: NDArray[np.float64],
    AK: NDArray[np.float64],
    ATh: NDArray[np.float64],
    AhW: NDArray[np.float64],
    AKW: NDArray[np.float64],
    AThW: NDArray[np.float64],
    ThRR: NDArray[np.float64],
    ConR: NDArray[np.float64],
    AKR: NDArray[np.float64],
    hRev: NDArray[np.float64],
    ThRev: NDArray[np.float64],
    nRev: NDArray[np.int64],
    iRev: NDArray[np.int64],
    KappaR: NDArray[np.int64],
    hRev0: NDArray[np.float64],
    ThRev0: NDArray[np.float64],
    KappaRev: NDArray[np.int64],
    hRev1: NDArray[np.float64],
    ThRev1: NDArray[np.float64],
    KappaRev1: NDArray[np.int64],
    ParD: NDArray[np.float64],
    ParW: NDArray[np.float64],
    iModel: int,
) -> None:
    """
    Update hysteresis state tracking.
    
    Direct port of hysteretic scanning curve logic in HYSTER.FOR.
    
    Parameters
    ----------
    N : int
        Number of nodes
    hNew : array, shape (N,)
        Current pressure head estimate
    hOld : array, shape (N,)
        Previous time step pressure heads
    hTemp : array, shape (N,)
        Temporary heads
    Kappa : array, shape (N,)
        Current curve type (1=drying, -1=wetting)
    KappaO : array, shape (N,)
        Previous curve type
    Ah : array, shape (N,)
        Head scaling factor
    AK : array, shape (N,)
        Conductivity scaling factor
    ATh : array, shape (N,)
        Water content scaling factor
    AhW : array, shape (N,)
        Wetting head scaling
    AKW : array, shape (N,)
        Wetting conductivity scaling
    AThW : array, shape (N,)
        Wetting water content scaling
    ThRR : array, shape (N,)
        Residual water content
    ConR : array, shape (N,)
        Residual conductivity
    AKR : array, shape (N,)
        Residual conductivity scaling
    hRev : array, shape (N, 20)
        Reversal point heads
    ThRev : array, shape (N, 20)
        Reversal point water contents
    nRev : array, shape (N,)
        Number of reversal points
    iRev : array, shape (N,)
        Current reversal index
    KappaR : array, shape (N,)
        Reversal curve type
    hRev0 : array, shape (N,)
        Initial reversal head
    ThRev0 : array, shape (N,)
        Initial reversal water content
    KappaRev : array, shape (N,)
        Reversal curve type
    hRev1 : array, shape (N,)
        Secondary reversal head
    ThRev1 : array, shape (N,)
        Secondary reversal water content
    KappaRev1 : array, shape (N,)
        Secondary reversal curve type
    ParD : array, shape (11, NMat)
        Drying hydraulic parameters
    ParW : array, shape (11, NMat)
        Wetting hydraulic parameters
    iModel : int
        Hydraulic model code
    """
    for i in range(N):
        M = 0  # MatNum[i] placeholder
        
        # Detect reversal
        dh = hNew[i] - hOld[i]
        
        if Kappa[i] > 0 and dh > 0:
            # Drying to wetting reversal
            Kappa[i] = -1
            nRev[i] = min(nRev[i] + 1, 19)
            hRev[i, nRev[i]] = hOld[i]
            ThRev[i, nRev[i]] = FQ(iModel, hOld[i], ParD[0, M])
            KappaR[i] = 1
        
        elif Kappa[i] < 0 and dh < 0:
            # Wetting to drying reversal
            Kappa[i] = 1
            nRev[i] = min(nRev[i] + 1, 19)
            hRev[i, nRev[i]] = hOld[i]
            ThRev[i, nRev[i]] = FQ(iModel, hOld[i], ParW[0, M])
            KappaR[i] = -1
        
        # Compute hysteretic scaling factors
        if nRev[i] == 0:
            # Main curve
            if Kappa[i] > 0:
                Ah[i] = 1.0
                AK[i] = 1.0
                ATh[i] = 1.0
            else:
                Ah[i] = AhW[i]
                AK[i] = AKW[i]
                ATh[i] = AThW[i]
        else:
            # Scanning curve
            _compute_scanning_factors(
                i, hNew[i], Kappa[i], nRev[i],
                hRev[i], ThRev[i], KappaR[i],
                hRev0[i], ThRev0[i],
                ParD, ParW, iModel,
                Ah, AK, ATh,
            )


def _compute_scanning_factors(
    i: int,
    h: float,
    Kappa: int,
    nRev: int,
    hRev: NDArray[np.float64],
    ThRev: NDArray[np.float64],
    KappaR: NDArray[np.int64],
    hRev0: float,
    ThRev0: float,
    ParD: NDArray[np.float64],
    ParW: NDArray[np.float64],
    iModel: int,
    Ah: NDArray[np.float64],
    AK: NDArray[np.float64],
    ATh: NDArray[np.float64],
) -> None:
    """
    Compute scanning curve scaling factors.
    
    Uses the parametric hysteresis model (van Genuchten 1980).
    
    Parameters
    ----------
    i : int
        Node index
    h : float
        Current pressure head
    Kappa : int
        Curve type
    nRev : int
        Number of reversal points
    hRev : array, shape (20,)
        Reversal point heads
    ThRev : array, shape (20,)
        Reversal point water contents
    KappaR : int
        Reversal curve type
    hRev0 : float
        Initial reversal head
    ThRev0 : float
        Initial reversal water content
    ParD : array, shape (11, NMat)
        Drying parameters
    ParW : array, shape (11, NMat)
        Wetting parameters
    iModel : int
        Model code
    Ah : array, shape (N,)
        Head scaling (output)
    AK : array, shape (N,)
        Conductivity scaling (output)
    ATh : array, shape (N,)
        Water content scaling (output)
    """
    M = 0
    
    if Kappa > 0:
        # Drying scanning curve
        Par = ParD[0, M]
        hR = max(hRev[i, nRev], hRev0)
        thR = max(ThRev[i, nRev], ThRev0)
    else:
        # Wetting scanning curve
        Par = ParW[0, M]
        hR = max(hRev[i, nRev], hRev0)
        thR = max(ThRev[i, nRev], ThRev0)
    
    # Compute scaling factors
    if abs(h - hR) > 1e-10:
        Ah[i] = (h - Par) / max(hR - Par, 1e-10)
        ATh[i] = (thR - ParD[0, M]) / max(ParW[1, M] - ParD[0, M], 1e-10)
        AK[i] = Ah[i] ** 0.5
    else:
        Ah[i] = 1.0
        AK[i] = 1.0
        ATh[i] = 1.0
    
    # Ensure positive values
    Ah[i] = max(Ah[i], 1e-10)
    AK[i] = max(AK[i], 1e-10)
    ATh[i] = max(ATh[i], 1e-10)


# ============================================================================
# Hysteretic property computation
# ============================================================================

def compute_hysteretic_properties(
    h: float,
    ParD: NDArray[np.float64],
    ParW: NDArray[np.float64],
    iModel: int,
    Ah: float,
    AK: float,
    ATh: float,
) -> Tuple[float, float, float]:
    """
    Compute hysteretic hydraulic properties.
    
    Parameters
    ----------
    h : float
        Pressure head
    ParD : array, shape (11,)
        Drying parameters
    ParW : array, shape (11,)
        Wetting parameters
    iModel : int
        Model code
    Ah : float
        Head scaling factor
    AK : float
        Conductivity scaling factor
    ATh : float
        Water content scaling factor
    
    Returns
    -------
    K : float
        Hydraulic conductivity
    C : float
        Specific water capacity
    theta : float
        Water content
    """
    h_eff = h / Ah
    
    K = FK(iModel, h_eff, ParD[0, 0]) * AK
    C = FC(iModel, h_eff, ParD[0, 0]) / Ah
    theta = FQ(iModel, h_eff, ParD[0, 0])
    
    return K, C, theta


# ============================================================================
# Lenhard gas relative permeability
# ============================================================================

def lenhard_gas_permeability(
    theta: float,
    theta_r: float,
    theta_s: float,
    alpha: float,
    n: float,
    m: float,
    kappa: int,
) -> float:
    """
    Lenhard model for gas relative permeability.
    
    Parameters
    ----------
    theta : float
        Water content
    theta_r : float
        Residual water content
    theta_s : float
        Saturated water content
    alpha : float
        van Genuchten alpha
    n : float
        van Genuchten n
    m : float
        van Genuchten m (= 1 - 1/n)
    kappa : int
        Curve type (1=drying, -1=wetting)
    
    Returns
    -------
    krg : float
        Gas relative permeability
    """
    Se = max((theta - theta_r) / max(theta_s - theta_r, 1e-10), 0.0)
    Se = min(Se, 1.0)
    
    # Brooks-Corey approximation for gas permeability
    krg = (1.0 - Se) ** (2.0 / 3.0) * (1.0 - Se ** (1.0 / m)) ** (2.0 * m)
    
    return max(krg, 0.0)
