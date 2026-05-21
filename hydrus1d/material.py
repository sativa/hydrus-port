"""
Soil hydraulic property models for HYDRUS-1D.
=============================================

Direct Python port of MATERIAL.FOR functions:
- FK(iModel, h, Par)   -> K(h) : hydraulic conductivity
- FC(iModel, h, Par)   -> C(h) : specific water capacity
- FQ(iModel, h, Par)   -> theta(h) : water content
- FH(iModel, Qe, Par)  -> h(theta) : inverse retention
- FS(iModel, h, Par)   -> S(h) : effective saturation
- FKQ(iModel, th, Par) -> K(theta) : conductivity from water content
- FKS(iModel, S, Par)  -> K(S) : conductivity from saturation

Models supported (iModel):
    0  = van Genuchten-Mualem
    1  = Modified van Genuchten (Vogel and Cislerova)
    2  = Brooks and Corey
    3  = van Genuchten with air entry value of 2 cm
    4  = Log-normal (Kosugi)
    5  = Dual-porosity (Durner)
   -1  = Fractal model (Shlomo Orr)

Parameter array Par (11 elements, 0-indexed in Python):
    Par[0]  = theta_r  (residual water content)
    Par[1]  = theta_s  (saturated water content)
    Par[2]  = alpha    (van Genuchten alpha, cm^-1)
    Par[3]  = n        (van Genuchten n)
    Par[4]  = Ks       (saturated hydraulic conductivity)
    Par[5]  = beta     (conductivity exponent)
    Par[6]  = theta_m  (modified VG m parameter)
    Par[7]  = theta_a  (modified VG air content)
    Par[8]  = theta_k  (modified VG knee point content)
    Par[9]  = Kk       (modified VG knee point conductivity)
    Par[10] = h_sat    (saturated pressure head)
"""

from __future__ import annotations
import numpy as np
from numpy.typing import NDArray
from typing import Union

# Minimum conductivity floor (matching Fortran 1.d-37)
K_MIN = 1e-37
# Minimum value floor for capacity/content
VAL_MIN = 1e-37
# Maximum head floor
H_MIN_FLOOR = -1e37
# Numerical limit for h
H_NUM_MIN = -1e300


# ============================================================================
# Core retention and conductivity functions
# ============================================================================

def FK(iModel: int, h: float, Par: NDArray[np.float64]) -> float:
    """
    Hydraulic conductivity K(h).
    
    Direct port of FK function in MATERIAL.FOR.
    
    Parameters
    ----------
    iModel : int
        Model selection code
    h : float
        Pressure head (negative for unsaturated)
    Par : array, shape (11,)
        Soil hydraulic parameters
    
    Returns
    -------
    K : float
        Hydraulic conductivity (always >= K_MIN)
    """
    Qr = Par[0]
    Qs = Par[1]
    Alfa = Par[2]
    n = Par[3]
    Ks = max(Par[4], 1e-37)
    BPar = Par[5]
    
    if iModel in (0, 1, 3):  # VG and modified VG
        PPar = 2
        if iModel in (0, 3):
            Qm = Qs
            Qa = Qr
            Qk = Qs
            Kk = Ks
        elif iModel == 1:
            Qm = Par[6]
            Qa = Par[7]
            Qk = Par[8]
            Kk = Par[9]
        if iModel == 3:
            Qm = Par[6]
        
        m = 1.0 - 1.0 / n
        HMin = max(h, _h_num_min(n, Alfa))
        Qees = min((Qs - Qa) / (Qm - Qa), 0.999999999999999)
        Qeek = min((Qk - Qa) / (Qm - Qa), Qees)
        Hs = -1.0 / Alfa * (Qees ** (-1.0 / m) - 1.0) ** (1.0 / n)
        Hk = -1.0 / Alfa * (Qeek ** (-1.0 / m) - 1.0) ** (1.0 / n)
        
        if h < Hk:
            # m = 1 - 1/n (standard)
            Qee = (1.0 + (-Alfa * HMin) ** n) ** (-m)
            Qe = (Qm - Qa) / (Qs - Qa) * Qee
            Qek = (Qm - Qa) / (Qs - Qa) * Qeek
            FFQ = 1.0 - (1.0 - Qee ** (1.0 / m)) ** m
            FFQk = 1.0 - (1.0 - Qeek ** (1.0 / m)) ** m
            if FFQ <= 0.0:
                FFQ = m * Qee ** (1.0 / m)
            Kr = (Qe / Qek) ** BPar * (FFQ / FFQk) ** PPar * Kk / Ks
            if iModel == 0:
                Kr = Qe ** BPar * (FFQ) ** PPar
            return max(Ks * Kr, K_MIN)
        
        if Hk <= h < Hs:
            Kr = (1.0 - Kk / Ks) / (Hs - Hk) * (h - Hs) + 1.0
            return Ks * Kr
        
        if h >= Hs:
            return Ks
    
    elif iModel == 2:  # Brooks and Corey
        Lambda = 2.0
        Hs = -1.0 / Alfa
        if h < Hs:
            Kr = 1.0 / (-Alfa * h) ** (n * (BPar + Lambda) + 2.0)
            return max(Ks * Kr, K_MIN)
        else:
            return Ks
    
    elif iModel == 4:  # Log-normal (Kosugi)
        Hs = 0.0
        if h < Hs:
            Qee = _qnorm(np.log(-h / Alfa) / n)
            t = _qnorm(np.log(-h / Alfa) / n + n)
            Kr = Qee ** BPar * t * t
            return max(Ks * Kr, K_MIN)
        else:
            return Ks
    
    elif iModel == 5:  # Dual-porosity (Durner)
        w2 = Par[6]
        Alfa2 = Par[7]
        n2 = Par[8]
        m = 1.0 - 1.0 / n
        m2 = 1.0 - 1.0 / n2
        w1 = 1.0 - w2
        Sw1 = w1 * (1.0 + (-Alfa * h) ** n) ** (-m)
        Sw2 = w2 * (1.0 + (-Alfa2 * h) ** n2) ** (-m2)
        Qe = Sw1 + Sw2
        Sv1 = (-Alfa * h) ** (n - 1)
        Sv2 = (-Alfa2 * h) ** (n2 - 1)
        Sk1 = w1 * Alfa * (1.0 - Sv1 * (1.0 + (-Alfa * h) ** n) ** (-m))
        Sk2 = w2 * Alfa2 * (1.0 - Sv2 * (1.0 + (-Alfa2 * h) ** n2) ** (-m2))
        rNumer = Sk1 + Sk2
        rDenom = w1 * Alfa + w2 * Alfa2
        if rDenom != 0.0:
            Kr = Qe ** BPar * (rNumer / rDenom) ** 2
        else:
            Kr = 1.0
        return max(Ks * Kr, K_MIN)
    
    elif iModel == -1:  # Fractal (Shlomo Orr)
        ha = Alfa
        D = n
        Kr = 1.0
        # Simplified - full implementation in original
        return max(Ks * Kr, K_MIN)
    
    return K_MIN


def FC(iModel: int, h: float, Par: NDArray[np.float64]) -> float:
    """
    Specific water capacity C(h) = dtheta/dh.
    
    Direct port of FC function in MATERIAL.FOR.
    """
    Qr = Par[0]
    Qs = Par[1]
    Alfa = Par[2]
    n = Par[3]
    
    if iModel in (0, 1, 3):
        if iModel in (0, 3):
            Qm = Qs
            Qa = Qr
        elif iModel == 1:
            Qm = Par[6]
            Qa = Par[7]
        if iModel == 3:
            Qm = Par[6]
        
        m = 1.0 - 1.0 / n
        HMin = max(h, _h_num_min(n, Alfa))
        Qees = min((Qs - Qa) / (Qm - Qa), 0.999999999999999)
        Hs = -1.0 / Alfa * (Qees ** (-1.0 / m) - 1.0) ** (1.0 / n)
        
        if h < Hs:
            C1 = (1.0 + (-Alfa * HMin) ** n) ** (-m - 1.0)
            C2 = (Qm - Qa) * m * n * (Alfa ** n) * (-HMin) ** (n - 1.0) * C1
            return max(C2, K_MIN)
        else:
            return 0.0
    
    elif iModel == 2:
        Hs = -1.0 / Alfa
        if h < Hs:
            C2 = (Qs - Qr) * n * Alfa ** (-n) * (-h) ** (-n - 1.0)
            return max(C2, K_MIN)
        else:
            return 0.0
    
    elif iModel == 4:
        Hs = 0.0
        if h < Hs:
            t = np.exp(-1.0 * (np.log(-h / Alfa)) ** 2.0 / (2.0 * n ** 2.0))
            C2 = (Qs - Qr) / (2.0 * 3.141592654) ** 0.5 / n / (-h) * t
            return max(C2, K_MIN)
        else:
            return 0.0
    
    elif iModel == 5:
        w2 = Par[6]
        Alfa2 = Par[7]
        n2 = Par[8]
        m = 1.0 - 1.0 / n
        m2 = 1.0 - 1.0 / n2
        C1a = (1.0 + (-Alfa * h) ** n) ** (-m - 1.0)
        C1b = (1.0 + (-Alfa2 * h) ** n2) ** (-m2 - 1.0)
        C2a = (Qs - Qr) * m * n * (Alfa ** n) * (-h) ** (n - 1.0) * C1a * (1.0 - w2)
        C2b = (Qs - Qr) * m2 * n2 * (Alfa2 ** n2) * (-h) ** (n2 - 1.0) * C1b * w2
        return C2a + C2b
    
    elif iModel == -1:
        ha = Alfa
        D = n
        if -h < ha:
            return 0.0
        else:
            C1 = -1.0 * ha ** (3.0 - D) * (D - 3.0) * (-h) ** (D - 4.0)
            return max(C1, K_MIN)
    
    return 0.0


def FQ(iModel: int, h: float, Par: NDArray[np.float64]) -> float:
    """
    Water content theta(h).
    
    Direct port of FQ function in MATERIAL.FOR.
    """
    Qr = Par[0]
    Qs = Par[1]
    Alfa = Par[2]
    n = Par[3]
    
    if iModel in (0, 1, 3):
        if iModel in (0, 3):
            Qm = Qs
            Qa = Qr
        elif iModel == 1:
            Qm = Par[6]
            Qa = Par[7]
        if iModel == 3:
            Qm = Par[6]
        
        m = 1.0 - 1.0 / n
        HMin = max(h, _h_num_min(n, Alfa))
        Qees = min((Qs - Qa) / (Qm - Qa), 0.999999999999999)
        Hs = -1.0 / Alfa * (Qees ** (-1.0 / m) - 1.0) ** (1.0 / n)
        
        if h < Hs:
            Qee = (1.0 + (-Alfa * HMin) ** n) ** (-m)
            return max(Qa + (Qm - Qa) * Qee, K_MIN)
        else:
            return Qs
    
    elif iModel == 2:
        Hs = -1.0 / Alfa
        if h < Hs:
            Qee = (-Alfa * h) ** (-n)
            return max(Qr + (Qs - Qr) * Qee, K_MIN)
        else:
            return Qs
    
    elif iModel == 4:
        Hs = 0.0
        if h < Hs:
            Qee = _qnorm(np.log(-h / Alfa) / n)
            return max(Qr + (Qs - Qr) * Qee, K_MIN)
        else:
            return Qs
    
    elif iModel == 5:
        w2 = Par[6]
        Alfa2 = Par[7]
        n2 = Par[8]
        m = 1.0 - 1.0 / n
        m2 = 1.0 - 1.0 / n2
        w1 = 1.0 - w2
        Sw1 = w1 * (1.0 + (-Alfa * h) ** n) ** (-m)
        Sw2 = w2 * (1.0 + (-Alfa2 * h) ** n2) ** (-m2)
        Qe = Sw1 + Sw2
        return max(Qr + (Qs - Qr) * Qe, K_MIN)
    
    elif iModel == -1:
        ha = Alfa
        D = n
        result = Qs
        if ha > 0.0:
            result = min(max(Qs + (-h / ha) ** (D - 3.0) - 1.0, Qr), Qs)
        return result
    
    return Qr


def FH(iModel: int, Qe: float, Par: NDArray[np.float64]) -> float:
    """
    Inverse retention: h(theta).
    
    Direct port of FH function in MATERIAL.FOR.
    """
    Qr = Par[0]
    Qs = Par[1]
    Alfa = Par[2]
    n = Par[3]
    
    if iModel in (0, 1, 3):
        if iModel in (0, 3):
            Qm = Qs
            Qa = Qr
        elif iModel == 1:
            Qm = Par[6]
            Qa = Par[7]
        if iModel == 3:
            Qm = Par[6]
        
        m = 1.0 - 1.0 / n
        HMin = _h_num_min(n, Alfa)
        QeeM = (1.0 + (-Alfa * HMin) ** n) ** (-m)
        Qee = min(max(Qe * (Qs - Qa) / (Qm - Qa), QeeM), 0.999999999999999)
        return max(-1.0 / Alfa * (Qee ** (-1.0 / m) - 1.0) ** (1.0 / n), H_MIN_FLOOR)
    
    elif iModel == 2:
        return max(-1.0 / Alfa * max(Qe, 1e-10) ** (-1.0 / n), H_MIN_FLOOR)
    
    elif iModel == 4:
        if Qe > 0.9999:
            return 0.0
        elif Qe < 0.00001:
            return -1e8
        else:
            y = Qe * 2.0
            if y < 1.0:
                p = np.sqrt(-np.log(y / 2.0))
            else:
                p = np.sqrt(-np.log(1.0 - y / 2.0))
            x = p - (1.881796 + 0.9425908 * p + 0.0546028 * p ** 3) / \
                (1.0 + 2.356868 * p + 0.3087091 * p ** 2 + 0.0937563 * p ** 3 + 0.021914 * p ** 4)
            if y >= 1.0:
                x = -x
            return -Alfa * np.exp(np.sqrt(2.0) * n * x)
    
    elif iModel == 5:
        # Newton-Raphson inversion for dual porosity
        if Qe > 0.9999:
            return 0.0
        elif Qe < 0.00001:
            return -1e8
        else:
            return _invert_dual_porosity(Qe, Par)
    
    elif iModel == -1:
        ha = Alfa
        D = n
        th = Qr + (Qs - Qr) * Qe
        if D != 3.0:
            return -ha * (1.0 - Qs + th) ** (1.0 / (D - 3.0))
        return 0.0
    
    return 0.0


def FS(iModel: int, h: float, Par: NDArray[np.float64]) -> float:
    """
    Effective saturation S(h).
    
    Direct port of FS function in MATERIAL.FOR.
    """
    Qr = Par[0]
    Qs = Par[1]
    Alfa = Par[2]
    n = Par[3]
    
    if iModel in (0, 1, 3):
        if iModel in (0, 3):
            Qm = Qs
            Qa = Qr
        elif iModel == 1:
            Qm = Par[6]
            Qa = Par[7]
        if iModel == 3:
            Qm = Par[6]
        
        m = 1.0 - 1.0 / n
        Qees = min((Qs - Qa) / (Qm - Qa), 0.999999999999999)
        Hs = -1.0 / Alfa * (Qees ** (-1.0 / m) - 1.0) ** (1.0 / n)
        
        if h < Hs:
            HMin = max(h, _h_num_min(n, Alfa))
            Qee = (1.0 + (-Alfa * HMin) ** n) ** (-m)
            Qe = Qee * (Qm - Qa) / (Qs - Qa)
            return max(Qe, K_MIN)
        else:
            return 1.0
    
    elif iModel == 2:
        Hs = -1.0 / Alfa
        if h < Hs:
            Qe = (-Alfa * h) ** (-n)
            return max(Qe, K_MIN)
        else:
            return 1.0
    
    elif iModel == 4:
        Hs = 0.0
        if h < Hs:
            Qee = _qnorm(np.log(-h / Alfa) / n)
            return max(Qee, K_MIN)
        else:
            return 1.0
    
    elif iModel == 5:
        w2 = Par[6]
        Alfa2 = Par[7]
        n2 = Par[8]
        m = 1.0 - 1.0 / n
        m2 = 1.0 - 1.0 / n2
        w1 = 1.0 - w2
        Sw1 = w1 * (1.0 + (-Alfa * h) ** n) ** (-m)
        Sw2 = w2 * (1.0 + (-Alfa2 * h) ** n2) ** (-m2)
        Qe = Sw1 + Sw2
        return max(Qe, K_MIN)
    
    elif iModel == -1:
        ha = Alfa
        D = n
        if ha > 0.0:
            return max(min(1.0, (Qs + (-h / ha) ** (D - 3.0) - 1.0 - Qr) / (Qs - Qr)), 0.0)
        return 1.0
    
    return 1.0


def FKQ(iModel: int, th: float, Par: NDArray[np.float64]) -> float:
    """
    Hydraulic conductivity from water content K(theta).
    
    Direct port of FKQ function in MATERIAL.FOR.
    """
    Qr = Par[0]
    Qs = Par[1]
    Alfa = Par[2]
    n = Par[3]
    Ks = max(Par[4], 1e-37)
    BPar = Par[5]
    
    if iModel in (0, 1, 3):
        PPar = 2
        if iModel in (0, 3):
            Qm = Qs
            Qa = Qr
            Qk = Qs
            Kk = Ks
        elif iModel == 1:
            Qm = Par[6]
            Qa = Par[7]
            Qk = Par[8]
            Kk = Par[9]
        if iModel == 3:
            Qm = Par[6]
        
        m = 1.0 - 1.0 / n
        Qees = min((Qs - Qa) / (Qm - Qa), 0.999999999999999)
        Qeek = min((Qk - Qa) / (Qm - Qa), Qees)
        
        if th < Qk:
            Qee = (th - Qa) / (Qm - Qa)
            Qe = (Qm - Qa) / (Qs - Qa) * Qee
            Qek = (Qm - Qa) / (Qs - Qa) * Qeek
            FFQ = 1.0 - (1.0 - Qee ** (1.0 / m)) ** m
            FFQk = 1.0 - (1.0 - Qeek ** (1.0 / m)) ** m
            if FFQ <= 0.0:
                FFQ = m * Qee ** (1.0 / m)
            Kr = (Qe / Qek) ** BPar * (FFQ / FFQk) ** PPar * Kk / Ks
            return max(Ks * Kr, K_MIN)
        
        if th >= Qs:
            return Ks
    
    elif iModel == -1:
        D = n
        Kr = 1.0
        Qx = 0.0
        if D != 3.0:
            Qx = Qr + 2.0 * (1.0 - Qs) / (D / (3.0 - D) - 2.0)
        if th > Qx:
            S = th / Qs
            if D != 3.0:
                Kr = (1.0 - Qs * (1.0 - S) / (1.0 - Qr)) ** (D / (3.0 - D))
        else:
            Sx = Qx / Qs
            if D != 3.0:
                Kr = (1.0 - Qs * (1.0 - Sx) / (1.0 - Qr)) ** (D / (3.0 - D))
                if Qx > Qr:
                    Kr = Kr * (th - Qr) ** 2 / (Qx - Qr) ** 2
        return max(Ks * Kr, K_MIN)
    
    return K_MIN


def FKS(iModel: int, S: float, Par: NDArray[np.float64]) -> float:
    """
    Hydraulic conductivity from effective saturation K(S).
    
    Direct port of FKS function in MATERIAL.FOR.
    """
    Qr = Par[0]
    Qs = Par[1]
    Ks = max(Par[4], 1e-37)
    BPar = Par[5]
    
    if iModel in (0, 1, 3):
        PPar = 2
        if iModel in (0, 3):
            Qm = Qs
            Qa = Qr
            Qk = Qs
            Kk = Ks
        elif iModel == 1:
            Qm = Par[6]
            Qa = Par[7]
            Qk = Par[8]
            Kk = Par[9]
        if iModel == 3:
            Qm = Par[6]
        
        m = 1.0 - 1.0 / Par[3]
        Qees = min((Qs - Qa) / (Qm - Qa), 0.999999999999999)
        Qeek = min((Qk - Qa) / (Qm - Qa), Qees)
        th = S * (Qs - Qr) + Qr
        
        if th < Qk:
            Qee = (th - Qa) / (Qm - Qa)
            Qe = (Qm - Qa) / (Qs - Qa) * Qee
            Qek = (Qm - Qa) / (Qs - Qa) * Qeek
            FFQ = 1.0 - (1.0 - Qee ** (1.0 / m)) ** m
            FFQk = 1.0 - (1.0 - Qeek ** (1.0 / m)) ** m
            if FFQ <= 0.0:
                FFQ = m * Qee ** (1.0 / m)
            Kr = (Qe / Qeek) ** BPar * (FFQ / FFQk) ** PPar * Kk / Ks
            return max(Ks * Kr, K_MIN)
        
        if th >= Qs:
            return Ks
    
    elif iModel == -1:
        D = Par[3]
        Kr = 1.0
        if D != 3.0:
            S_val = S
            Kr = (1.0 - Qs * (1.0 - S_val) / (1.0 - Qr)) ** (D / (3.0 - D))
        return max(Ks * Kr, K_MIN)
    
    return K_MIN


# ============================================================================
# Helper functions
# ============================================================================

def _h_num_min(n: float, Alfa: float) -> float:
    """Compute numerical minimum for h to prevent overflow."""
    return -1e300 ** (1.0 / n) / max(Alfa, 1.0)


def _qnorm(x: float) -> float:
    """
    Normal cumulative distribution function.
    
    Approximation matching Fortran qnorm used in Kosugi model.
    Uses Abramowitz and Stegun approximation 7.1.26.
    """
    # Abramowitz and Stegun approximation
    a1 = 0.254829592
    a2 = -0.284496736
    a3 = 1.421413741
    a4 = -1.453152027
    a5 = 1.061405429
    p = 0.3275911
    
    sign = 1 if x >= 0 else -1
    x_abs = abs(x)
    t = 1.0 / (1.0 + p * x_abs)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * np.exp(-x_abs * x_abs / 2.0)
    return 0.5 * (1.0 + sign * y)


def _invert_dual_porosity(Qe: float, Par: NDArray[np.float64]) -> float:
    """
    Invert dual-porosity retention using Newton-Raphson.
    
    Finds h such that FQ(5, h, Par) = Qe.
    """
    Qr = Par[0]
    Qs = Par[1]
    Alfa = Par[2]
    n = Par[3]
    w2 = Par[6]
    Alfa2 = Par[7]
    n2 = Par[8]
    m = 1.0 - 1.0 / n
    m2 = 1.0 - 1.0 / n2
    w1 = 1.0 - w2
    
    # Initial guess
    h = -10.0 / Alfa
    
    for _ in range(50):
        # Evaluate FQ
        Sw1 = w1 * (1.0 + (-Alfa * h) ** n) ** (-m)
        Sw2 = w2 * (1.0 + (-Alfa2 * h) ** n2) ** (-m2)
        Q_calc = Qr + (Qs - Qr) * (Sw1 + Sw2)
        
        # Evaluate FC (derivative)
        C1a = (1.0 + (-Alfa * h) ** n) ** (-m - 1.0)
        C1b = (1.0 + (-Alfa2 * h) ** n2) ** (-m2 - 1.0)
        C2a = (Qs - Qr) * m * n * (Alfa ** n) * (-h) ** (n - 1.0) * C1a * w1
        C2b = (Qs - Qr) * m2 * n2 * (Alfa2 ** n2) * (-h) ** (n2 - 1.0) * C1b * w2
        dQdh = C2a + C2b
        
        residual = Q_calc - Qe
        
        if abs(residual) < 1e-12:
            break
        
        if abs(dQdh) < 1e-30:
            break
        
        h = h - residual / dQdh
        
        # Prevent h from going positive
        if h > 0.0:
            h = -1e-6
    
    return h


# ============================================================================
# Vapor and temperature-dependent functions
# ============================================================================

def Fqv(h: float, Temp: float, xConv: float = 1.0, tConv: float = 1.0) -> float:
    """
    Vapor water content.
    
    Parameters
    ----------
    h : float
        Pressure head
    Temp : float
        Temperature (Celsius)
    xConv : float
        Length conversion factor
    tConv : float
        Time conversion factor
    
    Returns
    -------
    qv : float
        Vapor water content
    """
    # Saturation vapor pressure (Magnus formula, Pa)
    e_s = 610.78 * np.exp(17.2694 * Temp / (237.3 + Temp))
    
    # Vapor pressure in soil
    e = e_s * np.exp(h * xConv * 1000.0 * 0.01 / (461.5 * (Temp + 273.15)))
    
    # Air density
    rho_a = 1.293 * 273.15 / (Temp + 273.15)
    
    # Vapor water content
    qv = e / (461.5 * (Temp + 273.15)) / rho_a * 1000.0
    
    return qv


def FthetaV(h: float, ths: float, Temp: float, xConv: float = 1.0) -> float:
    """
    Vapor equivalent water content.
    
    Parameters
    ----------
    h : float
        Pressure head
    ths : float
        Saturated water content
    Temp : float
        Temperature (Celsius)
    xConv : float
        Length conversion factor
    
    Returns
    -------
    thetaV : float
        Vapor equivalent water content
    """
    if h >= 0.0:
        return 0.0
    
    # Saturation vapor pressure
    e_s = 610.78 * np.exp(17.2694 * Temp / (237.3 + Temp))
    
    # Vapor pressure ratio
    psi = np.exp(h * xConv * 100.0 / (461.5 * (Temp + 273.15)))
    
    # Vapor content in pore space
    thetaV = (1.0 - ths) * (1.0 - psi) * e_s / (461.5 * (Temp + 273.15))
    
    return max(thetaV, 0.0)


# ============================================================================
# Thermal conductivity (Campbell model)
# ============================================================================

def Fkappa(th: float, ParW: NDArray[np.float64]) -> float:
    """
    Thermal conductivity from water content (Campbell model).
    
    Parameters
    ----------
    th : float
        Water content
    ParW : array, shape (11,)
        Thermal parameters
    
    Returns
    -------
    kappa : float
        Thermal conductivity
    """
    theta_wr = ParW[0]
    theta_ws = ParW[1]
    lambda_d = ParW[2]
    delta_l = ParW[4]
    
    if th <= theta_wr:
        return lambda_d
    
    # Campbell's model
    Pf = (th - theta_wr) / (theta_ws - theta_wr)
    kappa = lambda_d + delta_l * np.sqrt(Pf)
    
    return kappa
