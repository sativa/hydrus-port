"""
Root water uptake for SWMS_2D Python port.
==========================================

Direct port of SINK2.FOR: nodal sink Sink(i) = Alfa(TPot,h) * Beta(i) * rLen * TPot.

Feddes-style piecewise-linear stress function FAlfa(TPot, h, ...) maps
pressure head to a 0-1 root activity factor.
"""

from __future__ import annotations
import numpy as np
from numpy.typing import NDArray

from .dataclasses import Mesh, SoilMaterial


def f_alfa(TPot: float, h: float,
           P0: float, P1: float, P2H: float, P2L: float, P3: float,
           r2H: float, r2L: float) -> float:
    """Stress response (FAlfa in SINK2.FOR L22-33).

    P0 (wet, ~0), P1 (= POptm[material]), P2H/P2L (limits), P3 (wilting).
    Returns 0 outside [P3, P0], peaks at 1 between P1 and P2.
    """
    if TPot < r2L:
        P2 = P2L
    elif TPot > r2H:
        P2 = P2H
    else:
        P2 = P2H + (r2H - TPot) / (r2H - r2L) * (P2L - P2H)

    if P3 < h < P2:
        return (h - P3) / (P2 - P3)
    if P2 <= h <= P1:
        return 1.0
    if P1 < h < P0:
        return (h - P0) / (P1 - P0)
    return 0.0


def set_snk(mesh: Mesh, materials: list[SoilMaterial],
            TPot: float, rLen: float,
            P0: float, POptm: NDArray[np.float64],
            P2H: float, P2L: float, P3: float,
            r2H: float, r2L: float,
            ) -> NDArray[np.float64]:
    """Per-node Sink(i) = Alfa * Beta(i) * rLen * TPot (SINK2.FOR L10-16)."""
    n = mesh.NumNP
    Sink = np.zeros(n, np.float64)
    Beta = mesh.nodes.Beta
    hNew = mesh.nodes.hNew
    MatNum = mesh.nodes.MatNum
    for i in range(n):
        if Beta[i] > 0.0:
            M = MatNum[i] - 1
            alfa = f_alfa(TPot, hNew[i], P0, POptm[M], P2H, P2L, P3, r2H, r2L)
            Sink[i] = alfa * Beta[i] * rLen * TPot
    return Sink


def normalize_beta(mesh: Mesh, KAT: int) -> None:
    """Beta(i) /= integral(Beta dA). SinkIn in INPUT2.FOR L508-541."""
    SBeta = 0.0
    KX = mesh.elements.KX
    x = mesh.nodes.x
    y = mesh.nodes.y
    Beta = mesh.nodes.Beta
    for e in range(mesh.NumEl):
        NUS = 3 if KX[e, 2] == KX[e, 3] else 4
        for k in range(NUS - 2):
            i, j, l = KX[e, 0], KX[e, k + 1], KX[e, k + 2]
            CJ = x[i] - x[l]
            CK = x[j] - x[i]
            BJ = y[l] - y[i]
            BK = y[i] - y[j]
            AE = (CK * BJ - CJ * BK) / 2.0
            xMul = 1.0
            if KAT == 1:
                xMul = 2.0 * 3.1416 * (x[i] + x[j] + x[l]) / 3.0
            BetaE = (Beta[i] + Beta[j] + Beta[l]) / 3.0
            SBeta += xMul * AE * BetaE
    if SBeta > 0.0:
        Beta /= SBeta
