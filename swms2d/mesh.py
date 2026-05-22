"""
Mesh utilities for SWMS_2D Python port.
=======================================

Derived/cached quantities computed from raw Node + Element data:
    - ListNE: per-node count of incident elements
    - Element areas, shape-function gradients (computed on-the-fly
      during Galerkin assembly, see watflow.py — not stored here)
    - Anisotropy tensor rotation (handled in input.py at read time)
"""

from __future__ import annotations
import numpy as np
from .dataclasses import Mesh


def build_listne(mesh: Mesh) -> None:
    """
    Populate mesh.ListNE: count of elements containing each node.

    Mirrors INPUT2.FOR L385-405. For triangles (KX[e,2] == KX[e,3]),
    only 3 unique corners are counted; for quads, all 4.
    """
    NumNP = mesh.NumNP
    NumEl = mesh.NumEl
    KX = mesh.elements.KX
    listne = np.zeros(NumNP, np.int32)
    for e in range(NumEl):
        # Determine corner count (3 for triangle degenerate quad, 4 for quad)
        ncorn = 3 if KX[e, 2] == KX[e, 3] else 4
        # Each sub-triangle (fan from KX[e,0]) contributes to its 3 nodes
        for sub in range(ncorn - 2):
            i = KX[e, 0]
            j = KX[e, sub + 1]
            k = KX[e, sub + 2]
            listne[i] += 1
            listne[j] += 1
            listne[k] += 1
    mesh.ListNE = listne


def rotate_anisotropy(angle_deg: float, aniz1: float, aniz2: float
                      ) -> tuple[float, float, float]:
    """
    Apply 2D rotation to diagonal anisotropy tensor.

    Input: principal axes Aniz1 (along x'), Aniz2 (along z'), rotated by
    `angle_deg` from global (x, z). Output: tensor components in global
    coordinates (Axx, Azz, Axz).

    Mirrors INPUT2.FOR L368-376.
    """
    ang = np.deg2rad(angle_deg)
    cs, sn = np.cos(ang), np.sin(ang)
    Axx = aniz1 * cs * cs + aniz2 * sn * sn
    Azz = aniz1 * sn * sn + aniz2 * cs * cs
    Axz = (aniz1 - aniz2) * sn * cs
    return Axx, Azz, Axz
