"""
SWMS_2D Python port.
====================

Direct 1:1 port of SWMS_2D v1.22 (Simunek, Vogel, van Genuchten, 1994/1996),
from the original Fortran 77 source in USDA-ARS USSL legacy repository.

Two-stage strategy:
    Stage 1 — 1:1 Fortran port (this package, in current state)
    Stage 2 — scikit-fem based modernized rewrite (future, validated
              against Stage 1 outputs)

License of underlying SWMS_2D: U.S. Public Domain / CC0 International.
This Python port inherits the same status.
"""

__version__ = "0.0.1-skeleton"

# Re-export the main simulation driver once it exists
# from .swms2d import SWMS2DSimulation  # TODO: enable in Phase 2e
