"""Sensitivity analysis (F2) — four SALib-backed methods sharing one
SensitivityResult schema.

Each method consumes the M0 narrow-waist `forward(theta) → y_sim` callable
and the M3 BatchRunner for parallel evaluation. Outputs are typed and
serializable for REST + GUI consumption.

See DOCS/superpowers/specs/2026-05-24-hydrus-research-platform-design.md §4.3.
"""
from .result import SensitivityResult
from .morris import morris_screen
from .sobol import sobol_decompose
from .fast import fast_indices
from .pawn import pawn_kde

__all__ = ["SensitivityResult",
           "morris_screen", "sobol_decompose", "fast_indices", "pawn_kde"]
