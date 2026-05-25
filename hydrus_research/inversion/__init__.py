"""Inversion (F3) — fit unknown soil parameters to observations.

Backends share one InversionResult schema:
  - lm_scipy   — scipy.optimize.least_squares (fast; ≤ 10 params, 1D)
  - pyemu_ies  — PESTPP-IES Iterative Ensemble Smoother (large D, posterior)
  - pymc_nuts  — Bayesian NUTS (P1; M9)

Dispatch via `fit(...)` with auto backend selection.

See DOCS/superpowers/specs/2026-05-24-hydrus-research-platform-design.md §4.4.
"""
from .base import InversionResult
from .lm_scipy import fit_lm
from .pyemu_pestpp import fit_pyemu
from .api import fit

__all__ = ["InversionResult", "fit_lm", "fit_pyemu", "fit"]
