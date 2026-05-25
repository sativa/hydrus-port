"""Batch runner (F4) — parallel forward-model evaluation.

Consumes the M0 narrow-waist `forward(theta) → y_sim` callable and stores
(θ, y_sim, wall_s, converged) tuples to parquet for downstream consumers
(M4 sensitivity, M5 inversion, M7 UQ, M8 surrogate training).

See DOCS/superpowers/specs/2026-05-24-hydrus-research-platform-design.md §4.2.
"""
from .result import BatchResult
from .runner import BatchRunner

__all__ = ["BatchResult", "BatchRunner"]
