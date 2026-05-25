"""Uncertainty quantification (F5).

Three independent methods that consume EXISTING artifacts:
  - propagate_ptf_uncertainty — runs forward N times sampling from PTFResult.covariance
  - predict_with_posterior   — reuses M5 InversionResult.posterior_ensemble
  - glue_filter              — filters an M3 BatchResult against obs (no new runs)

See DOCS/superpowers/specs/2026-05-24-hydrus-research-platform-design.md §4.5.
"""
from .result import UQResult
from .monte_carlo import propagate_ptf_uncertainty
from .posterior_predict import predict_with_posterior
from .glue import glue_filter

__all__ = ["UQResult", "propagate_ptf_uncertainty",
           "predict_with_posterior", "glue_filter"]
