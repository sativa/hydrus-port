"""Surrogate models — drop-in replacements for any Simulator.

Trained on M3 BatchResult (θ, y_sim) pairs, then any M4/M5/M6 workflow
that consumes the M0 `forward(theta) -> y` callable works on the
surrogate transparently.

Backends:
  - sklearn GP (default; Matérn 5/2; mean + std per prediction)
  - PCK (PC-Kriging via smt KPLS; from [research-3d] extras)

See DOCS/superpowers/specs/2026-05-24-hydrus-research-platform-design.md §4.6.
"""
from .base import SurrogateSimulator, SurrogateModel
from .api import evaluate
from .trainer import train_gp, train_pck

__all__ = ["SurrogateSimulator", "SurrogateModel",
           "train_gp", "train_pck", "evaluate"]
