"""Hydrus1DSimulator — wraps hydrus1d.hydrus.run_simulation behind the
Simulator ABC. The adapter receives a fully patched canonical scenario
dict (parameter application happens upstream in make_forward), serialises
it to HYDRUS-1D ASCII files, runs the solver in a temp directory, and
parses outputs into a SimResult."""
from __future__ import annotations
import copy
import shutil
import tempfile
from pathlib import Path
from typing import Any
import numpy as np

from .base import Simulator, Forcing, InitialState, SimResult


class Hydrus1DSimulator(Simulator):
    name = "hydrus1d"
    dimension = 1

    def __init__(self, work_root: Path | str | None = None):
        self.work_root = Path(work_root) if work_root else Path(tempfile.gettempdir()) / "hydrus_research"
        self.work_root.mkdir(parents=True, exist_ok=True)

    def run(self, scenario, forcing, ic):
        raise NotImplementedError("implemented in Task 11")

    def observable_at(self, result, spec):
        raise NotImplementedError("implemented in Task 12")
