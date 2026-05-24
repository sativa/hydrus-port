"""ObservationSet — aligned arrays of (spec, value, sigma) for use as the
data side of any inversion / sensitivity / UQ workflow."""
from __future__ import annotations
import csv
from pathlib import Path
from typing import Iterable
import numpy as np

from .spec import ObservationSpec


class ObservationSet:
    def __init__(self, specs: list[ObservationSpec],
                 values: np.ndarray, sigmas: np.ndarray):
        values = np.asarray(values, dtype=float)
        sigmas = np.asarray(sigmas, dtype=float)
        if values.shape != (len(specs),):
            raise ValueError(f"values shape {values.shape} mismatch len(specs)={len(specs)}")
        if sigmas.shape != (len(specs),):
            raise ValueError(f"sigmas shape {sigmas.shape} mismatch len(specs)={len(specs)}")
        if np.any(sigmas <= 0):
            raise ValueError("sigmas must be > 0")
        self.specs = list(specs)
        self.values = values
        self.sigmas = sigmas

    @property
    def M(self) -> int:
        return len(self.specs)

    def residuals(self, sim: np.ndarray) -> np.ndarray:
        """(sim - obs) / sigma — what scipy.least_squares wants."""
        sim = np.asarray(sim, dtype=float)
        if sim.shape != self.values.shape:
            raise ValueError(f"sim shape {sim.shape} mismatch obs {self.values.shape}")
        return (sim - self.values) / self.sigmas

    def objective_l2(self, sim: np.ndarray) -> float:
        """sum of squared standardized residuals (weighted by 1/sigma**2)."""
        r = self.residuals(sim)
        return float(np.sum(r * r))

    @classmethod
    def from_csv(cls, path: Path | str) -> "ObservationSet":
        """Columns: name, kind, z_cm (optional), node (optional),
        time_day, value, sigma, weight (optional), species (optional)."""
        path = Path(path)
        specs: list[ObservationSpec] = []
        vals: list[float] = []
        sigs: list[float] = []
        with path.open() as f:
            for row in csv.DictReader(f):
                loc: dict = {}
                if row.get("z_cm"):
                    loc["z_cm"] = float(row["z_cm"])
                if row.get("node"):
                    loc["node"] = int(row["node"])
                if row.get("xyz"):
                    loc["xyz"] = [float(x) for x in row["xyz"].split(";")]
                weight = float(row["weight"]) if row.get("weight") else 1.0
                species = row.get("species") or None
                specs.append(ObservationSpec(
                    name=row["name"], kind=row["kind"],
                    location=loc, time_day=float(row["time_day"]),
                    weight=weight, species=species,
                ))
                vals.append(float(row["value"]))
                sigs.append(float(row["sigma"]))
        return cls(specs=specs, values=np.array(vals), sigmas=np.array(sigs))
