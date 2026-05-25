"""BatchResult — aligned arrays of (θ, y_sim, wall_s, converged) for one sweep.

Serializes to parquet via pyarrow. The parquet schema stores:
  - thetas/ys flattened into named columns (theta__alpha, y__obs_name, ...)
  - param_names and obs_names stored in pyarrow schema metadata as JSON

Consumers (M4 sensitivity, M5 inversion, M7 UQ, M8 surrogate) read this
back via `BatchResult.from_parquet(path)` and use the param_names / obs_names
lists to align with their own ParameterMap / ObservationSet schemas."""
from __future__ import annotations
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


@dataclass
class BatchResult:
    thetas: np.ndarray                   # (N, D)
    ys: np.ndarray                       # (N, M)
    wall_s: np.ndarray                   # (N,)
    converged: np.ndarray                # (N,) bool
    param_names: list[str]
    obs_names: list[str]
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.thetas = np.asarray(self.thetas, dtype=float)
        self.ys = np.asarray(self.ys, dtype=float)
        self.wall_s = np.asarray(self.wall_s, dtype=float)
        self.converged = np.asarray(self.converged, dtype=bool)
        N = self.thetas.shape[0]
        if self.thetas.ndim != 2:
            raise ValueError(f"thetas must be 2-D, got shape {self.thetas.shape}")
        if self.ys.shape[0] != N:
            raise ValueError(f"ys.shape[0]={self.ys.shape[0]} != thetas.shape[0]={N}")
        if self.wall_s.shape != (N,):
            raise ValueError(f"wall_s.shape {self.wall_s.shape} != ({N},)")
        if self.converged.shape != (N,):
            raise ValueError(f"converged.shape {self.converged.shape} != ({N},)")
        if len(self.param_names) != self.thetas.shape[1]:
            raise ValueError(f"param_names length {len(self.param_names)} != thetas.shape[1]")
        if len(self.obs_names) != self.ys.shape[1]:
            raise ValueError(f"obs_names length {len(self.obs_names)} != ys.shape[1]")

    @property
    def N(self) -> int:
        return self.thetas.shape[0]

    @property
    def D(self) -> int:
        return self.thetas.shape[1]

    @property
    def M(self) -> int:
        return self.ys.shape[1]

    @property
    def n_converged(self) -> int:
        return int(self.converged.sum())

    @property
    def n_failed(self) -> int:
        return int((~self.converged).sum())

    # ------------------------------------------------------------------ I/O
    def to_parquet(self, path: Path | str) -> None:
        path = Path(path)
        # Flatten thetas + ys into named columns
        cols: dict[str, np.ndarray] = {}
        for j, name in enumerate(self.param_names):
            cols[f"theta__{name}"] = self.thetas[:, j]
        for j, name in enumerate(self.obs_names):
            cols[f"y__{name}"] = self.ys[:, j]
        cols["wall_s"] = self.wall_s
        cols["converged"] = self.converged

        table = pa.Table.from_pydict(cols)
        metadata = {
            b"hydrus_research_batch_meta": json.dumps(self.meta).encode("utf-8"),
            b"param_names": json.dumps(self.param_names).encode("utf-8"),
            b"obs_names": json.dumps(self.obs_names).encode("utf-8"),
        }
        table = table.replace_schema_metadata(metadata)
        pq.write_table(table, path)

    @classmethod
    def from_parquet(cls, path: Path | str) -> "BatchResult":
        table = pq.read_table(Path(path))
        md = table.schema.metadata or {}
        param_names = json.loads(md[b"param_names"].decode("utf-8"))
        obs_names = json.loads(md[b"obs_names"].decode("utf-8"))
        meta = json.loads(md.get(b"hydrus_research_batch_meta", b"{}").decode("utf-8"))
        N = table.num_rows
        thetas = (
            np.column_stack([table[f"theta__{n}"].to_numpy() for n in param_names])
            if param_names
            else np.zeros((N, 0))
        )
        ys = (
            np.column_stack([table[f"y__{n}"].to_numpy() for n in obs_names])
            if obs_names
            else np.zeros((N, 0))
        )
        return cls(
            thetas=thetas,
            ys=ys,
            wall_s=table["wall_s"].to_numpy(),
            converged=table["converged"].to_numpy().astype(bool),
            param_names=param_names,
            obs_names=obs_names,
            meta=meta,
        )
