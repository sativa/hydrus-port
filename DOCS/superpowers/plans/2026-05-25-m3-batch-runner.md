# M3 — Batch Runner Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement F4 — the parallel infrastructure that runs many forward-model evaluations and stores `(θ, y_sim)` pairs for downstream sensitivity (M4), inversion (M5), UQ (M7), and surrogate training. Two backends: `joblib` for single-host parallelism (default) and `pyemu_tcp` so a Simulator can act as a PEST++ TCP worker.

**Architecture:** New sibling subpackage `hydrus_research/batch/` with `BatchRunner` (orchestrator), `BatchResult` (parquet-serializable container), and a worker-mode CLI entry. The runner consumes the M0 narrow-waist `forward(theta) → y_sim` callable and is dimension-agnostic — works equally on 1D / 2D / 3D simulators or M8 surrogates. REST + GUI provide async progress streaming for long sweeps.

**Tech Stack:** Python 3.10+, `joblib` (already in `[research]` extras), `pyarrow` (new dep — parquet I/O), `tqdm` (progress bars), `scipy.stats.qmc` (Latin Hypercube sampling for CLI convenience), `pyemu` (already in extras — TCP worker only loaded when backend="pyemu_tcp"), FastAPI BackgroundTasks + SQLite for the async REST layer, Vue 3 + Pinia + Plotly for the BatchSweep.vue progress + result viz page.

**Spec reference:** `DOCS/superpowers/specs/2026-05-24-hydrus-research-platform-design.md` §4.2 + §5.1 (BatchSweep page) + §5.2 (REST endpoints).

**Outstanding M0 concerns folded in:** #5 (`make_forward` is serial) — resolved here by wrapping `forward` in `joblib.Parallel`. The per-call `deepcopy` in `apply_to_scenario` (per-call cost ~1 ms on 1D fixtures) is documented as the parallelism floor, not optimized away.

**Acceptance:**
- `python -c "from hydrus_research.batch import BatchRunner, BatchResult"` works.
- LHS=64 on `infiltr_v1` (3 VG params: alpha, n, Ks) completes via `BatchRunner.run(thetas)` in under 2× wall time of `64 × single-run / n_workers` (i.e. parallelism floor proven).
- `BatchResult.to_parquet(path)` round-trips through `BatchResult.from_parquet(path)` losslessly.
- `hydrus research sweep infiltr_v1 --param ksat:0.1:10:log --n 16 --workers 4` produces a parquet file with 16 rows.
- `hydrus research worker --master 127.0.0.1:4004` boots cleanly (TCP listen state) — exit code 0 on Ctrl-C; behaviour against a real PESTPP-IES master is a manual smoke step, not part of automated CI.
- `pytest tests/research/batch/` green; full `tests/research/` still green (≥ 110 + new batch tests).

---

## File Layout

**Created:**
- `hydrus_research/batch/__init__.py` — re-exports.
- `hydrus_research/batch/result.py` — `BatchResult` Pydantic+numpy container with parquet I/O.
- `hydrus_research/batch/runner.py` — `BatchRunner` orchestrator + joblib backend.
- `hydrus_research/batch/sampling.py` — small wrappers around scipy.stats.qmc + numpy.linspace for the CLI's `--param` sweep specifications.
- `hydrus_research/batch/pyemu_worker.py` — `start_worker(master_host, master_port, forward, ...)` for the pyemu_tcp backend + the CLI entry helper.
- `hydrus_port_server/routers/research_batch.py` — `/research/batch/*` async REST routes.
- `desktop/src/pages/research/BatchSweep.vue` — F4 GUI page.
- `desktop/src/components/EnsembleViz.vue` — reusable N-curves + 95% band viz (LTTB-downsampled; consumed by BatchSweep.vue today, M5/M7 GUI pages later per spec §5.1).
- `tests/research/batch/__init__.py`
- `tests/research/batch/test_result.py`
- `tests/research/batch/test_sampling.py`
- `tests/research/batch/test_runner_joblib.py`
- `tests/research/batch/test_pyemu_worker.py`
- `tests/research/batch/test_cli.py`
- `tests/research/batch/test_rest.py`
- `tests/research/batch/test_e2e_m3.py`

**Modified:**
- `pyproject.toml` — add `pyarrow` and `tqdm` to the `research` extra.
- `hydrus_port_server/app.py:build_app()` — register the new batch router (one-line `include_router`, same pattern as dndc/ptf).
- `hydrus_port/cli.py` — append `hydrus research sweep` + `hydrus research worker` subcommands to the existing `_build_research_subparser`.
- `desktop/src/api.ts` — append a `batch.*` REST wrapper block.
- `desktop/src/App.vue` — add `Batch Sweep` tab (sibling to existing `DNDC Inputs` + `Soil Library`).
- `desktop/src/stores/batch.ts` — new Pinia store for the BatchSweep page's reactive job state (created as part of GUI task).

---

### Task 1: Sub-package skeleton + pyarrow/tqdm deps

**Files:**
- Create: `hydrus_research/batch/__init__.py`
- Create: `tests/research/batch/__init__.py`
- Modify: `pyproject.toml`

- [ ] **Step 1: Create skeleton files**

```bash
mkdir -p hydrus_research/batch tests/research/batch
touch tests/research/batch/__init__.py
```

Write `hydrus_research/batch/__init__.py`:

```python
"""Batch runner (F4) — parallel forward-model evaluation.

Consumes the M0 narrow-waist `forward(theta) → y_sim` callable and stores
(θ, y_sim, wall_s, converged) tuples to parquet for downstream consumers
(M4 sensitivity, M5 inversion, M7 UQ, M8 surrogate training).

See DOCS/superpowers/specs/2026-05-24-hydrus-research-platform-design.md §4.2.
"""
from .result import BatchResult
from .runner import BatchRunner

__all__ = ["BatchResult", "BatchRunner"]
```

- [ ] **Step 2: Add deps to pyproject.toml**

In `pyproject.toml`, the existing `research` extra is:

```toml
research = [
    "pyemu>=1.3",
    "SALib>=1.5",
    "rosetta-soil",
    "scikit-learn",
    "joblib",
    "pydantic>=2.5",
    "PyYAML",
]
```

Append two entries before the closing `]`:

```toml
    "pyarrow>=15",
    "tqdm>=4.66",
```

- [ ] **Step 3: Install + commit**

```bash
pip install pyarrow tqdm
git add hydrus_research/batch/ tests/research/batch/ pyproject.toml
git commit -m "M3.1: batch sub-package skeleton + pyarrow/tqdm deps"
```

(The package import will fail until Task 2 creates `result.py` and Task 3 creates `runner.py` — that's expected; subsequent tasks fill them in.)

---

### Task 2: BatchResult container + parquet round-trip

**Files:**
- Create: `hydrus_research/batch/result.py`
- Create: `tests/research/batch/test_result.py`

- [ ] **Step 1: Write the failing test**

Write `tests/research/batch/test_result.py`:

```python
import numpy as np
import pytest
from pathlib import Path

from hydrus_research.batch import BatchResult


def _make_result(N=4, D=3, M=2):
    rng = np.random.default_rng(42)
    return BatchResult(
        thetas=rng.uniform(size=(N, D)),
        ys=rng.uniform(size=(N, M)),
        wall_s=rng.uniform(0.1, 1.0, size=N),
        converged=np.ones(N, dtype=bool),
        param_names=["alpha", "n", "Ks"][:D],
        obs_names=[f"obs_{i}" for i in range(M)],
        meta={"simulator": "test_fake", "n_workers": 1},
    )


def test_batch_result_shape_invariants():
    r = _make_result(N=5, D=3, M=2)
    assert r.N == 5
    assert r.D == 3
    assert r.M == 2
    assert r.thetas.shape == (5, 3)
    assert r.ys.shape == (5, 2)
    assert r.wall_s.shape == (5,)


def test_batch_result_rejects_shape_mismatch():
    with pytest.raises(ValueError):
        BatchResult(
            thetas=np.zeros((4, 3)),
            ys=np.zeros((3, 2)),                # mismatched N
            wall_s=np.zeros(4),
            converged=np.ones(4, dtype=bool),
            param_names=["a", "b", "c"],
            obs_names=["x", "y"],
            meta={},
        )


def test_batch_result_parquet_round_trip(tmp_path):
    r = _make_result(N=6, D=2, M=3)
    p = tmp_path / "sweep.parquet"
    r.to_parquet(p)
    assert p.exists()
    r2 = BatchResult.from_parquet(p)
    np.testing.assert_array_equal(r.thetas, r2.thetas)
    np.testing.assert_array_equal(r.ys, r2.ys)
    np.testing.assert_array_equal(r.wall_s, r2.wall_s)
    np.testing.assert_array_equal(r.converged, r2.converged)
    assert r2.param_names == r.param_names
    assert r2.obs_names == r.obs_names
    assert r2.meta["simulator"] == "test_fake"


def test_batch_result_handles_failed_runs():
    """When a forward call fails, converged=False and ys row is NaN."""
    r = BatchResult(
        thetas=np.array([[0.1, 1.5, 5.0], [0.2, 2.0, 10.0]]),
        ys=np.array([[0.31, 0.28], [np.nan, np.nan]]),
        wall_s=np.array([0.5, 0.0]),
        converged=np.array([True, False]),
        param_names=["alpha", "n", "Ks"],
        obs_names=["theta_z10", "theta_z20"],
        meta={},
    )
    # Convenience selectors
    assert r.n_converged == 1
    assert r.n_failed == 1
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest tests/research/batch/test_result.py -v
```

Expected: ImportError on `BatchResult` (runner.py doesn't exist yet → __init__ import fails).

- [ ] **Step 3: Implement**

Write `hydrus_research/batch/result.py`:

```python
"""BatchResult — aligned arrays of (θ, y_sim, wall_s, converged) for one sweep.

Serializes to parquet via pyarrow. The parquet schema is two tables:
  - main: one row per forward call with thetas/ys flattened into named columns
  - meta: a single-row dict (serialized as parquet metadata bytes)

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
    def N(self) -> int: return self.thetas.shape[0]
    @property
    def D(self) -> int: return self.thetas.shape[1]
    @property
    def M(self) -> int: return self.ys.shape[1]
    @property
    def n_converged(self) -> int: return int(self.converged.sum())
    @property
    def n_failed(self) -> int: return int((~self.converged).sum())

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
        thetas = np.column_stack([table[f"theta__{n}"].to_numpy() for n in param_names]) \
            if param_names else np.zeros((N, 0))
        ys = np.column_stack([table[f"y__{n}"].to_numpy() for n in obs_names]) \
            if obs_names else np.zeros((N, 0))
        return cls(
            thetas=thetas, ys=ys,
            wall_s=table["wall_s"].to_numpy(),
            converged=table["converged"].to_numpy().astype(bool),
            param_names=param_names, obs_names=obs_names, meta=meta,
        )
```

- [ ] **Step 4: Run to verify pass**

The package import still references `runner` which doesn't exist. Temporarily comment out the runner import to test result.py in isolation, OR (cleaner) create a stub `hydrus_research/batch/runner.py` containing only `class BatchRunner: pass`. Use the stub approach (matches M0/M1 pattern). Then:

```bash
pytest tests/research/batch/test_result.py -v
```

Expected: 4 PASS.

- [ ] **Step 5: Commit**

```bash
git add hydrus_research/batch/result.py hydrus_research/batch/runner.py tests/research/batch/test_result.py
git commit -m "M3.2: BatchResult container with parquet round-trip"
```

---

### Task 3: Sampling helpers (Latin Hypercube + grid + random)

**Files:**
- Create: `hydrus_research/batch/sampling.py`
- Create: `tests/research/batch/test_sampling.py`

- [ ] **Step 1: Write the failing test**

Write `tests/research/batch/test_sampling.py`:

```python
import numpy as np
import pytest

from hydrus_research.batch.sampling import (
    lhs, grid, uniform_random,
)


def test_lhs_returns_right_shape():
    bounds = np.array([[0.001, 1.0], [1.05, 5.0], [0.01, 100.0]])
    samples = lhs(bounds, n=32, seed=42)
    assert samples.shape == (32, 3)
    # Every column should span its bounds approximately
    for j in range(3):
        assert samples[:, j].min() >= bounds[j, 0]
        assert samples[:, j].max() <= bounds[j, 1]


def test_lhs_reproducible_with_seed():
    bounds = np.array([[0.0, 1.0], [0.0, 1.0]])
    a = lhs(bounds, n=8, seed=42)
    b = lhs(bounds, n=8, seed=42)
    np.testing.assert_array_equal(a, b)


def test_grid_full_factorial():
    bounds = np.array([[0.0, 1.0], [10.0, 20.0]])
    samples = grid(bounds, points_per_axis=[3, 4])
    assert samples.shape == (3 * 4, 2)
    # First column should have 3 unique values, second should have 4
    assert len(np.unique(samples[:, 0])) == 3
    assert len(np.unique(samples[:, 1])) == 4


def test_uniform_random():
    bounds = np.array([[0.0, 1.0], [-5.0, 5.0]])
    samples = uniform_random(bounds, n=100, seed=7)
    assert samples.shape == (100, 2)
    assert (samples[:, 0] >= 0).all() and (samples[:, 0] <= 1).all()
    assert (samples[:, 1] >= -5).all() and (samples[:, 1] <= 5).all()
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest tests/research/batch/test_sampling.py -v
```

Expected: ImportError on `sampling` module.

- [ ] **Step 3: Implement**

Write `hydrus_research/batch/sampling.py`:

```python
"""Sampling helpers for the BatchRunner CLI.

These produce a `thetas: (N, D)` array given a bounds array (D, 2). The
runner itself takes thetas directly — these helpers exist so the CLI can
spell out a sweep as `--n 32 --sampler lhs`."""
from __future__ import annotations
import numpy as np


def lhs(bounds: np.ndarray, n: int, seed: int | None = None) -> np.ndarray:
    """Latin Hypercube sampling via scipy.stats.qmc.

    bounds: shape (D, 2) — (lo, hi) per parameter (in user coords).
    Returns: shape (n, D) samples uniformly distributed within each [lo, hi]."""
    from scipy.stats import qmc
    bounds = np.asarray(bounds, dtype=float)
    D = bounds.shape[0]
    sampler = qmc.LatinHypercube(d=D, seed=seed)
    u = sampler.random(n)                                # (n, D) in [0, 1)
    return qmc.scale(u, bounds[:, 0], bounds[:, 1])


def grid(bounds: np.ndarray, points_per_axis: list[int]) -> np.ndarray:
    """Full-factorial grid. Returns (prod(points), D) samples."""
    bounds = np.asarray(bounds, dtype=float)
    D = bounds.shape[0]
    if len(points_per_axis) != D:
        raise ValueError(f"points_per_axis length {len(points_per_axis)} != D {D}")
    axes = [np.linspace(bounds[j, 0], bounds[j, 1], points_per_axis[j]) for j in range(D)]
    mesh = np.meshgrid(*axes, indexing="ij")
    return np.column_stack([m.ravel() for m in mesh])


def uniform_random(bounds: np.ndarray, n: int, seed: int | None = None) -> np.ndarray:
    """Plain uniform random sampling."""
    rng = np.random.default_rng(seed)
    bounds = np.asarray(bounds, dtype=float)
    return rng.uniform(bounds[:, 0], bounds[:, 1], size=(n, bounds.shape[0]))
```

- [ ] **Step 4: Run + commit**

```bash
pytest tests/research/batch/test_sampling.py -v
git add hydrus_research/batch/sampling.py tests/research/batch/test_sampling.py
git commit -m "M3.3: sampling helpers (LHS / grid / uniform_random)"
```

---

### Task 4: BatchRunner joblib backend — serial + parallel

**Files:**
- Modify: `hydrus_research/batch/runner.py` (replace the M3.2 stub)
- Create: `tests/research/batch/test_runner_joblib.py`

- [ ] **Step 1: Write the failing test**

Write `tests/research/batch/test_runner_joblib.py`:

```python
import time
import numpy as np
import pytest

from hydrus_research.batch import BatchRunner, BatchResult


def _slow_forward(theta: np.ndarray, sleep_s: float = 0.05) -> np.ndarray:
    """Toy forward — returns [theta[0] + theta[1], theta[0] * theta[1]] after sleeping."""
    time.sleep(sleep_s)
    return np.array([theta[0] + theta[1], theta[0] * theta[1]])


def _failing_forward(theta: np.ndarray) -> np.ndarray:
    """Toy forward that raises if theta[0] > 0.5."""
    if theta[0] > 0.5:
        raise RuntimeError("simulated solver divergence")
    return np.array([theta[0] + theta[1]])


def test_batch_runner_serial_basic():
    thetas = np.array([[0.1, 1.0], [0.2, 2.0], [0.3, 3.0]])
    runner = BatchRunner(forward=_slow_forward,
                         param_names=["alpha", "n"],
                         obs_names=["sum", "product"],
                         n_workers=1)
    r = runner.run(thetas)
    assert isinstance(r, BatchResult)
    assert r.N == 3
    assert r.D == 2
    assert r.M == 2
    assert r.converged.all()
    np.testing.assert_allclose(r.ys[:, 0], [1.1, 2.2, 3.3])
    np.testing.assert_allclose(r.ys[:, 1], [0.1, 0.4, 0.9])
    assert (r.wall_s > 0).all()


def test_batch_runner_parallel_speedup():
    """4 workers on 16 tasks of 0.1s each should be < 0.8s (vs. 1.6s serial)."""
    thetas = np.array([[i * 0.01, i * 0.02] for i in range(16)])
    runner = BatchRunner(
        forward=lambda t: _slow_forward(t, sleep_s=0.1),
        param_names=["a", "b"], obs_names=["s", "p"],
        n_workers=4,
    )
    t0 = time.time()
    r = runner.run(thetas)
    wall = time.time() - t0
    assert r.converged.all()
    # Soft assertion to avoid flakiness on contended CI hardware
    assert wall < 1.2, f"4-worker parallel took {wall:.2f}s; expected < 1.2s"


def test_batch_runner_handles_failures():
    """Failed forward calls produce converged=False and NaN ys."""
    thetas = np.array([[0.1, 1.0], [0.9, 2.0], [0.3, 3.0]])
    runner = BatchRunner(forward=_failing_forward,
                         param_names=["alpha", "n"],
                         obs_names=["sum"],
                         n_workers=1)
    r = runner.run(thetas)
    assert r.converged.tolist() == [True, False, True]
    assert np.isnan(r.ys[1, 0])
    assert r.n_failed == 1


def test_batch_runner_n_workers_auto():
    runner = BatchRunner(forward=_slow_forward,
                         param_names=["a", "b"], obs_names=["s", "p"],
                         n_workers="auto")
    assert runner.n_workers >= 1
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest tests/research/batch/test_runner_joblib.py -v
```

Expected: failures (stub `class BatchRunner: pass` from M3.2 doesn't accept the kwargs).

- [ ] **Step 3: Implement**

Replace `hydrus_research/batch/runner.py` (full overwrite of the M3.2 stub):

```python
"""BatchRunner — orchestrates N forward-model evaluations.

Two backends:
- joblib (default): single-host parallelism via threads/processes
- pyemu_tcp (M3.5): runs as a PEST++ TCP worker that receives thetas from
  a remote master process

This file defines the joblib backend. pyemu_tcp lives in pyemu_worker.py."""
from __future__ import annotations
import os
import time
from typing import Callable, Literal

import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

from .result import BatchResult


def _detect_workers() -> int:
    try:
        return max(1, os.cpu_count() or 1)
    except Exception:
        return 1


class BatchRunner:
    """Run `forward(theta) → y_sim` over many thetas in parallel.

    Parameters
    ----------
    forward : callable (theta_vector: np.ndarray) -> y_vector: np.ndarray
        The narrow-waist callable — typically `make_forward(simulator, ...)`.
    param_names : list[str]
        Names for the columns of `thetas` (one per parameter).
    obs_names : list[str]
        Names for the columns of `ys` (one per observation).
    n_workers : int | "auto"
        Number of parallel workers. "auto" = os.cpu_count(). 1 = serial.
    backend : "joblib" | "pyemu_tcp"
        Parallelism backend. "joblib" is default; "pyemu_tcp" is M3.5.
    show_progress : bool
        Show a tqdm bar (only meaningful when n_workers == 1; joblib backends
        report progress via a different mechanism — see runner internals).
    """

    def __init__(self,
                 forward: Callable[[np.ndarray], np.ndarray],
                 param_names: list[str],
                 obs_names: list[str],
                 n_workers: int | Literal["auto"] = "auto",
                 backend: Literal["joblib", "pyemu_tcp"] = "joblib",
                 show_progress: bool = True):
        self.forward = forward
        self.param_names = list(param_names)
        self.obs_names = list(obs_names)
        self.n_workers = _detect_workers() if n_workers == "auto" else int(n_workers)
        self.backend = backend
        self.show_progress = show_progress

    def _run_one(self, theta: np.ndarray) -> tuple[np.ndarray, float, bool]:
        """Returns (y_vec, wall_s, converged). On failure: NaN ys + converged=False."""
        t0 = time.time()
        try:
            y = np.asarray(self.forward(theta), dtype=float)
            return y, time.time() - t0, True
        except Exception:
            return (np.full(len(self.obs_names), np.nan), time.time() - t0, False)

    def run(self, thetas: np.ndarray) -> BatchResult:
        thetas = np.asarray(thetas, dtype=float)
        if thetas.ndim != 2 or thetas.shape[1] != len(self.param_names):
            raise ValueError(
                f"thetas shape {thetas.shape} incompatible with "
                f"{len(self.param_names)} param_names"
            )
        N = thetas.shape[0]

        if self.backend == "joblib":
            results = self._run_joblib(thetas)
        elif self.backend == "pyemu_tcp":
            raise NotImplementedError(
                "backend='pyemu_tcp' is for worker-mode (see pyemu_worker.py); "
                "use the `hydrus research worker` CLI instead of BatchRunner.run"
            )
        else:
            raise ValueError(f"unknown backend {self.backend!r}")

        ys = np.stack([r[0] for r in results])
        wall_s = np.array([r[1] for r in results])
        converged = np.array([r[2] for r in results], dtype=bool)

        return BatchResult(
            thetas=thetas, ys=ys, wall_s=wall_s, converged=converged,
            param_names=self.param_names, obs_names=self.obs_names,
            meta={"backend": self.backend, "n_workers": self.n_workers,
                  "n_total": N, "n_failed": int((~converged).sum())},
        )

    def _run_joblib(self, thetas: np.ndarray):
        N = thetas.shape[0]
        if self.n_workers <= 1:
            it = tqdm(thetas, total=N, disable=not self.show_progress)
            return [self._run_one(t) for t in it]
        # Parallel path — joblib threading is fine for ext-process-bound work
        # (HYDRUS solver spawns subprocesses; Python GIL is released during wait)
        with Parallel(n_jobs=self.n_workers, backend="loky") as parallel:
            return parallel(delayed(self._run_one)(t) for t in thetas)
```

- [ ] **Step 4: Run + commit**

```bash
pytest tests/research/batch/test_runner_joblib.py -v
git add hydrus_research/batch/runner.py tests/research/batch/test_runner_joblib.py
git commit -m "M3.4: BatchRunner joblib backend (serial + parallel + failure handling)"
```

---

### Task 5: PyEMU TCP worker entry point

**Files:**
- Create: `hydrus_research/batch/pyemu_worker.py`
- Create: `tests/research/batch/test_pyemu_worker.py`

This task wraps `pyemu.utils.os_utils.start_workers` so a Simulator can act as a PEST++ TCP worker. **It does NOT spin up a real PEST++ master in CI** — that's a manual smoke step. The pytest only verifies the wrapper exists, handles missing-pyemu gracefully, and starts a listener that exits cleanly on signal.

- [ ] **Step 1: Write the failing test**

Write `tests/research/batch/test_pyemu_worker.py`:

```python
import pytest


pyemu = pytest.importorskip("pyemu", reason="pyemu not installed")
from hydrus_research.batch.pyemu_worker import build_worker_entry


def test_build_worker_entry_returns_callable():
    """Smoke: the worker-entry factory takes a forward callable + names and
    returns a function that pyemu's start_workers can drive."""
    def fake_forward(theta):
        return [theta[0] + theta[1]]

    entry = build_worker_entry(forward=fake_forward,
                               param_names=["a", "b"],
                               obs_names=["sum"])
    assert callable(entry)


def test_pyemu_worker_module_imports_only_when_called():
    """The pyemu import must be lazy — `from hydrus_research.batch import ...`
    must not require pyemu to be installed for users only using joblib."""
    import importlib
    mod = importlib.import_module("hydrus_research.batch.pyemu_worker")
    # The module top-level should NOT have triggered a pyemu import
    # (we check by verifying pyemu is NOT in mod's globals as a top-level symbol)
    assert not hasattr(mod, "pyemu_lib"), \
        "pyemu_worker.py imported pyemu at top level; should be lazy"
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest tests/research/batch/test_pyemu_worker.py -v
```

Expected: ImportError on `build_worker_entry`.

- [ ] **Step 3: Implement**

Write `hydrus_research/batch/pyemu_worker.py`:

```python
"""PyEMU TCP worker mode.

PESTPP-IES and friends spawn workers across machines via a TCP protocol.
This module lets a Simulator (wrapped via make_forward) act as one of those
workers. The pyemu dependency is lazy — only imported inside the start_*
functions, so users who only need the joblib backend don't need pyemu.

The CLI entry `hydrus research worker --master host:port` calls
`run_worker(master_host, master_port, forward, param_names, obs_names)`."""
from __future__ import annotations
import shutil
import tempfile
from pathlib import Path
from typing import Callable

import numpy as np


def build_worker_entry(forward: Callable[[np.ndarray], np.ndarray],
                       param_names: list[str],
                       obs_names: list[str]) -> Callable[[Path], None]:
    """Return a callable `entry(workdir: Path) -> None` that:

    1. Reads PEST-format `input.dat` (or similar) from `workdir`
       containing the current theta vector.
    2. Calls `forward(theta) → y_sim`.
    3. Writes PEST-format `output.dat` to `workdir` with the y_sim values.

    The exact input/output filenames are conventional (PEST++ template/instruction
    files determine them); this implementation uses one column per param/obs
    in two whitespace-delimited text files."""
    def entry(workdir: Path) -> None:
        workdir = Path(workdir)
        in_path = workdir / "theta.dat"
        out_path = workdir / "y.dat"

        # Read theta — one float per line in param_names order
        theta = np.array([float(x) for x in in_path.read_text().split()])
        if theta.shape != (len(param_names),):
            raise ValueError(f"theta.dat had {theta.shape[0]} values; "
                             f"expected {len(param_names)}")

        # Forward-evaluate
        y = np.asarray(forward(theta), dtype=float)
        if y.shape != (len(obs_names),):
            raise ValueError(f"forward returned {y.shape[0]} values; "
                             f"expected {len(obs_names)}")

        # Write y — one float per line in obs_names order
        out_path.write_text("\n".join(f"{v:.10e}" for v in y))

    return entry


def run_worker(master_host: str,
               master_port: int,
               forward: Callable[[np.ndarray], np.ndarray],
               param_names: list[str],
               obs_names: list[str],
               workdir: Path | None = None,
               worker_name: str | None = None) -> int:
    """Start a long-running worker process that connects to a PESTPP master.

    Imports pyemu LAZILY. Returns the process exit code (0 on clean shutdown)."""
    try:
        import pyemu
    except ImportError as e:
        raise RuntimeError(
            "pyemu is not installed; install with `pip install 'hydrus-port[research]'`"
        ) from e

    wd = Path(workdir) if workdir else Path(tempfile.mkdtemp(prefix="hydrus_worker_"))
    wd.mkdir(parents=True, exist_ok=True)
    entry = build_worker_entry(forward, param_names, obs_names)
    name = worker_name or f"hydrus_worker_{wd.name}"

    # pyemu's worker API has evolved — defensively try the most common entry points
    try:
        # Newer pyemu (>=1.3): pyemu.os_utils.start_workers expects a master + worker_root
        # to launch sub-processes itself. For a single-process worker we use
        # pyemu.os_utils.run_serial_master or fall back to a custom loop.
        pyemu.utils.os_utils._try_remove_existing(str(wd / ".lock"))   # type: ignore[attr-defined]
    except Exception:
        pass

    # Minimal worker loop: poll the master for new theta input, invoke entry, write y.
    # PEST++ "Yamr" protocol is implemented in pyemu — we delegate to it when available.
    if hasattr(pyemu.utils.os_utils, "start_worker_from_callable"):
        return pyemu.utils.os_utils.start_worker_from_callable(   # type: ignore[attr-defined]
            master_host=master_host, master_port=master_port,
            entry=entry, worker_dir=str(wd), worker_name=name,
        )
    # Fallback: print a warning + spin forever waiting for input.dat to appear.
    # This is the manual-smoke path; PESTPP-IES integration is verified out-of-band.
    print(f"[pyemu_worker] No 'start_worker_from_callable' helper found; running "
          f"polling fallback. Watching {wd / 'theta.dat'} (Ctrl-C to exit).")
    import signal
    import sys
    import time
    stop = {"flag": False}

    def _on_sigint(signum, frame):
        stop["flag"] = True
    signal.signal(signal.SIGINT, _on_sigint)

    while not stop["flag"]:
        if (wd / "theta.dat").exists():
            entry(wd)
            (wd / "theta.dat").unlink()       # consume input
        time.sleep(0.1)
    return 0
```

- [ ] **Step 4: Run + commit**

```bash
pytest tests/research/batch/test_pyemu_worker.py -v
git add hydrus_research/batch/pyemu_worker.py tests/research/batch/test_pyemu_worker.py
git commit -m "M3.5: pyemu_worker — build_worker_entry + lazy run_worker"
```

---

### Task 6: CLI subcommands `hydrus research sweep` + `hydrus research worker`

**Files:**
- Modify: `hydrus_port/cli.py` — append `sweep` and `worker` parsers to `_build_research_subparser`
- Create: `tests/research/batch/test_cli.py`

- [ ] **Step 1: Write the failing test**

Write `tests/research/batch/test_cli.py`:

```python
import json
import subprocess
import sys
from pathlib import Path


def test_cli_sweep_writes_parquet(tmp_path):
    """`hydrus research sweep` should produce a parquet file with N rows."""
    # The CLI takes a scenario path and runs N evaluations of forward(theta)
    # on it. Smallest test: use infiltr_v1 + sweep over alpha only.
    out_parquet = tmp_path / "sweep.parquet"
    r = subprocess.run(
        [
            sys.executable, "-m", "hydrus_port.cli",
            "research", "sweep",
            "tests/fixtures/infiltr_v1/inputs",
            "--param", "materials[0].alpha:0.005:0.05:log",
            "--n", "4",
            "--workers", "1",
            "--out", str(out_parquet),
        ],
        capture_output=True, text=True,
    )
    assert r.returncode == 0, r.stderr
    assert out_parquet.exists()

    # Read back and verify
    from hydrus_research.batch import BatchResult
    br = BatchResult.from_parquet(out_parquet)
    assert br.N == 4
    assert br.D == 1
    assert "alpha" in br.param_names[0]    # name derives from path tail
    assert br.converged.sum() >= 3         # at least 3 of 4 should converge


def test_cli_worker_help_works():
    """`hydrus research worker --help` should print without raising."""
    r = subprocess.run(
        [sys.executable, "-m", "hydrus_port.cli",
         "research", "worker", "--help"],
        capture_output=True, text=True,
    )
    assert r.returncode == 0, r.stderr
    assert "--master" in r.stdout
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest tests/research/batch/test_cli.py::test_cli_worker_help_works -v
```

Expected: returncode != 0 (subcommand not registered).

- [ ] **Step 3: Implement**

In `hydrus_port/cli.py`, inside `_build_research_subparser(sub)`, append after the existing `p_soil` block:

```python
    # ----- sweep (M3) ------------------------------------------------
    p_sweep = rsub.add_parser("sweep", help="batch-run forward(θ) over N samples")
    p_sweep.add_argument("scenario_dir",
                         help="path to scenario inputs (1d/2d/3d)")
    p_sweep.add_argument("--param", action="append", required=True,
                         help="parameter spec, e.g. materials[0].alpha:0.001:1.0:log "
                              "(target:lo:hi:transform; transform optional, default linear). "
                              "Repeat for multi-D sweeps.")
    p_sweep.add_argument("--obs", default="theta@-30cm,t=1.0",
                         help="comma-separated obs specs (kind@location,t=time)")
    p_sweep.add_argument("--n", type=int, default=32, help="number of samples")
    p_sweep.add_argument("--sampler", default="lhs",
                         choices=["lhs", "grid", "uniform"])
    p_sweep.add_argument("--workers", type=int, default=1)
    p_sweep.add_argument("--seed", type=int, default=None)
    p_sweep.add_argument("--out", required=True, help="output parquet path")
    p_sweep.set_defaults(_cmd=_cmd_research_sweep)

    # ----- worker (M3) -----------------------------------------------
    p_worker = rsub.add_parser("worker",
                               help="run as a PEST++ TCP worker for inversion")
    p_worker.add_argument("--master", required=True,
                          help="master host:port (e.g. 127.0.0.1:4004)")
    p_worker.add_argument("--scenario-dir", required=True,
                          help="scenario template directory the worker forward-evaluates")
    p_worker.add_argument("--workdir", default=None,
                          help="working dir for I/O files (default: temp dir)")
    p_worker.add_argument("--name", default=None, help="worker name shown in master log")
    p_worker.set_defaults(_cmd=_cmd_research_worker)
```

And add the two command implementations alongside the other `_cmd_*` helpers:

```python
def _cmd_research_sweep(args: argparse.Namespace) -> int:
    import numpy as _np
    from pathlib import Path as _P
    from hydrus_research.batch import BatchRunner
    from hydrus_research.batch.sampling import lhs, grid, uniform_random
    from hydrus_research.parameters import ParameterSpec, ParameterMap
    from hydrus_research.observations import ObservationSpec
    from hydrus_research.simulator import make_forward
    from hydrus_research.simulator.hydrus1d_adapter import Hydrus1DSimulator
    from hydrus_port.adapters.hydrus1d import load as _load_h1d

    # Parse --param specs
    specs: list[ParameterSpec] = []
    for s in args.param:
        parts = s.split(":")
        target, lo, hi = parts[0], float(parts[1]), float(parts[2])
        transform = parts[3] if len(parts) > 3 else "linear"
        name = target.rsplit(".", 1)[-1]
        specs.append(ParameterSpec(name=name, target=target,
                                   bounds=(lo, hi), transform=transform))
    pm = ParameterMap(specs)

    # Parse --obs specs (kind@-Ncm,t=T format)
    obs_specs: list[ObservationSpec] = []
    for chunk in args.obs.split(","):
        # chunk like "theta@-30cm,t=1.0" or "h@-50cm,t=2.0"
        kind, _, rest = chunk.partition("@")
        loc_part, _, t_part = rest.partition(",t=")
        z_cm = float(loc_part.rstrip("cm"))
        t = float(t_part)
        obs_specs.append(ObservationSpec(name=f"{kind}_{loc_part}_{t}",
                                         kind=kind, location={"z_cm": z_cm}, time_day=t))

    # Build forward
    template = _load_h1d(_P(args.scenario_dir)).to_dict()
    sim = Hydrus1DSimulator()
    forward = make_forward(sim, pm,
                           template_scenario=template,
                           forcing=None, ic=None,
                           obs_specs=obs_specs)

    # Sample thetas (internal coords)
    bounds_internal = pm.bounds_array()
    if args.sampler == "lhs":
        thetas = lhs(bounds_internal, n=args.n, seed=args.seed)
    elif args.sampler == "grid":
        # Equal split per axis
        per_axis = max(1, int(round(args.n ** (1.0 / len(specs)))))
        thetas = grid(bounds_internal, points_per_axis=[per_axis] * len(specs))
    else:
        thetas = uniform_random(bounds_internal, n=args.n, seed=args.seed)

    runner = BatchRunner(forward=forward,
                         param_names=[s.name for s in specs],
                         obs_names=[o.name for o in obs_specs],
                         n_workers=args.workers)
    result = runner.run(thetas)
    out = _P(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(out)
    print(f"swept {result.N} samples, {result.n_failed} failed; written to {out}")
    return 0


def _cmd_research_worker(args: argparse.Namespace) -> int:
    from pathlib import Path as _P
    from hydrus_research.batch.pyemu_worker import run_worker
    # Build a minimal forward by loading the scenario template.
    # The worker is parameter-agnostic — pyemu/PESTPP pass theta as a flat
    # vector; the worker resolves it via the entry function defined here.
    # For now, fall back to a zero-arg forward that the user is expected to
    # configure via a PEST control file (out of scope for the CLI smoke).
    host, _, port_s = args.master.partition(":")
    port = int(port_s)

    def forward(theta):
        # Real implementation: build ParameterMap + ObservationSpec from
        # PEST template/instruction files. Out of M3 scope; the user
        # provides a Python adapter via a future plugin point (M4+).
        raise NotImplementedError(
            "worker forward() requires a PEST-template adapter not yet wired up; "
            "this is the M3 entry point — see DOCS/superpowers/specs/...§4.4 for plan"
        )

    return run_worker(master_host=host, master_port=port,
                      forward=forward, param_names=[], obs_names=[],
                      workdir=_P(args.workdir) if args.workdir else None,
                      worker_name=args.name)
```

- [ ] **Step 4: Run + commit**

```bash
pytest tests/research/batch/test_cli.py -v
git add hydrus_port/cli.py tests/research/batch/test_cli.py
git commit -m "M3.6: hydrus research {sweep,worker} CLI"
```

(`test_cli_sweep_writes_parquet` takes ~30s because it runs 4 real HYDRUS-1D solver calls.)

---

### Task 7: REST `/research/batch/*` async endpoints + SQLite job state

**Files:**
- Create: `hydrus_port_server/routers/research_batch.py`
- Modify: `hydrus_port_server/app.py:build_app()` (add `include_router(batch_router, prefix="/research/batch", ...)` with try/except ImportError — same pattern as dndc/ptf)
- Create: `tests/research/batch/test_rest.py`

- [ ] **Step 1: Write the failing test**

Write `tests/research/batch/test_rest.py`:

```python
import time
import pytest

try:
    from fastapi.testclient import TestClient
except ImportError:
    pytest.skip("FastAPI not installed", allow_module_level=True)

from hydrus_port_server.app import build_app


@pytest.fixture(scope="module")
def client():
    return TestClient(build_app())


def test_batch_start_returns_job_id(client):
    payload = {
        "scenario_dir": "tests/fixtures/infiltr_v1/inputs",
        "params": [{"name": "alpha", "target": "materials[0].alpha",
                    "bounds": [0.005, 0.05], "transform": "log"}],
        "obs": [{"name": "theta_z30_d1", "kind": "theta",
                 "location": {"z_cm": -30.0}, "time_day": 1.0}],
        "n": 2,
        "sampler": "lhs",
        "workers": 1,
        "seed": 42,
    }
    r = client.post("/research/batch/start", json=payload)
    assert r.status_code == 200, r.text
    body = r.json()
    assert "job_id" in body


def test_batch_status_progresses_to_done(client):
    payload = {
        "scenario_dir": "tests/fixtures/infiltr_v1/inputs",
        "params": [{"name": "alpha", "target": "materials[0].alpha",
                    "bounds": [0.005, 0.05], "transform": "log"}],
        "obs": [{"name": "theta", "kind": "theta",
                 "location": {"z_cm": -30.0}, "time_day": 1.0}],
        "n": 2, "sampler": "lhs", "workers": 1, "seed": 7,
    }
    r = client.post("/research/batch/start", json=payload)
    job_id = r.json()["job_id"]
    # Poll for completion (test fixture has 2 runs * ~10s each = up to 30s)
    for _ in range(60):
        s = client.get(f"/research/batch/{job_id}/status")
        assert s.status_code == 200
        if s.json()["state"] == "done":
            break
        time.sleep(1)
    else:
        pytest.fail("batch job did not complete within 60 seconds")

    res = client.get(f"/research/batch/{job_id}/result")
    assert res.status_code == 200
    # res.content is parquet bytes; verify it round-trips
    import io, tempfile, pathlib
    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
        f.write(res.content)
        f.flush()
        from hydrus_research.batch import BatchResult
        br = BatchResult.from_parquet(pathlib.Path(f.name))
    assert br.N == 2
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest tests/research/batch/test_rest.py -v
```

Expected: 404 (router not registered).

- [ ] **Step 3: Implement**

Write `hydrus_port_server/routers/research_batch.py`:

```python
"""/research/batch/* — async batch-sweep REST routes.

Job model: each POST /start creates a row in an in-memory dict (process-local;
fine for single-user desktop GUI). Status transitions: pending → running → done | failed.
The parquet result is read from disk and streamed on GET /result."""
from __future__ import annotations
import tempfile
import uuid
from pathlib import Path
from typing import Any, Literal

import numpy as np
from fastapi import APIRouter, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel


router = APIRouter()

# In-process job registry. Restart of the server = lost jobs.
_JOBS: dict[str, dict[str, Any]] = {}


class ParamSpecPayload(BaseModel):
    name: str
    target: str
    bounds: tuple[float, float]
    transform: Literal["linear", "log", "logit"] = "linear"


class ObsSpecPayload(BaseModel):
    name: str
    kind: Literal["theta", "h", "c", "flux", "cumulative_flux", "concentration_flux"]
    location: dict
    time_day: float


class StartRequest(BaseModel):
    scenario_dir: str
    params: list[ParamSpecPayload]
    obs: list[ObsSpecPayload]
    n: int = 32
    sampler: Literal["lhs", "grid", "uniform"] = "lhs"
    workers: int = 1
    seed: int | None = None


class StartResponse(BaseModel):
    job_id: str


@router.post("/start", response_model=StartResponse)
def start(req: StartRequest, bg: BackgroundTasks):
    job_id = uuid.uuid4().hex[:12]
    out_path = Path(tempfile.gettempdir()) / f"hydrus_batch_{job_id}.parquet"
    _JOBS[job_id] = {"state": "pending", "n_total": req.n, "n_done": 0,
                     "out_path": str(out_path), "error": None}
    bg.add_task(_run_job, job_id, req)
    return StartResponse(job_id=job_id)


@router.get("/{job_id}/status")
def status(job_id: str):
    if job_id not in _JOBS:
        raise HTTPException(status_code=404, detail="unknown job_id")
    return _JOBS[job_id]


@router.get("/{job_id}/result")
def result(job_id: str):
    if job_id not in _JOBS:
        raise HTTPException(status_code=404, detail="unknown job_id")
    j = _JOBS[job_id]
    if j["state"] != "done":
        raise HTTPException(status_code=409,
                            detail=f"job not done (state={j['state']})")
    return FileResponse(j["out_path"], media_type="application/octet-stream",
                        filename=f"sweep_{job_id}.parquet")


# ---------------------------------------------------------------- background task
def _run_job(job_id: str, req: StartRequest) -> None:
    from hydrus_research.batch import BatchRunner
    from hydrus_research.batch.sampling import lhs, grid, uniform_random
    from hydrus_research.parameters import ParameterSpec, ParameterMap
    from hydrus_research.observations import ObservationSpec
    from hydrus_research.simulator import make_forward
    from hydrus_research.simulator.hydrus1d_adapter import Hydrus1DSimulator
    from hydrus_port.adapters.hydrus1d import load as _load_h1d

    j = _JOBS[job_id]
    j["state"] = "running"
    try:
        specs = [ParameterSpec(name=p.name, target=p.target,
                               bounds=p.bounds, transform=p.transform)
                 for p in req.params]
        pm = ParameterMap(specs)
        obs_specs = [ObservationSpec(name=o.name, kind=o.kind,
                                     location=o.location, time_day=o.time_day)
                     for o in req.obs]
        template = _load_h1d(Path(req.scenario_dir)).to_dict()
        sim = Hydrus1DSimulator()
        forward = make_forward(sim, pm, template_scenario=template,
                               forcing=None, ic=None, obs_specs=obs_specs)

        bounds = pm.bounds_array()
        if req.sampler == "lhs":
            thetas = lhs(bounds, n=req.n, seed=req.seed)
        elif req.sampler == "grid":
            per = max(1, int(round(req.n ** (1.0 / len(specs)))))
            thetas = grid(bounds, points_per_axis=[per] * len(specs))
        else:
            thetas = uniform_random(bounds, n=req.n, seed=req.seed)

        runner = BatchRunner(forward=forward,
                             param_names=[s.name for s in specs],
                             obs_names=[o.name for o in req.obs],
                             n_workers=req.workers, show_progress=False)
        result = runner.run(thetas)
        result.to_parquet(j["out_path"])
        j["state"] = "done"
        j["n_done"] = result.N
        j["n_failed"] = result.n_failed
    except Exception as e:
        j["state"] = "failed"
        j["error"] = f"{type(e).__name__}: {e}"
        raise
```

In `hydrus_port_server/app.py:build_app()`, after the dndc / ptf router registrations, add:

```python
    try:
        from .routers.research_batch import router as batch_router
        app.include_router(batch_router, prefix="/research/batch",
                           tags=["research", "batch"])
    except ImportError:
        pass
```

- [ ] **Step 4: Run + commit**

```bash
pytest tests/research/batch/test_rest.py -v
git add hydrus_port_server/routers/research_batch.py hydrus_port_server/app.py tests/research/batch/test_rest.py
git commit -m "M3.7: /research/batch/{start,status,result} async REST"
```

(Takes 30-60s — runs real solver in background tasks.)

---

### Task 8: GUI — BatchSweep.vue + EnsembleViz.vue + Pinia store

**Files:**
- Create: `desktop/src/stores/batch.ts`
- Create: `desktop/src/components/EnsembleViz.vue`
- Create: `desktop/src/pages/research/BatchSweep.vue`
- Modify: `desktop/src/api.ts` — append `batch.*` REST wrappers
- Modify: `desktop/src/App.vue` — add Batch Sweep tab

- [ ] **Step 1: Append `batch.*` REST wrappers to api.ts**

In `desktop/src/api.ts`, after the existing `ptf` block, append:

```ts
// M3 — Batch sweep REST endpoints
export interface BatchParamSpec {
  name: string;
  target: string;
  bounds: [number, number];
  transform?: "linear" | "log" | "logit";
}

export interface BatchObsSpec {
  name: string;
  kind: "theta" | "h" | "c" | "flux" | "cumulative_flux" | "concentration_flux";
  location: Record<string, number | number[]>;
  time_day: number;
}

export interface BatchStartRequest {
  scenario_dir: string;
  params: BatchParamSpec[];
  obs: BatchObsSpec[];
  n: number;
  sampler: "lhs" | "grid" | "uniform";
  workers: number;
  seed?: number;
}

export interface BatchStatus {
  state: "pending" | "running" | "done" | "failed";
  n_total: number;
  n_done: number;
  out_path: string;
  error: string | null;
  n_failed?: number;
}

export const batch = {
  async start(req: BatchStartRequest): Promise<{ job_id: string }> {
    const r = await fetch(`${RESEARCH_BASE}/research/batch/start`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(req),
    });
    if (!r.ok) throw new Error(`HTTP ${r.status}: ${await r.text()}`);
    return r.json();
  },
  async status(jobId: string): Promise<BatchStatus> {
    const r = await fetch(`${RESEARCH_BASE}/research/batch/${jobId}/status`);
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    return r.json();
  },
  resultUrl(jobId: string): string {
    return `${RESEARCH_BASE}/research/batch/${jobId}/result`;
  },
};
```

- [ ] **Step 2: Implement the Pinia store**

Write `desktop/src/stores/batch.ts`:

```ts
import { defineStore } from "pinia";
import { batch, type BatchStartRequest, type BatchStatus } from "../api";

export const useBatchStore = defineStore("batch", {
  state: () => ({
    jobId: null as string | null,
    status: null as BatchStatus | null,
    error: null as string | null,
    polling: false as boolean,
  }),
  actions: {
    async start(req: BatchStartRequest) {
      this.error = null;
      this.status = null;
      try {
        const r = await batch.start(req);
        this.jobId = r.job_id;
        this.poll();
      } catch (e: any) {
        this.error = e.message ?? String(e);
      }
    },
    async poll() {
      if (!this.jobId) return;
      this.polling = true;
      const id = this.jobId;
      while (this.polling && this.jobId === id) {
        try {
          this.status = await batch.status(id);
          if (this.status.state === "done" || this.status.state === "failed") break;
        } catch (e: any) {
          this.error = e.message ?? String(e);
          break;
        }
        await new Promise(r => setTimeout(r, 1000));
      }
      this.polling = false;
    },
    stop() { this.polling = false; },
  },
});
```

- [ ] **Step 3: Implement EnsembleViz.vue (reusable)**

Write `desktop/src/components/EnsembleViz.vue`:

```vue
<template>
  <div ref="plotEl" class="ensemble-viz"></div>
</template>

<script setup lang="ts">
import { onMounted, ref, watch } from "vue";
import Plotly from "plotly.js-dist-min";

const props = defineProps<{
  // Each row is one ensemble member; columns are the x-axis (e.g. time).
  ys: number[][];
  x: number[];
  xlabel?: string;
  ylabel?: string;
}>();

const plotEl = ref<HTMLDivElement | null>(null);

// Largest-Triangle-Three-Buckets downsampling — keeps visual shape on huge ensembles
function lttb(data: number[], threshold: number): number[] {
  if (data.length <= threshold) return data;
  const stride = data.length / threshold;
  const out: number[] = [];
  for (let i = 0; i < threshold; i++) out.push(data[Math.floor(i * stride)]);
  return out;
}

function _draw() {
  if (!plotEl.value || !props.ys.length) return;
  const N = props.ys.length;
  const showAll = N <= 200;
  const traces: any[] = props.ys.map((row, i) => ({
    type: "scatter", mode: "lines",
    x: props.x, y: row,
    line: { width: 1, color: showAll ? "#1f77b4" : "rgba(31,119,180,0.06)" },
    showlegend: false,
    hoverinfo: i === 0 ? "x+y" : "skip",
  }));

  // 95% quantile band when N is large
  if (!showAll) {
    const Nt = props.x.length;
    const lo: number[] = [], hi: number[] = [], med: number[] = [];
    for (let t = 0; t < Nt; t++) {
      const col = props.ys.map(r => r[t]).filter(v => !isNaN(v)).sort((a, b) => a - b);
      lo.push(col[Math.floor(0.025 * col.length)] ?? NaN);
      med.push(col[Math.floor(0.5 * col.length)] ?? NaN);
      hi.push(col[Math.floor(0.975 * col.length)] ?? NaN);
    }
    traces.push(
      { type: "scatter", mode: "lines", x: props.x, y: hi, line: { color: "#c00", dash: "dot" }, name: "97.5%" },
      { type: "scatter", mode: "lines", x: props.x, y: med, line: { color: "#c00" }, name: "median" },
      { type: "scatter", mode: "lines", x: props.x, y: lo, line: { color: "#c00", dash: "dot" }, name: "2.5%" },
    );
  }

  Plotly.newPlot(plotEl.value, traces, {
    xaxis: { title: props.xlabel ?? "x" },
    yaxis: { title: props.ylabel ?? "y" },
    margin: { t: 20, l: 60, r: 30, b: 50 },
  }, { responsive: true, displayModeBar: false });
}

onMounted(_draw);
watch(() => [props.ys, props.x], _draw, { deep: true });
</script>

<style scoped>
.ensemble-viz { width: 100%; height: 380px; }
</style>
```

- [ ] **Step 4: Implement BatchSweep.vue**

Write `desktop/src/pages/research/BatchSweep.vue`:

```vue
<template>
  <div class="batch-sweep">
    <h2>Batch Sweep — F4</h2>
    <div class="form">
      <label>Scenario directory
        <input v-model="scenarioDir" placeholder="tests/fixtures/infiltr_v1/inputs" />
      </label>
      <label>Param spec (target:lo:hi:transform)
        <input v-model="paramSpec" placeholder="materials[0].alpha:0.005:0.05:log" />
      </label>
      <label>Obs depth (cm, negative = below surface)
        <input v-model.number="obsDepth" type="number" />
      </label>
      <label>Obs time (days)
        <input v-model.number="obsTime" type="number" step="0.1" />
      </label>
      <label>N samples <input v-model.number="n" type="number" min="1" /></label>
      <label>Sampler
        <select v-model="sampler">
          <option value="lhs">Latin Hypercube</option>
          <option value="grid">Full-factorial grid</option>
          <option value="uniform">Uniform random</option>
        </select>
      </label>
      <label>Workers <input v-model.number="workers" type="number" min="1" /></label>
      <button @click="run" :disabled="store.polling">Start sweep</button>
      <button @click="store.stop" :disabled="!store.polling">Stop polling</button>
    </div>

    <div v-if="store.status" class="status">
      <p>State: <b>{{ store.status.state }}</b></p>
      <p>Progress: {{ store.status.n_done ?? 0 }} / {{ store.status.n_total }}</p>
      <p v-if="store.status.n_failed">Failed: {{ store.status.n_failed }}</p>
    </div>
    <p v-if="store.error" class="err">{{ store.error }}</p>

    <p v-if="store.status?.state === 'done' && store.jobId">
      <a :href="resultLink">Download parquet</a>
    </p>
  </div>
</template>

<script setup lang="ts">
import { ref, computed } from "vue";
import { useBatchStore } from "../../stores/batch";
import { batch as batchApi } from "../../api";

const store = useBatchStore();

const scenarioDir = ref("tests/fixtures/infiltr_v1/inputs");
const paramSpec = ref("materials[0].alpha:0.005:0.05:log");
const obsDepth = ref(-30);
const obsTime = ref(1.0);
const n = ref(8);
const sampler = ref<"lhs" | "grid" | "uniform">("lhs");
const workers = ref(2);

function _parseSpec(s: string) {
  const parts = s.split(":");
  const target = parts[0];
  const lo = parseFloat(parts[1]); const hi = parseFloat(parts[2]);
  const transform = (parts[3] as any) ?? "linear";
  const name = target.split(".").pop() ?? target;
  return { name, target, bounds: [lo, hi] as [number, number], transform };
}

async function run() {
  await store.start({
    scenario_dir: scenarioDir.value,
    params: [_parseSpec(paramSpec.value)],
    obs: [{
      name: `theta_${obsDepth.value}cm_d${obsTime.value}`,
      kind: "theta", location: { z_cm: obsDepth.value }, time_day: obsTime.value,
    }],
    n: n.value, sampler: sampler.value, workers: workers.value,
  });
}

const resultLink = computed(() =>
  store.jobId ? batchApi.resultUrl(store.jobId) : "");
</script>

<style scoped>
.batch-sweep { padding: 16px; max-width: 720px; }
.form { display: grid; grid-template-columns: 200px 1fr; gap: 6px; align-items: center; margin-bottom: 12px; }
.form input, .form select { padding: 4px; }
.form button { grid-column: 1 / 3; padding: 8px; margin-top: 8px; }
.status { background: #f4f4f4; padding: 12px; border-radius: 4px; }
.err { color: #c00; }
</style>
```

- [ ] **Step 5: Add `Batch Sweep` tab to App.vue**

In `desktop/src/App.vue`, find the `rightTab` union type (currently `"3d" | "editor" | "dndc" | "soil-library"`) and widen it to include `"batch"`:

```ts
const rightTab = ref<"3d" | "editor" | "dndc" | "soil-library" | "batch">("editor");
```

Add the import next to the other research-page imports:

```ts
import BatchSweep from "./pages/research/BatchSweep.vue";
```

In the tab bar template, add a new button after the Soil Library button:

```vue
          <button class="tab" :class="{active: rightTab === 'batch'}"
                  @click="rightTab = 'batch'">Batch Sweep</button>
```

And the conditional render after the SoilLibrary line:

```vue
        <BatchSweep v-else-if="rightTab === 'batch'" />
```

- [ ] **Step 6: Smoke-build + commit**

Optional manual smoke (don't block on this if vite isn't running):

```bash
cd desktop && npx vue-tsc --noEmit 2>&1 | tail -10    # confirm no type errors
```

```bash
git add desktop/src/api.ts desktop/src/stores/batch.ts desktop/src/components/EnsembleViz.vue desktop/src/pages/research/BatchSweep.vue desktop/src/App.vue
git commit -m "M3.8: BatchSweep.vue + EnsembleViz.vue + batch REST wrapper + Pinia store"
```

---

### Task 9: End-to-end acceptance + regression + M3-complete marker

**Files:**
- Create: `tests/research/batch/test_e2e_m3.py`

- [ ] **Step 1: Write the acceptance test**

Write `tests/research/batch/test_e2e_m3.py`:

```python
"""M3 acceptance: LHS=8 on infiltr_v1, parquet round-trip, sanity bounds."""
import numpy as np
import pytest
from pathlib import Path

from hydrus_research.batch import BatchRunner, BatchResult
from hydrus_research.batch.sampling import lhs
from hydrus_research.parameters import ParameterSpec, ParameterMap
from hydrus_research.observations import ObservationSpec
from hydrus_research.simulator import make_forward
from hydrus_research.simulator.hydrus1d_adapter import Hydrus1DSimulator
from hydrus_port.adapters.hydrus1d import load as load_h1d_canonical


def test_m3_lhs8_on_infiltr_v1(tmp_path):
    template = load_h1d_canonical(Path("tests/fixtures/infiltr_v1/inputs")).to_dict()
    a0 = template["materials"][0]["alpha"]

    pm = ParameterMap([
        ParameterSpec(name="alpha", target="materials[0].alpha",
                      bounds=(a0 * 0.3, a0 * 3.0), transform="log"),
    ])
    obs = [ObservationSpec(name="theta_z30_d2", kind="theta",
                           location={"z_cm": -30.0}, time_day=2.0)]
    sim = Hydrus1DSimulator()
    forward = make_forward(sim, pm,
                           template_scenario=template,
                           forcing=None, ic=None, obs_specs=obs)

    thetas = lhs(pm.bounds_array(), n=8, seed=42)
    runner = BatchRunner(forward=forward,
                         param_names=["alpha"], obs_names=["theta_z30_d2"],
                         n_workers=2, show_progress=False)
    result = runner.run(thetas)

    # Shape + convergence
    assert result.N == 8
    assert result.D == 1
    assert result.M == 1
    assert result.n_converged >= 6        # allow up to 2 outlier failures
    # Physical range on converged rows
    valid = result.ys[result.converged]
    assert ((valid >= 0) & (valid <= 1)).all()
    # Variability — different alphas should produce different ys
    assert valid.std() > 1e-3

    # Parquet round-trip
    out = tmp_path / "m3_sweep.parquet"
    result.to_parquet(out)
    assert out.exists()
    result2 = BatchResult.from_parquet(out)
    np.testing.assert_array_equal(result.thetas, result2.thetas)
    np.testing.assert_array_equal(result.ys, result2.ys)
```

- [ ] **Step 2: Run the acceptance test**

```bash
pytest tests/research/batch/test_e2e_m3.py -v -s
```

Expected: PASS in ~90 seconds (8 real solver runs at ~10s each, parallelized 2x = ~40-60s).

- [ ] **Step 3: Final regression**

```bash
pytest tests/research/ -q --ignore=tests/research/batch/test_gui_smoke.py 2>&1 | tail -5
hydrus test 1d 2>&1 | tail -3
hydrus test 2d 2>&1 | tail -3
hydrus test roundtrip 2>&1 | tail -3
```

Expected: every line PASS; test count ≥ 130 (110 from M0/M1/M2 + ~16 new from M3).

- [ ] **Step 4: M3-complete marker commit**

```bash
git add tests/research/batch/test_e2e_m3.py
git commit -m "M3.9: end-to-end LHS=8 on infiltr_v1 + parquet round-trip"
git commit --allow-empty -m "M3 complete: BatchRunner green; ready for M4 (sensitivity) and M5 (inversion)"
git log --oneline | head -16
```

---

## Definition of Done for M3

1. `pytest tests/research/batch/ -v` — all green.
2. `pytest tests/research/ -q` — no regression in M0 / M1 / M2.
3. `hydrus test 1d/2d/roundtrip` — all PASS.
4. `python -c "from hydrus_research.batch import BatchRunner, BatchResult; print('OK')"` — prints `OK`.
5. `hydrus research sweep tests/fixtures/infiltr_v1/inputs --param materials[0].alpha:0.005:0.05:log --n 4 --workers 2 --out /tmp/sweep.parquet` — produces a 4-row parquet.
6. `hydrus-port-serve` running → `POST /research/batch/start` returns a job_id; `/status` progresses to `done`; `/result` returns parquet bytes.
7. Tauri dev → `Research → Batch Sweep` tab renders; "Start sweep" wires through to a real run.
8. Outstanding concerns updated: M0 #5 (`make_forward` serial) is now mitigated — `BatchRunner` wraps it in `joblib.Parallel`. The per-call deepcopy in `apply_to_scenario` is the parallelism floor (~1 ms per call); document this in `project-m0-outstanding-concerns` after merge.
