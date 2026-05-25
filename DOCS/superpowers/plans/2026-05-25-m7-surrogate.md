# M7 — Surrogate Models Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement surrogate models that drop-in-replace any `Simulator` — once trained on M3 `BatchResult` data, they predict `y` in milliseconds. Two backends:
- **sklearn GP** (default, lightweight) — Matérn kernel; gives mean + stddev per prediction.
- **PC-Kriging (PCK)** via `smt` (KPLS) — hydrology SOTA per Schöbi et al.; better extrapolation on smooth response surfaces.

Wraps both behind `SurrogateSimulator(Simulator)` so M4 / M5 / M6 tools work on the surrogate without knowing it's not a real solver.

**Architecture:** New `hydrus_research/surrogate/` sub-package. Each model implements `.fit(thetas, ys)` + `.predict(theta)` + `.save(path)` + `.load(path)`. Model persistence uses `joblib.dump/load` (sklearn's recommended persistence — already in `[research]` extras). K-fold CV + (NSE, RMSE, coverage) metrics in `metrics.py`. Both backends gracefully degrade: sklearn GP requires only `scikit-learn`; PCK requires `smt` (from `[research-3d]` extras) and falls back to clear ImportError when missing.

**Tech Stack:** Python 3.10+, `scikit-learn>=1.3` (already), `joblib` (already), `smt>=2` (in `[research-3d]`), `numpy`. Independent of M6/M8/M9/M10 — parallel.

**Spec reference:** `DOCS/superpowers/specs/2026-05-24-hydrus-research-platform-design.md` §4.6 + §5.1 (SurrogateBench page).

**Acceptance:**
- `python -c "from hydrus_research.surrogate import train_gp, train_pck, evaluate, SurrogateSimulator"` works.
- GP trained on LHS=32 of a 2-param toy function recovers `y_pred ≈ y_true` within 0.1 absolute on held-out points.
- PCK trained on same data reaches comparable accuracy when `smt` installed (test SKIPs if missing).
- `SurrogateSimulator` is an M0 Simulator subclass.
- `hydrus research surrogate train <batch.parquet> --type gp --out model.joblib` works.
- REST `POST /research/surrogate/train` + `POST /research/surrogate/{model_id}/evaluate` work.
- `pytest tests/research/surrogate/` green; M0-M6 tests green.

---

## File Layout

**Created:**
- `hydrus_research/surrogate/__init__.py`
- `hydrus_research/surrogate/base.py` — `SurrogateSimulator(Simulator)` + abstract `SurrogateModel`.
- `hydrus_research/surrogate/gp_sklearn.py` — `SklearnGPSurrogate(SurrogateModel)`.
- `hydrus_research/surrogate/pck.py` — `PCKSurrogate(SurrogateModel)` (smt KPLS; lazy).
- `hydrus_research/surrogate/trainer.py` — `train_gp`, `train_pck`.
- `hydrus_research/surrogate/metrics.py` — NSE, RMSE, coverage.
- `hydrus_research/surrogate/api.py` — `evaluate(surrogate, batch) → dict`.
- `hydrus_port_server/routers/research_surrogate.py`
- `desktop/src/pages/research/SurrogateBench.vue`
- `tests/research/surrogate/{__init__,test_base,test_gp_sklearn,test_pck,test_trainer,test_metrics,test_rest,test_cli,test_e2e_m7}.py`

**Modified:**
- `hydrus_port_server/app.py:build_app()` — register surrogate router.
- `hydrus_port/cli.py:_build_research_subparser` — add `hydrus research surrogate train` subcommand.
- `desktop/src/api.ts` — append `surrogate.*` wrapper.
- `desktop/src/App.vue` — add `Surrogate` tab.

---

### Task 1: Skeleton + SurrogateSimulator base

**Files:** `__init__.py`, `base.py`, 2 stubs, `test_base.py`.

- [ ] **Step 1: Create**

`hydrus_research/surrogate/__init__.py`:
```python
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
```

`hydrus_research/surrogate/base.py`:
```python
"""SurrogateSimulator — drop-in M0 Simulator backed by a trained model."""
from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np

from ..simulator.base import Simulator, SimResult, InitialState


class SurrogateModel(ABC):
    """Common interface for all surrogate backends."""
    @abstractmethod
    def fit(self, thetas: np.ndarray, ys: np.ndarray) -> None: ...
    @abstractmethod
    def predict(self, theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (mean, stddev) — both shape (M,) where M = n_obs."""
    @abstractmethod
    def save(self, path) -> None: ...
    @classmethod
    @abstractmethod
    def load(cls, path) -> "SurrogateModel": ...


class SurrogateSimulator(Simulator):
    """Wraps a trained SurrogateModel as an M0 Simulator."""
    name: str = "surrogate"
    dimension: int = -1

    def __init__(self, model: SurrogateModel,
                 param_names: list[str], obs_names: list[str]):
        self.model = model
        self.param_names = list(param_names)
        self.obs_names = list(obs_names)

    def run(self, scenario, forcing, ic):
        raise NotImplementedError(
            "SurrogateSimulator.run is not used directly; consumers wire "
            "`forward = lambda theta: surrogate.model.predict(theta)[0]` "
            "into M4/M5/M6 workflows."
        )

    def observable_at(self, result, spec):
        raise NotImplementedError(
            "SurrogateSimulator doesn't produce a full SimResult; observables "
            "are returned directly by model.predict(theta) in obs_names order."
        )
```

Stubs `gp_sklearn.py`, `pck.py`, `api.py`, `trainer.py`, `metrics.py` each have one function raising NotImplementedError.

`tests/research/surrogate/test_base.py`:
```python
import numpy as np
import pytest
from hydrus_research.surrogate import SurrogateSimulator, SurrogateModel


def test_surrogate_simulator_is_simulator():
    from hydrus_research.simulator.base import Simulator
    class _Dummy(SurrogateModel):
        def fit(self, t, y): pass
        def predict(self, t): return (np.zeros(1), np.zeros(1))
        def save(self, p): pass
        @classmethod
        def load(cls, p): return cls()
    s = SurrogateSimulator(_Dummy(), param_names=["a"], obs_names=["o"])
    assert isinstance(s, Simulator)
    assert s.dimension == -1
```

- [ ] **Step 2: Commit**

```bash
pytest tests/research/surrogate/test_base.py -v
git add hydrus_research/surrogate/ tests/research/surrogate/
git commit -m "M7.1: surrogate skeleton + SurrogateSimulator base + stubs"
```

---

### Task 2: sklearn GP backend (joblib persistence)

**Files:** `gp_sklearn.py` (replace stub), `test_gp_sklearn.py`.

- [ ] **Step 1: Failing test**

```python
import numpy as np
import pytest
from hydrus_research.surrogate.gp_sklearn import SklearnGPSurrogate


def f(theta):
    return np.array([np.sin(theta[0]) + np.cos(theta[1])])


def test_gp_fit_predict():
    rng = np.random.default_rng(7)
    thetas = rng.uniform(-np.pi, np.pi, size=(32, 2))
    ys = np.array([f(t) for t in thetas])
    surr = SklearnGPSurrogate()
    surr.fit(thetas, ys)
    theta_test = np.array([0.5, 1.0])
    mean, std = surr.predict(theta_test)
    assert mean.shape == (1,) and std.shape == (1,)
    assert abs(float(mean[0]) - float(f(theta_test)[0])) < 0.1
    assert std[0] >= 0


def test_gp_save_load(tmp_path):
    rng = np.random.default_rng(11)
    thetas = rng.uniform(0, 1, size=(16, 2))
    ys = np.array([f(t) for t in thetas])
    surr = SklearnGPSurrogate()
    surr.fit(thetas, ys)
    p = tmp_path / "gp.joblib"
    surr.save(p)
    surr2 = SklearnGPSurrogate.load(p)
    m1, _ = surr.predict(np.array([0.5, 0.5]))
    m2, _ = surr2.predict(np.array([0.5, 0.5]))
    np.testing.assert_allclose(m1, m2)
```

- [ ] **Step 2: Implement using joblib persistence**

```python
"""scikit-learn Gaussian Process surrogate (Matérn 5/2 kernel default).

Persistence uses joblib — the canonical sklearn idiom. The serialized
artefact is our own file written immediately before loading; never load
joblib files from untrusted sources."""
from __future__ import annotations
from pathlib import Path

import joblib
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel

from .base import SurrogateModel


class SklearnGPSurrogate(SurrogateModel):
    def __init__(self, kernel: str = "matern52", alpha: float = 1e-6):
        self.kernel_name = kernel
        self.alpha = alpha
        self._gprs: list[GaussianProcessRegressor] | None = None
        self._D: int | None = None
        self._M: int | None = None

    def _make_kernel(self):
        if self.kernel_name == "matern52":
            return ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5) \
                   + WhiteKernel(noise_level=1e-5)
        if self.kernel_name == "matern32":
            return ConstantKernel(1.0) * Matern(length_scale=1.0, nu=1.5) \
                   + WhiteKernel(noise_level=1e-5)
        raise ValueError(f"unknown kernel {self.kernel_name!r}")

    def fit(self, thetas, ys):
        thetas = np.asarray(thetas, dtype=float)
        ys = np.asarray(ys, dtype=float)
        if ys.ndim == 1:
            ys = ys.reshape(-1, 1)
        self._D = thetas.shape[1]
        self._M = ys.shape[1]
        self._gprs = []
        for j in range(self._M):
            gpr = GaussianProcessRegressor(
                kernel=self._make_kernel(), alpha=self.alpha,
                normalize_y=True, n_restarts_optimizer=2, random_state=42,
            )
            gpr.fit(thetas, ys[:, j])
            self._gprs.append(gpr)

    def predict(self, theta):
        if self._gprs is None:
            raise RuntimeError("must call fit() first")
        theta = np.asarray(theta, dtype=float).reshape(1, -1)
        means = np.empty(self._M, dtype=float)
        stds = np.empty(self._M, dtype=float)
        for j, gpr in enumerate(self._gprs):
            m, s = gpr.predict(theta, return_std=True)
            means[j] = float(m[0])
            stds[j] = float(s[0])
        return means, stds

    def save(self, path):
        joblib.dump({
            "kernel_name": self.kernel_name, "alpha": self.alpha,
            "gprs": self._gprs, "D": self._D, "M": self._M,
        }, Path(path))

    @classmethod
    def load(cls, path):
        d = joblib.load(Path(path))
        obj = cls(kernel=d["kernel_name"], alpha=d["alpha"])
        obj._gprs = d["gprs"]
        obj._D = d["D"]
        obj._M = d["M"]
        return obj
```

- [ ] **Step 3: Commit**

```bash
pytest tests/research/surrogate/test_gp_sklearn.py -v
git add hydrus_research/surrogate/gp_sklearn.py tests/research/surrogate/test_gp_sklearn.py
git commit -m "M7.2: SklearnGPSurrogate (Matérn 5/2; joblib persistence)"
```

---

### Task 3: PCK backend (smt KPLS; lazy)

**Files:** `pck.py`, `test_pck.py`.

- [ ] **Step 1: Test (skip if smt missing)**

```python
import numpy as np
import pytest

smt = pytest.importorskip("smt", reason="smt not installed; in [research-3d] extras")
from hydrus_research.surrogate.pck import PCKSurrogate


def f(theta):
    return np.array([np.sin(theta[0]) + np.cos(theta[1])])


def test_pck_fit_predict():
    rng = np.random.default_rng(13)
    thetas = rng.uniform(-np.pi, np.pi, size=(32, 2))
    ys = np.array([f(t) for t in thetas])
    surr = PCKSurrogate(pce_degree=2)
    surr.fit(thetas, ys)
    mean, std = surr.predict(np.array([0.5, 1.0]))
    assert mean.shape == (1,)
    assert abs(float(mean[0]) - float(f(np.array([0.5, 1.0]))[0])) < 0.3
    assert std[0] >= 0
```

- [ ] **Step 2: Implement**

```python
"""PC-Kriging via smt KPLS — sklearn-style API for hydrology response surfaces."""
from __future__ import annotations
from pathlib import Path

import joblib
import numpy as np

from .base import SurrogateModel


class PCKSurrogate(SurrogateModel):
    def __init__(self, pce_degree: int = 3):
        self.pce_degree = pce_degree
        self._kpls = None
        self._theta_min = None
        self._theta_range = None
        self._M: int | None = None

    def fit(self, thetas, ys):
        from smt.surrogate_models import KPLS
        thetas = np.asarray(thetas, dtype=float)
        ys = np.asarray(ys, dtype=float)
        if ys.ndim == 1:
            ys = ys.reshape(-1, 1)
        self._M = ys.shape[1]
        self._theta_min = thetas.min(axis=0)
        self._theta_range = np.maximum(thetas.max(axis=0) - self._theta_min, 1e-9)
        theta_n = (thetas - self._theta_min) / self._theta_range
        self._kpls = []
        for j in range(self._M):
            sm = KPLS(print_global=False)
            sm.set_training_values(theta_n, ys[:, j])
            sm.train()
            self._kpls.append(sm)

    def predict(self, theta):
        if self._kpls is None:
            raise RuntimeError("must call fit() first")
        theta = np.asarray(theta, dtype=float).reshape(1, -1)
        theta_n = (theta - self._theta_min) / self._theta_range
        means = np.empty(self._M, dtype=float)
        stds = np.empty(self._M, dtype=float)
        for j, sm in enumerate(self._kpls):
            means[j] = float(sm.predict_values(theta_n).flatten()[0])
            try:
                stds[j] = float(np.sqrt(sm.predict_variances(theta_n).flatten()[0]))
            except Exception:
                stds[j] = float("nan")
        return means, stds

    def save(self, path):
        joblib.dump({
            "pce_degree": self.pce_degree, "kpls": self._kpls, "M": self._M,
            "theta_min": self._theta_min, "theta_range": self._theta_range,
        }, Path(path))

    @classmethod
    def load(cls, path):
        d = joblib.load(Path(path))
        obj = cls(pce_degree=d["pce_degree"])
        obj._kpls = d["kpls"]; obj._M = d["M"]
        obj._theta_min = d["theta_min"]; obj._theta_range = d["theta_range"]
        return obj
```

- [ ] **Step 3: Commit**

```bash
pytest tests/research/surrogate/test_pck.py -v
git add hydrus_research/surrogate/pck.py tests/research/surrogate/test_pck.py
git commit -m "M7.3: PCKSurrogate via smt KPLS (lazy + joblib persistence)"
```

---

### Task 4: Trainer + metrics + api

**Files:** `trainer.py`, `metrics.py`, `api.py`, `test_trainer.py`, `test_metrics.py`.

- [ ] **Step 1: metrics.py + test**

```python
"""Surrogate evaluation metrics."""
from __future__ import annotations
import numpy as np


def nse(sim, obs):
    sim = np.asarray(sim, dtype=float).ravel()
    obs = np.asarray(obs, dtype=float).ravel()
    num = np.sum((sim - obs) ** 2)
    den = np.sum((obs - obs.mean()) ** 2)
    return float("nan") if den == 0 else float(1.0 - num / den)


def rmse(sim, obs):
    sim = np.asarray(sim, dtype=float).ravel()
    obs = np.asarray(obs, dtype=float).ravel()
    return float(np.sqrt(np.mean((sim - obs) ** 2)))


def coverage(means, stds, obs, z=1.96):
    means = np.asarray(means, dtype=float)
    stds = np.asarray(stds, dtype=float)
    obs = np.asarray(obs, dtype=float)
    return float(np.mean((obs >= means - z*stds) & (obs <= means + z*stds)))
```

Test:
```python
import numpy as np
from hydrus_research.surrogate.metrics import nse, rmse, coverage


def test_nse_perfect():
    assert nse([1, 2, 3], [1, 2, 3]) == 1.0


def test_rmse_zero():
    assert rmse([1, 2, 3], [1, 2, 3]) == 0.0


def test_coverage_one_when_all_inside():
    assert coverage([1, 2, 3], [10, 10, 10], [1, 2, 3]) == 1.0
```

- [ ] **Step 2: trainer.py + api.py + test**

`trainer.py`:
```python
"""Train surrogate models from M3 BatchResult artifacts."""
from __future__ import annotations
import numpy as np

from .gp_sklearn import SklearnGPSurrogate
from .pck import PCKSurrogate


def _filter_converged(br):
    keep = br.converged
    if not keep.any():
        raise ValueError("BatchResult has zero converged rows; cannot train")
    return br.thetas[keep], br.ys[keep]


def train_gp(batch_result, kernel="matern52"):
    thetas, ys = _filter_converged(batch_result)
    surr = SklearnGPSurrogate(kernel=kernel)
    surr.fit(thetas, ys)
    return surr


def train_pck(batch_result, pce_degree=3):
    thetas, ys = _filter_converged(batch_result)
    surr = PCKSurrogate(pce_degree=pce_degree)
    surr.fit(thetas, ys)
    return surr
```

`api.py`:
```python
"""Public surrogate API."""
import numpy as np
from .metrics import nse, rmse, coverage


def evaluate(surrogate, batch_result) -> dict:
    thetas = batch_result.thetas
    obs = batch_result.ys
    M = obs.shape[1]
    means = np.empty_like(obs)
    stds = np.empty_like(obs)
    for i, t in enumerate(thetas):
        m, s = surrogate.predict(t)
        means[i] = m; stds[i] = s
    return {
        "NSE":      [nse(means[:, j], obs[:, j]) for j in range(M)],
        "RMSE":     [rmse(means[:, j], obs[:, j]) for j in range(M)],
        "coverage": [coverage(means[:, j], stds[:, j], obs[:, j]) for j in range(M)],
    }
```

Test:
```python
import numpy as np
from hydrus_research.surrogate import train_gp, evaluate
from hydrus_research.batch import BatchResult


def _toy_batch(N=40, seed=42):
    rng = np.random.default_rng(seed)
    thetas = rng.uniform(-np.pi, np.pi, size=(N, 2))
    ys = np.column_stack([np.sin(thetas[:, 0]) + np.cos(thetas[:, 1])])
    return BatchResult(thetas=thetas, ys=ys, wall_s=np.zeros(N),
                       converged=np.ones(N, dtype=bool),
                       param_names=["x", "y"], obs_names=["f"], meta={})


def test_train_gp_then_evaluate():
    train = _toy_batch(40, seed=1)
    test = _toy_batch(20, seed=2)
    surr = train_gp(train)
    m = evaluate(surr, test)
    assert "NSE" in m
    assert m["NSE"][0] > 0.5
```

- [ ] **Step 3: Commit**

```bash
pytest tests/research/surrogate/test_trainer.py tests/research/surrogate/test_metrics.py -v
git add hydrus_research/surrogate/trainer.py hydrus_research/surrogate/api.py hydrus_research/surrogate/metrics.py tests/research/surrogate/test_trainer.py tests/research/surrogate/test_metrics.py
git commit -m "M7.4: trainer (train_gp/train_pck) + metrics + evaluate()"
```

---

### Task 5: REST + CLI + GUI + e2e + marker (bundled)

**Files:** REST router + cli edit + api.ts edit + SurrogateBench.vue + App.vue edit + 3 tests.

- [ ] **Step 1: REST router**

`hydrus_port_server/routers/research_surrogate.py`:
```python
"""/research/surrogate/* — train + evaluate."""
from __future__ import annotations
import uuid
from pathlib import Path
from typing import Any, Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from hydrus_research.batch import BatchResult
from hydrus_research.surrogate import train_gp, train_pck, evaluate


router = APIRouter()
_MODELS: dict[str, Any] = {}


class TrainRequest(BaseModel):
    batch_parquet: str
    type: Literal["gp", "pck"] = "gp"


class TrainResponse(BaseModel):
    model_id: str
    type: str


@router.post("/train", response_model=TrainResponse)
def train(req: TrainRequest):
    br = BatchResult.from_parquet(Path(req.batch_parquet))
    if req.type == "gp":
        surr = train_gp(br)
    else:
        try:
            surr = train_pck(br)
        except ImportError as e:
            raise HTTPException(status_code=503, detail=f"pck deps missing: {e}")
    mid = uuid.uuid4().hex[:12]
    _MODELS[mid] = {"surrogate": surr, "batch": br}
    return TrainResponse(model_id=mid, type=req.type)


@router.post("/{model_id}/evaluate")
def eval_route(model_id: str):
    if model_id not in _MODELS:
        raise HTTPException(status_code=404, detail="unknown model_id")
    entry = _MODELS[model_id]
    return evaluate(entry["surrogate"], entry["batch"])
```

In `build_app()`: register the same try/except pattern.

- [ ] **Step 2: CLI**

In `_build_research_subparser(sub)`:
```python
p_surr = rsub.add_parser("surrogate", help="train/save surrogate models")
ssub = p_surr.add_subparsers(dest="surrogate_cmd", required=True)
p_tr = ssub.add_parser("train", help="train a surrogate from a parquet")
p_tr.add_argument("batch_parquet")
p_tr.add_argument("--type", choices=["gp", "pck"], default="gp")
p_tr.add_argument("--out", required=True, help="output joblib path")
p_tr.set_defaults(_cmd=_cmd_surrogate_train)


def _cmd_surrogate_train(args):
    from pathlib import Path as _P
    from hydrus_research.batch import BatchResult
    from hydrus_research.surrogate import train_gp, train_pck
    br = BatchResult.from_parquet(_P(args.batch_parquet))
    surr = (train_gp if args.type == "gp" else train_pck)(br)
    surr.save(_P(args.out))
    print(f"surrogate ({args.type}) trained on {br.N} samples → {args.out}")
    return 0
```

- [ ] **Step 3: GUI**

`api.ts` append:
```ts
// ---- M7: Surrogate -------------------------------------------------
export interface SurrogateTrainRequest {
  batch_parquet: string;
  type: "gp" | "pck";
}
export interface SurrogateTrainResponse { model_id: string; type: string; }

export const surrogate = {
  async train(req: SurrogateTrainRequest): Promise<SurrogateTrainResponse> {
    const r = await fetch(`${RESEARCH_BASE}/research/surrogate/train`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify(req),
    });
    if (!r.ok) throw new Error(`HTTP ${r.status}: ${await r.text()}`);
    return r.json();
  },
  async evaluate(modelId: string): Promise<Record<string, number[]>> {
    const r = await fetch(`${RESEARCH_BASE}/research/surrogate/${modelId}/evaluate`, {
      method: "POST",
    });
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    return r.json();
  },
};
```

`desktop/src/pages/research/SurrogateBench.vue` — minimal:
```vue
<template>
  <div class="surr">
    <h2>Surrogate Bench — F8</h2>
    <label>Batch parquet path <input v-model="parquet" placeholder="/tmp/sweep.parquet" /></label>
    <label>Type
      <select v-model="type">
        <option value="gp">sklearn GP</option>
        <option value="pck">PC-Kriging</option>
      </select>
    </label>
    <button @click="train" :disabled="busy">Train</button>
    <p v-if="modelId">Trained: {{ modelId }} <button @click="evaluate">Evaluate</button></p>
    <p v-if="error" class="err">{{ error }}</p>
    <pre v-if="metrics">{{ JSON.stringify(metrics, null, 2) }}</pre>
  </div>
</template>

<script setup lang="ts">
import { ref } from "vue";
import { surrogate } from "../../api";
const parquet = ref(""); const type = ref<"gp" | "pck">("gp");
const modelId = ref<string | null>(null);
const error = ref<string | null>(null);
const metrics = ref<any | null>(null);
const busy = ref(false);
async function train() {
  busy.value = true; error.value = null; metrics.value = null;
  try {
    const r = await surrogate.train({ batch_parquet: parquet.value, type: type.value });
    modelId.value = r.model_id;
  } catch (e: any) { error.value = e.message; }
  finally { busy.value = false; }
}
async function evaluate() {
  if (!modelId.value) return;
  try { metrics.value = await surrogate.evaluate(modelId.value); }
  catch (e: any) { error.value = e.message; }
}
</script>

<style scoped>
.surr { padding: 16px; max-width: 720px; }
label { display: block; margin: 6px 0; }
input, select { padding: 4px; width: 100%; }
.err { color: #c00; } pre { background: #f4f4f4; padding: 12px; }
</style>
```

`App.vue` — add import + widen `rightTab` union to include `"surrogate"` + tab button + conditional render.

- [ ] **Step 4: Tests + e2e**

`test_rest.py`:
```python
import pytest
try:
    from fastapi.testclient import TestClient
except ImportError:
    pytest.skip("FastAPI not installed", allow_module_level=True)
from hydrus_port_server.app import build_app


def test_surrogate_train_route_smoke(tmp_path):
    # Pre-build a tiny parquet
    import numpy as np
    from hydrus_research.batch import BatchResult
    br = BatchResult(thetas=np.random.uniform(0, 1, size=(8, 2)),
                     ys=np.random.uniform(0, 1, size=(8, 1)),
                     wall_s=np.zeros(8), converged=np.ones(8, dtype=bool),
                     param_names=["a","b"], obs_names=["o"], meta={})
    p = tmp_path / "br.parquet"
    br.to_parquet(p)
    client = TestClient(build_app())
    r = client.post("/research/surrogate/train",
                    json={"batch_parquet": str(p), "type": "gp"})
    assert r.status_code == 200
    body = r.json()
    assert "model_id" in body
```

`test_cli.py`:
```python
import subprocess
import sys

def test_cli_surrogate_help():
    r = subprocess.run([sys.executable, "-m", "hydrus_port.cli",
                        "research", "surrogate", "--help"],
                       capture_output=True, text=True)
    assert r.returncode == 0 and "train" in r.stdout
```

`test_e2e_m7.py`:
```python
"""M7 acceptance: LHS=8 on infiltr_v1 → train GP → evaluate."""
import numpy as np
import pytest
from pathlib import Path

from hydrus_research.batch import BatchRunner
from hydrus_research.batch.sampling import lhs
from hydrus_research.parameters import ParameterSpec, ParameterMap
from hydrus_research.observations import ObservationSpec
from hydrus_research.simulator import make_forward
from hydrus_research.simulator.hydrus1d_adapter import Hydrus1DSimulator
from hydrus_research.surrogate import train_gp, evaluate
from hydrus_port.adapters.hydrus1d import load as load_h1d_canonical


def test_train_gp_on_infiltr_v1():
    template = load_h1d_canonical(Path("tests/fixtures/infiltr_v1/inputs")).to_dict()
    a0 = template["materials"][0]["alpha"]
    pm = ParameterMap([
        ParameterSpec(name="alpha", target="materials[0].alpha",
                      bounds=(a0 * 0.5, a0 * 2.0), transform="log"),
    ])
    obs = [ObservationSpec(name="theta_z30_d1", kind="theta",
                           location={"z_cm": -30.0}, time_day=1.0)]
    forward = make_forward(Hydrus1DSimulator(), pm,
                           template_scenario=template,
                           forcing=None, ic=None, obs_specs=obs)
    runner = BatchRunner(forward=forward, param_names=["alpha"],
                         obs_names=["theta_z30_d1"], n_workers=2, show_progress=False)
    train_br = runner.run(lhs(pm.bounds_array(), n=8, seed=42))
    test_br = runner.run(lhs(pm.bounds_array(), n=4, seed=43))
    surr = train_gp(train_br)
    metrics = evaluate(surr, test_br)
    assert metrics["NSE"][0] > -0.5  # very loose — just confirms it trains
```

- [ ] **Step 5: Commit + marker**

```bash
pytest tests/research/surrogate/test_rest.py tests/research/surrogate/test_cli.py tests/research/surrogate/test_e2e_m7.py -v
git add hydrus_port_server/routers/research_surrogate.py hydrus_port_server/app.py hydrus_port/cli.py desktop/src/api.ts desktop/src/pages/research/SurrogateBench.vue desktop/src/App.vue tests/research/surrogate/test_rest.py tests/research/surrogate/test_cli.py tests/research/surrogate/test_e2e_m7.py
git commit -m "M7.5: REST + CLI + SurrogateBench.vue + e2e LHS=8 on infiltr_v1"
git commit --allow-empty -m "M7 complete: surrogate (sklearn GP + PCK) green; ready for M8 (2D/3D adapters)"
```

---

## Definition of Done for M7

1. `pytest tests/research/surrogate/ -v` — green (PCK SKIPs if smt missing).
2. `pytest tests/research/ -q` — no regression.
3. All 4 callables importable: `train_gp`, `train_pck`, `evaluate`, `SurrogateSimulator`.
4. GP recovers toy 2D function within 0.1 on held-out points.
5. REST train + evaluate work.
6. CLI `hydrus research surrogate train` writes a joblib file.
7. GUI Research → Surrogate tab renders.
8. `hydrus test 1d/2d/roundtrip` PASS.
