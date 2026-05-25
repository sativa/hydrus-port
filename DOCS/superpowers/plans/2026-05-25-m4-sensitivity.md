# M4 — Sensitivity Analysis Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement F2 — four global sensitivity analysis methods (Morris elementary effects, Sobol variance-based, FAST/eFAST, PAWN distribution-based) consuming the M3 `BatchRunner` via the M0 `forward(theta) → y_sim` callable. Outputs go to a shared `SensitivityResult` schema rendered by a `SensitivityReport.vue` GUI page.

**Architecture:** Each method is a thin wrapper over `SALib` that (1) generates samples via the method's required design, (2) hands them to `BatchRunner.run()`, (3) post-processes the resulting `BatchResult.ys` through the SALib analyzer, (4) returns a typed `SensitivityResult`. The four methods share zero internal code beyond `SensitivityResult` — each lives in its own file so a bug in PAWN can't break Morris.

**Tech Stack:** Python 3.10+, `SALib>=1.5` (already in `[research]` extras), `numpy`, `scipy`, FastAPI + Pydantic, Vue 3 + Plotly. Independent of M5 — can be developed and merged in parallel.

**Spec reference:** `DOCS/superpowers/specs/2026-05-24-hydrus-research-platform-design.md` §4.3 + §5.1 (SensitivityReport page) + §5.2 (REST endpoints).

**Acceptance:**
- `python -c "from hydrus_research.sensitivity import morris_screen, sobol_decompose, fast_indices, pawn_kde"` works.
- Morris EE on the Ishigami analytical test function recovers the documented ranking (`x1` < `x2` < `x3` in importance; SALib examples).
- Sobol decomposition on Ishigami reproduces SALib's example `S1 ≈ [0.314, 0.443, 0]` and `ST ≈ [0.557, 0.443, 0.244]` within 5%.
- All four methods deliver a `SensitivityResult` with `indices` dict, `param_names`, `method`, `sample_size`.
- REST `POST /research/sensitivity/{method}` returns JSON with the same fields.
- `hydrus research sensitize <scenario_dir> --method morris --params alpha,n,Ks --n 100` writes a parquet of `(θ, y_sim)` plus a JSON summary of indices.
- `pytest tests/research/sensitivity/` green; full `tests/research/` still green.

---

## File Layout

**Created:**
- `hydrus_research/sensitivity/__init__.py` — re-exports.
- `hydrus_research/sensitivity/result.py` — `SensitivityResult` Pydantic model.
- `hydrus_research/sensitivity/morris.py` — `morris_screen(forward, param_map, ...)`.
- `hydrus_research/sensitivity/sobol.py` — `sobol_decompose(forward, param_map, ...)`.
- `hydrus_research/sensitivity/fast.py` — `fast_indices(forward, param_map, ...)`.
- `hydrus_research/sensitivity/pawn.py` — `pawn_kde(forward, param_map, ...)`.
- `hydrus_research/sensitivity/_runner.py` — small internal helper that wraps `BatchRunner.run(thetas)` for the four methods (shared bookkeeping; no public API).
- `hydrus_port_server/routers/research_sensitivity.py` — `/research/sensitivity/{method}` REST routes.
- `desktop/src/pages/research/SensitivityReport.vue` — F2 GUI page.
- `desktop/src/components/SensitivityIndicesBar.vue` — reusable bar chart (used by Sobol/FAST/PAWN tabs).
- `desktop/src/components/MorrisEEPlot.vue` — Morris elementary-effects scatter (μ* vs σ).
- `tests/research/sensitivity/__init__.py`
- `tests/research/sensitivity/test_result.py`
- `tests/research/sensitivity/test_morris.py`
- `tests/research/sensitivity/test_sobol.py`
- `tests/research/sensitivity/test_fast.py`
- `tests/research/sensitivity/test_pawn.py`
- `tests/research/sensitivity/test_cli.py`
- `tests/research/sensitivity/test_rest.py`
- `tests/research/sensitivity/test_e2e_m4.py`

**Modified:**
- `hydrus_port_server/app.py:build_app()` — register the sensitivity router (try/except ImportError, same pattern as dndc/ptf/batch).
- `hydrus_port/cli.py:_build_research_subparser` — add `hydrus research sensitize` subcommand.
- `desktop/src/api.ts` — append `sensitivity.*` REST wrapper block.
- `desktop/src/App.vue` — add `Sensitivity` tab (sibling to existing 5 tabs).

---

### Task 1: Sub-package skeleton + SensitivityResult

**Files:**
- Create: `hydrus_research/sensitivity/__init__.py`
- Create: `hydrus_research/sensitivity/result.py`
- Create: `tests/research/sensitivity/__init__.py`
- Create: `tests/research/sensitivity/test_result.py`

- [ ] **Step 1: Skeleton + result schema**

```bash
mkdir -p hydrus_research/sensitivity tests/research/sensitivity
touch tests/research/sensitivity/__init__.py
```

Write `hydrus_research/sensitivity/__init__.py`:

```python
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
```

Write `hydrus_research/sensitivity/result.py`:

```python
"""SensitivityResult — typed output of every sensitivity-analysis call."""
from __future__ import annotations
from typing import Literal
from pydantic import BaseModel, ConfigDict


SensMethod = Literal["morris", "sobol", "fast", "pawn"]


class SensitivityResult(BaseModel):
    """One index dict per observable (or aggregated)."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    method: SensMethod
    param_names: list[str]
    obs_names: list[str]
    # indices[index_name] -> list of length D (param indices) OR list of M lists of D
    # Common keys per method:
    #   morris: mu, mu_star, sigma, mu_star_conf
    #   sobol:  S1, S1_conf, ST, ST_conf, [S2 if calc_second_order]
    #   fast:   S1, ST
    #   pawn:   minimum, mean, median, maximum, CV
    indices: dict[str, list[list[float]] | list[float]]
    sample_size: int
    forward_cost_s: float
    diagnostics: dict = {}
```

- [ ] **Step 2: Write test**

Write `tests/research/sensitivity/test_result.py`:

```python
import pytest
from hydrus_research.sensitivity import SensitivityResult


def test_sensitivity_result_construction():
    r = SensitivityResult(
        method="sobol",
        param_names=["alpha", "n", "Ks"],
        obs_names=["theta_z30_d1"],
        indices={"S1": [[0.31, 0.44, 0.0]],
                 "ST": [[0.56, 0.44, 0.24]],
                 "S1_conf": [[0.05, 0.05, 0.01]]},
        sample_size=1024,
        forward_cost_s=12.5,
    )
    assert r.method == "sobol"
    assert len(r.indices["S1"][0]) == 3


def test_sensitivity_result_rejects_unknown_method():
    with pytest.raises(Exception):
        SensitivityResult(method="banana", param_names=[], obs_names=[],
                          indices={}, sample_size=0, forward_cost_s=0)
```

- [ ] **Step 3: Run + commit**

The eager imports in `__init__.py` will fail until Tasks 2-5 create morris/sobol/fast/pawn modules. Run only the result test:

```bash
pytest tests/research/sensitivity/test_result.py -v
```

If the test errors on the package-level import, create empty stubs for the four method files:

```bash
for m in morris sobol fast pawn; do
  echo "def ${m}_screen(*args, **kwargs): raise NotImplementedError('M4.${m} stub')" \
    > hydrus_research/sensitivity/${m}.py 2>/dev/null
done
```

Actually the names differ — use correct stubs:

```python
# hydrus_research/sensitivity/morris.py
def morris_screen(*args, **kwargs):
    raise NotImplementedError("M4.2 stub")
```

```python
# hydrus_research/sensitivity/sobol.py
def sobol_decompose(*args, **kwargs):
    raise NotImplementedError("M4.3 stub")
```

```python
# hydrus_research/sensitivity/fast.py
def fast_indices(*args, **kwargs):
    raise NotImplementedError("M4.4 stub")
```

```python
# hydrus_research/sensitivity/pawn.py
def pawn_kde(*args, **kwargs):
    raise NotImplementedError("M4.5 stub")
```

```bash
pytest tests/research/sensitivity/test_result.py -v
git add hydrus_research/sensitivity/ tests/research/sensitivity/
git commit -m "M4.1: sensitivity sub-package skeleton + SensitivityResult + 4 method stubs"
```

---

### Task 2: Internal `_runner` helper (consume BatchRunner)

**Files:**
- Create: `hydrus_research/sensitivity/_runner.py`

This is an internal helper that the four methods all use to convert SALib's sample matrix into a `BatchRunner.run()` call and extract `ys`. Not exported.

- [ ] **Step 1: Write the helper**

Write `hydrus_research/sensitivity/_runner.py`:

```python
"""Internal — shared bookkeeping for the four sensitivity methods.

Given a SALib sample matrix `samples (N, D)` and a forward callable,
runs them through BatchRunner and returns `ys (N, M)` ready for SALib
analyzer functions."""
from __future__ import annotations
import time
from typing import Callable
import numpy as np

from ..batch import BatchRunner


def evaluate_samples(forward: Callable[[np.ndarray], np.ndarray],
                     samples: np.ndarray,
                     param_names: list[str],
                     obs_names: list[str],
                     n_workers: int = 1) -> tuple[np.ndarray, float]:
    """Run forward over `samples` and return (ys, wall_s_total).

    Failed runs propagate as NaN rows — SALib analyzers handle these
    by raising; callers should filter the input first or accept the
    failure as a flagged result. This helper does NOT silently drop
    rows."""
    runner = BatchRunner(forward=forward,
                         param_names=param_names,
                         obs_names=obs_names,
                         n_workers=n_workers,
                         show_progress=False)
    t0 = time.time()
    br = runner.run(samples)
    wall = time.time() - t0
    return br.ys, wall
```

- [ ] **Step 2: Commit (no test — exercised by Tasks 3-6)**

```bash
git add hydrus_research/sensitivity/_runner.py
git commit -m "M4.2: sensitivity._runner helper — BatchRunner glue"
```

---

### Task 3: Morris elementary effects

**Files:**
- Modify: `hydrus_research/sensitivity/morris.py` (replace stub)
- Create: `tests/research/sensitivity/test_morris.py`

- [ ] **Step 1: Write the failing test**

Write `tests/research/sensitivity/test_morris.py`:

```python
import numpy as np
import pytest
from hydrus_research.sensitivity import morris_screen, SensitivityResult


def ishigami(theta: np.ndarray) -> np.ndarray:
    """SALib's canonical 3-param test function: f = sin(x1) + 7sin²(x2) + 0.1·x3⁴·sin(x1)."""
    x1, x2, x3 = theta
    return np.array([np.sin(x1) + 7.0 * np.sin(x2) ** 2
                     + 0.1 * x3 ** 4 * np.sin(x1)])


class _IshigamiParamMap:
    """Minimal duck-typed ParameterMap for SALib's bounds_array() call."""
    def __init__(self):
        self.specs = []                     # not used by Morris
    @property
    def names(self): return ["x1", "x2", "x3"]
    def bounds_array(self):
        return np.array([[-np.pi, np.pi]] * 3)


def test_morris_returns_sensitivity_result():
    r = morris_screen(forward=ishigami,
                      param_map=_IshigamiParamMap(),
                      obs_names=["f"],
                      n_trajectories=20,
                      num_levels=4,
                      seed=42)
    assert isinstance(r, SensitivityResult)
    assert r.method == "morris"
    assert r.param_names == ["x1", "x2", "x3"]
    assert r.obs_names == ["f"]
    for key in ("mu", "mu_star", "sigma", "mu_star_conf"):
        assert key in r.indices
        # Each index is per observable, length D=3
        assert len(r.indices[key]) == 1            # 1 observable
        assert len(r.indices[key][0]) == 3         # 3 params


def test_morris_ishigami_ranking():
    """On Ishigami: x3 has interactions only (mu_star high; mu low), x2 has
    direct effects, x1 weakest. Morris should rank x3 highest by mu_star."""
    r = morris_screen(forward=ishigami,
                      param_map=_IshigamiParamMap(),
                      obs_names=["f"],
                      n_trajectories=40, num_levels=4, seed=42)
    mu_star = np.array(r.indices["mu_star"][0])
    ranked = np.argsort(-mu_star)             # descending
    # Documented Ishigami: |x3| > |x2| > |x1| in mu_star
    assert ranked[0] == 2, f"expected x3 ranked first; got {ranked}"
    assert ranked[-1] == 0, f"expected x1 ranked last; got {ranked}"
```

- [ ] **Step 2: Verify failure**

```bash
pytest tests/research/sensitivity/test_morris.py -v
```

Expected: NotImplementedError from the stub.

- [ ] **Step 3: Implement**

Replace `hydrus_research/sensitivity/morris.py`:

```python
"""Morris elementary effects — screening method.

Wraps SALib's morris.sample + morris.analyze. Useful for screening dozens
of parameters cheaply (~10·(D+1) forward calls)."""
from __future__ import annotations
from typing import Callable
import numpy as np

from .result import SensitivityResult
from ._runner import evaluate_samples


def morris_screen(forward: Callable[[np.ndarray], np.ndarray],
                  param_map,
                  obs_names: list[str],
                  n_trajectories: int = 20,
                  num_levels: int = 4,
                  seed: int | None = None,
                  n_workers: int = 1) -> SensitivityResult:
    """Morris elementary effects via SALib.

    `param_map` only needs `.names` and `.bounds_array()` (the duck-typed
    minimum). Pass an internal-coords ParameterMap if your forward expects
    internal coords; pass a user-coords ParameterMap if it expects user."""
    from SALib.sample import morris as morris_sample
    from SALib.analyze import morris as morris_analyze

    bounds = np.asarray(param_map.bounds_array(), dtype=float)
    problem = {
        "num_vars": len(param_map.names),
        "names": list(param_map.names),
        "bounds": bounds.tolist(),
    }
    samples = morris_sample.sample(problem, N=n_trajectories,
                                   num_levels=num_levels, seed=seed)
    ys, wall = evaluate_samples(forward, samples,
                                param_names=list(param_map.names),
                                obs_names=obs_names,
                                n_workers=n_workers)
    # SALib analyses one y vector at a time
    indices: dict[str, list[list[float]]] = {k: [] for k in
                                              ("mu", "mu_star", "sigma", "mu_star_conf")}
    for j in range(ys.shape[1]):
        Si = morris_analyze.analyze(problem, samples, ys[:, j],
                                    num_levels=num_levels, seed=seed)
        for key in indices:
            indices[key].append([float(v) for v in Si[key]])

    return SensitivityResult(
        method="morris",
        param_names=list(param_map.names),
        obs_names=obs_names,
        indices=indices,
        sample_size=samples.shape[0],
        forward_cost_s=wall,
        diagnostics={"n_trajectories": n_trajectories, "num_levels": num_levels},
    )
```

- [ ] **Step 4: Run + commit**

```bash
pytest tests/research/sensitivity/test_morris.py -v
git add hydrus_research/sensitivity/morris.py tests/research/sensitivity/test_morris.py
git commit -m "M4.3: Morris elementary effects (SALib wrapper + Ishigami ranking test)"
```

---

### Task 4: Sobol decomposition

**Files:**
- Modify: `hydrus_research/sensitivity/sobol.py` (replace stub)
- Create: `tests/research/sensitivity/test_sobol.py`

- [ ] **Step 1: Write the failing test**

Write `tests/research/sensitivity/test_sobol.py`:

```python
import numpy as np
import pytest
from hydrus_research.sensitivity import sobol_decompose, SensitivityResult


def ishigami(theta: np.ndarray) -> np.ndarray:
    x1, x2, x3 = theta
    return np.array([np.sin(x1) + 7.0 * np.sin(x2) ** 2
                     + 0.1 * x3 ** 4 * np.sin(x1)])


class _IshigamiParamMap:
    @property
    def names(self): return ["x1", "x2", "x3"]
    def bounds_array(self):
        return np.array([[-np.pi, np.pi]] * 3)


def test_sobol_returns_sensitivity_result():
    r = sobol_decompose(forward=ishigami, param_map=_IshigamiParamMap(),
                        obs_names=["f"], n_base=512, seed=42)
    assert isinstance(r, SensitivityResult)
    assert r.method == "sobol"
    for key in ("S1", "ST", "S1_conf", "ST_conf"):
        assert key in r.indices


def test_sobol_ishigami_indices_within_5pct():
    """Documented analytic Ishigami: S1 ≈ [0.314, 0.443, 0.0],
    ST ≈ [0.557, 0.443, 0.244]. With n_base=1024 we should be within ~5%."""
    r = sobol_decompose(forward=ishigami, param_map=_IshigamiParamMap(),
                        obs_names=["f"], n_base=1024, seed=42)
    S1 = np.array(r.indices["S1"][0])
    ST = np.array(r.indices["ST"][0])
    np.testing.assert_allclose(S1, [0.314, 0.443, 0.0], atol=0.05)
    np.testing.assert_allclose(ST, [0.557, 0.443, 0.244], atol=0.05)
```

- [ ] **Step 2: Verify failure**

```bash
pytest tests/research/sensitivity/test_sobol.py -v
```

Expected: NotImplementedError.

- [ ] **Step 3: Implement**

Replace `hydrus_research/sensitivity/sobol.py`:

```python
"""Sobol variance-based decomposition.

Wraps SALib's saltelli.sample + sobol.analyze. Cost: (2D+2)·N forward
calls. Use Morris first to screen down to ≤ 10 params, THEN Sobol for
quantitative decomposition."""
from __future__ import annotations
from typing import Callable
import numpy as np

from .result import SensitivityResult
from ._runner import evaluate_samples


def sobol_decompose(forward: Callable[[np.ndarray], np.ndarray],
                    param_map,
                    obs_names: list[str],
                    n_base: int = 1024,
                    calc_second_order: bool = False,
                    seed: int | None = None,
                    n_workers: int = 1) -> SensitivityResult:
    from SALib.sample import saltelli
    from SALib.analyze import sobol

    bounds = np.asarray(param_map.bounds_array(), dtype=float)
    problem = {
        "num_vars": len(param_map.names),
        "names": list(param_map.names),
        "bounds": bounds.tolist(),
    }
    samples = saltelli.sample(problem, n_base,
                              calc_second_order=calc_second_order, seed=seed)
    ys, wall = evaluate_samples(forward, samples,
                                param_names=list(param_map.names),
                                obs_names=obs_names,
                                n_workers=n_workers)
    base_keys = ["S1", "S1_conf", "ST", "ST_conf"]
    if calc_second_order:
        base_keys += ["S2", "S2_conf"]
    indices: dict[str, list] = {k: [] for k in base_keys}
    for j in range(ys.shape[1]):
        Si = sobol.analyze(problem, ys[:, j],
                           calc_second_order=calc_second_order,
                           seed=seed)
        for key in base_keys:
            v = Si[key]
            # S2 is a (D, D) matrix; flatten to list-of-lists
            indices[key].append(
                v.tolist() if hasattr(v, "tolist") else list(v))

    return SensitivityResult(
        method="sobol",
        param_names=list(param_map.names),
        obs_names=obs_names,
        indices=indices,
        sample_size=samples.shape[0],
        forward_cost_s=wall,
        diagnostics={"n_base": n_base, "calc_second_order": calc_second_order},
    )
```

- [ ] **Step 4: Run + commit**

```bash
pytest tests/research/sensitivity/test_sobol.py -v
git add hydrus_research/sensitivity/sobol.py tests/research/sensitivity/test_sobol.py
git commit -m "M4.4: Sobol decomposition (SALib + Ishigami within 5% tolerance)"
```

---

### Task 5: FAST (Fourier Amplitude Sensitivity Test)

**Files:**
- Modify: `hydrus_research/sensitivity/fast.py` (replace stub)
- Create: `tests/research/sensitivity/test_fast.py`

- [ ] **Step 1: Write the failing test**

Write `tests/research/sensitivity/test_fast.py`:

```python
import numpy as np
import pytest
from hydrus_research.sensitivity import fast_indices, SensitivityResult


def ishigami(theta: np.ndarray) -> np.ndarray:
    x1, x2, x3 = theta
    return np.array([np.sin(x1) + 7.0 * np.sin(x2) ** 2
                     + 0.1 * x3 ** 4 * np.sin(x1)])


class _IshigamiParamMap:
    @property
    def names(self): return ["x1", "x2", "x3"]
    def bounds_array(self):
        return np.array([[-np.pi, np.pi]] * 3)


def test_fast_returns_sensitivity_result():
    r = fast_indices(forward=ishigami, param_map=_IshigamiParamMap(),
                     obs_names=["f"], n=512, seed=42)
    assert isinstance(r, SensitivityResult)
    assert r.method == "fast"
    assert "S1" in r.indices and "ST" in r.indices


def test_fast_ranks_x2_x1_x3_correctly():
    """FAST on Ishigami: x2 strongest direct effect (sin² interaction);
    S1 ranking should be x2 > x1 > x3."""
    r = fast_indices(forward=ishigami, param_map=_IshigamiParamMap(),
                     obs_names=["f"], n=1024, seed=42)
    S1 = np.array(r.indices["S1"][0])
    assert S1[1] > S1[0] > 0, f"unexpected ranking; S1 = {S1}"
```

- [ ] **Step 2: Verify failure**

```bash
pytest tests/research/sensitivity/test_fast.py -v
```

- [ ] **Step 3: Implement**

Replace `hydrus_research/sensitivity/fast.py`:

```python
"""FAST / eFAST Fourier-based sensitivity indices."""
from __future__ import annotations
from typing import Callable
import numpy as np

from .result import SensitivityResult
from ._runner import evaluate_samples


def fast_indices(forward: Callable[[np.ndarray], np.ndarray],
                 param_map,
                 obs_names: list[str],
                 n: int = 1024,
                 m: int = 4,
                 seed: int | None = None,
                 n_workers: int = 1) -> SensitivityResult:
    from SALib.sample import fast_sampler
    from SALib.analyze import fast

    bounds = np.asarray(param_map.bounds_array(), dtype=float)
    problem = {
        "num_vars": len(param_map.names),
        "names": list(param_map.names),
        "bounds": bounds.tolist(),
    }
    samples = fast_sampler.sample(problem, N=n, M=m, seed=seed)
    ys, wall = evaluate_samples(forward, samples,
                                param_names=list(param_map.names),
                                obs_names=obs_names,
                                n_workers=n_workers)
    indices: dict[str, list[list[float]]] = {"S1": [], "ST": []}
    for j in range(ys.shape[1]):
        Si = fast.analyze(problem, ys[:, j], M=m)
        indices["S1"].append([float(v) for v in Si["S1"]])
        indices["ST"].append([float(v) for v in Si["ST"]])

    return SensitivityResult(
        method="fast",
        param_names=list(param_map.names),
        obs_names=obs_names,
        indices=indices,
        sample_size=samples.shape[0],
        forward_cost_s=wall,
        diagnostics={"n": n, "m": m},
    )
```

- [ ] **Step 4: Run + commit**

```bash
pytest tests/research/sensitivity/test_fast.py -v
git add hydrus_research/sensitivity/fast.py tests/research/sensitivity/test_fast.py
git commit -m "M4.5: FAST/eFAST sensitivity (SALib + Ishigami ranking)"
```

---

### Task 6: PAWN distribution-based sensitivity

**Files:**
- Modify: `hydrus_research/sensitivity/pawn.py` (replace stub)
- Create: `tests/research/sensitivity/test_pawn.py`

- [ ] **Step 1: Write the failing test**

Write `tests/research/sensitivity/test_pawn.py`:

```python
import numpy as np
import pytest
from hydrus_research.sensitivity import pawn_kde, SensitivityResult


def ishigami(theta: np.ndarray) -> np.ndarray:
    x1, x2, x3 = theta
    return np.array([np.sin(x1) + 7.0 * np.sin(x2) ** 2
                     + 0.1 * x3 ** 4 * np.sin(x1)])


class _IshigamiParamMap:
    @property
    def names(self): return ["x1", "x2", "x3"]
    def bounds_array(self):
        return np.array([[-np.pi, np.pi]] * 3)


def test_pawn_returns_sensitivity_result():
    r = pawn_kde(forward=ishigami, param_map=_IshigamiParamMap(),
                 obs_names=["f"], n=2000, s=10, seed=42)
    assert isinstance(r, SensitivityResult)
    assert r.method == "pawn"
    # PAWN returns minimum, mean, median, maximum, CV per param
    for key in ("minimum", "mean", "median", "maximum", "CV"):
        assert key in r.indices
        assert len(r.indices[key][0]) == 3
```

- [ ] **Step 2: Verify failure**

```bash
pytest tests/research/sensitivity/test_pawn.py -v
```

- [ ] **Step 3: Implement**

Replace `hydrus_research/sensitivity/pawn.py`:

```python
"""PAWN distribution-based sensitivity — non-parametric, robust to
non-linear / non-monotonic responses."""
from __future__ import annotations
from typing import Callable
import numpy as np

from .result import SensitivityResult
from ._runner import evaluate_samples


def pawn_kde(forward: Callable[[np.ndarray], np.ndarray],
             param_map,
             obs_names: list[str],
             n: int = 2000,
             s: int = 10,
             seed: int | None = None,
             n_workers: int = 1) -> SensitivityResult:
    from SALib.sample import latin
    from SALib.analyze import pawn

    bounds = np.asarray(param_map.bounds_array(), dtype=float)
    problem = {
        "num_vars": len(param_map.names),
        "names": list(param_map.names),
        "bounds": bounds.tolist(),
    }
    samples = latin.sample(problem, n, seed=seed)
    ys, wall = evaluate_samples(forward, samples,
                                param_names=list(param_map.names),
                                obs_names=obs_names,
                                n_workers=n_workers)
    keys = ["minimum", "mean", "median", "maximum", "CV"]
    indices: dict[str, list[list[float]]] = {k: [] for k in keys}
    for j in range(ys.shape[1]):
        Si = pawn.analyze(problem, samples, ys[:, j], S=s)
        for key in keys:
            indices[key].append([float(v) for v in Si[key]])

    return SensitivityResult(
        method="pawn",
        param_names=list(param_map.names),
        obs_names=obs_names,
        indices=indices,
        sample_size=samples.shape[0],
        forward_cost_s=wall,
        diagnostics={"n": n, "S": s},
    )
```

- [ ] **Step 4: Run + commit**

```bash
pytest tests/research/sensitivity/test_pawn.py -v
git add hydrus_research/sensitivity/pawn.py tests/research/sensitivity/test_pawn.py
git commit -m "M4.6: PAWN distribution-based sensitivity (SALib)"
```

---

### Task 7: REST `/research/sensitivity/{method}` endpoints

**Files:**
- Create: `hydrus_port_server/routers/research_sensitivity.py`
- Modify: `hydrus_port_server/app.py:build_app()`
- Create: `tests/research/sensitivity/test_rest.py`

- [ ] **Step 1: Write the failing test**

Write `tests/research/sensitivity/test_rest.py`:

```python
import pytest

try:
    from fastapi.testclient import TestClient
except ImportError:
    pytest.skip("FastAPI not installed", allow_module_level=True)

from hydrus_port_server.app import build_app


@pytest.fixture(scope="module")
def client():
    return TestClient(build_app())


def _minimal_payload():
    return {
        "scenario_dir": "tests/fixtures/infiltr_v1/inputs",
        "params": [
            {"name": "alpha", "target": "materials[0].alpha",
             "bounds": [0.005, 0.05], "transform": "log"},
        ],
        "obs": [
            {"name": "theta_z30_d1", "kind": "theta",
             "location": {"z_cm": -30.0}, "time_day": 1.0},
        ],
        "n": 8,                             # tiny — just to confirm the route wiring
        "workers": 1,
        "seed": 42,
    }


def test_morris_endpoint_returns_indices(client):
    r = client.post("/research/sensitivity/morris", json=_minimal_payload())
    # 8 trajectories * (1+1) = 16 forward calls; ~2 minutes on a laptop —
    # the test is slow but verifies the wiring end-to-end.
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["method"] == "morris"
    assert "mu_star" in body["indices"]


def test_unknown_method_404(client):
    r = client.post("/research/sensitivity/cubism", json=_minimal_payload())
    assert r.status_code == 404
```

- [ ] **Step 2: Verify failure**

```bash
pytest tests/research/sensitivity/test_rest.py -v
```

Expected: 404 on both (router not registered).

- [ ] **Step 3: Implement**

Write `hydrus_port_server/routers/research_sensitivity.py`:

```python
"""/research/sensitivity/{method} — F2 REST surface.

Synchronous endpoint: runs the sweep + analysis in-process and returns
the SensitivityResult. For long sweeps the client should use the M3
/research/batch/* async pattern instead."""
from __future__ import annotations
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel


router = APIRouter()


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


class SensitivityRequest(BaseModel):
    scenario_dir: str
    params: list[ParamSpecPayload]
    obs: list[ObsSpecPayload]
    n: int = 100
    workers: int = 1
    seed: int | None = None
    # Method-specific knobs (optional)
    num_levels: int = 4                    # morris
    calc_second_order: bool = False        # sobol
    m: int = 4                              # fast
    s: int = 10                             # pawn


_VALID_METHODS = {"morris", "sobol", "fast", "pawn"}


@router.post("/{method}")
def run(method: str, req: SensitivityRequest):
    if method not in _VALID_METHODS:
        raise HTTPException(status_code=404,
                            detail=f"unknown method {method!r}; "
                                   f"available: {sorted(_VALID_METHODS)}")
    from hydrus_research.parameters import ParameterSpec, ParameterMap
    from hydrus_research.observations import ObservationSpec
    from hydrus_research.simulator import make_forward
    from hydrus_research.simulator.hydrus1d_adapter import Hydrus1DSimulator
    from hydrus_port.adapters.hydrus1d import load as _load_h1d

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
    obs_names = [o.name for o in req.obs]

    if method == "morris":
        from hydrus_research.sensitivity import morris_screen
        r = morris_screen(forward, pm, obs_names,
                          n_trajectories=req.n, num_levels=req.num_levels,
                          seed=req.seed, n_workers=req.workers)
    elif method == "sobol":
        from hydrus_research.sensitivity import sobol_decompose
        r = sobol_decompose(forward, pm, obs_names,
                            n_base=req.n,
                            calc_second_order=req.calc_second_order,
                            seed=req.seed, n_workers=req.workers)
    elif method == "fast":
        from hydrus_research.sensitivity import fast_indices
        r = fast_indices(forward, pm, obs_names,
                         n=req.n, m=req.m,
                         seed=req.seed, n_workers=req.workers)
    else:                                    # pawn
        from hydrus_research.sensitivity import pawn_kde
        r = pawn_kde(forward, pm, obs_names,
                     n=req.n, s=req.s,
                     seed=req.seed, n_workers=req.workers)

    return r.model_dump()
```

In `hydrus_port_server/app.py:build_app()`, append:

```python
    try:
        from .routers.research_sensitivity import router as sens_router
        app.include_router(sens_router, prefix="/research/sensitivity",
                           tags=["research", "sensitivity"])
    except ImportError:
        pass
```

- [ ] **Step 4: Run + commit**

```bash
pytest tests/research/sensitivity/test_rest.py -v
git add hydrus_port_server/routers/research_sensitivity.py hydrus_port_server/app.py tests/research/sensitivity/test_rest.py
git commit -m "M4.7: /research/sensitivity/{morris,sobol,fast,pawn} REST"
```

(Test takes ~3-5 minutes due to real solver calls; expected.)

---

### Task 8: CLI `hydrus research sensitize`

**Files:**
- Modify: `hydrus_port/cli.py:_build_research_subparser` (append `p_sens` parser)
- Create: `tests/research/sensitivity/test_cli.py`

- [ ] **Step 1: Write failing test**

Write `tests/research/sensitivity/test_cli.py`:

```python
import json
import subprocess
import sys
from pathlib import Path


def test_cli_sensitize_morris(tmp_path):
    out_json = tmp_path / "morris.json"
    r = subprocess.run(
        [sys.executable, "-m", "hydrus_port.cli",
         "research", "sensitize",
         "tests/fixtures/infiltr_v1/inputs",
         "--method", "morris",
         "--param", "materials[0].alpha:0.005:0.05:log",
         "--obs", "theta@-30cm,t=1.0",
         "--n", "4", "--workers", "1", "--seed", "42",
         "--out", str(out_json)],
        capture_output=True, text=True,
    )
    assert r.returncode == 0, r.stderr
    assert out_json.exists()
    body = json.loads(out_json.read_text())
    assert body["method"] == "morris"
    assert "mu_star" in body["indices"]


def test_cli_sensitize_help_lists_methods():
    r = subprocess.run(
        [sys.executable, "-m", "hydrus_port.cli",
         "research", "sensitize", "--help"],
        capture_output=True, text=True,
    )
    assert r.returncode == 0
    out = r.stdout
    for m in ("morris", "sobol", "fast", "pawn"):
        assert m in out
```

- [ ] **Step 2: Implement**

In `hydrus_port/cli.py`, inside `_build_research_subparser(sub)`, after the existing sweep/worker blocks, append:

```python
    # ----- sensitize (M4) --------------------------------------------
    p_sens = rsub.add_parser("sensitize", help="global sensitivity analysis")
    p_sens.add_argument("scenario_dir", help="scenario inputs directory")
    p_sens.add_argument("--method", required=True,
                        choices=["morris", "sobol", "fast", "pawn"])
    p_sens.add_argument("--param", action="append", required=True,
                        help="target:lo:hi[:transform]; repeat for multi-D")
    p_sens.add_argument("--obs", required=True,
                        help="semicolon-separated obs specs (kind@-Ncm,t=T;...)")
    p_sens.add_argument("--n", type=int, default=100)
    p_sens.add_argument("--workers", type=int, default=1)
    p_sens.add_argument("--seed", type=int, default=None)
    p_sens.add_argument("--out", required=True, help="output JSON path")
    p_sens.set_defaults(_cmd=_cmd_research_sensitize)
```

And add the helper alongside other `_cmd_research_*`:

```python
def _cmd_research_sensitize(args: argparse.Namespace) -> int:
    import json as _json
    from pathlib import Path as _P
    from hydrus_research.parameters import ParameterSpec, ParameterMap
    from hydrus_research.observations import ObservationSpec
    from hydrus_research.simulator import make_forward
    from hydrus_research.simulator.hydrus1d_adapter import Hydrus1DSimulator
    from hydrus_port.adapters.hydrus1d import load as _load_h1d

    specs = []
    for s in args.param:
        parts = s.split(":")
        target, lo, hi = parts[0], float(parts[1]), float(parts[2])
        transform = parts[3] if len(parts) > 3 else "linear"
        name = target.rsplit(".", 1)[-1]
        specs.append(ParameterSpec(name=name, target=target,
                                   bounds=(lo, hi), transform=transform))
    pm = ParameterMap(specs)

    obs_specs = []
    for chunk in args.obs.split(";"):
        kind, _, rest = chunk.partition("@")
        loc_part, _, t_part = rest.partition(",t=")
        z_cm = float(loc_part.rstrip("cm"))
        t = float(t_part)
        obs_specs.append(ObservationSpec(name=f"{kind}_{loc_part}_{t}",
                                         kind=kind, location={"z_cm": z_cm},
                                         time_day=t))

    template = _load_h1d(_P(args.scenario_dir)).to_dict()
    sim = Hydrus1DSimulator()
    forward = make_forward(sim, pm, template_scenario=template,
                           forcing=None, ic=None, obs_specs=obs_specs)
    obs_names = [o.name for o in obs_specs]

    if args.method == "morris":
        from hydrus_research.sensitivity import morris_screen
        r = morris_screen(forward, pm, obs_names,
                          n_trajectories=args.n, seed=args.seed,
                          n_workers=args.workers)
    elif args.method == "sobol":
        from hydrus_research.sensitivity import sobol_decompose
        r = sobol_decompose(forward, pm, obs_names,
                            n_base=args.n, seed=args.seed,
                            n_workers=args.workers)
    elif args.method == "fast":
        from hydrus_research.sensitivity import fast_indices
        r = fast_indices(forward, pm, obs_names,
                         n=args.n, seed=args.seed,
                         n_workers=args.workers)
    else:
        from hydrus_research.sensitivity import pawn_kde
        r = pawn_kde(forward, pm, obs_names,
                     n=args.n, seed=args.seed,
                     n_workers=args.workers)

    out = _P(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(_json.dumps(r.model_dump(), indent=2))
    print(f"sensitivity ({args.method}) written to {out}")
    return 0
```

- [ ] **Step 3: Run + commit**

```bash
pytest tests/research/sensitivity/test_cli.py -v
git add hydrus_port/cli.py tests/research/sensitivity/test_cli.py
git commit -m "M4.8: hydrus research sensitize CLI"
```

---

### Task 9: GUI — SensitivityReport.vue + 2 reusable viz components

**Files:**
- Create: `desktop/src/components/SensitivityIndicesBar.vue`
- Create: `desktop/src/components/MorrisEEPlot.vue`
- Create: `desktop/src/pages/research/SensitivityReport.vue`
- Modify: `desktop/src/api.ts` — append `sensitivity.*` wrapper
- Modify: `desktop/src/App.vue` — add `Sensitivity` tab

- [ ] **Step 1: Append `sensitivity` wrapper to `desktop/src/api.ts`**

```ts
// ---- M4: Sensitivity analysis ---------------------------------------
export interface SensitivityRequest {
  scenario_dir: string;
  params: BatchParamSpec[];
  obs: BatchObsSpec[];
  n: number;
  workers: number;
  seed?: number;
  num_levels?: number;
  calc_second_order?: boolean;
  m?: number;
  s?: number;
}

export interface SensitivityResult {
  method: "morris" | "sobol" | "fast" | "pawn";
  param_names: string[];
  obs_names: string[];
  indices: Record<string, number[][] | number[]>;
  sample_size: number;
  forward_cost_s: number;
  diagnostics: Record<string, any>;
}

export const sensitivity = {
  async run(method: "morris" | "sobol" | "fast" | "pawn",
            req: SensitivityRequest): Promise<SensitivityResult> {
    const r = await fetch(`${RESEARCH_BASE}/research/sensitivity/${method}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(req),
    });
    if (!r.ok) throw new Error(`HTTP ${r.status}: ${await r.text()}`);
    return r.json();
  },
};
```

- [ ] **Step 2: Write `SensitivityIndicesBar.vue`**

```vue
<template>
  <div ref="el" class="bars"></div>
</template>

<script setup lang="ts">
import { onMounted, ref, watch } from "vue";
import Plotly from "plotly.js-dist-min";

const props = defineProps<{
  paramNames: string[];
  indices: Record<string, number[]>;
  title?: string;
}>();
const el = ref<HTMLDivElement | null>(null);

function _draw() {
  if (!el.value) return;
  const traces = Object.entries(props.indices).map(([key, vals]) => ({
    type: "bar", x: props.paramNames, y: vals, name: key,
  }));
  Plotly.newPlot(el.value, traces, {
    title: props.title ?? "",
    barmode: "group",
    yaxis: { title: "index value" },
    margin: { t: 40, l: 60, r: 20, b: 60 },
  }, { responsive: true, displayModeBar: false });
}

onMounted(_draw);
watch(() => [props.paramNames, props.indices], _draw, { deep: true });
</script>

<style scoped>
.bars { width: 100%; height: 360px; }
</style>
```

- [ ] **Step 3: Write `MorrisEEPlot.vue`**

```vue
<template>
  <div ref="el" class="ee"></div>
</template>

<script setup lang="ts">
import { onMounted, ref, watch } from "vue";
import Plotly from "plotly.js-dist-min";

const props = defineProps<{
  paramNames: string[];
  muStar: number[];
  sigma: number[];
}>();
const el = ref<HTMLDivElement | null>(null);

function _draw() {
  if (!el.value) return;
  Plotly.newPlot(el.value, [{
    type: "scatter", mode: "markers+text",
    x: props.muStar, y: props.sigma,
    text: props.paramNames, textposition: "top center",
    marker: { size: 12, color: "#1f77b4" },
  }], {
    xaxis: { title: "μ* (overall effect magnitude)" },
    yaxis: { title: "σ (interaction / nonlinearity)" },
    margin: { t: 20, l: 60, r: 20, b: 50 },
  }, { responsive: true, displayModeBar: false });
}
onMounted(_draw);
watch(() => [props.paramNames, props.muStar, props.sigma], _draw, { deep: true });
</script>

<style scoped>
.ee { width: 100%; height: 380px; }
</style>
```

- [ ] **Step 4: Write `SensitivityReport.vue`**

```vue
<template>
  <div class="sens-report">
    <h2>Sensitivity Report — F2</h2>
    <div class="form">
      <label>Scenario dir
        <input v-model="scenarioDir" />
      </label>
      <label>Params (one per line, target:lo:hi[:transform])
        <textarea v-model="paramText" rows="3"></textarea>
      </label>
      <label>Obs depth (cm) <input v-model.number="obsDepth" type="number" /></label>
      <label>Obs time (days) <input v-model.number="obsTime" type="number" step="0.1" /></label>
      <label>Method
        <select v-model="method">
          <option value="morris">Morris EE (screening)</option>
          <option value="sobol">Sobol (variance)</option>
          <option value="fast">FAST</option>
          <option value="pawn">PAWN (distribution)</option>
        </select>
      </label>
      <label>N <input v-model.number="n" type="number" min="1" /></label>
      <label>Workers <input v-model.number="workers" type="number" min="1" /></label>
      <button @click="run" :disabled="running">Run</button>
    </div>

    <p v-if="running">Running… ({{ Math.round(elapsed) }}s)</p>
    <p v-if="error" class="err">{{ error }}</p>

    <div v-if="result">
      <p>{{ result.method }} — {{ result.sample_size }} samples, {{ result.forward_cost_s.toFixed(1) }}s</p>
      <MorrisEEPlot v-if="result.method === 'morris'"
                    :param-names="result.param_names"
                    :mu-star="(result.indices.mu_star as number[][])[0]"
                    :sigma="(result.indices.sigma as number[][])[0]" />
      <SensitivityIndicesBar v-else
                             :param-names="result.param_names"
                             :indices="firstObsIndices"
                             :title="`${result.method} indices`" />
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed } from "vue";
import { sensitivity, type SensitivityResult } from "../../api";
import SensitivityIndicesBar from "../../components/SensitivityIndicesBar.vue";
import MorrisEEPlot from "../../components/MorrisEEPlot.vue";

const scenarioDir = ref("tests/fixtures/infiltr_v1/inputs");
const paramText = ref("materials[0].alpha:0.005:0.05:log\nmaterials[0].n:1.1:2.0:linear");
const obsDepth = ref(-30);
const obsTime = ref(1.0);
const method = ref<"morris" | "sobol" | "fast" | "pawn">("morris");
const n = ref(20);
const workers = ref(2);
const result = ref<SensitivityResult | null>(null);
const error = ref<string | null>(null);
const running = ref(false);
const elapsed = ref(0);

function _parseParams() {
  return paramText.value.trim().split("\n").map(line => {
    const parts = line.split(":");
    const target = parts[0].trim();
    const lo = parseFloat(parts[1]); const hi = parseFloat(parts[2]);
    const transform = (parts[3]?.trim() as any) ?? "linear";
    const name = target.split(".").pop() ?? target;
    return { name, target, bounds: [lo, hi] as [number, number], transform };
  });
}

async function run() {
  running.value = true; error.value = null; result.value = null;
  const t0 = Date.now();
  const tick = setInterval(() => { elapsed.value = (Date.now() - t0) / 1000; }, 200);
  try {
    result.value = await sensitivity.run(method.value, {
      scenario_dir: scenarioDir.value,
      params: _parseParams(),
      obs: [{
        name: `theta_${obsDepth.value}_${obsTime.value}`,
        kind: "theta", location: { z_cm: obsDepth.value }, time_day: obsTime.value,
      }],
      n: n.value, workers: workers.value,
    });
  } catch (e: any) {
    error.value = e.message ?? String(e);
  } finally {
    clearInterval(tick); running.value = false;
  }
}

const firstObsIndices = computed(() => {
  if (!result.value) return {};
  const out: Record<string, number[]> = {};
  for (const [k, v] of Object.entries(result.value.indices)) {
    if (k.endsWith("_conf")) continue;
    const first = (v as number[][])[0];
    if (first) out[k] = first;
  }
  return out;
});
</script>

<style scoped>
.sens-report { padding: 16px; max-width: 900px; }
.form { display: grid; grid-template-columns: 200px 1fr; gap: 6px; align-items: start; margin-bottom: 12px; }
.form input, .form select, .form textarea { padding: 4px; font-family: inherit; }
.form button { grid-column: 1 / 3; padding: 8px; margin-top: 8px; }
.err { color: #c00; }
</style>
```

- [ ] **Step 5: Add tab to App.vue**

In `desktop/src/App.vue`:

1. Add import: `import SensitivityReport from "./pages/research/SensitivityReport.vue";`
2. Widen `rightTab` union: `"3d" | "editor" | "dndc" | "soil-library" | "batch" | "sensitivity"`
3. Add tab button after Batch Sweep: `<button class="tab" :class="{active: rightTab === 'sensitivity'}" @click="rightTab = 'sensitivity'">Sensitivity</button>`
4. Add conditional render: `<SensitivityReport v-else-if="rightTab === 'sensitivity'" />`

- [ ] **Step 6: Commit**

```bash
git add desktop/src/api.ts desktop/src/components/SensitivityIndicesBar.vue desktop/src/components/MorrisEEPlot.vue desktop/src/pages/research/SensitivityReport.vue desktop/src/App.vue
git commit -m "M4.9: SensitivityReport.vue + bar/EE viz + sensitivity REST wrapper + nav tab"
```

---

### Task 10: End-to-end + regression + M4-complete marker

**Files:**
- Create: `tests/research/sensitivity/test_e2e_m4.py`

- [ ] **Step 1: Write acceptance test**

Write `tests/research/sensitivity/test_e2e_m4.py`:

```python
"""M4 acceptance: Morris on infiltr_v1 with 3 params + sanity ranking."""
import numpy as np
import pytest
from pathlib import Path

from hydrus_research.sensitivity import morris_screen
from hydrus_research.parameters import ParameterSpec, ParameterMap
from hydrus_research.observations import ObservationSpec
from hydrus_research.simulator import make_forward
from hydrus_research.simulator.hydrus1d_adapter import Hydrus1DSimulator
from hydrus_port.adapters.hydrus1d import load as load_h1d_canonical


def test_morris_on_infiltr_v1():
    template = load_h1d_canonical(Path("tests/fixtures/infiltr_v1/inputs")).to_dict()
    a0 = template["materials"][0]["alpha"]
    n0 = template["materials"][0]["n"]
    Ks0 = template["materials"][0]["Ks"]
    pm = ParameterMap([
        ParameterSpec(name="alpha", target="materials[0].alpha",
                      bounds=(a0 * 0.5, a0 * 2.0), transform="log"),
        ParameterSpec(name="n", target="materials[0].n",
                      bounds=(max(1.05, n0 * 0.8), n0 * 1.5), transform="linear"),
        ParameterSpec(name="Ks", target="materials[0].Ks",
                      bounds=(Ks0 * 0.2, Ks0 * 5.0), transform="log"),
    ])
    obs = [ObservationSpec(name="theta_z30_d1", kind="theta",
                           location={"z_cm": -30.0}, time_day=1.0)]
    sim = Hydrus1DSimulator()
    forward = make_forward(sim, pm, template_scenario=template,
                           forcing=None, ic=None, obs_specs=obs)
    r = morris_screen(forward, pm, ["theta_z30_d1"],
                      n_trajectories=4, num_levels=4, seed=42, n_workers=2)
    # 4 trajectories × (3+1) = 16 forward calls; ~2 min on a laptop
    assert r.method == "morris"
    assert len(r.indices["mu_star"][0]) == 3
    # All mu_star ≥ 0; at least one parameter has visible effect
    mu_star = np.array(r.indices["mu_star"][0])
    assert (mu_star >= 0).all()
    assert mu_star.max() > 1e-6, f"all mu_star ~ 0; sweep had no effect: {mu_star}"
```

- [ ] **Step 2: Run + regression + marker**

```bash
pytest tests/research/sensitivity/test_e2e_m4.py -v -s
pytest tests/research/ -q --ignore=tests/research/dndc_seam/test_gui_smoke.py 2>&1 | tail -5
hydrus test 1d 2>&1 | tail -3
git add tests/research/sensitivity/test_e2e_m4.py
git commit -m "M4.10: end-to-end Morris on infiltr_v1 + regression"
git commit --allow-empty -m "M4 complete: SALib sensitivity (Morris/Sobol/FAST/PAWN) green; ready for M5"
```

---

## Definition of Done for M4

1. `pytest tests/research/sensitivity/ -v` — all green.
2. `pytest tests/research/ -q` — no regression in M0/M1/M2/M3.
3. All four `*screen / *decompose / *indices / *kde` callables importable.
4. Ishigami benchmarks pass within stated tolerances (Sobol within 5%; Morris ranks `x3 > x1`; FAST has `S1[x2] > S1[x1]`).
5. `hydrus research sensitize ... --method morris ... --out file.json` works.
6. REST `POST /research/sensitivity/morris` returns 200 + indices.
7. GUI `Research → Sensitivity` tab renders; Morris EE scatter + bar chart visible.
8. `hydrus test 1d/2d/roundtrip` still PASS.
