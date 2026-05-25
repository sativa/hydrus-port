# M10 — Decision Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement F6 — irrigation / fertigation schedule optimization. Two backends:
- **pymoo NSGA-II** — multi-objective Pareto-front search (e.g. minimise N leaching AND minimise water use simultaneously).
- **Optuna** — single-objective (e.g. maximise WUE or NUE under constraints).

Decision variables encode irrigation amounts + fert events as a flat θ vector; the forward model evaluates one candidate schedule and returns the objective(s).

**Architecture:** New `hydrus_research/optimization/` sub-package. Each backend has a thin wrapper that converts the user's schedule-decision-vars → θ vector → forward call → objective values. Constraints (field capacity, regulatory N caps) handled by pymoo's constraint API or Optuna's pruner. P2 scope per spec — keeps the GUI minimal.

**Tech Stack:** Python 3.10+, `pymoo>=0.6` + `optuna` (in `[research-opt]` extras; both lazy), `numpy`. Independent of M6-M9 — parallel.

**Spec reference:** `DOCS/superpowers/specs/2026-05-24-hydrus-research-platform-design.md` §4.7 + §0.2 (P2 scope).

**Acceptance:**
- `python -c "from hydrus_research.optimization import nsga_optimize, optuna_optimize, OptimizationResult"` works.
- `nsga_optimize` on a synthetic 2-objective 2-param toy problem returns a non-trivial Pareto front (≥ 5 non-dominated points after 20 generations).
- `optuna_optimize` on a synthetic 1-objective minimisation problem converges to within 5% of the analytic minimum after 30 trials.
- `pytest tests/research/optimization/` green (both tests SKIP if libs missing).
- M0-M9 tests green.

---

## File Layout

**Created:**
- `hydrus_research/optimization/__init__.py`
- `hydrus_research/optimization/result.py` — `OptimizationResult` Pydantic.
- `hydrus_research/optimization/pymoo_nsga.py` — `nsga_optimize(forward, bounds, ...)`.
- `hydrus_research/optimization/optuna_single.py` — `optuna_optimize(forward, bounds, ...)`.
- `hydrus_research/optimization/decision_vars.py` — `encode_schedule(events) → theta` + `decode(theta) → events`.
- `hydrus_research/optimization/constraints.py` — simple checker helpers.
- `hydrus_research/optimization/api.py` — re-exports.
- `tests/research/optimization/{__init__,test_result,test_pymoo,test_optuna,test_decision_vars,test_e2e_m10}.py`

**Modified:**
- `hydrus_port/cli.py:_build_research_subparser` — add `hydrus research optimize` subcommand (delegates to `nsga` or `optuna`).

(No REST or GUI in this milestone — P2 spec scope.)

---

### Task 1: Skeleton + OptimizationResult + decision_vars

**Files:** `__init__.py`, `result.py`, `decision_vars.py`, 2 method stubs, `test_result.py`, `test_decision_vars.py`.

- [ ] **Step 1: Schema + decision encoder**

`hydrus_research/optimization/__init__.py`:
```python
"""Decision optimization (F6, P2).

Two backends:
  - pymoo NSGA-II — multi-objective Pareto search
  - Optuna       — single-objective (TPE / random / CMA-ES)

Decision variables encode irrigation / fert event schedules.

See DOCS/superpowers/specs/2026-05-24-hydrus-research-platform-design.md §4.7.
"""
from .result import OptimizationResult
from .pymoo_nsga import nsga_optimize
from .optuna_single import optuna_optimize
from .decision_vars import encode_schedule, decode_schedule

__all__ = ["OptimizationResult", "nsga_optimize", "optuna_optimize",
           "encode_schedule", "decode_schedule"]
```

`hydrus_research/optimization/result.py`:
```python
"""OptimizationResult — typed output of nsga/optuna runs."""
from __future__ import annotations
from typing import Literal
from pydantic import BaseModel, ConfigDict


OptMethod = Literal["nsga2", "nsga3", "optuna_tpe", "optuna_random", "optuna_cmaes"]


class OptimizationResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    method: OptMethod
    param_names: list[str]
    objective_names: list[str]
    pareto_thetas: list[list[float]]       # (N_pareto, D); for single-obj, just the best as [[θ*]]
    pareto_objectives: list[list[float]]   # (N_pareto, n_obj)
    history: list[list[float]] = []         # objective trace per evaluation (single-obj) or generation (multi)
    n_evaluations: int
    wall_s: float
    diagnostics: dict = {}
```

`hydrus_research/optimization/decision_vars.py`:
```python
"""Encode irrigation/fert event schedules as flat θ vectors for optimizers.

Convention: each event contributes (amount, day_offset) to the vector;
events are ordered by their slot in the schedule. Optimizers see a
flat 2N-vector for an N-event schedule."""
from __future__ import annotations
import numpy as np


def encode_schedule(events: list[dict]) -> np.ndarray:
    """events: list of {"amount": float, "day": float}; returns flat 2N array."""
    out = np.empty(2 * len(events), dtype=float)
    for i, e in enumerate(events):
        out[2*i] = float(e["amount"])
        out[2*i + 1] = float(e["day"])
    return out


def decode_schedule(theta: np.ndarray) -> list[dict]:
    if theta.shape[0] % 2 != 0:
        raise ValueError("theta length must be even (amount, day pairs)")
    n = theta.shape[0] // 2
    return [{"amount": float(theta[2*i]), "day": float(theta[2*i + 1])}
            for i in range(n)]
```

Stubs `pymoo_nsga.py`, `optuna_single.py`: each has one function raising NotImplementedError.

- [ ] **Step 2: Tests + commit**

`test_result.py`:
```python
import pytest
from hydrus_research.optimization import OptimizationResult


def test_result_construction():
    r = OptimizationResult(method="nsga2",
                           param_names=["a", "b"], objective_names=["o1", "o2"],
                           pareto_thetas=[[0.1, 0.2], [0.3, 0.4]],
                           pareto_objectives=[[1.0, 2.0], [1.5, 1.0]],
                           n_evaluations=200, wall_s=5.0)
    assert r.method == "nsga2"
    assert len(r.pareto_thetas) == 2
```

`test_decision_vars.py`:
```python
import numpy as np
from hydrus_research.optimization import encode_schedule, decode_schedule


def test_roundtrip():
    events = [{"amount": 1.5, "day": 10}, {"amount": 2.0, "day": 30}]
    theta = encode_schedule(events)
    assert theta.shape == (4,)
    events2 = decode_schedule(theta)
    assert events2[0]["amount"] == 1.5
    assert events2[1]["day"] == 30
```

```bash
mkdir -p hydrus_research/optimization tests/research/optimization
touch tests/research/optimization/__init__.py
pytest tests/research/optimization/test_result.py tests/research/optimization/test_decision_vars.py -v
git add hydrus_research/optimization/ tests/research/optimization/
git commit -m "M10.1: optimization skeleton + OptimizationResult + decision_vars + stubs"
```

---

### Task 2: pymoo NSGA-II wrapper

**Files:** `pymoo_nsga.py` (replace stub), `test_pymoo.py`.

- [ ] **Step 1: Test (skip if pymoo missing)**

```python
import numpy as np
import pytest

pymoo = pytest.importorskip("pymoo", reason="pymoo not installed; in [research-opt]")
from hydrus_research.optimization import nsga_optimize, OptimizationResult


def test_nsga_on_toy_problem():
    """Minimise two conflicting objectives:
       f1 = (theta[0] - 1)^2 + (theta[1] - 2)^2
       f2 = (theta[0] + 1)^2 + (theta[1] + 2)^2
    Pareto front is a non-trivial curve in objective space."""
    def fwd(theta):
        f1 = (theta[0] - 1) ** 2 + (theta[1] - 2) ** 2
        f2 = (theta[0] + 1) ** 2 + (theta[1] + 2) ** 2
        return np.array([f1, f2])
    bounds = np.array([[-3.0, 3.0], [-3.0, 3.0]])
    r = nsga_optimize(forward=fwd, bounds=bounds,
                      param_names=["x", "y"], objective_names=["f1", "f2"],
                      pop_size=20, n_gen=10, seed=42)
    assert isinstance(r, OptimizationResult)
    assert r.method in ("nsga2", "nsga3")
    assert len(r.pareto_thetas) >= 5
    # Pareto front: non-dominated points
    objs = np.array(r.pareto_objectives)
    assert objs.shape[1] == 2
```

- [ ] **Step 2: Implement**

```python
"""pymoo NSGA-II / NSGA-III multi-objective optimization."""
from __future__ import annotations
import time
from typing import Callable
import numpy as np

from .result import OptimizationResult


def nsga_optimize(forward: Callable[[np.ndarray], np.ndarray],
                  bounds: np.ndarray,
                  param_names: list[str],
                  objective_names: list[str],
                  pop_size: int = 50,
                  n_gen: int = 20,
                  seed: int | None = None,
                  variant: str = "nsga2") -> OptimizationResult:
    try:
        from pymoo.core.problem import Problem
        from pymoo.algorithms.moo.nsga2 import NSGA2
        from pymoo.algorithms.moo.nsga3 import NSGA3
        from pymoo.optimize import minimize
        from pymoo.util.ref_dirs import get_reference_directions
    except ImportError as e:
        raise ImportError(
            "nsga_optimize requires pymoo. Install with:\n"
            "    pip install 'hydrus-port[research,research-opt]'"
        ) from e

    bounds = np.asarray(bounds, dtype=float)
    D = bounds.shape[0]
    M = len(objective_names)

    class _UserProblem(Problem):
        def __init__(self):
            super().__init__(n_var=D, n_obj=M, xl=bounds[:, 0], xu=bounds[:, 1])

        def _evaluate(self, X, out, *args, **kwargs):
            F = np.array([forward(x) for x in X])
            out["F"] = F

    if variant == "nsga3":
        ref_dirs = get_reference_directions("das-dennis", M, n_partitions=12)
        algo = NSGA3(pop_size=pop_size, ref_dirs=ref_dirs)
    else:
        algo = NSGA2(pop_size=pop_size)

    t0 = time.time()
    res = minimize(_UserProblem(), algo,
                   termination=("n_gen", n_gen),
                   seed=seed, verbose=False)
    wall = time.time() - t0

    pareto_thetas = res.X.tolist() if res.X is not None else []
    pareto_objs = res.F.tolist() if res.F is not None else []
    return OptimizationResult(
        method=variant,                      # type: ignore[arg-type]
        param_names=param_names,
        objective_names=objective_names,
        pareto_thetas=pareto_thetas,
        pareto_objectives=pareto_objs,
        n_evaluations=int(pop_size * n_gen),
        wall_s=float(wall),
        diagnostics={"pop_size": pop_size, "n_gen": n_gen},
    )
```

- [ ] **Step 3: Commit**

```bash
pytest tests/research/optimization/test_pymoo.py -v
git add hydrus_research/optimization/pymoo_nsga.py tests/research/optimization/test_pymoo.py
git commit -m "M10.2: nsga_optimize (NSGA-II / NSGA-III via pymoo)"
```

---

### Task 3: Optuna wrapper

**Files:** `optuna_single.py`, `test_optuna.py`.

- [ ] **Step 1: Test**

```python
import numpy as np
import pytest

optuna = pytest.importorskip("optuna", reason="optuna not installed; in [research-opt]")
from hydrus_research.optimization import optuna_optimize, OptimizationResult


def test_optuna_finds_min_of_parabola():
    def f(theta): return float((theta[0] - 2.5) ** 2)
    bounds = np.array([[-5.0, 5.0]])
    r = optuna_optimize(forward_scalar=f, bounds=bounds,
                        param_names=["a"], objective_name="quadratic",
                        n_trials=30, seed=42)
    assert isinstance(r, OptimizationResult)
    assert r.method.startswith("optuna_")
    # Best should be near 2.5
    assert abs(r.pareto_thetas[0][0] - 2.5) < 0.5
```

- [ ] **Step 2: Implement**

```python
"""Optuna single-objective optimization (TPE / random / CMA-ES)."""
from __future__ import annotations
import time
from typing import Callable, Literal
import numpy as np

from .result import OptimizationResult


def optuna_optimize(forward_scalar: Callable[[np.ndarray], float],
                    bounds: np.ndarray,
                    param_names: list[str],
                    objective_name: str = "objective",
                    n_trials: int = 100,
                    sampler: Literal["tpe", "random", "cmaes"] = "tpe",
                    direction: Literal["minimize", "maximize"] = "minimize",
                    seed: int | None = None) -> OptimizationResult:
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError as e:
        raise ImportError(
            "optuna_optimize requires optuna. Install with:\n"
            "    pip install 'hydrus-port[research,research-opt]'"
        ) from e

    bounds = np.asarray(bounds, dtype=float)
    D = bounds.shape[0]

    if sampler == "tpe":
        samp = optuna.samplers.TPESampler(seed=seed)
        method_label = "optuna_tpe"
    elif sampler == "random":
        samp = optuna.samplers.RandomSampler(seed=seed)
        method_label = "optuna_random"
    elif sampler == "cmaes":
        samp = optuna.samplers.CmaEsSampler(seed=seed)
        method_label = "optuna_cmaes"
    else:
        raise ValueError(f"unknown sampler {sampler!r}")

    study = optuna.create_study(direction=direction, sampler=samp)

    def _objective(trial):
        theta = np.array([
            trial.suggest_float(name, lo, hi)
            for name, (lo, hi) in zip(param_names, bounds)
        ])
        return forward_scalar(theta)

    t0 = time.time()
    study.optimize(_objective, n_trials=n_trials)
    wall = time.time() - t0

    best_theta = np.array([study.best_params[n] for n in param_names])
    history = [float(t.value) for t in study.trials if t.value is not None]

    return OptimizationResult(
        method=method_label,                  # type: ignore[arg-type]
        param_names=param_names,
        objective_names=[objective_name],
        pareto_thetas=[best_theta.tolist()],
        pareto_objectives=[[float(study.best_value)]],
        history=[[v] for v in history],
        n_evaluations=int(n_trials),
        wall_s=float(wall),
        diagnostics={"sampler": sampler, "direction": direction,
                     "best_value": float(study.best_value)},
    )
```

- [ ] **Step 3: Commit**

```bash
pytest tests/research/optimization/test_optuna.py -v
git add hydrus_research/optimization/optuna_single.py tests/research/optimization/test_optuna.py
git commit -m "M10.3: optuna_optimize (TPE / random / CMA-ES via Optuna)"
```

---

### Task 4: CLI + e2e + marker

**Files:** modify `hydrus_port/cli.py`, `test_e2e_m10.py`.

- [ ] **Step 1: CLI**

In `_build_research_subparser(sub)`:
```python
p_opt = rsub.add_parser("optimize", help="decision optimization (P2)")
osub = p_opt.add_subparsers(dest="opt_cmd", required=True)
p_nsga = osub.add_parser("nsga", help="NSGA-II multi-objective")
p_nsga.add_argument("--bounds", required=True,
                    help="comma-separated lo:hi pairs (one per param)")
p_nsga.add_argument("--n-gen", type=int, default=20)
p_nsga.add_argument("--pop", type=int, default=50)
p_nsga.add_argument("--out", required=True)
p_nsga.set_defaults(_cmd=_cmd_optimize_nsga)


def _cmd_optimize_nsga(args):
    import json as _json
    from pathlib import Path as _P
    import numpy as _np
    from hydrus_research.optimization import nsga_optimize
    bnds = _np.array([list(map(float, b.split(":")))
                      for b in args.bounds.split(",")])
    # Toy multi-obj for CLI smoke; real schedule optimization wires
    # forward to a Hydrus1DSimulator → WUE + N-leaching computation.
    def fwd(theta): return _np.array([theta[0] ** 2 + theta[1] ** 2,
                                       (theta[0] - 1) ** 2 + (theta[1] - 1) ** 2])
    r = nsga_optimize(forward=fwd, bounds=bnds,
                      param_names=[f"x{i}" for i in range(bnds.shape[0])],
                      objective_names=["f1", "f2"],
                      pop_size=args.pop, n_gen=args.n_gen)
    _P(args.out).write_text(_json.dumps(r.model_dump(), indent=2))
    print(f"NSGA-II Pareto front: {len(r.pareto_thetas)} points → {args.out}")
    return 0
```

- [ ] **Step 2: E2E**

```python
"""M10 acceptance: NSGA-II + Optuna on synthetic problems."""
import numpy as np
import pytest

pymoo = pytest.importorskip("pymoo")
from hydrus_research.optimization import nsga_optimize, optuna_optimize


def test_nsga_finds_pareto_front_on_zdt1():
    """ZDT1 — classic 2-objective benchmark; Pareto front is the curve
    f2 = 1 - sqrt(f1) for theta[0] in [0, 1] (other thetas = 0)."""
    def zdt1(theta):
        f1 = theta[0]
        g = 1 + 9 * np.sum(theta[1:]) / (len(theta) - 1)
        f2 = g * (1 - np.sqrt(f1 / g))
        return np.array([f1, f2])
    bounds = np.array([[0.0, 1.0]] * 3)
    r = nsga_optimize(forward=zdt1, bounds=bounds,
                      param_names=["x0", "x1", "x2"],
                      objective_names=["f1", "f2"],
                      pop_size=30, n_gen=20, seed=42)
    objs = np.array(r.pareto_objectives)
    # f1 axis should span most of [0, 1]
    assert objs[:, 0].min() < 0.3
    assert objs[:, 0].max() > 0.7


def test_optuna_finds_min_of_quadratic():
    optuna = pytest.importorskip("optuna")
    def f(theta): return float((theta[0] - 2.5) ** 2 + (theta[1] + 1.0) ** 2)
    r = optuna_optimize(forward_scalar=f,
                       bounds=np.array([[-5.0, 5.0], [-5.0, 5.0]]),
                       param_names=["a", "b"], objective_name="q",
                       n_trials=50, seed=11)
    assert abs(r.pareto_thetas[0][0] - 2.5) < 0.5
    assert abs(r.pareto_thetas[0][1] + 1.0) < 0.5
```

- [ ] **Step 3: Marker**

```bash
pytest tests/research/optimization/test_e2e_m10.py -v
pytest tests/research/ -q --ignore=tests/research/dndc_seam/test_gui_smoke.py 2>&1 | tail -5
git add hydrus_port/cli.py tests/research/optimization/test_e2e_m10.py
git commit -m "M10.4: hydrus research optimize nsga CLI + e2e (ZDT1 + parabola)"
git commit --allow-empty -m "M10 complete: optimization (NSGA-II + Optuna) green; P2 done"
```

---

## Definition of Done for M10

1. `pytest tests/research/optimization/ -v` — green (SKIPs if pymoo / optuna missing).
2. `pytest tests/research/ -q` — no regression in M0-M9.
3. `nsga_optimize` returns ≥ 5 non-dominated points on the ZDT1 benchmark.
4. `optuna_optimize` recovers the minimum of a 2-param quadratic within 0.5.
5. `decision_vars.encode_schedule` / `decode_schedule` round-trip.
6. CLI `hydrus research optimize nsga` writes a JSON Pareto front.
7. No REST or GUI in this milestone (deferred to a follow-up; P2 spec scope).
