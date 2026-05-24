# M0 — Abstraction Layer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stand up the `hydrus_research/` package with the §2 abstraction layer (Simulator ABC, ParameterMap, ObservationSet, `make_forward` closure) and a working `Hydrus1DSimulator` adapter, verified by an end-to-end test that runs `forward(θ) → y` on the real `infiltr_v1` fixture.

**Architecture:** New sibling Python package. All research modules (added in M1+) consume the same triad — `Simulator + ParameterMap + ObservationSet` — through a single `forward(theta) -> y_sim` callable. Adapters reuse the existing canonical Scenario schema (`hydrus_port/schema.py`) so parameter patching happens on the schema dict, never on solver internals.

**Tech Stack:** Python 3.10+, `numpy`, `scipy`, `pydantic>=2`, `pytest`. No new heavy deps in M0.

**Spec reference:** `docs/superpowers/specs/2026-05-24-hydrus-research-platform-design.md` §§ 1–2.

**Acceptance:**
- `forward(theta)` runs on `infiltr_v1` and returns the expected number of observations.
- `ParameterMap.apply_to_scenario` round-trips a canonical scenario through `to_dict()` ↔ `_scenario_from_dict()` without loss.
- `pytest tests/research/` passes green.

---

## File Layout

**Created in this plan:**
- `hydrus_research/__init__.py` — package marker, version string.
- `hydrus_research/simulator/__init__.py` — re-exports.
- `hydrus_research/simulator/base.py` — `Forcing`, `InitialState`, `SimResult`, `Event`, `Simulator` ABC.
- `hydrus_research/simulator/closure.py` — `make_forward()`.
- `hydrus_research/simulator/hydrus1d_adapter.py` — `Hydrus1DSimulator`.
- `hydrus_research/parameters/__init__.py` — re-exports.
- `hydrus_research/parameters/spec.py` — `ParameterSpec`, transforms.
- `hydrus_research/parameters/map.py` — `ParameterMap`.
- `hydrus_research/observations/__init__.py` — re-exports.
- `hydrus_research/observations/spec.py` — `ObservationSpec`.
- `hydrus_research/observations/set.py` — `ObservationSet`.
- `hydrus_research/observations/loaders.py` — `from_hydrus_obsnod`.
- `tests/research/__init__.py`
- `tests/research/conftest.py` — shared fixtures.
- `tests/research/test_parameters.py`
- `tests/research/test_observations.py`
- `tests/research/test_simulator_base.py`
- `tests/research/test_closure.py`
- `tests/research/test_hydrus1d_adapter.py`
- `tests/research/test_e2e_m0.py` — the M0 acceptance test.

**Modified:**
- `pyproject.toml` — add `hydrus_research` to hatch packages; add four `research*` optional-deps groups.

---

### Task 1: Package skeleton and pyproject

**Files:**
- Create: `hydrus_research/__init__.py`
- Create: `hydrus_research/simulator/__init__.py`
- Create: `hydrus_research/parameters/__init__.py`
- Create: `hydrus_research/observations/__init__.py`
- Create: `tests/research/__init__.py`
- Modify: `pyproject.toml`

- [ ] **Step 1: Create empty package files**

```bash
mkdir -p hydrus_research/simulator hydrus_research/parameters hydrus_research/observations tests/research
```

Write `hydrus_research/__init__.py`:

```python
"""HYDRUS research platform — abstractions + research tools built on top of
the verified HYDRUS-1D / SWMS_2D / 3D Richards solvers.

See docs/superpowers/specs/2026-05-24-hydrus-research-platform-design.md
"""
__version__ = "0.1.0"
```

Write `hydrus_research/simulator/__init__.py`:

```python
from .base import Forcing, InitialState, SimResult, Event, Simulator
from .closure import make_forward

__all__ = ["Forcing", "InitialState", "SimResult", "Event", "Simulator", "make_forward"]
```

Write `hydrus_research/parameters/__init__.py`:

```python
from .spec import ParameterSpec
from .map import ParameterMap

__all__ = ["ParameterSpec", "ParameterMap"]
```

Write `hydrus_research/observations/__init__.py`:

```python
from .spec import ObservationSpec
from .set import ObservationSet

__all__ = ["ObservationSpec", "ObservationSet"]
```

Write `tests/research/__init__.py`:

```python
```

- [ ] **Step 2: Modify pyproject.toml — add hydrus_research to hatch packages**

In `pyproject.toml`, find:

```toml
[tool.hatch.build.targets.wheel]
packages = ["hydrus1d", "swms2d", "hydrus_port", "hydrus_port_server"]
```

Replace with:

```toml
[tool.hatch.build.targets.wheel]
packages = ["hydrus1d", "swms2d", "hydrus_port", "hydrus_port_server", "hydrus_research"]
```

- [ ] **Step 3: Modify pyproject.toml — add four research optional-deps groups**

In `pyproject.toml`, find the existing `[project.optional-dependencies]` block ending with:

```toml
dev = [
    "pytest>=7.4",
    "hatchling>=1.21",
]
```

Append after that line, still inside `[project.optional-dependencies]`:

```toml
research = [
    "pyemu>=1.3",
    "SALib>=1.5",
    "rosetta-soil",
    "scikit-learn",
    "joblib",
    "pydantic>=2.5",
]
research-uq = [
    "pymc>=5",
    "arviz",
]
research-3d = [
    "smt>=2",
    "chaospy",
]
research-opt = [
    "pymoo>=0.6",
    "optuna",
]
```

- [ ] **Step 4: Install editable + verify package import**

Run:

```bash
pip install -e '.[dev,gui]'
python -c "import hydrus_research; print(hydrus_research.__version__)"
```

Expected: `0.1.0` (no errors). `pydantic>=2.5` comes from the already-installed `[gui]` extras and is enough for M0 (we don't need pyemu/salib yet).

- [ ] **Step 5: Commit**

```bash
git add hydrus_research/ tests/research/ pyproject.toml
git commit -m "M0.1: hydrus_research package skeleton + research optional extras"
```

---

### Task 2: Event / Forcing / InitialState / SimResult dataclasses

**Files:**
- Create: `hydrus_research/simulator/base.py`
- Test: `tests/research/test_simulator_base.py`

- [ ] **Step 1: Write the failing test for dataclass construction**

Write `tests/research/test_simulator_base.py`:

```python
import numpy as np
import pytest
from hydrus_research.simulator import Forcing, InitialState, SimResult, Event


def test_event_is_frozen_dataclass():
    e = Event(time_day=1.0, depth_cm=0.0, amount=10.0,
              method="drip", solute_concs_mg_l={"NO3": 50.0})
    assert e.time_day == 1.0
    with pytest.raises(Exception):
        e.time_day = 2.0  # frozen


def test_forcing_minimum_construction():
    f = Forcing(
        times_days=np.array([0.0, 1.0, 2.0]),
        precip_cm_per_day=np.zeros(3),
        pet_cm_per_day=np.full(3, 0.5),
        lai=np.full(3, 2.0),
        root_depth_cm=np.full(3, 30.0),
        root_density_fn=lambda z, t: np.exp(-z / 30.0),
        irrigation_events=[],
        fert_events=[],
        n_source_terms=lambda z, t, theta, c: (0.0, 0.0),
        air_temp_c=None,
    )
    assert f.times_days.shape == (3,)
    assert f.air_temp_c is None
    assert f.root_density_fn(np.array([0.0, 30.0]), 0.0)[0] == 1.0


def test_initial_state_holds_profile():
    ic = InitialState(
        z_cm=np.linspace(0, 100, 11),
        theta=None,
        h_cm=np.full(11, -100.0),
        c_mg_per_L=None,
        t_celsius=None,
    )
    assert ic.h_cm[0] == -100.0
    assert ic.theta is None


def test_sim_result_holds_arrays_and_meta():
    z = np.linspace(0, 100, 5)
    t = np.array([0.0, 1.0])
    theta = np.zeros((2, 5))
    sr = SimResult(
        times=t, z=z, theta=theta, h=np.zeros_like(theta),
        c=None, fluxes={}, mass_balance={"total": 0.0},
        final_state=InitialState(z_cm=z, theta=theta[-1], h_cm=None,
                                 c_mg_per_L=None, t_celsius=None),
        meta={"solver": "test", "wall_s": 0.0},
    )
    assert sr.theta.shape == (2, 5)
    assert sr.meta["solver"] == "test"
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
pytest tests/research/test_simulator_base.py -v
```

Expected: ImportError — `Forcing` / `InitialState` / `SimResult` / `Event` not defined.

- [ ] **Step 3: Write minimal implementation**

Write `hydrus_research/simulator/base.py`:

```python
"""Core dataclasses + Simulator ABC for the research platform.

Forcing  — time-varying drivers (populated by dndc_seam.to_forcing()).
InitialState — z-profiles of theta / h / c / T at t=0 (also used for the
               returned final state, enabling DNDC day-step restart).
SimResult — what every Simulator.run() returns.
Event — irrigation or fertilizer event.
Simulator — ABC; subclasses wrap the real solvers (1D / 2D / 3D / surrogate).
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable
import numpy as np


@dataclass(frozen=True)
class Event:
    """Irrigation or fertilizer event. `method` distinguishes the two."""
    time_day: float
    depth_cm: float                              # 0 = surface
    amount: float                                # cm for irrig; kg N / ha for fert
    method: str                                  # "drip" | "sprinkler" | "flood" | "subsurface" | "fert"
    solute_concs_mg_l: dict[str, float] = field(default_factory=dict)
    form: str | None = None                      # "NH4" | "NO3" | "urea" | ... (fert only)


@dataclass(frozen=True)
class Forcing:
    """All time-varying drivers for one Simulator.run()."""
    times_days: np.ndarray
    precip_cm_per_day: np.ndarray
    pet_cm_per_day: np.ndarray
    lai: np.ndarray
    root_depth_cm: np.ndarray
    root_density_fn: Callable[[np.ndarray, float], np.ndarray]   # (z, t) -> normalized beta(z)
    irrigation_events: list[Event]
    fert_events: list[Event]
    n_source_terms: Callable[..., tuple[float, float]]           # (z, t, theta, c) -> (gamma_w, gamma_s); B2 hook
    air_temp_c: np.ndarray | None


@dataclass(frozen=True)
class InitialState:
    z_cm: np.ndarray
    theta: np.ndarray | None
    h_cm: np.ndarray | None
    c_mg_per_L: np.ndarray | None
    t_celsius: np.ndarray | None


@dataclass(frozen=True)
class SimResult:
    """Raw simulator output. 1D adapters store theta/h with shape (NT, NZ);
    2D/3D adapters store mesh-node arrays of shape (NT, Nnode) and use
    `z` slot for mesh metadata or a dummy axis."""
    times: np.ndarray
    z: np.ndarray
    theta: np.ndarray
    h: np.ndarray
    c: np.ndarray | None
    fluxes: dict[str, np.ndarray]
    mass_balance: dict[str, float]
    final_state: InitialState
    meta: dict[str, Any]


class Simulator(ABC):
    """Pure-function interface. No hidden state between runs.

    Subclasses must set class attributes `name` and `dimension`.

    `run` takes a **fully patched canonical scenario dict** (parameter
    application happens upstream in `make_forward` via
    `ParameterMap.apply_to_scenario`). `forcing=None` means "use whatever
    atmospheric / sink data is already inside the scenario"; non-None
    forcing overrides them (DNDC seam consumes this from M1 on).
    """

    name: str = ""
    dimension: int = 0

    @abstractmethod
    def run(self, scenario: dict, forcing: Forcing | None,
            ic: InitialState | None) -> SimResult: ...

    @abstractmethod
    def observable_at(self, result: SimResult,
                      spec: "ObservationSpec") -> float: ...

    def batch_observables(self, result: SimResult,
                          specs: "list[ObservationSpec]") -> np.ndarray:
        return np.array([self.observable_at(result, s) for s in specs])
```

- [ ] **Step 4: Run test to verify it passes**

Run:

```bash
pytest tests/research/test_simulator_base.py -v
```

Expected: 4 PASS.

- [ ] **Step 5: Commit**

```bash
git add hydrus_research/simulator/base.py tests/research/test_simulator_base.py
git commit -m "M0.2: Forcing / InitialState / SimResult / Event / Simulator ABC"
```

---

### Task 3: ParameterSpec with transforms

**Files:**
- Create: `hydrus_research/parameters/spec.py`
- Test: `tests/research/test_parameters.py`

- [ ] **Step 1: Write the failing test**

Write `tests/research/test_parameters.py`:

```python
import numpy as np
import pytest
from hydrus_research.parameters import ParameterSpec


def test_parameter_spec_linear_default():
    s = ParameterSpec(name="alpha", target="materials[0].alpha", bounds=(0.001, 1.0))
    assert s.transform == "linear"
    assert s.to_internal(0.5) == 0.5
    assert s.from_internal(0.5) == 0.5


def test_parameter_spec_log_transform():
    s = ParameterSpec(name="Ks", target="materials[0].Ks", bounds=(0.01, 100.0),
                      transform="log")
    internal = s.to_internal(10.0)
    assert internal == pytest.approx(np.log(10.0))
    assert s.from_internal(internal) == pytest.approx(10.0)


def test_parameter_spec_logit_transform():
    s = ParameterSpec(name="frac", target="x", bounds=(0.0, 1.0), transform="logit")
    internal = s.to_internal(0.5)
    assert internal == pytest.approx(0.0)
    assert s.from_internal(0.0) == pytest.approx(0.5)


def test_parameter_spec_internal_bounds():
    s = ParameterSpec(name="Ks", target="x", bounds=(0.01, 100.0), transform="log")
    lo, hi = s.internal_bounds()
    assert lo == pytest.approx(np.log(0.01))
    assert hi == pytest.approx(np.log(100.0))


def test_parameter_spec_rejects_invalid_transform():
    with pytest.raises(ValueError):
        ParameterSpec(name="x", target="x", bounds=(0, 1), transform="cubic")
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
pytest tests/research/test_parameters.py -v
```

Expected: ImportError — `ParameterSpec` not defined.

- [ ] **Step 3: Write minimal implementation**

Write `hydrus_research/parameters/spec.py`:

```python
"""ParameterSpec — one calibrated/sampled parameter, in user units.

Optimizers see an *internal* coordinate (linear / log / logit) that is
unbounded or trivially bounded, making bound-handling robust. The
back-transform happens here, not in the optimizer."""
from __future__ import annotations
from typing import Literal
import numpy as np
from pydantic import BaseModel, Field, field_validator


Transform = Literal["linear", "log", "logit"]


class ParameterSpec(BaseModel):
    name: str
    target: str                              # path into canonical scenario, e.g. "materials[0].alpha"
    bounds: tuple[float, float]
    transform: Transform = "linear"
    prior_mean: float | None = None          # in user units
    prior_std: float | None = None
    group: str = "default"

    @field_validator("bounds")
    @classmethod
    def _bounds_ordered(cls, v):
        lo, hi = v
        if lo >= hi:
            raise ValueError(f"bounds must satisfy lo < hi; got {v}")
        return v

    @field_validator("transform")
    @classmethod
    def _transform_known(cls, v):
        if v not in ("linear", "log", "logit"):
            raise ValueError(f"unknown transform: {v!r}")
        return v

    def to_internal(self, user: float) -> float:
        if self.transform == "linear":
            return user
        if self.transform == "log":
            return float(np.log(user))
        # logit
        lo, hi = self.bounds
        u = (user - lo) / (hi - lo)
        return float(np.log(u / (1.0 - u)))

    def from_internal(self, internal: float) -> float:
        if self.transform == "linear":
            return internal
        if self.transform == "log":
            return float(np.exp(internal))
        # logit
        lo, hi = self.bounds
        u = 1.0 / (1.0 + np.exp(-internal))
        return float(lo + u * (hi - lo))

    def internal_bounds(self) -> tuple[float, float]:
        lo, hi = self.bounds
        if self.transform == "linear":
            return lo, hi
        if self.transform == "log":
            return float(np.log(lo)), float(np.log(hi))
        # logit: maps (lo, hi) -> (-inf, inf); choose wide finite bounds for optimizers
        return -1e6, 1e6
```

- [ ] **Step 4: Run test to verify it passes**

Run:

```bash
pytest tests/research/test_parameters.py -v
```

Expected: 5 PASS.

- [ ] **Step 5: Commit**

```bash
git add hydrus_research/parameters/spec.py tests/research/test_parameters.py
git commit -m "M0.3: ParameterSpec with linear / log / logit transforms"
```

---

### Task 4: ParameterMap — vector ↔ named, bounds, midpoints

**Files:**
- Create: `hydrus_research/parameters/map.py`
- Test: append to `tests/research/test_parameters.py`

- [ ] **Step 1: Append failing tests**

Append to `tests/research/test_parameters.py`:

```python
from hydrus_research.parameters import ParameterMap


def _three_specs():
    return [
        ParameterSpec(name="alpha", target="materials[0].alpha",
                      bounds=(0.001, 1.0), transform="log"),
        ParameterSpec(name="n",     target="materials[0].n",
                      bounds=(1.05, 5.0),  transform="linear"),
        ParameterSpec(name="Ks",    target="materials[0].Ks",
                      bounds=(0.01, 100.0), transform="log"),
    ]


def test_parameter_map_roundtrip():
    pm = ParameterMap(_three_specs())
    named = {"alpha": 0.05, "n": 1.5, "Ks": 10.0}
    theta = pm.to_vector(named)
    back = pm.from_vector(theta)
    for k, v in named.items():
        assert back[k] == pytest.approx(v)


def test_parameter_map_bounds_array_internal():
    pm = ParameterMap(_three_specs())
    bnds = pm.bounds_array()                 # shape (D, 2), internal coords
    assert bnds.shape == (3, 2)
    assert bnds[0, 0] == pytest.approx(np.log(0.001))
    assert bnds[1, 0] == pytest.approx(1.05)


def test_parameter_map_midpoints():
    pm = ParameterMap(_three_specs())
    mids = pm.midpoints()
    assert mids.shape == (3,)
    # n is linear: midpoint of (1.05, 5.0) is 3.025
    assert mids[1] == pytest.approx(3.025)


def test_parameter_map_requires_unique_names():
    dup = [
        ParameterSpec(name="x", target="a", bounds=(0, 1)),
        ParameterSpec(name="x", target="b", bounds=(0, 1)),
    ]
    with pytest.raises(ValueError):
        ParameterMap(dup)
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
pytest tests/research/test_parameters.py -v
```

Expected: 4 new tests fail with ImportError on `ParameterMap`.

- [ ] **Step 3: Write minimal ParameterMap (without apply_to_scenario yet)**

Write `hydrus_research/parameters/map.py`:

```python
"""ParameterMap — bijection between an optimizer's theta vector and a
named-parameter dict that the Simulator (and scenario JSON) understands."""
from __future__ import annotations
from typing import Any
import numpy as np

from .spec import ParameterSpec


class ParameterMap:
    def __init__(self, specs: list[ParameterSpec]):
        names = [s.name for s in specs]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate ParameterSpec names: {names}")
        self.specs = list(specs)
        self._index = {s.name: i for i, s in enumerate(self.specs)}

    @property
    def names(self) -> list[str]:
        return [s.name for s in self.specs]

    @property
    def D(self) -> int:
        return len(self.specs)

    def to_vector(self, named: dict[str, float]) -> np.ndarray:
        theta = np.empty(self.D, dtype=float)
        for s in self.specs:
            if s.name not in named:
                raise KeyError(f"missing parameter {s.name!r} in {named!r}")
            theta[self._index[s.name]] = s.to_internal(named[s.name])
        return theta

    def from_vector(self, theta: np.ndarray) -> dict[str, float]:
        theta = np.asarray(theta, dtype=float)
        if theta.shape != (self.D,):
            raise ValueError(f"theta shape {theta.shape}, expected ({self.D},)")
        return {s.name: s.from_internal(theta[self._index[s.name]]) for s in self.specs}

    def bounds_array(self) -> np.ndarray:
        """(D, 2) array of internal-coord bounds, ready for scipy / pymoo."""
        return np.array([s.internal_bounds() for s in self.specs], dtype=float)

    def midpoints(self) -> np.ndarray:
        """In *user* coords, midpoint of each spec's bounds, expressed back as
        an internal theta vector. Useful as x0 for optimizers when no prior."""
        mids_user = {s.name: 0.5 * (s.bounds[0] + s.bounds[1]) for s in self.specs}
        return self.to_vector(mids_user)
```

- [ ] **Step 4: Run test to verify it passes**

Run:

```bash
pytest tests/research/test_parameters.py -v
```

Expected: all 9 tests in the file PASS.

- [ ] **Step 5: Commit**

```bash
git add hydrus_research/parameters/map.py tests/research/test_parameters.py
git commit -m "M0.4: ParameterMap vector<->named, bounds, midpoints"
```

---

### Task 5: ParameterMap.apply_to_scenario — canonical JSON path patching

**Files:**
- Modify: `hydrus_research/parameters/map.py`
- Test: append to `tests/research/test_parameters.py`

- [ ] **Step 1: Append failing tests**

Append to `tests/research/test_parameters.py`:

```python
def test_apply_to_scenario_patches_dict_path():
    pm = ParameterMap(_three_specs())
    scenario = {
        "materials": [
            {"alpha": 0.01, "n": 1.4, "Ks": 1.0},
            {"alpha": 0.05, "n": 1.5, "Ks": 5.0},
        ],
    }
    patched = pm.apply_to_scenario(scenario, {"alpha": 0.123, "n": 2.0, "Ks": 50.0})
    assert patched["materials"][0]["alpha"] == 0.123
    assert patched["materials"][0]["n"] == 2.0
    assert patched["materials"][0]["Ks"] == 50.0
    # original untouched (we return a deep copy)
    assert scenario["materials"][0]["alpha"] == 0.01
    # other-index material untouched
    assert patched["materials"][1]["alpha"] == 0.05


def test_apply_to_scenario_supports_nested_path():
    pm = ParameterMap([
        ParameterSpec(name="tol", target="solver.tol_theta", bounds=(1e-6, 1e-2)),
    ])
    scenario = {"solver": {"tol_theta": 0.001, "max_picard": 20}}
    patched = pm.apply_to_scenario(scenario, {"tol": 0.005})
    assert patched["solver"]["tol_theta"] == 0.005
    assert patched["solver"]["max_picard"] == 20


def test_apply_to_scenario_rejects_unknown_path():
    pm = ParameterMap([
        ParameterSpec(name="x", target="does.not.exist", bounds=(0, 1)),
    ])
    with pytest.raises(KeyError):
        pm.apply_to_scenario({"materials": []}, {"x": 0.5})
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
pytest tests/research/test_parameters.py::test_apply_to_scenario_patches_dict_path -v
```

Expected: AttributeError — `apply_to_scenario` not defined.

- [ ] **Step 3: Implement apply_to_scenario**

Append to `hydrus_research/parameters/map.py`:

```python
import copy
import re

_INDEX_RE = re.compile(r"^([^\[]+)\[(\d+)\]$")


def _walk(d: Any, parts: list[str]) -> tuple[Any, str | int]:
    """Walk `parts` over a nested dict / list; return (container, last_key).
    `parts` are dotted keys, optionally suffixed with `[N]` for list indexing."""
    cur = d
    for i, part in enumerate(parts[:-1]):
        m = _INDEX_RE.match(part)
        if m:
            key, idx = m.group(1), int(m.group(2))
            cur = cur[key]
            cur = cur[idx]
        else:
            cur = cur[part]
    last = parts[-1]
    m = _INDEX_RE.match(last)
    if m:
        key, idx = m.group(1), int(m.group(2))
        return cur[key], idx
    return cur, last


class _ApplyMixin:
    def apply_to_scenario(self, scenario: dict, named: dict[str, float]) -> dict:
        """Return a deep copy of `scenario` with each named value patched into
        its ParameterSpec.target path. `target` syntax: dotted keys with
        optional `[N]` indexing, e.g. `materials[0].alpha`, `solver.tol_theta`,
        `geometry.nodes[12].h_init`."""
        out = copy.deepcopy(scenario)
        for s in self.specs:
            if s.name not in named:
                continue
            value = named[s.name]
            parts = s.target.split(".")
            try:
                container, last_key = _walk(out, parts)
            except (KeyError, IndexError, TypeError) as e:
                raise KeyError(f"target {s.target!r} not found in scenario: {e}") from e
            try:
                container[last_key] = value
            except (KeyError, IndexError, TypeError) as e:
                raise KeyError(f"cannot set {s.target!r} in scenario: {e}") from e
        return out
```

Then at the bottom of the file, after the `ParameterMap` class definition, mix it in:

```python
ParameterMap.__bases__ = (_ApplyMixin,) + ParameterMap.__bases__
```

Or alternatively (cleaner): edit the `class ParameterMap:` line to `class ParameterMap(_ApplyMixin):` and put `_ApplyMixin` above `ParameterMap`.

Use the cleaner form. Final structure of `map.py`:

```python
# ... existing imports + helpers (_INDEX_RE, _walk) ...

class _ApplyMixin:
    def apply_to_scenario(self, scenario, named): ...   # as above

class ParameterMap(_ApplyMixin):
    # ... existing body ...
```

- [ ] **Step 4: Run tests to verify they pass**

Run:

```bash
pytest tests/research/test_parameters.py -v
```

Expected: all 12 tests PASS.

- [ ] **Step 5: Verify round-trip with the real canonical Scenario**

Append one more test to `tests/research/test_parameters.py`:

```python
def test_apply_to_scenario_roundtrip_with_real_scenario():
    """The patched dict must round-trip through hydrus_port.schema, proving
    the path format is compatible with the canonical schema."""
    from hydrus_port.schema import Scenario, ScenarioMeta, Units, Solver, \
        HydraulicMaterial, TimeControl, Geometry1D, _scenario_from_dict
    s = Scenario(
        meta=ScenarioMeta(name="t"),
        units=Units(),
        solver=Solver(),
        materials=[HydraulicMaterial(theta_r=0.05, theta_s=0.4,
                                     alpha=0.02, n=1.5, Ks=10.0)],
        time=TimeControl(t_init=0.0, t_max=1.0),
        geometry=Geometry1D(z=[0.0, 50.0, 100.0],
                            h_init=[-100.0, -100.0, -100.0],
                            mat_num=[1, 1, 1]),
    )
    d = s.to_dict()
    pm = ParameterMap([
        ParameterSpec(name="alpha", target="materials[0].alpha", bounds=(0.001, 1.0)),
    ])
    patched = pm.apply_to_scenario(d, {"alpha": 0.099})
    s2 = _scenario_from_dict(patched)
    assert s2.materials[0].alpha == 0.099
    # original Scenario object untouched
    assert s.materials[0].alpha == 0.02
```

Run:

```bash
pytest tests/research/test_parameters.py -v
```

Expected: all 13 PASS.

- [ ] **Step 6: Commit**

```bash
git add hydrus_research/parameters/map.py tests/research/test_parameters.py
git commit -m "M0.5: ParameterMap.apply_to_scenario — dotted-path patcher with list indexing"
```

---

### Task 6: ObservationSpec

**Files:**
- Create: `hydrus_research/observations/spec.py`
- Test: `tests/research/test_observations.py`

- [ ] **Step 1: Write the failing test**

Write `tests/research/test_observations.py`:

```python
import pytest
from hydrus_research.observations import ObservationSpec


def test_observation_spec_theta_at_depth():
    s = ObservationSpec(name="theta_z20_d5",
                        kind="theta",
                        location={"z_cm": 20.0},
                        time_day=5.0)
    assert s.weight == 1.0
    assert s.species is None
    assert s.location["z_cm"] == 20.0


def test_observation_spec_concentration_with_species():
    s = ObservationSpec(name="no3_z30_d10",
                        kind="c",
                        location={"z_cm": 30.0},
                        time_day=10.0,
                        species="NO3",
                        weight=2.5)
    assert s.species == "NO3"
    assert s.weight == 2.5


def test_observation_spec_rejects_bad_kind():
    with pytest.raises(Exception):
        ObservationSpec(name="x", kind="banana",
                        location={"z_cm": 0}, time_day=1.0)


def test_observation_spec_2d_location_node():
    s = ObservationSpec(name="h_node17_d3", kind="h",
                        location={"node": 17}, time_day=3.0)
    assert s.location == {"node": 17}
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
pytest tests/research/test_observations.py -v
```

Expected: ImportError.

- [ ] **Step 3: Implement ObservationSpec**

Write `hydrus_research/observations/spec.py`:

```python
"""ObservationSpec — one scalar observation point.

A spec carries enough information for any Simulator.observable_at()
implementation to sample the right scalar from a SimResult."""
from __future__ import annotations
from typing import Literal
from pydantic import BaseModel, Field


ObsKind = Literal[
    "theta", "h", "c",
    "flux", "cumulative_flux", "concentration_flux",
]


class ObservationSpec(BaseModel):
    name: str
    kind: ObsKind
    location: dict                       # 1D: {"z_cm": float}; 2D/3D: {"node": int} or {"xyz": [x,y,z]}
    time_day: float
    weight: float = 1.0
    species: str | None = None           # solute species name, only for kind == "c" or "concentration_flux"
```

- [ ] **Step 4: Run test to verify it passes**

Run:

```bash
pytest tests/research/test_observations.py -v
```

Expected: 4 PASS.

- [ ] **Step 5: Commit**

```bash
git add hydrus_research/observations/spec.py tests/research/test_observations.py
git commit -m "M0.6: ObservationSpec (theta / h / c / flux types; location dict)"
```

---

### Task 7: ObservationSet — residuals, objective_l2, from_csv

**Files:**
- Create: `hydrus_research/observations/set.py`
- Create: `tests/research/data/obs_minimal.csv`
- Test: append to `tests/research/test_observations.py`

- [ ] **Step 1: Create the test CSV fixture**

Make directory and file:

```bash
mkdir -p tests/research/data
```

Write `tests/research/data/obs_minimal.csv`:

```csv
name,kind,z_cm,time_day,value,sigma,weight,species
theta_z20_d1,theta,20,1.0,0.31,0.02,1.0,
theta_z40_d1,theta,40,1.0,0.28,0.02,1.0,
no3_z20_d5,c,20,5.0,12.5,1.5,1.0,NO3
```

- [ ] **Step 2: Append failing tests**

Append to `tests/research/test_observations.py`:

```python
import numpy as np
from pathlib import Path
from hydrus_research.observations import ObservationSet


def test_observation_set_residuals_and_objective():
    specs = [
        ObservationSpec(name="a", kind="theta", location={"z_cm": 10}, time_day=1.0),
        ObservationSpec(name="b", kind="theta", location={"z_cm": 20}, time_day=1.0),
    ]
    obs = ObservationSet(specs=specs,
                         values=np.array([0.30, 0.35]),
                         sigmas=np.array([0.02, 0.02]))
    sim = np.array([0.32, 0.33])
    res = obs.residuals(sim)
    assert res == pytest.approx([(0.32 - 0.30) / 0.02, (0.33 - 0.35) / 0.02])
    assert obs.objective_l2(sim) == pytest.approx(sum(res ** 2))


def test_observation_set_from_csv():
    path = Path(__file__).parent / "data" / "obs_minimal.csv"
    obs = ObservationSet.from_csv(path)
    assert len(obs.specs) == 3
    assert obs.specs[0].kind == "theta"
    assert obs.specs[2].species == "NO3"
    assert obs.values[2] == 12.5
    assert obs.sigmas[2] == 1.5


def test_observation_set_shape_mismatch_raises():
    specs = [ObservationSpec(name="a", kind="theta", location={"z_cm": 1}, time_day=0)]
    with pytest.raises(ValueError):
        ObservationSet(specs=specs,
                       values=np.array([0.1, 0.2]),    # wrong length
                       sigmas=np.array([0.01, 0.01]))
```

- [ ] **Step 3: Run to verify failure**

Run:

```bash
pytest tests/research/test_observations.py -v
```

Expected: new tests fail with ImportError on `ObservationSet`.

- [ ] **Step 4: Implement ObservationSet**

Write `hydrus_research/observations/set.py`:

```python
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
```

- [ ] **Step 5: Run to verify pass**

Run:

```bash
pytest tests/research/test_observations.py -v
```

Expected: 7 PASS.

- [ ] **Step 6: Commit**

```bash
git add hydrus_research/observations/set.py tests/research/test_observations.py tests/research/data/obs_minimal.csv
git commit -m "M0.7: ObservationSet (residuals / objective_l2 / from_csv)"
```

---

### Task 8: ObservationSet.from_hydrus_obsnod — parse HYDRUS-1D OBS_NODE.OUT

**Files:**
- Create: `hydrus_research/observations/loaders.py`
- Modify: `hydrus_research/observations/__init__.py` and `set.py`
- Test: append to `tests/research/test_observations.py`

- [ ] **Step 1: Eyeball the fixture file**

Run:

```bash
head -25 tests/fixtures/infiltr_v1/reference_out/OBS_NODE.OUT
```

This will print the file's header. The format (HYDRUS-1D 4.08): comment lines starting with `#`, then a header row with columns `time`, then for each observation node `h`, `theta`, plus optional `Conc`, `Temp`. Confirm column count and presence of `Conc` before proceeding.

- [ ] **Step 2: Append failing test**

Append to `tests/research/test_observations.py`:

```python
def test_obs_node_loader_reads_infiltr_v1():
    """Load the real HYDRUS-1D OBS_NODE.OUT and verify spec count + sample value."""
    from hydrus_research.observations.loaders import from_hydrus_obsnod
    path = Path("tests/fixtures/infiltr_v1/reference_out/OBS_NODE.OUT")
    if not path.exists():
        pytest.skip("infiltr_v1 reference output not present")
    times_to_sample = [0.5, 1.0, 2.0]   # any times present in the file
    obs = from_hydrus_obsnod(path, kinds=("theta", "h"), times_day=times_to_sample)
    # number of obs = n_nodes * n_kinds * n_times
    assert obs.M > 0
    # spec names follow the "<kind>_node<N>_d<t>" format
    for s in obs.specs:
        assert "_node" in s.name and "_d" in s.name
        assert "node" in s.location
    # theta values lie in a physical range [0, 1)
    theta_vals = np.array([v for s, v in zip(obs.specs, obs.values) if s.kind == "theta"])
    assert (theta_vals >= 0).all() and (theta_vals < 1.0).all()
```

- [ ] **Step 3: Run to verify failure**

Run:

```bash
pytest tests/research/test_observations.py::test_obs_node_loader_reads_infiltr_v1 -v
```

Expected: ImportError on `from_hydrus_obsnod`.

- [ ] **Step 4: Implement the loader**

Write `hydrus_research/observations/loaders.py`:

```python
"""Loaders that build an ObservationSet from existing HYDRUS / SWMS output."""
from __future__ import annotations
from pathlib import Path
import re
import numpy as np

from .spec import ObservationSpec
from .set import ObservationSet


# OBS_NODE.OUT layout (HYDRUS-1D 4.08):
#   # comment lines start with '#'
#   one header line listing per-node block names: "Node( 5)  Node( 10) ..."
#   one sub-header listing per-node column names: "h  theta  Conc  Temp"  (Conc/Temp optional)
#   data rows: "time   h1 theta1 [c1] [T1]   h2 theta2 [c2] [T2] ..."
# The exact spacing varies between Fortran builds, so we tokenize on whitespace.

_NODE_RE = re.compile(r"Node\s*\(\s*(\d+)\s*\)")


def _parse_obsnod(path: Path) -> tuple[list[int], list[str], np.ndarray]:
    """Returns (node_ids, per_node_columns, data_array).

    data_array has shape (NT, 1 + n_nodes * n_cols) where col 0 is time."""
    lines = [ln.rstrip() for ln in path.read_text().splitlines() if ln.strip()]
    # find the line listing Node(N) tokens
    node_line_idx = None
    for i, ln in enumerate(lines):
        if _NODE_RE.search(ln):
            node_line_idx = i
            break
    if node_line_idx is None:
        raise ValueError(f"no 'Node(N)' header found in {path}")
    node_ids = [int(m.group(1)) for m in _NODE_RE.finditer(lines[node_line_idx])]

    # the next non-comment, non-empty line is the per-node column header
    col_idx = node_line_idx + 1
    while col_idx < len(lines) and (lines[col_idx].startswith("#") or not lines[col_idx].strip()):
        col_idx += 1
    header_tokens = lines[col_idx].split()
    # tokens look like: time h theta Conc Temp h theta Conc Temp ...
    # n_cols_per_node = (len(header_tokens) - 1) // len(node_ids)
    n_nodes = len(node_ids)
    n_cols = (len(header_tokens) - 1) // n_nodes
    per_node_cols = header_tokens[1 : 1 + n_cols]   # ["h", "theta", "Conc"?, "Temp"?]

    # data rows
    data_rows: list[list[float]] = []
    for ln in lines[col_idx + 1 :]:
        s = ln.strip()
        if not s or s.startswith("#") or s.startswith("end"):
            continue
        try:
            data_rows.append([float(x) for x in s.split()])
        except ValueError:
            break    # ran past the numeric section
    return node_ids, per_node_cols, np.array(data_rows, dtype=float)


def from_hydrus_obsnod(path: Path | str,
                       kinds: tuple[str, ...] = ("theta",),
                       times_day: list[float] | None = None,
                       default_sigma: dict[str, float] | None = None
                       ) -> ObservationSet:
    """Build an ObservationSet from a HYDRUS-1D OBS_NODE.OUT file.

    Parameters
    ----------
    path : OBS_NODE.OUT location.
    kinds : which observable columns to harvest. Choose any of
        {"theta", "h", "c", "T"} that are present in the file.
    times_day : list of times to sample (linear interp on the file's time axis).
        If None, every printed time is used.
    default_sigma : per-kind measurement-error stddev; default 0.01 for theta,
        1.0 for h, 0.5 for c, 0.5 for T.
    """
    path = Path(path)
    node_ids, cols, data = _parse_obsnod(path)
    times = data[:, 0]
    n_nodes = len(node_ids)
    n_cols_per_node = len(cols)
    # column index helper inside one node block
    col_pos = {name.lower(): i for i, name in enumerate(cols)}

    sigma_defaults = {"theta": 0.01, "h": 1.0, "c": 0.5, "T": 0.5}
    if default_sigma:
        sigma_defaults.update(default_sigma)

    if times_day is None:
        times_day = list(times)

    specs: list[ObservationSpec] = []
    vals: list[float] = []
    sigs: list[float] = []
    for kind in kinds:
        key = "conc" if kind == "c" else ("temp" if kind == "T" else kind)
        if key not in col_pos:
            raise KeyError(f"requested kind {kind!r} (column {key!r}) not in OBS_NODE.OUT")
        col_in_node = col_pos[key]
        for node_id, node_block in zip(node_ids, range(n_nodes)):
            col_in_data = 1 + node_block * n_cols_per_node + col_in_node
            series = data[:, col_in_data]
            for t in times_day:
                v = float(np.interp(t, times, series))
                specs.append(ObservationSpec(
                    name=f"{kind}_node{node_id}_d{t:g}",
                    kind=kind if kind in ("theta", "h", "c") else "h",  # T not in M0
                    location={"node": node_id},
                    time_day=float(t),
                ))
                vals.append(v)
                sigs.append(sigma_defaults.get(kind, 1.0))
    return ObservationSet(specs=specs,
                          values=np.array(vals),
                          sigmas=np.array(sigs))
```

- [ ] **Step 5: Re-export from package**

Edit `hydrus_research/observations/__init__.py`:

```python
from .spec import ObservationSpec
from .set import ObservationSet
from .loaders import from_hydrus_obsnod

__all__ = ["ObservationSpec", "ObservationSet", "from_hydrus_obsnod"]
```

- [ ] **Step 6: Run test**

Run:

```bash
pytest tests/research/test_observations.py -v
```

Expected: all 8 PASS (one of them may `skip` if reference_out is missing, but `infiltr_v1` was confirmed to have OBS_NODE.OUT).

- [ ] **Step 7: Commit**

```bash
git add hydrus_research/observations/loaders.py hydrus_research/observations/__init__.py tests/research/test_observations.py
git commit -m "M0.8: ObservationSet.from_hydrus_obsnod — parse OBS_NODE.OUT"
```

---

### Task 9: make_forward closure

**Files:**
- Create: `hydrus_research/simulator/closure.py`
- Test: `tests/research/test_closure.py`

- [ ] **Step 1: Write failing test using a fake Simulator**

Write `tests/research/test_closure.py`:

```python
import numpy as np
import pytest
from dataclasses import replace

from hydrus_research.simulator import (
    Forcing, InitialState, SimResult, Simulator, make_forward,
)
from hydrus_research.parameters import ParameterSpec, ParameterMap
from hydrus_research.observations import ObservationSpec


class _FakeSimulator(Simulator):
    """Pretends to be a 1-D solver. theta_at(z, t) = alpha * z + n * t.
    Reads alpha/n straight out of the patched scenario dict."""
    name = "fake"
    dimension = 1

    def run(self, scenario, forcing, ic):
        alpha = scenario["x"]
        n = scenario["y"]
        z = np.linspace(0.0, 100.0, 11)
        t = np.array([0.0, 1.0, 2.0, 5.0])
        theta = np.outer(t * n, np.ones_like(z)) + np.outer(np.ones_like(t), z * alpha)
        return SimResult(
            times=t, z=z, theta=theta, h=np.zeros_like(theta),
            c=None, fluxes={}, mass_balance={},
            final_state=InitialState(z_cm=z, theta=theta[-1], h_cm=None,
                                     c_mg_per_L=None, t_celsius=None),
            meta={"solver": "fake", "wall_s": 0.0},
        )

    def observable_at(self, result, spec):
        z_target = spec.location["z_cm"]
        t_target = spec.time_day
        theta_at_z = np.array([np.interp(z_target, result.z, row) for row in result.theta])
        return float(np.interp(t_target, result.times, theta_at_z))


def test_make_forward_returns_callable_of_right_shape():
    sim = _FakeSimulator()
    pm = ParameterMap([
        ParameterSpec(name="alpha", target="x", bounds=(0.0, 1.0)),
        ParameterSpec(name="n",     target="y", bounds=(0.0, 5.0)),
    ])
    template = {"x": 0.0, "y": 0.0}     # FakeSimulator reads scenario["x"], scenario["y"]
    obs_specs = [
        ObservationSpec(name="a", kind="theta", location={"z_cm": 50.0}, time_day=1.0),
        ObservationSpec(name="b", kind="theta", location={"z_cm": 100.0}, time_day=2.0),
    ]
    forward = make_forward(sim, pm,
                           template_scenario=template,
                           forcing=None, ic=None,
                           obs_specs=obs_specs)
    theta = pm.to_vector({"alpha": 0.1, "n": 1.5})
    y = forward(theta)
    assert y.shape == (2,)
    # at (z=50, t=1): 0.1*50 + 1.5*1 = 6.5
    assert y[0] == pytest.approx(6.5, rel=1e-6)
    # at (z=100, t=2): 0.1*100 + 1.5*2 = 13.0
    assert y[1] == pytest.approx(13.0, rel=1e-6)
```

- [ ] **Step 2: Run to verify failure**

Run:

```bash
pytest tests/research/test_closure.py -v
```

Expected: ImportError on `make_forward`.

- [ ] **Step 3: Implement make_forward**

Write `hydrus_research/simulator/closure.py`:

```python
"""make_forward — produces the single (theta -> y) callable that every
research module consumes. This is the narrow waist of the architecture."""
from __future__ import annotations
from typing import Callable
import numpy as np

from .base import Simulator, Forcing, InitialState
from ..parameters import ParameterMap
from ..observations import ObservationSpec


def make_forward(
    simulator: Simulator,
    param_map: ParameterMap,
    template_scenario: dict,
    forcing: Forcing | None,
    ic: InitialState | None,
    obs_specs: list[ObservationSpec],
) -> Callable[[np.ndarray], np.ndarray]:
    """Build a pure function `forward(theta) -> y_sim` of length M = len(obs_specs).

    `theta` is in *internal* coords (see ParameterSpec transforms). On each
    call: from_vector → apply_to_scenario(template_scenario, named) → run
    → batch_observables. The adapter never sees parameter specs — its only
    contract is "take patched scenario dict → return SimResult"."""
    def forward(theta: np.ndarray) -> np.ndarray:
        named = param_map.from_vector(theta)
        scenario = param_map.apply_to_scenario(template_scenario, named)
        result = simulator.run(scenario, forcing, ic)
        return simulator.batch_observables(result, obs_specs)
    return forward
```

- [ ] **Step 4: Run test**

Run:

```bash
pytest tests/research/test_closure.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add hydrus_research/simulator/closure.py tests/research/test_closure.py
git commit -m "M0.9: make_forward — the narrow-waist callable shared by all research tools"
```

---

### Task 10: Hydrus1DSimulator scaffolding

**Files:**
- Create: `hydrus_research/simulator/hydrus1d_adapter.py`
- Test: `tests/research/test_hydrus1d_adapter.py`

- [ ] **Step 1: Write failing test for scaffolding**

Write `tests/research/test_hydrus1d_adapter.py`:

```python
import pytest
from pathlib import Path

from hydrus_research.simulator.hydrus1d_adapter import Hydrus1DSimulator
from hydrus_port.adapters.hydrus1d import load as load_h1d_canonical


def _infiltr_v1_inputs() -> Path:
    return Path("tests/fixtures/infiltr_v1/inputs")


def _infiltr_v1_template_dict() -> dict:
    sc = load_h1d_canonical(_infiltr_v1_inputs())
    return sc.to_dict()


def test_hydrus1d_adapter_construction():
    sim = Hydrus1DSimulator(work_root=Path("/tmp"))
    assert sim.name == "hydrus1d"
    assert sim.dimension == 1
```

- [ ] **Step 2: Run to verify failure**

Run:

```bash
pytest tests/research/test_hydrus1d_adapter.py -v
```

Expected: ImportError.

- [ ] **Step 3: Implement scaffolding only (no run yet)**

Write `hydrus_research/simulator/hydrus1d_adapter.py`:

```python
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
```

- [ ] **Step 4: Run test**

Run:

```bash
pytest tests/research/test_hydrus1d_adapter.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add hydrus_research/simulator/hydrus1d_adapter.py tests/research/test_hydrus1d_adapter.py
git commit -m "M0.10: Hydrus1DSimulator scaffolding (run/observable_at stubs)"
```

---

### Task 11: Hydrus1DSimulator.run — invoke real solver via canonical adapter

**Files:**
- Modify: `hydrus_research/simulator/hydrus1d_adapter.py`
- Test: append to `tests/research/test_hydrus1d_adapter.py`

- [ ] **Step 1: Append failing test**

Append to `tests/research/test_hydrus1d_adapter.py`:

```python
import numpy as np


def test_hydrus1d_run_returns_sim_result_on_infiltr_v1():
    template = _infiltr_v1_template_dict()
    sim = Hydrus1DSimulator(work_root=Path("/tmp/hres_m0"))
    # Run the unpatched template (no parameter sweep needed for this test).
    result = sim.run(template, forcing=None, ic=None)
    assert result.theta.ndim == 2
    assert result.times.ndim == 1
    assert result.z.ndim == 1
    assert result.theta.shape[0] == result.times.shape[0]
    assert result.theta.shape[1] == result.z.shape[0]
    assert "wall_s" in result.meta
    assert result.meta["solver"] == "hydrus1d"
    assert isinstance(result.mass_balance, dict)
```

- [ ] **Step 2: Run to verify failure**

Run:

```bash
pytest tests/research/test_hydrus1d_adapter.py::test_hydrus1d_run_returns_sim_result_on_infiltr_v1 -v
```

Expected: NotImplementedError.

- [ ] **Step 3: Implement run()**

Replace the body of `Hydrus1DSimulator.run` in `hydrus_research/simulator/hydrus1d_adapter.py` with:

```python
    def run(self, scenario, forcing, ic):
        """`scenario` is a fully patched canonical Scenario dict (i.e. what
        `Scenario.to_dict()` returns, after any parameter patching done by
        ParameterMap.apply_to_scenario). `forcing` and `ic` are reserved for
        M1 (DNDC seam); in M0 the adapter uses whatever atmospheric / initial
        data lives inside `scenario` itself."""
        import time as _time

        # ----- 1. round-trip through canonical schema and write HYDRUS-1D inputs
        from hydrus_port.schema import _scenario_from_dict
        from hydrus_port.adapters.hydrus1d import save as save_h1d
        sc = _scenario_from_dict(scenario)
        run_dir = Path(tempfile.mkdtemp(prefix="h1d_", dir=self.work_root))
        out_dir = run_dir / "out"
        save_h1d(sc, run_dir)
        out_dir.mkdir(exist_ok=True)

        # ----- 2. invoke the real solver
        from hydrus1d.hydrus import run_simulation
        t0 = _time.time()
        _h1d = run_simulation(input_dir=str(run_dir), output_dir=str(out_dir))
        wall = _time.time() - t0

        # ----- 3. parse outputs into a SimResult
        return self._load_outputs(out_dir, wall)

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------
    def _load_outputs(self, out_dir: Path, wall_s: float) -> SimResult:
        """Parse NOD_INF.OUT (per-node profiles per print time) + BALANCE.OUT."""
        nod_path = out_dir / "NOD_INF.OUT"
        if not nod_path.exists():
            # case-insensitive find
            for p in out_dir.iterdir():
                if p.name.lower() == "nod_inf.out":
                    nod_path = p
                    break
        times, z, theta, h = _parse_nod_inf(nod_path)
        mb = _parse_balance_total(out_dir)
        final = InitialState(z_cm=z, theta=theta[-1], h_cm=h[-1],
                             c_mg_per_L=None, t_celsius=None)
        return SimResult(
            times=times, z=z, theta=theta, h=h, c=None,
            fluxes={}, mass_balance=mb, final_state=final,
            meta={"solver": "hydrus1d", "wall_s": wall_s, "out_dir": str(out_dir)},
        )
```

At the bottom of `hydrus1d_adapter.py`, add the two parsing helpers:

```python
def _parse_nod_inf(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Parse HYDRUS-1D NOD_INF.OUT into (times, z, theta(NT,NZ), h(NT,NZ)).

    File layout: repeated blocks, each preceded by 'Time:  <value>' and a
    header with columns including 'Node', 'Depth', 'Head', 'Moisture'. A
    'end' line closes each block."""
    text = path.read_text().splitlines()
    blocks: list[tuple[float, list[tuple[float, float, float]]]] = []
    i = 0
    cur_time: float | None = None
    rows: list[tuple[float, float, float]] = []
    while i < len(text):
        line = text[i].strip()
        if line.lower().startswith("time"):
            # commit previous block
            if cur_time is not None and rows:
                blocks.append((cur_time, rows))
            tokens = line.replace(":", " ").split()
            cur_time = float(tokens[1])
            rows = []
            i += 1
            continue
        if line.lower().startswith("end") or not line:
            i += 1
            continue
        parts = line.split()
        # numeric data row: Node Depth Head Moisture K C ...
        if len(parts) >= 4:
            try:
                # Node int, Depth float, Head float, Moisture float
                int(parts[0])
                depth = float(parts[1])
                head = float(parts[2])
                moist = float(parts[3])
                rows.append((depth, head, moist))
            except ValueError:
                pass
        i += 1
    if cur_time is not None and rows:
        blocks.append((cur_time, rows))

    if not blocks:
        raise ValueError(f"no time blocks found in {path}")

    # depth axis from the first block
    z = np.array([d for d, _, _ in blocks[0][1]], dtype=float)
    times = np.array([t for t, _ in blocks], dtype=float)
    NT, NZ = len(times), len(z)
    theta = np.empty((NT, NZ), dtype=float)
    h = np.empty((NT, NZ), dtype=float)
    for ti, (_, rs) in enumerate(blocks):
        for zi, (_, head, moist) in enumerate(rs):
            h[ti, zi] = head
            theta[ti, zi] = moist
    return times, z, theta, h


def _parse_balance_total(out_dir: Path) -> dict[str, float]:
    """Pull final-row 'Volume' from BALANCE.OUT; tolerant of missing file."""
    for name in ("BALANCE.OUT", "balance.out", "Balance.out"):
        p = out_dir / name
        if p.exists():
            try:
                lines = p.read_text().splitlines()
                vol_lines = [ln for ln in lines if "Volume" in ln and "[" in ln]
                if not vol_lines:
                    return {}
                last = vol_lines[-1].split()
                # final number on that line
                return {"volume_last": float(last[-1])}
            except Exception:
                return {}
    return {}
```

- [ ] **Step 4: Run test**

Run:

```bash
pytest tests/research/test_hydrus1d_adapter.py::test_hydrus1d_run_returns_sim_result_on_infiltr_v1 -v -s
```

Expected: PASS. The run may take a few seconds.

- [ ] **Step 5: Commit**

```bash
git add hydrus_research/simulator/hydrus1d_adapter.py tests/research/test_hydrus1d_adapter.py
git commit -m "M0.11: Hydrus1DSimulator.run — patch scenario, invoke hydrus1d, parse NOD_INF/BALANCE"
```

---

### Task 12: Hydrus1DSimulator.observable_at — 1D interpolation

**Files:**
- Modify: `hydrus_research/simulator/hydrus1d_adapter.py`
- Test: append to `tests/research/test_hydrus1d_adapter.py`

- [ ] **Step 1: Append failing test**

Append to `tests/research/test_hydrus1d_adapter.py`:

```python
from hydrus_research.observations import ObservationSpec


def test_hydrus1d_observable_at_theta_interp():
    template = _infiltr_v1_template_dict()
    sim = Hydrus1DSimulator(work_root=Path("/tmp/hres_m0"))
    result = sim.run(template, forcing=None, ic=None)
    # Pick a depth midway and a time present in the run
    z_target = float(result.z[len(result.z) // 2])
    t_target = float(result.times[len(result.times) // 2])
    spec = ObservationSpec(name="theta_mid", kind="theta",
                           location={"z_cm": z_target}, time_day=t_target)
    v = sim.observable_at(result, spec)
    # Should equal the exact array value (no interp needed at a sample point)
    expected = result.theta[len(result.times) // 2, len(result.z) // 2]
    assert v == pytest.approx(expected, rel=1e-6)


def test_hydrus1d_observable_at_h_interp_between_nodes():
    template = _infiltr_v1_template_dict()
    sim = Hydrus1DSimulator(work_root=Path("/tmp/hres_m0"))
    result = sim.run(template, forcing=None, ic=None)
    # Halfway between first two depth nodes, midway in time
    z_target = 0.5 * (float(result.z[0]) + float(result.z[1]))
    t_target = float(result.times[len(result.times) // 2])
    spec = ObservationSpec(name="h_between", kind="h",
                           location={"z_cm": z_target}, time_day=t_target)
    v = sim.observable_at(result, spec)
    # Expected: bilinear interp along z then along t (here t is exact)
    h_at_t = result.h[len(result.times) // 2]
    expected = float(np.interp(z_target, result.z, h_at_t))
    assert v == pytest.approx(expected, rel=1e-6)
```

- [ ] **Step 2: Run to verify failure**

Run:

```bash
pytest tests/research/test_hydrus1d_adapter.py -v
```

Expected: NotImplementedError on observable_at tests.

- [ ] **Step 3: Implement observable_at**

In `hydrus_research/simulator/hydrus1d_adapter.py`, replace the body of `observable_at` with:

```python
    def observable_at(self, result, spec):
        """1D interp: locate `z_cm` along the depth axis (linear), then locate
        `time_day` along the time axis (linear). 2D/3D adapters override this."""
        z_target = float(spec.location["z_cm"])
        t_target = float(spec.time_day)

        if spec.kind == "theta":
            field = result.theta
        elif spec.kind == "h":
            field = result.h
        elif spec.kind == "c":
            if result.c is None:
                raise ValueError("observable kind='c' requested but result has no solute field")
            field = result.c
        else:
            raise NotImplementedError(f"observable kind {spec.kind!r} not supported in M0")

        # interp along z for each time row, then along t
        col_at_z = np.array([np.interp(z_target, result.z, row) for row in field])
        return float(np.interp(t_target, result.times, col_at_z))
```

- [ ] **Step 4: Run test**

Run:

```bash
pytest tests/research/test_hydrus1d_adapter.py -v
```

Expected: all 4 tests in the file PASS.

- [ ] **Step 5: Commit**

```bash
git add hydrus_research/simulator/hydrus1d_adapter.py tests/research/test_hydrus1d_adapter.py
git commit -m "M0.12: Hydrus1DSimulator.observable_at — 1D linear interp in z and t"
```

---

### Task 13: End-to-end M0 acceptance test on infiltr_v1

**Files:**
- Create: `tests/research/test_e2e_m0.py`

- [ ] **Step 1: Write the acceptance test**

Write `tests/research/test_e2e_m0.py`:

```python
"""M0 acceptance: forward(theta) -> y_sim end-to-end on infiltr_v1."""
import numpy as np
import pytest
from pathlib import Path

from hydrus_research.simulator import make_forward
from hydrus_research.simulator.hydrus1d_adapter import Hydrus1DSimulator
from hydrus_research.parameters import ParameterSpec, ParameterMap
from hydrus_research.observations import ObservationSpec
from hydrus_port.adapters.hydrus1d import load as load_h1d_canonical


@pytest.fixture(scope="module")
def template_dict():
    return load_h1d_canonical(Path("tests/fixtures/infiltr_v1/inputs")).to_dict()


def test_forward_theta_to_y_end_to_end(template_dict):
    sim = Hydrus1DSimulator(work_root=Path("/tmp/hres_m0_e2e"))

    alpha0 = template_dict["materials"][0]["alpha"]
    n0     = template_dict["materials"][0]["n"]
    Ks0    = template_dict["materials"][0]["Ks"]

    pm = ParameterMap([
        ParameterSpec(name="alpha",
                      target="materials[0].alpha",
                      bounds=(alpha0 * 0.5, alpha0 * 2.0),
                      transform="log"),
        ParameterSpec(name="n",
                      target="materials[0].n",
                      bounds=(max(1.05, n0 * 0.8), n0 * 1.5),
                      transform="linear"),
        ParameterSpec(name="Ks",
                      target="materials[0].Ks",
                      bounds=(Ks0 * 0.1, Ks0 * 10.0),
                      transform="log"),
    ])

    # Get the available z extent + simulation duration from the template, so
    # we observe at points the run will actually visit.
    z_nodes = template_dict["geometry"]["z"]
    z_min, z_max = min(z_nodes), max(z_nodes)
    t_max = template_dict["time"]["t_max"]

    obs_specs = [
        ObservationSpec(name=f"theta_mid_t_{t:g}",
                        kind="theta",
                        location={"z_cm": 0.5 * (z_min + z_max)},
                        time_day=t)
        for t in np.linspace(0.0, t_max, 5)[1:]    # drop t=0
    ]

    forward = make_forward(sim, pm,
                           template_scenario=template_dict,
                           forcing=None, ic=None,
                           obs_specs=obs_specs)

    # Reference run at nominal parameters
    theta_ref = pm.to_vector({"alpha": alpha0, "n": n0, "Ks": Ks0})
    y_ref = forward(theta_ref)

    # Perturbed run: increase alpha by 20%
    theta_pert = pm.to_vector({"alpha": alpha0 * 1.2, "n": n0, "Ks": Ks0})
    y_pert = forward(theta_pert)

    # ---- acceptance checks ----
    assert y_ref.shape == (len(obs_specs),)
    assert np.all(np.isfinite(y_ref))
    assert np.all((y_ref >= 0) & (y_ref < 1.0)), "theta out of physical range"
    # Parameter perturbation must change at least one observable
    assert not np.allclose(y_ref, y_pert), \
        "alpha perturbation did not change any observable; abstraction is broken"
```

- [ ] **Step 2: Run the acceptance test**

Run:

```bash
pytest tests/research/test_e2e_m0.py -v -s
```

Expected: PASS. May take 10-30 seconds (two real solver runs).

If the perturbation assertion fails because the perturbation is too small for visible change on this fixture, increase the alpha multiplier (e.g. 2.0) before considering it broken.

- [ ] **Step 3: Run the full research test suite to confirm green**

Run:

```bash
pytest tests/research/ -v
```

Expected: all tests across `test_parameters.py`, `test_observations.py`, `test_simulator_base.py`, `test_closure.py`, `test_hydrus1d_adapter.py`, `test_e2e_m0.py` PASS.

- [ ] **Step 4: Commit**

```bash
git add tests/research/test_e2e_m0.py
git commit -m "M0.13: end-to-end acceptance — forward(theta)->y on infiltr_v1"
```

---

### Task 14: Run all existing test suites — regression check

**Files:** none (verification only)

- [ ] **Step 1: Confirm existing test suites still pass**

Run:

```bash
pytest tests/ -x -q --ignore=tests/research --timeout=120
```

Expected: same green status as before this branch was opened. If any existing test breaks, the failure must be diagnosed before this plan can be considered complete — the M0 changes are additive only (new package + pyproject extras) and must not regress anything in `hydrus1d/`, `swms2d/`, or `hydrus_port/`.

- [ ] **Step 2: Run the canonical CLI smoke**

Run:

```bash
hydrus test 1d
```

Expected: PASS line for `1d` as in `README.md`.

- [ ] **Step 3: If everything green, commit a tag-style empty marker**

Run:

```bash
git commit --allow-empty -m "M0 complete: abstraction layer green; ready for M1/M2/M3"
git log --oneline | head -15
```

---

## Definition of Done for M0

All of these must hold before declaring M0 complete and writing M1/M2/M3 plans:

1. `pytest tests/research/ -v` — all green.
2. `pytest tests/ -x -q --ignore=tests/research` — all green (no regression).
3. `hydrus test 1d` — PASS.
4. `python -c "from hydrus_research.simulator import make_forward; print('OK')"` — prints `OK`.
5. `git log --oneline | grep '^.\{8\} M0\.'` — shows commits M0.1 through M0.13 (and a final empty M0-complete commit).

If any of those fail, the milestone is not done. Fix the failure before moving on.
