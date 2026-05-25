# M5 — Inversion (Parameter Calibration) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement F3 — fit unknown soil parameters to observed measurements. Two backends share one `InversionResult` schema:
- **scipy LM** — fast in-process trust-region Levenberg-Marquardt for ≤ 10 params on 1D fixtures (~5 min, no posterior).
- **PEST/PyEMU IES** — Iterative Ensemble Smoother orchestrating subprocess workers via a JSON-config file (no pickle); gives a posterior ensemble.

Auto-backend selection (params < 10 ∧ 1D → LM; else → IES) is exposed via a single `fit()` dispatcher.

**Architecture:** Sub-package `hydrus_research/inversion/` mirrors M4's structure — one file per backend behind a shared schema. PyEMU runs PEST++-IES as an external process; each PESTPP worker invocation is a `python -m hydrus_research.inversion.worker_script <theta.dat> <y.dat> <config.json>` call that rebuilds `forward()` locally from the JSON config (scenario_dir + param specs + obs specs). No cross-process Python-object serialization. PyMC Bayesian (P1) is stubbed for M9.

**Tech Stack:** Python 3.10+, `scipy.optimize.least_squares` (already required), `pyemu>=1.3` (already in `[research]` extras), FastAPI + Pydantic, Vue 3 + Plotly. Independent of M4 — can be developed and merged in parallel.

**Spec reference:** `DOCS/superpowers/specs/2026-05-24-hydrus-research-platform-design.md` §4.4 + §5.1 (InversionStudio page) + §5.2 (REST endpoints).

**Acceptance:**
- `python -c "from hydrus_research.inversion import fit_lm, fit_pyemu, fit"` works.
- **LM synthetic recovery**: generate y_obs from a known θ_true via `forward(θ_true)`; perturb starting θ; LM recovers θ_true within 5% on a 1-param case on `infiltr_v1`.
- **PyEMU IES smoke**: when pyemu + `pestpp-ies` binary available, dispatcher launches IES via the JSON-config worker script; when missing, raises a clear actionable error. (Real IES convergence test is OPT-IN — slow.)
- **Auto-backend**: `fit(...)` with 3 params on 1D simulator picks `lm_scipy`; with 12 params picks `pyemu_ies`.
- REST `POST /research/inversion/{backend}` returns 200 + InversionResult JSON.
- `hydrus research invert <scenario> --obs obs.csv --backend lm` produces an InversionResult JSON.
- `pytest tests/research/inversion/` green (real PESTPP test is skipped if binary missing); full `tests/research/` still green.

---

## File Layout

**Created:**
- `hydrus_research/inversion/__init__.py` — re-exports.
- `hydrus_research/inversion/base.py` — `InversionResult` Pydantic model.
- `hydrus_research/inversion/lm_scipy.py` — `fit_lm(forward, param_map, obs, ...)`.
- `hydrus_research/inversion/pyemu_pestpp.py` — `fit_pyemu(forward, param_map, obs, method="ies", ...)`.
- `hydrus_research/inversion/worker_script.py` — entry-point script PEST++ runs per-realization (reads config.json, rebuilds forward, runs it).
- `hydrus_research/inversion/pymc_bayes.py` — P1 stub.
- `hydrus_research/inversion/api.py` — `fit(forward, param_map, obs, backend="auto", ...)` dispatcher.
- `hydrus_port_server/routers/research_inversion.py` — REST routes.
- `desktop/src/pages/research/InversionStudio.vue` — F3 GUI page.
- `desktop/src/components/ParamPosteriorBar.vue` — best-fit + CI bar chart.
- `desktop/src/components/ResidualsPlot.vue` — convergence (objective vs iter).
- `tests/research/inversion/__init__.py`
- `tests/research/inversion/test_base.py`
- `tests/research/inversion/test_lm_scipy.py`
- `tests/research/inversion/test_worker_script.py`
- `tests/research/inversion/test_pyemu_pestpp.py`
- `tests/research/inversion/test_api.py`
- `tests/research/inversion/test_cli.py`
- `tests/research/inversion/test_rest.py`
- `tests/research/inversion/test_e2e_m5.py`

**Modified:**
- `hydrus_port_server/app.py:build_app()` — register the inversion router.
- `hydrus_port/cli.py:_build_research_subparser` — add `hydrus research invert` subcommand.
- `desktop/src/api.ts` — append `inversion.*` wrapper.
- `desktop/src/App.vue` — add `Inversion` tab.

---

### Task 1: Sub-package skeleton + InversionResult

**Files:**
- Create: `hydrus_research/inversion/__init__.py`
- Create: `hydrus_research/inversion/base.py`
- Create: `tests/research/inversion/__init__.py`
- Create: `tests/research/inversion/test_base.py`

- [ ] **Step 1: Skeleton + 4 stubs + result schema**

```bash
mkdir -p hydrus_research/inversion tests/research/inversion
touch tests/research/inversion/__init__.py
```

Write `hydrus_research/inversion/__init__.py`:

```python
"""Inversion (F3) — fit unknown soil parameters to observations.

Backends share one InversionResult schema:
  - lm_scipy   — scipy.optimize.least_squares (fast; ≤ 10 params, 1D)
  - pyemu_ies  — PESTPP-IES Iterative Ensemble Smoother (large D, posterior)
  - pymc_nuts  — Bayesian NUTS (P1; M9)

Dispatch via `fit(...)` with auto backend selection.

See DOCS/superpowers/specs/2026-05-24-hydrus-research-platform-design.md §4.4.
"""
from .base import InversionResult
from .lm_scipy import fit_lm
from .pyemu_pestpp import fit_pyemu
from .api import fit

__all__ = ["InversionResult", "fit_lm", "fit_pyemu", "fit"]
```

Write `hydrus_research/inversion/base.py`:

```python
"""Shared InversionResult schema."""
from __future__ import annotations
from typing import Any, Literal
from pydantic import BaseModel, ConfigDict


Backend = Literal["lm_scipy", "pyemu_glm", "pyemu_ies", "pymc_nuts"]


class InversionResult(BaseModel):
    """Result of one inversion run. Fields are nullable when a backend
    can't provide them (LM gives no posterior; IES gives no jacobian)."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    backend: Backend
    best_params: dict[str, float]
    parameter_ci_lo: dict[str, float] = {}
    parameter_ci_hi: dict[str, float] = {}
    posterior_ensemble: list[list[float]] | None = None       # (N_real, D); None for LM
    posterior_param_names: list[str] = []
    objective_history: list[float] = []
    n_forward_calls: int
    wall_s: float
    pest_workspace: str | None = None
    diagnostics: dict[str, Any] = {}
```

Create stubs (will be replaced by Tasks 2-4):

`hydrus_research/inversion/lm_scipy.py`:
```python
def fit_lm(*args, **kwargs):
    raise NotImplementedError("M5.2 stub")
```

`hydrus_research/inversion/pyemu_pestpp.py`:
```python
def fit_pyemu(*args, **kwargs):
    raise NotImplementedError("M5.3 stub")
```

`hydrus_research/inversion/api.py`:
```python
def fit(*args, **kwargs):
    raise NotImplementedError("M5.4 stub")
```

`hydrus_research/inversion/pymc_bayes.py`:
```python
def fit_pymc(*args, **kwargs):
    raise NotImplementedError("P1 — PyMC Bayesian inversion lives in M9")
```

- [ ] **Step 2: Write test**

Write `tests/research/inversion/test_base.py`:

```python
import pytest
from hydrus_research.inversion import InversionResult


def test_inversion_result_lm_minimum():
    r = InversionResult(
        backend="lm_scipy",
        best_params={"alpha": 0.036, "n": 1.56},
        parameter_ci_lo={"alpha": 0.030, "n": 1.50},
        parameter_ci_hi={"alpha": 0.042, "n": 1.62},
        n_forward_calls=42,
        wall_s=12.3,
    )
    assert r.backend == "lm_scipy"
    assert r.posterior_ensemble is None


def test_inversion_result_ies_with_posterior():
    r = InversionResult(
        backend="pyemu_ies",
        best_params={"alpha": 0.036, "n": 1.56},
        posterior_ensemble=[[0.034, 1.55], [0.036, 1.56], [0.038, 1.57]],
        posterior_param_names=["alpha", "n"],
        n_forward_calls=300,
        wall_s=120.0,
    )
    assert len(r.posterior_ensemble) == 3


def test_inversion_result_rejects_unknown_backend():
    with pytest.raises(Exception):
        InversionResult(backend="grpc", best_params={}, n_forward_calls=0, wall_s=0)
```

- [ ] **Step 3: Run + commit**

```bash
pytest tests/research/inversion/test_base.py -v
git add hydrus_research/inversion/ tests/research/inversion/
git commit -m "M5.1: inversion sub-package skeleton + InversionResult + 4 stubs"
```

---

### Task 2: scipy LM backend

**Files:**
- Modify: `hydrus_research/inversion/lm_scipy.py` (replace stub)
- Create: `tests/research/inversion/test_lm_scipy.py`

- [ ] **Step 1: Write failing test**

Write `tests/research/inversion/test_lm_scipy.py`:

```python
import numpy as np
import pytest
from hydrus_research.inversion import fit_lm, InversionResult
from hydrus_research.parameters import ParameterSpec, ParameterMap
from hydrus_research.observations import ObservationSpec, ObservationSet


def quadratic(theta):
    a, b = theta
    return np.array([a + b, a * b, b ** 2])


def test_lm_recovers_synthetic_theta_within_5pct():
    pm = ParameterMap([
        ParameterSpec(name="a", target="a", bounds=(-5.0, 5.0)),
        ParameterSpec(name="b", target="b", bounds=(-5.0, 5.0)),
    ])
    theta_true = np.array([1.5, 2.0])
    y_obs = quadratic(theta_true)
    obs = ObservationSet(
        specs=[ObservationSpec(name=f"o{i}", kind="theta",
                               location={"z_cm": 0.0}, time_day=0.0)
               for i in range(3)],
        values=y_obs,
        sigmas=np.array([0.01, 0.01, 0.01]),
    )
    result = fit_lm(forward=quadratic, param_map=pm, obs=obs,
                    x0=pm.to_vector({"a": 0.5, "b": 0.5}),
                    max_nfev=200)
    assert isinstance(result, InversionResult)
    assert result.backend == "lm_scipy"
    np.testing.assert_allclose(result.best_params["a"], 1.5, atol=0.1)
    np.testing.assert_allclose(result.best_params["b"], 2.0, atol=0.1)
    assert result.n_forward_calls > 0
    assert result.wall_s > 0


def test_lm_returns_ci_from_hessian():
    pm = ParameterMap([ParameterSpec(name="a", target="a", bounds=(-5, 5))])
    obs = ObservationSet(
        specs=[ObservationSpec(name="o", kind="theta",
                               location={"z_cm": 0.0}, time_day=0.0)],
        values=np.array([4.0]), sigmas=np.array([0.01]),
    )
    result = fit_lm(forward=lambda t: np.array([t[0] ** 2]),
                    param_map=pm, obs=obs,
                    x0=pm.to_vector({"a": 1.0}), max_nfev=100)
    assert result.parameter_ci_lo
    assert result.parameter_ci_hi
    assert result.parameter_ci_lo["a"] < result.best_params["a"]
    assert result.parameter_ci_hi["a"] > result.best_params["a"]
```

- [ ] **Step 2: Verify failure**

```bash
pytest tests/research/inversion/test_lm_scipy.py -v
```

- [ ] **Step 3: Implement**

Replace `hydrus_research/inversion/lm_scipy.py`:

```python
"""Scipy Levenberg-Marquardt — the in-process fast inversion path.

Uses scipy.optimize.least_squares with the 'trf' (trust-region) variant,
which handles bounded problems more robustly than pure LM."""
from __future__ import annotations
import time
from typing import Callable
import numpy as np

from .base import InversionResult


def fit_lm(forward: Callable[[np.ndarray], np.ndarray],
           param_map,
           obs,
           x0: np.ndarray | None = None,
           max_nfev: int = 200) -> InversionResult:
    """LM-fit `forward(theta) → y_sim` to `obs.values` via least_squares.

    Returns InversionResult with best_params + jacobian-derived CIs."""
    from scipy.optimize import least_squares

    if x0 is None:
        x0 = param_map.midpoints()

    bounds = param_map.bounds_array()
    history: list[float] = []
    nfev = {"count": 0}

    def residuals(theta: np.ndarray) -> np.ndarray:
        nfev["count"] += 1
        sim = forward(theta)
        r = obs.residuals(sim)
        history.append(float(np.sum(r * r)))
        return r

    t0 = time.time()
    res = least_squares(
        residuals, x0=np.asarray(x0, dtype=float),
        bounds=(bounds[:, 0], bounds[:, 1]),
        method="trf", jac="2-point", x_scale="jac",
        max_nfev=max_nfev,
    )
    wall = time.time() - t0

    best_named = param_map.from_vector(res.x)

    # Jacobian-based ±1σ CIs via the pseudo-inverse of J^T J
    ci_lo: dict[str, float] = {}
    ci_hi: dict[str, float] = {}
    try:
        J = res.jac
        if J is not None and J.size > 0:
            JTJ = J.T @ J
            cov = np.linalg.pinv(JTJ)
            sigmas = np.sqrt(np.maximum(np.diag(cov), 0.0))
            for spec, sigma in zip(param_map.specs, sigmas):
                mean_user = best_named[spec.name]
                if spec.transform == "log":
                    su = mean_user * sigma
                elif spec.transform == "logit":
                    lo_b, hi_b = spec.bounds
                    range_w = hi_b - lo_b
                    u = (mean_user - lo_b) / range_w
                    su = u * (1 - u) * range_w * sigma
                else:
                    su = sigma
                ci_lo[spec.name] = float(mean_user - su)
                ci_hi[spec.name] = float(mean_user + su)
    except (np.linalg.LinAlgError, ValueError):
        pass

    return InversionResult(
        backend="lm_scipy",
        best_params={k: float(v) for k, v in best_named.items()},
        parameter_ci_lo=ci_lo,
        parameter_ci_hi=ci_hi,
        posterior_ensemble=None,
        objective_history=history,
        n_forward_calls=int(nfev["count"]),
        wall_s=float(wall),
        diagnostics={"scipy_status": int(res.status),
                     "scipy_message": str(res.message),
                     "scipy_optimality": float(res.optimality)},
    )
```

- [ ] **Step 4: Run + commit**

```bash
pytest tests/research/inversion/test_lm_scipy.py -v
git add hydrus_research/inversion/lm_scipy.py tests/research/inversion/test_lm_scipy.py
git commit -m "M5.2: scipy LM backend with synthetic recovery + jacobian CIs"
```

---

### Task 3: PESTPP-IES worker script (JSON-config; no cross-process Python serialization)

**Files:**
- Create: `hydrus_research/inversion/worker_script.py`
- Create: `tests/research/inversion/test_worker_script.py`

This script is what `pestpp-ies` invokes once per realization. It reads (theta.dat, y.dat path, config.json) — config.json contains scenario_dir + param specs + obs specs as plain JSON. The worker rebuilds `make_forward` locally, runs it, writes y.dat. No Python-object serialization across processes.

- [ ] **Step 1: Write failing test**

Write `tests/research/inversion/test_worker_script.py`:

```python
import json
import subprocess
import sys
import tempfile
from pathlib import Path
import numpy as np


def test_worker_script_runs_forward_and_writes_y():
    """Smoke-test the worker by invoking it like PESTPP would: with a theta
    file + an output y file + a JSON config that describes the forward."""
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        theta_path = td / "theta.dat"
        y_path = td / "y.dat"
        cfg_path = td / "config.json"

        # Use infiltr_v1; alpha at the template's nominal value
        from hydrus_port.adapters.hydrus1d import load
        template = load(Path("tests/fixtures/infiltr_v1/inputs")).to_dict()
        a0 = template["materials"][0]["alpha"]

        cfg = {
            "scenario_dir": "tests/fixtures/infiltr_v1/inputs",
            "param_specs": [
                {"name": "alpha", "target": "materials[0].alpha",
                 "bounds": [a0 * 0.5, a0 * 2.0], "transform": "log"},
            ],
            "obs_specs": [
                {"name": "theta_z30_d1", "kind": "theta",
                 "location": {"z_cm": -30.0}, "time_day": 1.0},
            ],
        }
        cfg_path.write_text(json.dumps(cfg))
        # Theta is in INTERNAL coords (log for alpha)
        theta_internal = float(np.log(a0))
        theta_path.write_text(f"{theta_internal:.10e}\n")

        r = subprocess.run(
            [sys.executable, "-m", "hydrus_research.inversion.worker_script",
             str(theta_path), str(y_path), str(cfg_path)],
            capture_output=True, text=True, timeout=120,
        )
        assert r.returncode == 0, r.stderr
        assert y_path.exists()
        y_value = float(y_path.read_text().strip())
        assert 0 <= y_value < 1.0          # plausible θ
```

- [ ] **Step 2: Implement**

Write `hydrus_research/inversion/worker_script.py`:

```python
"""PEST++ per-realization worker entry point.

Invoked by PESTPP-IES as: `python -m hydrus_research.inversion.worker_script
<theta.dat> <y.dat> <config.json>`.

The script reads:
  - theta.dat   — one float per line, internal coords, in param-spec order
  - config.json — {scenario_dir, param_specs[], obs_specs[]}

Rebuilds `make_forward(...)` locally, runs it, writes y.dat (one float per
line, in obs-spec order).

No Python-object serialization crosses the process boundary — only JSON
config and plain-text numerical I/O. This is the contract PEST++ workers
operate under."""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np


def _run(theta_path: Path, y_path: Path, cfg_path: Path) -> int:
    cfg = json.loads(cfg_path.read_text())
    scenario_dir = Path(cfg["scenario_dir"])
    param_specs_json = cfg["param_specs"]
    obs_specs_json = cfg["obs_specs"]

    # Local imports — the worker pulls in the full engine when invoked
    from hydrus_research.parameters import ParameterSpec, ParameterMap
    from hydrus_research.observations import ObservationSpec
    from hydrus_research.simulator import make_forward
    from hydrus_research.simulator.hydrus1d_adapter import Hydrus1DSimulator
    from hydrus_port.adapters.hydrus1d import load as _load_h1d

    specs = [ParameterSpec(**s) for s in param_specs_json]
    pm = ParameterMap(specs)
    obs_specs = [ObservationSpec(**o) for o in obs_specs_json]
    template = _load_h1d(scenario_dir).to_dict()
    sim = Hydrus1DSimulator()
    forward = make_forward(sim, pm, template_scenario=template,
                           forcing=None, ic=None, obs_specs=obs_specs)

    theta = np.array([float(x) for x in theta_path.read_text().split()])
    if theta.shape != (len(specs),):
        print(f"theta.dat has {theta.shape[0]} values, expected {len(specs)}",
              file=sys.stderr)
        return 2

    y = np.asarray(forward(theta), dtype=float)
    y_path.write_text("\n".join(f"{v:.10e}" for v in y))
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    if len(argv) != 3:
        print("usage: worker_script <theta.dat> <y.dat> <config.json>",
              file=sys.stderr)
        return 64
    return _run(Path(argv[0]), Path(argv[1]), Path(argv[2]))


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 3: Run + commit**

```bash
pytest tests/research/inversion/test_worker_script.py -v -s
git add hydrus_research/inversion/worker_script.py tests/research/inversion/test_worker_script.py
git commit -m "M5.3: PEST++ worker entry point (JSON-config; no Python serialization)"
```

(Test takes ~10s — one real solver call.)

---

### Task 4: PyEMU PESTPP-IES wrapper (using the worker_script)

**Files:**
- Modify: `hydrus_research/inversion/pyemu_pestpp.py` (replace stub)
- Create: `tests/research/inversion/test_pyemu_pestpp.py`

- [ ] **Step 1: Write failing test**

Write `tests/research/inversion/test_pyemu_pestpp.py`:

```python
import shutil
import pytest
import numpy as np

pyemu = pytest.importorskip("pyemu", reason="pyemu not installed")
from hydrus_research.inversion import fit_pyemu, InversionResult
from hydrus_research.parameters import ParameterSpec, ParameterMap
from hydrus_research.observations import ObservationSpec, ObservationSet


def test_pyemu_fit_raises_when_binary_missing():
    """If pestpp-ies isn't on PATH, the wrapper must raise a clear actionable
    error — never silently fabricate a result."""
    binary = shutil.which("pestpp-ies") or shutil.which("PESTPP-IES")
    if binary:
        pytest.skip("pestpp-ies IS available; can't test the missing-binary path")

    pm = ParameterMap([ParameterSpec(name="alpha", target="materials[0].alpha",
                                     bounds=(0.005, 0.05), transform="log")])
    obs = ObservationSet(
        specs=[ObservationSpec(name="theta_z30_d1", kind="theta",
                               location={"z_cm": -30.0}, time_day=1.0)],
        values=np.array([0.31]), sigmas=np.array([0.02]),
    )
    with pytest.raises(RuntimeError, match="pestpp-ies"):
        fit_pyemu(scenario_dir="tests/fixtures/infiltr_v1/inputs",
                  param_map=pm, obs=obs, method="ies",
                  n_real=4, n_iter=1)


@pytest.mark.skipif(
    not (shutil.which("pestpp-ies") or shutil.which("PESTPP-IES")),
    reason="pestpp-ies binary not on PATH",
)
def test_pyemu_ies_runs_synthetic_recovery():
    """Real IES smoke (slow; opt-in). Skipped if pestpp-ies missing."""
    pm = ParameterMap([ParameterSpec(name="alpha", target="materials[0].alpha",
                                     bounds=(0.005, 0.05), transform="log")])
    obs = ObservationSet(
        specs=[ObservationSpec(name="theta_z30_d1", kind="theta",
                               location={"z_cm": -30.0}, time_day=1.0)],
        values=np.array([0.31]), sigmas=np.array([0.02]),
    )
    result = fit_pyemu(scenario_dir="tests/fixtures/infiltr_v1/inputs",
                      param_map=pm, obs=obs,
                      method="ies", n_real=4, n_iter=1)
    assert isinstance(result, InversionResult)
    assert result.backend == "pyemu_ies"
    assert result.n_forward_calls > 0
```

- [ ] **Step 2: Implement**

Replace `hydrus_research/inversion/pyemu_pestpp.py`:

```python
"""PESTPP-IES Iterative Ensemble Smoother via PyEMU.

This wrapper builds a minimal PEST control file (.pst) in a temporary
workspace, configures PESTPP-IES to invoke our `worker_script` for each
realization, runs the binary, and parses the resulting posterior
ensemble.

The worker script receives a JSON config — no cross-process Python
serialization. See `worker_script.py`."""
from __future__ import annotations
import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Literal
import numpy as np

from .base import InversionResult


def _check_binary(name: str) -> str:
    path = shutil.which(name) or shutil.which(name.upper())
    if not path:
        raise RuntimeError(
            f"`{name}` binary not on PATH. Install PEST++ from "
            "https://github.com/usgs/pestpp/releases and add to PATH, "
            "or use backend='lm_scipy' for the in-process LM path."
        )
    return path


def fit_pyemu(scenario_dir: str | Path,
              param_map,
              obs,
              method: Literal["glm", "ies", "opt"] = "ies",
              n_real: int = 200,
              n_iter: int = 4,
              workspace: Path | None = None) -> InversionResult:
    """Calibrate via PESTPP-{IES,GLM,OPT}.

    Unlike `fit_lm`, this wrapper takes `scenario_dir` directly (not a
    forward callable) because the worker subprocess must rebuild forward
    locally from a JSON config — see `worker_script.py`."""
    import pyemu
    binary = _check_binary(f"pestpp-{method}")
    scenario_dir = Path(scenario_dir).resolve()

    ws = Path(workspace) if workspace else Path(tempfile.mkdtemp(prefix="hr_pestpp_"))
    ws.mkdir(parents=True, exist_ok=True)

    # Write the JSON config the worker script will consume
    cfg = {
        "scenario_dir": str(scenario_dir),
        "param_specs": [s.model_dump() for s in param_map.specs],
        "obs_specs": [o.model_dump() for o in obs.specs],
    }
    (ws / "config.json").write_text(json.dumps(cfg, indent=2))

    par_names = list(param_map.names)
    obs_names = [s.name for s in obs.specs]
    bounds = param_map.bounds_array()                            # internal coords

    # Build a minimal PEST control file
    pst = pyemu.Pst.from_par_obs_names(par_names=par_names, obs_names=obs_names)
    pst.parameter_data.loc[par_names, "parlbnd"] = bounds[:, 0]
    pst.parameter_data.loc[par_names, "parubnd"] = bounds[:, 1]
    pst.parameter_data.loc[par_names, "parval1"] = param_map.midpoints()
    pst.observation_data.loc[obs_names, "obsval"] = obs.values
    pst.observation_data.loc[obs_names, "weight"] = 1.0 / np.maximum(obs.sigmas, 1e-12)
    pst.control_data.noptmax = int(n_iter)
    pst.pestpp_options["ies_num_reals"] = n_real

    # model_command: invoke the worker script via -m so it's importable
    pst.model_command = [
        sys.executable, "-m", "hydrus_research.inversion.worker_script",
        "theta.dat", "y.dat", "config.json",
    ]
    pst_path = ws / "case.pst"
    pst.write(str(pst_path))

    t0 = time.time()
    rc = subprocess.run([binary, "case.pst"], cwd=str(ws), capture_output=True)
    wall = time.time() - t0
    if rc.returncode != 0:
        raise RuntimeError(
            f"PESTPP-{method.upper()} failed (rc={rc.returncode}). stderr:\n"
            + rc.stderr.decode("utf-8", errors="replace")[-2000:]
        )

    # Parse posterior from .par.csv (highest iteration number)
    candidates = sorted(ws.glob("case.*.par.csv"))
    posterior: list[list[float]] | None = None
    if candidates:
        import csv
        with candidates[-1].open() as f:
            reader = csv.DictReader(f)
            posterior = [[float(row[name]) for name in par_names] for row in reader]

    best_named: dict[str, float] = {}
    if posterior:
        arr = np.asarray(posterior)
        for j, name in enumerate(par_names):
            best_named[name] = float(arr[:, j].mean())
    else:
        for name in par_names:
            best_named[name] = float(pst.parameter_data.loc[name, "parval1"])

    return InversionResult(
        backend=f"pyemu_{method}",                                  # type: ignore[arg-type]
        best_params=best_named,
        posterior_ensemble=posterior,
        posterior_param_names=par_names,
        n_forward_calls=int(n_real * n_iter),
        wall_s=float(wall),
        pest_workspace=str(ws),
        diagnostics={"pst": str(pst_path), "binary": binary,
                     "n_real": n_real, "n_iter": n_iter},
    )
```

- [ ] **Step 3: Run + commit**

```bash
pytest tests/research/inversion/test_pyemu_pestpp.py -v
git add hydrus_research/inversion/pyemu_pestpp.py tests/research/inversion/test_pyemu_pestpp.py
git commit -m "M5.4: PESTPP-IES backend via pyemu (JSON-config worker; binary-skip when missing)"
```

(The "missing binary" test always runs; the IES smoke test SKIPs unless `pestpp-ies` is on PATH.)

---

### Task 5: Auto-backend `fit()` dispatcher

**Files:**
- Modify: `hydrus_research/inversion/api.py` (replace stub)
- Create: `tests/research/inversion/test_api.py`

- [ ] **Step 1: Write failing test**

Write `tests/research/inversion/test_api.py`:

```python
import pytest
import numpy as np
from hydrus_research.inversion import fit, InversionResult
from hydrus_research.parameters import ParameterSpec, ParameterMap
from hydrus_research.observations import ObservationSpec, ObservationSet


def quadratic(theta):
    return np.array([theta[0] + theta[1], theta[0] * theta[1]])


def test_fit_auto_picks_lm_for_small_d():
    pm = ParameterMap([
        ParameterSpec(name="a", target="a", bounds=(-5, 5)),
        ParameterSpec(name="b", target="b", bounds=(-5, 5)),
    ])
    obs = ObservationSet(
        specs=[ObservationSpec(name=f"o{i}", kind="theta",
                               location={"z_cm": 0.0}, time_day=0.0)
               for i in range(2)],
        values=np.array([3.5, 3.0]), sigmas=np.array([0.01, 0.01]),
    )
    r = fit(forward=quadratic, param_map=pm, obs=obs,
            scenario_dir=None, backend="auto", simulator_dimension=1)
    assert r.backend == "lm_scipy"


def test_fit_explicit_backend_lm():
    pm = ParameterMap([ParameterSpec(name="a", target="a", bounds=(-5, 5))])
    obs = ObservationSet(
        specs=[ObservationSpec(name="o", kind="theta",
                               location={"z_cm": 0.0}, time_day=0.0)],
        values=np.array([4.0]), sigmas=np.array([0.01]),
    )
    r = fit(forward=lambda t: np.array([t[0] ** 2]),
            param_map=pm, obs=obs, scenario_dir=None, backend="lm")
    assert r.backend == "lm_scipy"


def test_fit_unknown_backend_raises():
    pm = ParameterMap([ParameterSpec(name="a", target="a", bounds=(-5, 5))])
    obs = ObservationSet(
        specs=[ObservationSpec(name="o", kind="theta",
                               location={"z_cm": 0.0}, time_day=0.0)],
        values=np.array([1.0]), sigmas=np.array([0.01]),
    )
    with pytest.raises(ValueError):
        fit(forward=lambda t: np.array([t[0]]),
            param_map=pm, obs=obs, scenario_dir=None, backend="quantum")
```

- [ ] **Step 2: Implement**

Replace `hydrus_research/inversion/api.py`:

```python
"""fit() dispatcher with auto backend selection.

Selection rule (per spec §4.4):
  - params < 10 AND 1D simulator           → lm_scipy
  - params ≥ 10  OR  2D/3D simulator       → pyemu_ies
  - user requests posterior explicitly      → pymc_nuts (P1; raises until M9)

The two backends have different argument shapes:
  - lm_scipy needs `forward` (any callable).
  - pyemu_ies needs `scenario_dir` (path) — its workers rebuild forward
    from JSON config; passing a Python callable is not supported because
    PEST++ subprocess workers cannot share Python objects.

The dispatcher requires BOTH `forward` and `scenario_dir` and forwards
each to the relevant backend."""
from __future__ import annotations
from pathlib import Path
from typing import Callable, Literal
import numpy as np

from .base import InversionResult
from .lm_scipy import fit_lm
from .pyemu_pestpp import fit_pyemu


def fit(forward: Callable[[np.ndarray], np.ndarray] | None,
        param_map,
        obs,
        scenario_dir: str | Path | None = None,
        backend: Literal["auto", "lm", "lm_scipy",
                         "ies", "pyemu_ies", "glm", "pyemu_glm",
                         "nuts", "pymc_nuts"] = "auto",
        simulator_dimension: int = 1,
        **kwargs) -> InversionResult:
    """Dispatch to the right inversion backend."""
    if backend == "auto":
        D = len(param_map.names) if hasattr(param_map, "names") else len(param_map.specs)
        backend = "lm" if (D < 10 and simulator_dimension == 1) else "ies"

    if backend in ("lm", "lm_scipy"):
        if forward is None:
            raise ValueError("LM backend requires a `forward` callable")
        lm_kwargs = {k: v for k, v in kwargs.items() if k in ("x0", "max_nfev")}
        return fit_lm(forward=forward, param_map=param_map, obs=obs, **lm_kwargs)
    if backend in ("ies", "pyemu_ies"):
        if scenario_dir is None:
            raise ValueError(
                "PESTPP-IES backend requires a `scenario_dir` path (workers "
                "rebuild forward locally; Python callables can't cross subprocess "
                "boundaries). Pass scenario_dir=Path(...)."
            )
        pe_kwargs = {k: v for k, v in kwargs.items()
                     if k in ("n_real", "n_iter", "workspace")}
        return fit_pyemu(scenario_dir=scenario_dir, param_map=param_map, obs=obs,
                         method="ies", **pe_kwargs)
    if backend in ("glm", "pyemu_glm"):
        if scenario_dir is None:
            raise ValueError("PESTPP-GLM backend requires a `scenario_dir` path")
        pe_kwargs = {k: v for k, v in kwargs.items()
                     if k in ("n_real", "n_iter", "workspace")}
        return fit_pyemu(scenario_dir=scenario_dir, param_map=param_map, obs=obs,
                         method="glm", **pe_kwargs)
    if backend in ("nuts", "pymc_nuts"):
        from .pymc_bayes import fit_pymc
        return fit_pymc(forward=forward, param_map=param_map, obs=obs, **kwargs)
    raise ValueError(f"unknown backend {backend!r}")
```

- [ ] **Step 3: Run + commit**

```bash
pytest tests/research/inversion/test_api.py -v
git add hydrus_research/inversion/api.py tests/research/inversion/test_api.py
git commit -m "M5.5: fit() auto-backend dispatcher"
```

---

### Task 6: REST `/research/inversion/{backend}`

**Files:**
- Create: `hydrus_port_server/routers/research_inversion.py`
- Modify: `hydrus_port_server/app.py:build_app()`
- Create: `tests/research/inversion/test_rest.py`

- [ ] **Step 1: Write failing test**

Write `tests/research/inversion/test_rest.py`:

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


def test_lm_endpoint_synthetic(client):
    payload = {
        "scenario_dir": "tests/fixtures/infiltr_v1/inputs",
        "params": [
            {"name": "alpha", "target": "materials[0].alpha",
             "bounds": [0.005, 0.05], "transform": "log"},
        ],
        "obs_inline": {
            "specs": [{"name": "theta_z30_d1", "kind": "theta",
                       "location": {"z_cm": -30.0}, "time_day": 1.0}],
            "values": [0.31],
            "sigmas": [0.02],
        },
        "max_nfev": 10,                  # tight cap for CI
    }
    r = client.post("/research/inversion/lm", json=payload)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["backend"] == "lm_scipy"
    assert "alpha" in body["best_params"]


def test_unknown_backend_404(client):
    r = client.post("/research/inversion/quantum", json={})
    assert r.status_code in (404, 422)
```

- [ ] **Step 2: Implement router**

Write `hydrus_port_server/routers/research_inversion.py`:

```python
"""/research/inversion/{backend} — F3 REST surface."""
from __future__ import annotations
from pathlib import Path
from typing import Any, Literal

import numpy as np
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


class ObsInline(BaseModel):
    specs: list[ObsSpecPayload]
    values: list[float]
    sigmas: list[float]


class InversionRequest(BaseModel):
    scenario_dir: str
    params: list[ParamSpecPayload]
    obs_inline: ObsInline | None = None
    obs_csv: str | None = None
    max_nfev: int = 200
    n_real: int = 200
    n_iter: int = 4


_VALID = {"lm", "lm_scipy", "ies", "pyemu_ies", "glm", "pyemu_glm",
          "nuts", "pymc_nuts", "auto"}


@router.post("/{backend}")
def run(backend: str, req: InversionRequest):
    if backend not in _VALID:
        raise HTTPException(status_code=404,
                            detail=f"unknown backend {backend!r}; "
                                   f"available: {sorted(_VALID)}")
    from hydrus_research.parameters import ParameterSpec, ParameterMap
    from hydrus_research.observations import ObservationSpec, ObservationSet
    from hydrus_research.simulator import make_forward
    from hydrus_research.simulator.hydrus1d_adapter import Hydrus1DSimulator
    from hydrus_research.inversion import fit
    from hydrus_port.adapters.hydrus1d import load as _load_h1d

    specs = [ParameterSpec(name=p.name, target=p.target,
                           bounds=p.bounds, transform=p.transform)
             for p in req.params]
    pm = ParameterMap(specs)

    if req.obs_inline:
        obs_specs = [ObservationSpec(name=s.name, kind=s.kind,
                                     location=s.location, time_day=s.time_day)
                     for s in req.obs_inline.specs]
        obs = ObservationSet(specs=obs_specs,
                             values=np.array(req.obs_inline.values),
                             sigmas=np.array(req.obs_inline.sigmas))
    elif req.obs_csv:
        obs = ObservationSet.from_csv(Path(req.obs_csv))
    else:
        raise HTTPException(status_code=400,
                            detail="must provide obs_inline or obs_csv")

    template = _load_h1d(Path(req.scenario_dir)).to_dict()
    sim = Hydrus1DSimulator()
    forward = make_forward(sim, pm, template_scenario=template,
                           forcing=None, ic=None, obs_specs=obs.specs)

    try:
        result = fit(forward=forward, param_map=pm, obs=obs,
                     scenario_dir=req.scenario_dir,
                     backend=backend, simulator_dimension=1,
                     max_nfev=req.max_nfev,
                     n_real=req.n_real, n_iter=req.n_iter)
    except RuntimeError as e:                    # pestpp-ies missing, etc.
        raise HTTPException(status_code=503, detail=str(e))

    return result.model_dump()
```

In `hydrus_port_server/app.py:build_app()`:

```python
    try:
        from .routers.research_inversion import router as inv_router
        app.include_router(inv_router, prefix="/research/inversion",
                           tags=["research", "inversion"])
    except ImportError:
        pass
```

- [ ] **Step 3: Run + commit**

```bash
pytest tests/research/inversion/test_rest.py -v
git add hydrus_port_server/routers/research_inversion.py hydrus_port_server/app.py tests/research/inversion/test_rest.py
git commit -m "M5.6: /research/inversion/{backend} REST"
```

(Takes ~1 min for LM test with max_nfev=10.)

---

### Task 7: CLI `hydrus research invert`

**Files:**
- Modify: `hydrus_port/cli.py:_build_research_subparser`
- Create: `tests/research/inversion/test_cli.py`

- [ ] **Step 1: Write failing test**

Write `tests/research/inversion/test_cli.py`:

```python
import subprocess
import sys


def test_cli_invert_help():
    r = subprocess.run(
        [sys.executable, "-m", "hydrus_port.cli",
         "research", "invert", "--help"],
        capture_output=True, text=True,
    )
    assert r.returncode == 0
    for k in ("--obs", "--backend", "--out"):
        assert k in r.stdout
```

- [ ] **Step 2: Implement**

In `hydrus_port/cli.py`, inside `_build_research_subparser(sub)`, append:

```python
    # ----- invert (M5) -----------------------------------------------
    p_inv = rsub.add_parser("invert", help="parameter inversion / calibration")
    p_inv.add_argument("scenario_dir")
    p_inv.add_argument("--param", action="append", required=True,
                       help="target:lo:hi[:transform]; repeat for multi-D")
    p_inv.add_argument("--obs", required=True, help="observations CSV path")
    p_inv.add_argument("--backend", default="auto",
                       choices=["auto", "lm", "ies", "glm", "nuts"])
    p_inv.add_argument("--max-nfev", type=int, default=200)
    p_inv.add_argument("--n-real", type=int, default=200)
    p_inv.add_argument("--n-iter", type=int, default=4)
    p_inv.add_argument("--out", required=True, help="output JSON path")
    p_inv.set_defaults(_cmd=_cmd_research_invert)
```

Add helper alongside other `_cmd_research_*`:

```python
def _cmd_research_invert(args: argparse.Namespace) -> int:
    import json as _json
    from pathlib import Path as _P
    from hydrus_research.parameters import ParameterSpec, ParameterMap
    from hydrus_research.observations import ObservationSet
    from hydrus_research.simulator import make_forward
    from hydrus_research.simulator.hydrus1d_adapter import Hydrus1DSimulator
    from hydrus_research.inversion import fit
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
    obs = ObservationSet.from_csv(_P(args.obs))

    template = _load_h1d(_P(args.scenario_dir)).to_dict()
    sim = Hydrus1DSimulator()
    forward = make_forward(sim, pm, template_scenario=template,
                           forcing=None, ic=None, obs_specs=obs.specs)

    result = fit(forward=forward, param_map=pm, obs=obs,
                 scenario_dir=args.scenario_dir,
                 backend=args.backend,
                 simulator_dimension=1,
                 max_nfev=args.max_nfev,
                 n_real=args.n_real, n_iter=args.n_iter)
    out = _P(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(_json.dumps(result.model_dump(), indent=2))
    print(f"inversion ({result.backend}): best_params = {result.best_params}")
    print(f"  written to {out}")
    return 0
```

- [ ] **Step 3: Run + commit**

```bash
pytest tests/research/inversion/test_cli.py -v
git add hydrus_port/cli.py tests/research/inversion/test_cli.py
git commit -m "M5.7: hydrus research invert CLI"
```

---

### Task 8: GUI — InversionStudio.vue + 2 viz components

**Files:**
- Create: `desktop/src/components/ParamPosteriorBar.vue`
- Create: `desktop/src/components/ResidualsPlot.vue`
- Create: `desktop/src/pages/research/InversionStudio.vue`
- Modify: `desktop/src/api.ts`
- Modify: `desktop/src/App.vue`

- [ ] **Step 1: Append `inversion` wrapper to api.ts**

```ts
// ---- M5: Inversion --------------------------------------------------
export interface InversionRequest {
  scenario_dir: string;
  params: BatchParamSpec[];
  obs_inline: {
    specs: BatchObsSpec[];
    values: number[];
    sigmas: number[];
  };
  max_nfev?: number;
  n_real?: number;
  n_iter?: number;
}

export interface InversionResult {
  backend: string;
  best_params: Record<string, number>;
  parameter_ci_lo: Record<string, number>;
  parameter_ci_hi: Record<string, number>;
  posterior_ensemble: number[][] | null;
  posterior_param_names: string[];
  objective_history: number[];
  n_forward_calls: number;
  wall_s: number;
  diagnostics: Record<string, any>;
}

export const inversion = {
  async run(backend: "auto" | "lm" | "ies" | "glm" | "nuts",
            req: InversionRequest): Promise<InversionResult> {
    const r = await fetch(`${RESEARCH_BASE}/research/inversion/${backend}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(req),
    });
    if (!r.ok) throw new Error(`HTTP ${r.status}: ${await r.text()}`);
    return r.json();
  },
};
```

- [ ] **Step 2: Implement `ParamPosteriorBar.vue`**

```vue
<template>
  <div ref="el" class="bars"></div>
</template>

<script setup lang="ts">
import { onMounted, ref, watch } from "vue";
import Plotly from "plotly.js-dist-min";

const props = defineProps<{
  paramNames: string[];
  bestParams: Record<string, number>;
  ciLo?: Record<string, number>;
  ciHi?: Record<string, number>;
}>();

const el = ref<HTMLDivElement | null>(null);

function _draw() {
  if (!el.value) return;
  const best = props.paramNames.map(n => props.bestParams[n]);
  const lo = props.ciLo ? props.paramNames.map(n => props.bestParams[n] - props.ciLo![n]) : null;
  const hi = props.ciHi ? props.paramNames.map(n => props.ciHi![n] - props.bestParams[n]) : null;

  Plotly.newPlot(el.value, [{
    type: "bar", x: props.paramNames, y: best,
    error_y: lo && hi ? { type: "data", symmetric: false, array: hi, arrayminus: lo } : undefined,
    marker: { color: "#1f77b4" },
  }], {
    yaxis: { title: "best-fit value" },
    margin: { t: 20, l: 60, r: 20, b: 50 },
  }, { responsive: true, displayModeBar: false });
}

onMounted(_draw);
watch(() => [props.paramNames, props.bestParams, props.ciLo, props.ciHi], _draw, { deep: true });
</script>

<style scoped>.bars { width: 100%; height: 340px; }</style>
```

- [ ] **Step 3: Implement `ResidualsPlot.vue`**

```vue
<template>
  <div ref="el" class="resid"></div>
</template>

<script setup lang="ts">
import { onMounted, ref, watch } from "vue";
import Plotly from "plotly.js-dist-min";

const props = defineProps<{ history: number[] }>();
const el = ref<HTMLDivElement | null>(null);

function _draw() {
  if (!el.value) return;
  Plotly.newPlot(el.value, [{
    type: "scatter", mode: "lines+markers",
    x: props.history.map((_, i) => i + 1),
    y: props.history,
    line: { color: "#1f77b4" },
  }], {
    xaxis: { title: "forward call #" },
    yaxis: { title: "Σ(residuals²)", type: "log" },
    margin: { t: 20, l: 70, r: 20, b: 50 },
  }, { responsive: true, displayModeBar: false });
}
onMounted(_draw);
watch(() => props.history, _draw);
</script>

<style scoped>.resid { width: 100%; height: 280px; }</style>
```

- [ ] **Step 4: Implement `InversionStudio.vue`**

```vue
<template>
  <div class="inversion">
    <h2>Inversion Studio — F3</h2>
    <div class="form">
      <label>Scenario dir <input v-model="scenarioDir" /></label>
      <label>Params (target:lo:hi[:transform], one per line)
        <textarea v-model="paramText" rows="3"></textarea>
      </label>
      <label>Obs: name, z_cm, time_day, value, sigma (one per line)
        <textarea v-model="obsText" rows="4"></textarea>
      </label>
      <label>Backend
        <select v-model="backend">
          <option value="auto">auto</option>
          <option value="lm">scipy LM (fast)</option>
          <option value="ies">PEST/PyEMU IES</option>
        </select>
      </label>
      <label>Max nfev (LM) / N real (IES)
        <input v-model.number="nIter" type="number" min="1" /></label>
      <button @click="run" :disabled="running">Calibrate</button>
    </div>

    <p v-if="running">Running… ({{ Math.round(elapsed) }}s)</p>
    <p v-if="error" class="err">{{ error }}</p>

    <div v-if="result">
      <p>{{ result.backend }} — {{ result.n_forward_calls }} forward calls, {{ result.wall_s.toFixed(1) }}s</p>
      <h3>Best-fit parameters</h3>
      <ParamPosteriorBar :param-names="paramNames"
                         :best-params="result.best_params"
                         :ci-lo="result.parameter_ci_lo"
                         :ci-hi="result.parameter_ci_hi" />
      <h3>Convergence</h3>
      <ResidualsPlot :history="result.objective_history" />
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed } from "vue";
import { inversion, type InversionResult } from "../../api";
import ParamPosteriorBar from "../../components/ParamPosteriorBar.vue";
import ResidualsPlot from "../../components/ResidualsPlot.vue";

const scenarioDir = ref("tests/fixtures/infiltr_v1/inputs");
const paramText = ref("materials[0].alpha:0.005:0.05:log");
const obsText = ref("theta_z30_d1, -30, 1.0, 0.31, 0.02");
const backend = ref<"auto" | "lm" | "ies">("lm");
const nIter = ref(50);
const result = ref<InversionResult | null>(null);
const error = ref<string | null>(null);
const running = ref(false);
const elapsed = ref(0);

const paramNames = computed(() =>
  paramText.value.trim().split("\n").map(l => {
    const target = l.split(":")[0].trim();
    return target.split(".").pop() ?? target;
  }));

function _parseParams() {
  return paramText.value.trim().split("\n").map(l => {
    const parts = l.split(":");
    const target = parts[0].trim();
    const lo = parseFloat(parts[1]); const hi = parseFloat(parts[2]);
    const transform = (parts[3]?.trim() as any) ?? "linear";
    const name = target.split(".").pop() ?? target;
    return { name, target, bounds: [lo, hi] as [number, number], transform };
  });
}

function _parseObs() {
  const lines = obsText.value.trim().split("\n");
  const specs = lines.map(l => {
    const [name, z, t] = l.split(",").map(x => x.trim());
    return { name, kind: "theta" as const, location: { z_cm: parseFloat(z) }, time_day: parseFloat(t) };
  });
  const values = lines.map(l => parseFloat(l.split(",")[3].trim()));
  const sigmas = lines.map(l => parseFloat(l.split(",")[4].trim()));
  return { specs, values, sigmas };
}

async function run() {
  running.value = true; error.value = null; result.value = null;
  const t0 = Date.now();
  const tick = setInterval(() => { elapsed.value = (Date.now() - t0) / 1000; }, 200);
  try {
    result.value = await inversion.run(backend.value, {
      scenario_dir: scenarioDir.value,
      params: _parseParams(),
      obs_inline: _parseObs(),
      max_nfev: nIter.value,
      n_real: nIter.value, n_iter: 3,
    });
  } catch (e: any) {
    error.value = e.message ?? String(e);
  } finally {
    clearInterval(tick); running.value = false;
  }
}
</script>

<style scoped>
.inversion { padding: 16px; max-width: 900px; }
.form { display: grid; grid-template-columns: 200px 1fr; gap: 6px; align-items: start; margin-bottom: 12px; }
.form input, .form select, .form textarea { padding: 4px; font-family: inherit; }
.form button { grid-column: 1 / 3; padding: 8px; margin-top: 8px; }
.err { color: #c00; }
h3 { margin-top: 16px; font-size: 13px; }
</style>
```

- [ ] **Step 5: Add tab to App.vue**

In `desktop/src/App.vue`:
1. Add import: `import InversionStudio from "./pages/research/InversionStudio.vue";`
2. Widen `rightTab` union: append `"inversion"`.
3. Add tab button + conditional render after existing tabs.

- [ ] **Step 6: Commit**

```bash
git add desktop/src/api.ts desktop/src/components/ParamPosteriorBar.vue desktop/src/components/ResidualsPlot.vue desktop/src/pages/research/InversionStudio.vue desktop/src/App.vue
git commit -m "M5.8: InversionStudio.vue + posterior bar + residuals viz + nav tab"
```

---

### Task 9: End-to-end + regression + M5-complete marker

**Files:**
- Create: `tests/research/inversion/test_e2e_m5.py`

- [ ] **Step 1: Write acceptance test**

Write `tests/research/inversion/test_e2e_m5.py`:

```python
"""M5 acceptance: synthetic recovery on infiltr_v1 via LM.

Generate y_obs from a perturbed alpha_true (= alpha_nominal × 1.5);
run LM from the nominal alpha and verify it recovers alpha_true within 5%."""
import numpy as np
import pytest
from pathlib import Path

from hydrus_research.parameters import ParameterSpec, ParameterMap
from hydrus_research.observations import ObservationSpec, ObservationSet
from hydrus_research.simulator import make_forward
from hydrus_research.simulator.hydrus1d_adapter import Hydrus1DSimulator
from hydrus_research.inversion import fit_lm
from hydrus_port.adapters.hydrus1d import load as load_h1d_canonical


def test_lm_recovers_perturbed_alpha_on_infiltr_v1():
    template = load_h1d_canonical(Path("tests/fixtures/infiltr_v1/inputs")).to_dict()
    a0 = template["materials"][0]["alpha"]
    alpha_true = a0 * 1.5

    pm = ParameterMap([
        ParameterSpec(name="alpha", target="materials[0].alpha",
                      bounds=(a0 * 0.5, a0 * 3.0), transform="log"),
    ])
    obs_specs = [ObservationSpec(name=f"theta_z30_d{t}", kind="theta",
                                 location={"z_cm": -30.0}, time_day=t)
                 for t in (0.5, 1.0, 2.0)]
    sim = Hydrus1DSimulator()
    forward = make_forward(sim, pm, template_scenario=template,
                           forcing=None, ic=None, obs_specs=obs_specs)

    # Generate synthetic y_obs at alpha_true
    y_obs = forward(pm.to_vector({"alpha": alpha_true}))
    obs = ObservationSet(specs=obs_specs, values=y_obs,
                         sigmas=np.full(len(y_obs), 0.01))

    # Start from the nominal alpha; LM should walk to alpha_true
    result = fit_lm(forward=forward, param_map=pm, obs=obs,
                    x0=pm.to_vector({"alpha": a0}),
                    max_nfev=30)
    rel_err = abs(result.best_params["alpha"] - alpha_true) / alpha_true
    assert rel_err < 0.05, \
        f"recovered alpha={result.best_params['alpha']:.5g} vs true {alpha_true:.5g} (err={rel_err:.1%})"
```

- [ ] **Step 2: Run + regression + marker**

```bash
pytest tests/research/inversion/test_e2e_m5.py -v -s
pytest tests/research/ -q --ignore=tests/research/dndc_seam/test_gui_smoke.py 2>&1 | tail -5
hydrus test 1d 2>&1 | tail -3
git add tests/research/inversion/test_e2e_m5.py
git commit -m "M5.9: end-to-end LM synthetic recovery on infiltr_v1"
git commit --allow-empty -m "M5 complete: inversion (LM + IES + auto dispatcher) green; ready for M6 (UQ + Surrogate)"
```

---

## Definition of Done for M5

1. `pytest tests/research/inversion/ -v` — all green (pyemu test SKIPs if binary missing; "missing binary" test always runs).
2. `pytest tests/research/ -q` — no regression in M0/M1/M2/M3.
3. All three public callables importable: `fit_lm`, `fit_pyemu`, `fit`.
4. Synthetic recovery test: LM recovers `alpha_true × 1.5` within 5% on infiltr_v1.
5. `hydrus research invert ... --backend lm ... --out file.json` works.
6. REST `POST /research/inversion/lm` returns 200 + InversionResult.
7. GUI `Research → Inversion` tab renders; "Calibrate" wires through to a real run.
8. `hydrus test 1d/2d/roundtrip` still PASS.
9. PyMC backend stubbed (raises NotImplementedError pointing at M9).
10. **No cross-process Python serialization**: PESTPP-IES workers consume only JSON config + plain-text I/O via `hydrus_research.inversion.worker_script`. The `pestpp-ies` test that proves "binary missing → clear error" always runs; the "real IES smoke" is OPT-IN.
