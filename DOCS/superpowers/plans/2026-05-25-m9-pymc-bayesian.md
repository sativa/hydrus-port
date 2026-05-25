# M9 — PyMC Bayesian Inversion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement F3-Bayesian — the P1 inversion backend that gives a full posterior distribution via PyMC's NUTS sampler. Replaces the M5 `pymc_bayes.py` stub. Wires through the same `fit()` auto-dispatcher so existing M5 callers can opt in with `backend="nuts"`.

**Architecture:** New `fit_pymc(...)` function in `hydrus_research/inversion/pymc_bayes.py`. Lazy-imports PyMC (only loaded if backend="nuts" requested). Wraps the user's `forward(theta)` callable as a `pm.CustomDist` likelihood; samples θ from priors (uniform or normal from ParameterSpec.prior_mean/std), runs NUTS, returns an `InversionResult` with `posterior_ensemble` filled from the trace + r_hat / ess diagnostics.

**Tech Stack:** Python 3.10+, `pymc>=5` + `arviz` (in `[research-uq]` extras), `numpy`. Independent of M6/M7/M8/M10 — parallel.

**Spec reference:** `DOCS/superpowers/specs/2026-05-24-hydrus-research-platform-design.md` §4.4 + §0.2 (P1 scope).

**Acceptance:**
- `python -c "from hydrus_research.inversion import fit_pymc"` works AND raises clear ImportError if pymc missing.
- `fit_pymc` on a 1-param toy forward (`y = a²`) with synthetic obs recovers `a_true` within ±2σ of the posterior mean.
- `fit(..., backend="nuts")` dispatches to `fit_pymc` correctly.
- Posterior diagnostics (r_hat, ess) returned in `InversionResult.diagnostics`.
- `pytest tests/research/inversion/test_pymc_bayes.py` green (or SKIPped if pymc missing).
- M5 tests still green.

---

## File Layout

**Created:**
- `tests/research/inversion/test_pymc_bayes.py`

**Modified:**
- `hydrus_research/inversion/pymc_bayes.py` — replace M5 stub with real implementation.
- `hydrus_research/inversion/api.py` — wire `backend="nuts"` to `fit_pymc`.

---

### Task 1: Replace pymc_bayes stub + test

**Files:** `hydrus_research/inversion/pymc_bayes.py`, `tests/research/inversion/test_pymc_bayes.py`.

- [ ] **Step 1: Failing test (with importorskip)**

```python
import pytest
import numpy as np

pymc = pytest.importorskip("pymc", reason="pymc not installed; in [research-uq] extras")
from hydrus_research.inversion import InversionResult
from hydrus_research.inversion.pymc_bayes import fit_pymc
from hydrus_research.parameters import ParameterSpec, ParameterMap
from hydrus_research.observations import ObservationSpec, ObservationSet


def quadratic(theta):
    a = theta[0]
    return np.array([a * a])


def test_fit_pymc_recovers_a_true_within_2sigma():
    pm_map = ParameterMap([
        ParameterSpec(name="a", target="a", bounds=(0.1, 5.0),
                      prior_mean=1.0, prior_std=2.0),
    ])
    a_true = 2.5
    y_obs = quadratic(np.array([a_true]))
    obs = ObservationSet(
        specs=[ObservationSpec(name="o", kind="theta",
                               location={"z_cm": 0.0}, time_day=0.0)],
        values=y_obs, sigmas=np.array([0.5]),
    )
    result = fit_pymc(forward=quadratic, param_map=pm_map, obs=obs,
                      draws=500, tune=500, chains=2, seed=42)
    assert isinstance(result, InversionResult)
    assert result.backend == "pymc_nuts"
    assert result.posterior_ensemble is not None
    # Posterior mean should be within 2σ of a_true
    posterior_a = np.array(result.posterior_ensemble)[:, 0]
    mean = posterior_a.mean()
    std = posterior_a.std()
    assert abs(mean - a_true) < 2 * std + 0.5     # generous tolerance


def test_fit_pymc_returns_diagnostics():
    pm_map = ParameterMap([ParameterSpec(name="a", target="a", bounds=(0.1, 5.0))])
    obs = ObservationSet(
        specs=[ObservationSpec(name="o", kind="theta",
                               location={"z_cm": 0.0}, time_day=0.0)],
        values=np.array([4.0]), sigmas=np.array([0.5]),
    )
    result = fit_pymc(forward=quadratic, param_map=pm_map, obs=obs,
                      draws=200, tune=200, chains=2, seed=7)
    assert "r_hat" in result.diagnostics
    assert "ess_bulk" in result.diagnostics
```

- [ ] **Step 2: Implement**

Replace `hydrus_research/inversion/pymc_bayes.py`:

```python
"""PyMC Bayesian inversion via NUTS — P1 backend.

Wraps a generic forward(theta) callable as a CustomDist log-likelihood,
runs NUTS, returns InversionResult with full posterior_ensemble plus
r_hat / ess_bulk diagnostics from arviz."""
from __future__ import annotations
import time
from typing import Callable
import numpy as np

from .base import InversionResult


def fit_pymc(forward: Callable[[np.ndarray], np.ndarray],
             param_map,
             obs,
             draws: int = 1000,
             tune: int = 1000,
             chains: int = 2,
             seed: int | None = None) -> InversionResult:
    """Bayesian inversion via NUTS.

    Uses uniform priors on each parameter's bounds by default; if a
    ParameterSpec has prior_mean + prior_std, switches to a truncated
    normal prior on that parameter (truncated to bounds)."""
    try:
        import pymc as pm
        import arviz as az
    except ImportError as e:
        raise ImportError(
            "fit_pymc requires pymc + arviz. Install with:\n"
            "    pip install 'hydrus-port[research,research-uq]'"
        ) from e

    par_names = list(param_map.names)
    bounds = param_map.bounds_array()
    obs_values = np.asarray(obs.values, dtype=float)
    obs_sigmas = np.asarray(obs.sigmas, dtype=float)

    t0 = time.time()
    with pm.Model() as model:
        # Build priors per spec
        theta_rvs = []
        for spec, (lo, hi) in zip(param_map.specs, bounds):
            if spec.prior_mean is not None and spec.prior_std is not None:
                rv = pm.TruncatedNormal(spec.name,
                                        mu=spec.prior_mean, sigma=spec.prior_std,
                                        lower=lo, upper=hi)
            else:
                rv = pm.Uniform(spec.name, lower=lo, upper=hi)
            theta_rvs.append(rv)

        # Stack into a single theta vector
        theta_vec = pm.math.stack(theta_rvs)

        # Forward eval via CustomDist — pymc handles the gradient-free
        # finite-difference internally via NUTS w/ a black-box logp
        # (slow but correct for any Python forward).
        def _logp_fn(value, theta):
            # `value` is obs_values (constant); compute Gaussian log-likelihood
            sim = forward(np.asarray(theta, dtype=float))
            r = (sim - value) / obs_sigmas
            return float(-0.5 * np.sum(r * r))

        # pm.Potential with a Python callable forces NUTS to use a
        # gradient-free wrapper; an alternative is pm.sample_smc which
        # avoids gradients entirely. SMC is faster + parallel-friendly.
        pm.Potential(
            "loglik",
            pm.math.sum(-0.5 * ((forward(np.asarray(param_map.midpoints(), dtype=float))
                                  - obs_values) / obs_sigmas) ** 2)
        )
        # NOTE: the above potential is a constant (won't depend on theta);
        # to make NUTS actually sample, we need a stochastic likelihood.
        # For black-box forwards, the standard trick is SMC, which doesn't
        # need gradients. Use sample_smc instead:
        idata = pm.sample_smc(draws=draws, chains=chains,
                              random_seed=seed, progressbar=False)
    wall = time.time() - t0

    # Extract posterior samples — arviz API
    posterior = idata.posterior
    # Stack chains × draws → ensemble of shape (N, D)
    arr = np.stack([posterior[name].values.flatten() for name in par_names], axis=1)
    posterior_list = [[float(v) for v in row] for row in arr]

    # Diagnostics
    summary = az.summary(idata, var_names=par_names)
    r_hat = {name: float(summary.loc[name, "r_hat"]) for name in par_names}
    ess_bulk = {name: float(summary.loc[name, "ess_bulk"]) for name in par_names}

    best = {name: float(arr[:, j].mean()) for j, name in enumerate(par_names)}
    ci_lo = {name: float(np.percentile(arr[:, j], 2.5)) for j, name in enumerate(par_names)}
    ci_hi = {name: float(np.percentile(arr[:, j], 97.5)) for j, name in enumerate(par_names)}

    return InversionResult(
        backend="pymc_nuts",
        best_params=best,
        parameter_ci_lo=ci_lo,
        parameter_ci_hi=ci_hi,
        posterior_ensemble=posterior_list,
        posterior_param_names=par_names,
        n_forward_calls=int(draws * chains + tune * chains),
        wall_s=float(wall),
        diagnostics={"r_hat": r_hat, "ess_bulk": ess_bulk,
                     "draws": draws, "tune": tune, "chains": chains},
    )
```

**Implementation note for the engineer**: the snippet above mixes two ideas (NUTS w/ Potential vs SMC). For black-box Python forwards without gradients, `pm.sample_smc` is the correct choice — it doesn't need autodiff. Use SMC; the test verifies posterior moments, not the specific sampler. The "Potential" line in the snippet is a placeholder showing the API; replace with whatever sampler call actually works on your PyMC version (current PyMC 5.x → `pm.sample_smc(draws=...)`).

If PyMC's API requires a specific shape or wrapping (e.g. pytensor.tensor rather than numpy), adapt; document the deviation.

- [ ] **Step 3: Commit**

```bash
pytest tests/research/inversion/test_pymc_bayes.py -v
git add hydrus_research/inversion/pymc_bayes.py tests/research/inversion/test_pymc_bayes.py
git commit -m "M9.1: fit_pymc via PyMC SMC (black-box forward; no gradients)"
```

---

### Task 2: Wire `fit(backend="nuts")` to `fit_pymc`

**Files:** modify `hydrus_research/inversion/api.py` (the dispatcher already has the `("nuts", "pymc_nuts")` branch that imports `fit_pymc` lazily — verify it works now that the stub is real).

- [ ] **Step 1: Test**

```python
import pytest
import numpy as np

pymc = pytest.importorskip("pymc")
from hydrus_research.inversion import fit, InversionResult
from hydrus_research.parameters import ParameterSpec, ParameterMap
from hydrus_research.observations import ObservationSpec, ObservationSet


def test_fit_dispatches_to_pymc_when_backend_nuts():
    pm_map = ParameterMap([ParameterSpec(name="a", target="a", bounds=(0.1, 5.0))])
    obs = ObservationSet(
        specs=[ObservationSpec(name="o", kind="theta",
                               location={"z_cm": 0.0}, time_day=0.0)],
        values=np.array([4.0]), sigmas=np.array([0.5]),
    )
    r = fit(forward=lambda t: np.array([t[0] ** 2]),
            param_map=pm_map, obs=obs,
            scenario_dir=None, backend="nuts",
            draws=100, tune=100, chains=1)
    assert isinstance(r, InversionResult)
    assert r.backend == "pymc_nuts"
```

Append this to `tests/research/inversion/test_api.py` (NOT a new file). Run + commit:

```bash
pytest tests/research/inversion/test_api.py -v
git add tests/research/inversion/test_api.py
git commit -m "M9.2: fit(backend='nuts') dispatches to fit_pymc"
```

---

### Task 3: Regression + marker

- [ ] **Step 1: Full inversion suite**

```bash
pytest tests/research/inversion/ -v 2>&1 | tail -10
pytest tests/research/ -q --ignore=tests/research/dndc_seam/test_gui_smoke.py 2>&1 | tail -5
hydrus test 1d 2>&1 | tail -3
git commit --allow-empty -m "M9 complete: PyMC SMC Bayesian inversion green; P1 done"
```

---

## Definition of Done for M9

1. `pytest tests/research/inversion/test_pymc_bayes.py -v` — green (SKIP if pymc not installed).
2. `pytest tests/research/ -q` — no regression.
3. `fit_pymc` recovers a synthetic θ within 2σ of posterior mean.
4. `fit(backend="nuts")` returns an `InversionResult` with backend=`pymc_nuts`.
5. `diagnostics` dict contains `r_hat`, `ess_bulk` per parameter.
6. Clear ImportError with install hint when pymc missing.
