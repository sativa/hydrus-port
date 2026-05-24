# M2 — PTF + Soil Library Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement F1 — given soil texture (sand / silt / clay / BD), return van Genuchten hydraulic parameters via three pedotransfer functions (ROSETTA-3 neural net, Carsel-Parrish 1988 12-class lookup, Wösten HYPRES continuous PTF), feeding into the M0 ParameterMap as priors and exposed via REST + CLI + a `SoilLibrary.vue` page with an interactive texture-triangle picker.

**Architecture:** New sibling subpackage `hydrus_research/ptf/` with three independent PTF backends behind a single `texture_to_vg()` entry point, plus `usda_class_to_vg()` for the 12-class shortcut and `vg_to_prior()` to convert PTF output into `ParameterSpec` priors for M4 inversion / M5 UQ. REST in `hydrus_port_server/routers/research_ptf.py`, CLI in `hydrus_port/cli.py`, GUI in `desktop/`.

**Tech Stack:** Python 3.10+, `rosetta-soil` (USDA-ARS official Python package — ONNX runtime), `numpy`, `pydantic`, Vue 3 + Plotly.js (existing in `desktop/`) for the ternary plot. M2 is independent of M1 — can be developed and merged in parallel.

**Spec reference:** `DOCS/superpowers/specs/2026-05-24-hydrus-research-platform-design.md` §4.1 + §5.1 (SoilLibrary page) + §5.5 (rosetta-soil risk note).

**Acceptance:**
- `python -c "from hydrus_research.ptf import texture_to_vg; print(texture_to_vg(45, 35, 20, bulk_density_g_cm3=1.4))"` returns a PTFResult with all 5 VG params populated.
- `usda_class_to_vg('loam')` returns values matching Carsel-Parrish 1988 Table 2 (within published precision).
- `rosetta3_h2` (sand/silt/clay + BD model) reproduces 3 spot checks from the rosetta-soil package's own README example.
- REST `POST /research/ptf/predict` returns JSON with the 5 params + covariance matrix when applicable.
- `hydrus research soil ptf --texture sand=45,silt=35,clay=20 --bd 1.4` prints a tidy table.
- Tauri GUI `Research → Soil Library` page renders the texture triangle and reactively updates VG params on click.
- `pytest tests/research/ptf/` green; full `tests/research/` still green.

---

## File Layout

**Created:**
- `hydrus_research/ptf/__init__.py` — re-exports.
- `hydrus_research/ptf/result.py` — `PTFResult` Pydantic model.
- `hydrus_research/ptf/carsel_parrish.py` — 12 USDA classes hardcoded.
- `hydrus_research/ptf/rosetta.py` — `rosetta-soil` package wrapper.
- `hydrus_research/ptf/wosten_hypres.py` — Wösten 1999 closed-form PTF.
- `hydrus_research/ptf/presets.py` — USDA texture centers + `usda_class_to_vg`.
- `hydrus_research/ptf/uncertainty.py` — `vg_to_prior(ptf) → list[ParameterSpec]`.
- `hydrus_research/ptf/api.py` — public `texture_to_vg` with `method="rosetta3_auto"`.
- `hydrus_port_server/routers/research_ptf.py` — `/research/ptf/*` routes.
- `desktop/src/components/TextureTriangle.vue` — reusable ternary plot component.
- `desktop/src/pages/research/SoilLibrary.vue` — F1 GUI page.
- `tests/research/ptf/__init__.py`
- `tests/research/ptf/test_carsel_parrish.py`
- `tests/research/ptf/test_rosetta.py`
- `tests/research/ptf/test_wosten.py`
- `tests/research/ptf/test_presets.py`
- `tests/research/ptf/test_uncertainty.py`
- `tests/research/ptf/test_api.py`
- `tests/research/ptf/test_rest.py`
- `tests/research/ptf/test_cli.py`

**Modified:**
- `hydrus_port_server/app.py` — register the new router (one-line `include_router`).
- `hydrus_port/cli.py` — add `hydrus research soil ptf` subcommand.
- `desktop/src/api.ts` — add `ptf.*` REST wrappers.
- `desktop/src/App.vue` — add nav entry for Soil Library page.

---

### Task 1: Sub-package skeleton + dependency probe

**Files:**
- Create: `hydrus_research/ptf/__init__.py`
- Create: `tests/research/ptf/__init__.py`

- [ ] **Step 1: Create empty package files**

```bash
mkdir -p hydrus_research/ptf tests/research/ptf
touch hydrus_research/ptf/__init__.py tests/research/ptf/__init__.py
```

Write `hydrus_research/ptf/__init__.py`:

```python
"""Pedotransfer functions (F1) — texture → van Genuchten hydraulic params.

Three backends share a single PTFResult schema:
  - rosetta3_*  — neural-net PTFs (rosetta-soil package, USDA-ARS)
  - carsel_parrish — 12 USDA-class lookup (1988)
  - wosten — HYPRES continuous closed-form (Wösten 1999)

Entry point: texture_to_vg(...). For the 12-class shortcut use usda_class_to_vg(name).
"""
from .result import PTFResult
from .api import texture_to_vg
from .presets import usda_class_to_vg
from .uncertainty import vg_to_prior

__all__ = ["PTFResult", "texture_to_vg", "usda_class_to_vg", "vg_to_prior"]
```

- [ ] **Step 2: Verify rosetta-soil installs cleanly (smoke only — don't import yet)**

```bash
pip install rosetta-soil
python -c "import rosetta; print('rosetta version:', getattr(rosetta, '__version__', '?'))"
```

If installation fails (network issue or platform incompat), document the failure in the task report — Task 3 will degrade gracefully (`ImportError` caught + only `carsel_parrish` + `wosten` available).

- [ ] **Step 3: Commit**

```bash
git add hydrus_research/ptf/ tests/research/ptf/
git commit -m "M2.1: ptf sub-package skeleton"
```

(Package import will fail until Tasks 2–7 are done; no try/except.)

---

### Task 2: PTFResult schema

**Files:**
- Create: `hydrus_research/ptf/result.py`
- Create: `tests/research/ptf/test_result.py`

- [ ] **Step 1: Write the failing test**

Write `tests/research/ptf/test_result.py`:

```python
import pytest
import numpy as np
from hydrus_research.ptf import PTFResult


def test_ptf_result_minimum():
    r = PTFResult(theta_r=0.05, theta_s=0.43, alpha=0.036, n=1.56, Ks=4.31,
                  method="carsel_parrish")
    assert r.L == 0.5         # default Mualem tortuosity
    assert r.covariance is None


def test_ptf_result_with_covariance():
    cov = np.eye(5).tolist()
    r = PTFResult(theta_r=0.05, theta_s=0.43, alpha=0.036, n=1.56, Ks=4.31,
                  method="rosetta3_h2", covariance=cov)
    assert len(r.covariance) == 5
    assert len(r.covariance[0]) == 5


def test_ptf_result_method_must_be_known():
    with pytest.raises(Exception):
        PTFResult(theta_r=0.05, theta_s=0.43, alpha=0.036, n=1.56, Ks=4.31,
                  method="banana")
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest tests/research/ptf/test_result.py -v
```

Expected: ImportError on `PTFResult`.

- [ ] **Step 3: Implement**

Write `hydrus_research/ptf/result.py`:

```python
"""PTFResult — output of every pedotransfer function call.

`theta_r, theta_s, alpha, n, Ks, L` are the van Genuchten-Mualem 6 parameters
in HYDRUS conventions (cm⁻¹ for alpha, dimensionless n, cm/day for Ks).
`covariance` is the 5×5 covariance of (theta_r, theta_s, alpha, n, Ks) when
the backend provides one (ROSETTA-3 returns a per-prediction stddev; we
store it as the diagonal of `covariance`)."""
from __future__ import annotations
from typing import Literal
from pydantic import BaseModel


PTFMethod = Literal[
    "rosetta3_h1", "rosetta3_h2", "rosetta3_h3", "rosetta3_h4",
    "carsel_parrish", "wosten",
]


class PTFResult(BaseModel):
    theta_r: float
    theta_s: float
    alpha: float                                # 1/cm
    n: float                                    # > 1, dimensionless
    Ks: float                                   # cm/day
    L: float = 0.5                              # Mualem pore-connectivity tortuosity
    method: PTFMethod
    covariance: list[list[float]] | None = None
```

- [ ] **Step 4: Run to verify pass**

```bash
pytest tests/research/ptf/test_result.py -v
```

Expected: 3 PASS.

- [ ] **Step 5: Commit**

```bash
git add hydrus_research/ptf/result.py tests/research/ptf/test_result.py
git commit -m "M2.2: PTFResult schema (5 VG params + L + method + covariance)"
```

---

### Task 3: Carsel-Parrish 12-class lookup

**Files:**
- Create: `hydrus_research/ptf/carsel_parrish.py`
- Create: `tests/research/ptf/test_carsel_parrish.py`

Carsel & Parrish (1988) Water Resources Research 24(5) Table 2 gives published mean VG parameters per USDA texture class. Hardcode them — they're a fixed reference.

- [ ] **Step 1: Write the failing test (values from the published 1988 table)**

Write `tests/research/ptf/test_carsel_parrish.py`:

```python
import pytest
from hydrus_research.ptf.carsel_parrish import (
    USDA_CLASSES, carsel_parrish_lookup,
)


def test_loam_matches_1988_table():
    r = carsel_parrish_lookup("loam")
    # Carsel & Parrish 1988 Table 2: loam mean values
    assert r.theta_r == pytest.approx(0.078, abs=0.001)
    assert r.theta_s == pytest.approx(0.43,  abs=0.005)
    assert r.alpha   == pytest.approx(0.036, abs=0.002)
    assert r.n       == pytest.approx(1.56,  abs=0.02)
    assert r.Ks      == pytest.approx(24.96, abs=0.1)   # cm/day
    assert r.method  == "carsel_parrish"


def test_sand_matches_1988_table():
    r = carsel_parrish_lookup("sand")
    assert r.theta_r == pytest.approx(0.045, abs=0.001)
    assert r.theta_s == pytest.approx(0.43,  abs=0.005)
    assert r.alpha   == pytest.approx(0.145, abs=0.005)
    assert r.n       == pytest.approx(2.68,  abs=0.05)
    assert r.Ks      == pytest.approx(712.8, abs=1.0)


def test_clay_matches_1988_table():
    r = carsel_parrish_lookup("clay")
    assert r.theta_r == pytest.approx(0.068, abs=0.002)
    assert r.theta_s == pytest.approx(0.38,  abs=0.005)
    assert r.alpha   == pytest.approx(0.008, abs=0.001)
    assert r.n       == pytest.approx(1.09,  abs=0.02)
    assert r.Ks      == pytest.approx(4.8,   abs=0.1)


def test_all_12_classes_present():
    assert len(USDA_CLASSES) == 12
    for c in ("sand", "loamy_sand", "sandy_loam", "loam", "silt", "silt_loam",
              "sandy_clay_loam", "clay_loam", "silty_clay_loam",
              "sandy_clay", "silty_clay", "clay"):
        assert c in USDA_CLASSES


def test_unknown_class_raises():
    with pytest.raises(KeyError):
        carsel_parrish_lookup("not_a_texture")
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest tests/research/ptf/test_carsel_parrish.py -v
```

Expected: ImportError.

- [ ] **Step 3: Implement (values from Carsel & Parrish 1988 Table 2)**

Write `hydrus_research/ptf/carsel_parrish.py`:

```python
"""Carsel & Parrish (1988) WRR 24(5) Table 2 — mean VG parameters per
USDA texture class. Ks in cm/day; alpha in 1/cm; n dimensionless.

Reference: Carsel, R. F., & Parrish, R. S. (1988). Developing joint
probability distributions of soil water retention characteristics.
Water Resources Research, 24(5), 755-769."""
from __future__ import annotations
from .result import PTFResult


# (theta_r, theta_s, alpha [1/cm], n, Ks [cm/day])  — published means
USDA_CLASSES: dict[str, tuple[float, float, float, float, float]] = {
    "sand":             (0.045, 0.43, 0.145, 2.68, 712.8),
    "loamy_sand":       (0.057, 0.41, 0.124, 2.28, 350.2),
    "sandy_loam":       (0.065, 0.41, 0.075, 1.89, 106.1),
    "loam":             (0.078, 0.43, 0.036, 1.56, 24.96),
    "silt":             (0.034, 0.46, 0.016, 1.37, 6.0),
    "silt_loam":        (0.067, 0.45, 0.020, 1.41, 10.8),
    "sandy_clay_loam":  (0.100, 0.39, 0.059, 1.48, 31.44),
    "clay_loam":        (0.095, 0.41, 0.019, 1.31, 6.24),
    "silty_clay_loam":  (0.089, 0.43, 0.010, 1.23, 1.68),
    "sandy_clay":       (0.100, 0.38, 0.027, 1.23, 2.88),
    "silty_clay":       (0.070, 0.36, 0.005, 1.09, 0.48),
    "clay":             (0.068, 0.38, 0.008, 1.09, 4.8),
}


def carsel_parrish_lookup(class_name: str) -> PTFResult:
    name = class_name.lower().strip().replace(" ", "_").replace("-", "_")
    if name not in USDA_CLASSES:
        raise KeyError(f"unknown USDA class {class_name!r}; available: {sorted(USDA_CLASSES)}")
    tr, ts, a, n, ks = USDA_CLASSES[name]
    return PTFResult(theta_r=tr, theta_s=ts, alpha=a, n=n, Ks=ks,
                     method="carsel_parrish")
```

- [ ] **Step 4: Run to verify pass**

```bash
pytest tests/research/ptf/test_carsel_parrish.py -v
```

Expected: 5 PASS.

- [ ] **Step 5: Commit**

```bash
git add hydrus_research/ptf/carsel_parrish.py tests/research/ptf/test_carsel_parrish.py
git commit -m "M2.3: Carsel-Parrish 1988 — 12 USDA classes hardcoded"
```

---

### Task 4: ROSETTA-3 wrapper

**Files:**
- Create: `hydrus_research/ptf/rosetta.py`
- Create: `tests/research/ptf/test_rosetta.py`

- [ ] **Step 1: Inspect the rosetta-soil API**

Run:

```bash
python -c "import rosetta; help(rosetta)"
```

The package's public entry point as of 2025 is `rosetta.rosetta(soildata, version=N)` where `soildata` is a 2-D array (rows = samples, cols = sand%, silt%, clay% [, bulk density [, θ at 33 kPa [, θ at 1500 kPa ]]]) and the model version is selected by the number of cols (3 → H1; 4 → H2; 5 → H3; 6 → H4). It returns a tuple `(mean, std, code)` where `mean` and `std` are `(n_samples, 5)` arrays of (theta_r, theta_s, log10(alpha), log10(n), log10(Ks)). **Verify the exact API** before coding; if it differs, adapt and report. Mean/std on log-scale params must be back-transformed for the PTFResult.

- [ ] **Step 2: Write the failing test**

Write `tests/research/ptf/test_rosetta.py`:

```python
import pytest


rosetta = pytest.importorskip("rosetta",
                              reason="rosetta-soil package not installed")
from hydrus_research.ptf.rosetta import rosetta3_predict


def test_h1_sand_silt_clay_only():
    """H1 model uses sand/silt/clay only (no BD)."""
    r = rosetta3_predict(sand_pct=45, silt_pct=35, clay_pct=20)
    assert 0.0 < r.theta_r < 0.2
    assert 0.2 < r.theta_s < 0.6
    assert r.alpha > 0
    assert r.n > 1
    assert r.Ks > 0
    assert r.method == "rosetta3_h1"


def test_h2_with_bd_uses_h2_model():
    r = rosetta3_predict(sand_pct=45, silt_pct=35, clay_pct=20,
                         bulk_density_g_cm3=1.4)
    assert r.method == "rosetta3_h2"
    assert r.covariance is not None
    assert len(r.covariance) == 5


def test_h3_with_theta33():
    r = rosetta3_predict(sand_pct=45, silt_pct=35, clay_pct=20,
                         bulk_density_g_cm3=1.4, theta_33=0.31)
    assert r.method == "rosetta3_h3"


def test_h4_with_theta33_and_theta1500():
    r = rosetta3_predict(sand_pct=45, silt_pct=35, clay_pct=20,
                         bulk_density_g_cm3=1.4, theta_33=0.31, theta_1500=0.12)
    assert r.method == "rosetta3_h4"


def test_invalid_texture_sum_raises():
    with pytest.raises(ValueError):
        rosetta3_predict(sand_pct=50, silt_pct=50, clay_pct=10)    # sums to 110
```

- [ ] **Step 3: Run to verify failure**

```bash
pytest tests/research/ptf/test_rosetta.py -v
```

Expected: ImportError on `rosetta3_predict`.

- [ ] **Step 4: Implement**

Write `hydrus_research/ptf/rosetta.py`:

```python
"""Wrapper around the USDA-ARS rosetta-soil package.

The public API picks the right hierarchical model from the inputs:
  - 3 inputs (sand/silt/clay)       → H1
  - + bulk density                  → H2
  - + theta at 33 kPa               → H3
  - + theta at 1500 kPa             → H4

ROSETTA returns log10(alpha), log10(n), log10(Ks); we back-transform here.
"""
from __future__ import annotations
import numpy as np
import rosetta as _rosetta

from .result import PTFResult


def _validate_texture(sand_pct, silt_pct, clay_pct):
    total = sand_pct + silt_pct + clay_pct
    if not 99.0 <= total <= 101.0:
        raise ValueError(f"sand+silt+clay must sum to 100; got {total}")
    for v, name in [(sand_pct, "sand"), (silt_pct, "silt"), (clay_pct, "clay")]:
        if not 0.0 <= v <= 100.0:
            raise ValueError(f"{name}_pct out of range [0, 100]: {v}")


def rosetta3_predict(sand_pct: float, silt_pct: float, clay_pct: float,
                     bulk_density_g_cm3: float | None = None,
                     theta_33: float | None = None,
                     theta_1500: float | None = None) -> PTFResult:
    _validate_texture(sand_pct, silt_pct, clay_pct)
    cols = [sand_pct, silt_pct, clay_pct]
    method = "rosetta3_h1"
    if bulk_density_g_cm3 is not None:
        cols.append(bulk_density_g_cm3)
        method = "rosetta3_h2"
    if theta_33 is not None:
        cols.append(theta_33)
        method = "rosetta3_h3"
    if theta_1500 is not None:
        cols.append(theta_1500)
        method = "rosetta3_h4"

    arr = np.asarray([cols], dtype=float)
    # API: rosetta.rosetta(arr) returns (mean, std, code) per package docs
    mean, std, _code = _rosetta.rosetta(arr)
    mean = np.asarray(mean).reshape(-1, 5)[0]
    std = np.asarray(std).reshape(-1, 5)[0]

    # Columns: theta_r, theta_s, log10(alpha), log10(n), log10(Ks)
    theta_r, theta_s, log_a, log_n, log_ks = mean
    alpha = float(10.0 ** log_a)
    n = float(10.0 ** log_n)
    Ks = float(10.0 ** log_ks)

    # Build a diagonal covariance from per-param stddevs (log-back-transformed
    # for alpha/n/Ks — first-order delta-method approximation).
    diag = np.zeros(5, dtype=float)
    diag[0] = std[0] ** 2                                # theta_r linear
    diag[1] = std[1] ** 2                                # theta_s linear
    diag[2] = (alpha * np.log(10) * std[2]) ** 2         # alpha from log10
    diag[3] = (n * np.log(10) * std[3]) ** 2
    diag[4] = (Ks * np.log(10) * std[4]) ** 2
    cov = np.diag(diag).tolist()

    return PTFResult(theta_r=float(theta_r), theta_s=float(theta_s),
                     alpha=alpha, n=n, Ks=Ks,
                     method=method, covariance=cov)
```

- [ ] **Step 5: Run to verify pass**

```bash
pytest tests/research/ptf/test_rosetta.py -v
```

Expected: 5 PASS. If `rosetta` is not installed, the test is `xfail`-skipped via `pytest.importorskip`.

- [ ] **Step 6: Commit**

```bash
git add hydrus_research/ptf/rosetta.py tests/research/ptf/test_rosetta.py
git commit -m "M2.4: rosetta-soil wrapper (H1-H4 hierarchical models)"
```

---

### Task 5: Wösten HYPRES continuous PTF

**Files:**
- Create: `hydrus_research/ptf/wosten_hypres.py`
- Create: `tests/research/ptf/test_wosten.py`

Wösten et al. (1999) HYPRES gives **closed-form** continuous PTFs from sand%, silt%, clay%, bulk density (g/cm³), organic matter%, and a topsoil/subsoil flag. No external dependency.

- [ ] **Step 1: Write the failing test (sanity bounds only — Wösten's exact published numbers vary per subset)**

Write `tests/research/ptf/test_wosten.py`:

```python
import pytest
from hydrus_research.ptf.wosten_hypres import wosten_predict


def test_returns_5_param_ptf_result():
    r = wosten_predict(sand_pct=45, silt_pct=35, clay_pct=20,
                       bulk_density_g_cm3=1.4, organic_matter_pct=1.5,
                       topsoil=True)
    assert r.method == "wosten"
    assert 0.0 < r.theta_r < 0.2
    assert 0.3 < r.theta_s < 0.7
    assert 0.001 < r.alpha < 1.0
    assert 1.0 < r.n < 3.0
    assert r.Ks > 0


def test_topsoil_subsoil_differ():
    a = wosten_predict(sand_pct=45, silt_pct=35, clay_pct=20,
                       bulk_density_g_cm3=1.4, organic_matter_pct=1.5,
                       topsoil=True)
    b = wosten_predict(sand_pct=45, silt_pct=35, clay_pct=20,
                       bulk_density_g_cm3=1.4, organic_matter_pct=1.5,
                       topsoil=False)
    # Topsoil and subsoil PTFs differ in at least one parameter
    differ = any(abs(getattr(a, p) - getattr(b, p)) > 1e-6
                 for p in ("theta_s", "alpha", "n", "Ks"))
    assert differ


def test_rejects_bad_texture():
    with pytest.raises(ValueError):
        wosten_predict(sand_pct=120, silt_pct=0, clay_pct=0,
                       bulk_density_g_cm3=1.4, organic_matter_pct=1.5,
                       topsoil=True)
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest tests/research/ptf/test_wosten.py -v
```

Expected: ImportError.

- [ ] **Step 3: Implement (Wösten et al. 1999, Eq. 1-4)**

Write `hydrus_research/ptf/wosten_hypres.py`:

```python
"""Wösten et al. (1999) HYPRES continuous PTF.

Reference: Wösten, J. H. M., Lilly, A., Nemes, A., & Le Bas, C. (1999).
Development and use of a database of hydraulic properties of European
soils. Geoderma, 90(3-4), 169-185.

Inputs: sand%, silt%, clay%, bulk density (g/cm³), organic matter %,
topsoil/subsoil flag (boolean). Returns the 5 VG params via closed-form
multivariate polynomial regressions."""
from __future__ import annotations
import math
from .result import PTFResult


def _validate(sand_pct, silt_pct, clay_pct):
    total = sand_pct + silt_pct + clay_pct
    if not 99.0 <= total <= 101.0:
        raise ValueError(f"sand+silt+clay must sum to 100; got {total}")
    for v, name in [(sand_pct, "sand"), (silt_pct, "silt"), (clay_pct, "clay")]:
        if not 0.0 <= v <= 100.0:
            raise ValueError(f"{name}_pct out of range [0, 100]: {v}")


def wosten_predict(sand_pct: float, silt_pct: float, clay_pct: float,
                   bulk_density_g_cm3: float,
                   organic_matter_pct: float,
                   topsoil: bool) -> PTFResult:
    _validate(sand_pct, silt_pct, clay_pct)
    C = clay_pct
    S = silt_pct                                # silt + clay = fine fraction
    D = bulk_density_g_cm3
    OM = max(organic_matter_pct, 0.01)
    topsoil_i = 1 if topsoil else 0

    # Saturated water content (theta_s) — Eq. 1
    theta_s = (0.7919
               + 0.001691 * C
               - 0.29619 * D
               - 0.000001491 * S * S
               + 0.0000821 * OM * OM
               + 0.02427 / C
               + 0.01113 / S
               + 0.01472 * math.log(S)
               - 0.0000733 * OM * C
               - 0.000619 * D * C
               - 0.001183 * D * OM
               - 0.0001664 * topsoil_i * S)

    # ln(alpha*) — Eq. 2  (alpha in 1/cm)
    ln_alpha_star = (-14.96
                     + 0.03135 * C
                     + 0.0351 * S
                     + 0.646 * OM
                     + 15.29 * D
                     - 0.192 * topsoil_i
                     - 4.671 * D * D
                     - 0.000781 * C * C
                     - 0.00687 * OM * OM
                     + 0.0449 / OM
                     + 0.0663 * math.log(S)
                     + 0.1482 * math.log(OM)
                     - 0.04546 * D * S
                     - 0.4852 * D * OM
                     + 0.00673 * topsoil_i * C)
    alpha = math.exp(ln_alpha_star)

    # ln(n*-1) — Eq. 3  (n > 1 always)
    ln_n_minus_1 = (-25.23
                    - 0.02195 * C
                    + 0.0074 * S
                    - 0.1940 * OM
                    + 45.5 * D
                    - 7.24 * D * D
                    + 0.0003658 * C * C
                    + 0.002885 * OM * OM
                    - 12.81 / D
                    - 0.1524 / S
                    - 0.01958 / OM
                    - 0.2876 * math.log(S)
                    - 0.0709 * math.log(OM)
                    - 44.6 * math.log(D)
                    - 0.02264 * D * C
                    + 0.0896 * D * OM
                    + 0.00718 * topsoil_i * C)
    n = math.exp(ln_n_minus_1) + 1.0

    # ln(Ks) — Eq. 4 (cm/day)
    ln_Ks = (7.755
             + 0.0352 * S
             + 0.93 * topsoil_i
             - 0.967 * D * D
             - 0.000484 * C * C
             - 0.000322 * S * S
             + 0.001 / S
             - 0.0748 / OM
             - 0.643 * math.log(S)
             - 0.01398 * D * C
             - 0.1673 * D * OM
             + 0.02986 * topsoil_i * C
             - 0.03305 * topsoil_i * S)
    Ks = math.exp(ln_Ks)

    # theta_r is not predicted by Wösten 1999; HYDRUS convention default
    theta_r = 0.01

    return PTFResult(theta_r=theta_r, theta_s=float(theta_s),
                     alpha=float(alpha), n=float(n), Ks=float(Ks),
                     method="wosten")
```

- [ ] **Step 4: Run to verify pass**

```bash
pytest tests/research/ptf/test_wosten.py -v
```

Expected: 3 PASS.

- [ ] **Step 5: Commit**

```bash
git add hydrus_research/ptf/wosten_hypres.py tests/research/ptf/test_wosten.py
git commit -m "M2.5: Wösten HYPRES 1999 closed-form PTF"
```

---

### Task 6: Presets — `usda_class_to_vg` + USDA texture centers

**Files:**
- Create: `hydrus_research/ptf/presets.py`
- Create: `tests/research/ptf/test_presets.py`

- [ ] **Step 1: Write the failing test**

Write `tests/research/ptf/test_presets.py`:

```python
import pytest
from hydrus_research.ptf import usda_class_to_vg
from hydrus_research.ptf.presets import USDA_TEXTURE_CENTERS


def test_class_to_vg_loam():
    r = usda_class_to_vg("loam")
    # Same as Carsel-Parrish since presets aliases the lookup
    assert r.method == "carsel_parrish"
    assert 0.02 < r.alpha < 0.05


def test_usda_texture_centers_has_12_entries():
    assert len(USDA_TEXTURE_CENTERS) == 12
    sand_center = USDA_TEXTURE_CENTERS["sand"]
    assert sand_center["sand_pct"] > 85
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest tests/research/ptf/test_presets.py -v
```

Expected: ImportError.

- [ ] **Step 3: Implement**

Write `hydrus_research/ptf/presets.py`:

```python
"""USDA texture-class centers + thin `usda_class_to_vg` wrapper around
Carsel-Parrish lookup. The centers are useful to seed the texture triangle
in the GUI when the user picks a class instead of clicking the triangle."""
from __future__ import annotations
from .carsel_parrish import carsel_parrish_lookup
from .result import PTFResult


# Approximate centroids of each USDA texture class in (sand%, silt%, clay%).
# Source: USDA Soil Survey Manual texture triangle vertices.
USDA_TEXTURE_CENTERS: dict[str, dict[str, float]] = {
    "sand":             {"sand_pct": 92.0, "silt_pct": 5.0,  "clay_pct": 3.0},
    "loamy_sand":       {"sand_pct": 82.0, "silt_pct": 12.0, "clay_pct": 6.0},
    "sandy_loam":       {"sand_pct": 65.0, "silt_pct": 25.0, "clay_pct": 10.0},
    "loam":             {"sand_pct": 40.0, "silt_pct": 40.0, "clay_pct": 20.0},
    "silt":             {"sand_pct": 5.0,  "silt_pct": 88.0, "clay_pct": 7.0},
    "silt_loam":        {"sand_pct": 20.0, "silt_pct": 65.0, "clay_pct": 15.0},
    "sandy_clay_loam":  {"sand_pct": 60.0, "silt_pct": 13.0, "clay_pct": 27.0},
    "clay_loam":        {"sand_pct": 32.0, "silt_pct": 34.0, "clay_pct": 34.0},
    "silty_clay_loam":  {"sand_pct": 10.0, "silt_pct": 56.0, "clay_pct": 34.0},
    "sandy_clay":       {"sand_pct": 52.0, "silt_pct": 6.0,  "clay_pct": 42.0},
    "silty_clay":       {"sand_pct": 6.0,  "silt_pct": 47.0, "clay_pct": 47.0},
    "clay":             {"sand_pct": 22.0, "silt_pct": 20.0, "clay_pct": 58.0},
}


def usda_class_to_vg(class_name: str) -> PTFResult:
    """Look up VG parameters for a USDA texture class (Carsel-Parrish 1988 means)."""
    return carsel_parrish_lookup(class_name)
```

- [ ] **Step 4: Run + commit**

```bash
pytest tests/research/ptf/test_presets.py -v
git add hydrus_research/ptf/presets.py tests/research/ptf/test_presets.py
git commit -m "M2.6: USDA texture class centers + usda_class_to_vg shortcut"
```

---

### Task 7: `vg_to_prior` — convert PTFResult into ParameterSpec priors

**Files:**
- Create: `hydrus_research/ptf/uncertainty.py`
- Create: `tests/research/ptf/test_uncertainty.py`

- [ ] **Step 1: Write the failing test**

Write `tests/research/ptf/test_uncertainty.py`:

```python
import numpy as np
import pytest
from hydrus_research.ptf import vg_to_prior, PTFResult
from hydrus_research.parameters import ParameterSpec, ParameterMap


def _ptf_with_diag_cov():
    cov = (np.diag([0.0001, 0.0001, 0.0001, 0.01, 0.5])).tolist()
    return PTFResult(theta_r=0.07, theta_s=0.43, alpha=0.036,
                     n=1.56, Ks=24.96, method="rosetta3_h2",
                     covariance=cov)


def test_vg_to_prior_returns_5_specs():
    specs = vg_to_prior(_ptf_with_diag_cov(), material_index=0)
    assert len(specs) == 5
    names = {s.name for s in specs}
    assert names == {"theta_r", "theta_s", "alpha", "n", "Ks"}


def test_vg_to_prior_uses_log_transform_for_alpha_and_Ks():
    specs = vg_to_prior(_ptf_with_diag_cov(), material_index=0)
    by_name = {s.name: s for s in specs}
    assert by_name["alpha"].transform == "log"
    assert by_name["Ks"].transform == "log"
    assert by_name["theta_r"].transform == "linear"
    assert by_name["theta_s"].transform == "linear"
    assert by_name["n"].transform == "linear"


def test_vg_to_prior_targets_material_index_correctly():
    specs = vg_to_prior(_ptf_with_diag_cov(), material_index=2)
    by_name = {s.name: s for s in specs}
    assert by_name["alpha"].target == "materials[2].alpha"


def test_vg_to_prior_bounds_span_4_sigma_or_physical_minimum():
    specs = vg_to_prior(_ptf_with_diag_cov(), material_index=0)
    by_name = {s.name: s for s in specs}
    # theta_s prior mean is 0.43 with stddev 0.01 (from cov diag); 4σ = 0.04
    # so bounds should be roughly (0.39, 0.47)
    lo, hi = by_name["theta_s"].bounds
    assert lo > 0.34 and hi < 0.5


def test_vg_to_prior_works_without_covariance():
    """Without covariance, falls back to ±50% factor bounds."""
    ptf = PTFResult(theta_r=0.07, theta_s=0.43, alpha=0.036, n=1.56, Ks=24.96,
                    method="carsel_parrish")
    specs = vg_to_prior(ptf, material_index=0)
    by_name = {s.name: s for s in specs}
    assert by_name["alpha"].bounds[0] < 0.036 < by_name["alpha"].bounds[1]
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest tests/research/ptf/test_uncertainty.py -v
```

Expected: ImportError.

- [ ] **Step 3: Implement**

Write `hydrus_research/ptf/uncertainty.py`:

```python
"""Convert a PTFResult into M0 ParameterSpec priors for use as a starting
point in F3 inversion / F5 UQ."""
from __future__ import annotations
import math
import numpy as np

from .result import PTFResult
from ..parameters import ParameterSpec


def vg_to_prior(ptf: PTFResult, material_index: int = 0) -> list[ParameterSpec]:
    """Build 5 ParameterSpec entries (theta_r/theta_s/alpha/n/Ks) from a PTF.

    `alpha` and `Ks` are log-transformed (always positive); `theta_r`, `theta_s`
    and `n` are linear. Bounds span ±4σ when covariance is available; otherwise
    fall back to ±50% multiplicative for log params and ±20% additive for theta_r
    and theta_s, ±0.5 for n."""
    # Diagonal stddevs from covariance, or None
    if ptf.covariance is not None:
        diag = [math.sqrt(max(ptf.covariance[i][i], 0.0)) for i in range(5)]
    else:
        diag = [None] * 5

    means = (ptf.theta_r, ptf.theta_s, ptf.alpha, ptf.n, ptf.Ks)
    names = ("theta_r", "theta_s", "alpha", "n", "Ks")
    transforms = ("linear", "linear", "log", "linear", "log")
    fallback_bounds = (
        lambda m: (max(m - 0.04, 0.0), m + 0.04),                   # theta_r
        lambda m: (max(m - 0.04, 0.0), min(m + 0.04, 1.0)),         # theta_s
        lambda m: (m * 0.5, m * 2.0),                                # alpha
        lambda m: (max(m - 0.3, 1.05), m + 0.5),                    # n
        lambda m: (m * 0.2, m * 5.0),                                # Ks
    )

    specs: list[ParameterSpec] = []
    for i, (name, mean, trans) in enumerate(zip(names, means, transforms)):
        sigma = diag[i]
        if sigma is not None and sigma > 0:
            if trans == "log":
                # 4-sigma window in user units (multiplicative-ish)
                lo = max(mean - 4 * sigma, mean * 0.05)
                hi = mean + 4 * sigma
            else:
                lo, hi = mean - 4 * sigma, mean + 4 * sigma
        else:
            lo, hi = fallback_bounds[i](mean)

        if trans == "log":
            lo = max(lo, mean * 0.01)              # never let lo touch zero
        if name == "theta_r":
            lo = max(lo, 0.0)
        if name == "theta_s":
            hi = min(hi, 1.0)
        if name == "n":
            lo = max(lo, 1.05)

        specs.append(ParameterSpec(
            name=name,
            target=f"materials[{material_index}].{name}",
            bounds=(float(lo), float(hi)),
            transform=trans,
            prior_mean=float(mean),
            prior_std=float(sigma) if sigma is not None else None,
            group=f"mat{material_index}_vg",
        ))
    return specs
```

- [ ] **Step 4: Run + commit**

```bash
pytest tests/research/ptf/test_uncertainty.py -v
git add hydrus_research/ptf/uncertainty.py tests/research/ptf/test_uncertainty.py
git commit -m "M2.7: vg_to_prior — PTFResult → 5 ParameterSpec priors (with covariance window)"
```

---

### Task 8: Public API — `texture_to_vg(method="rosetta3_auto")`

**Files:**
- Create: `hydrus_research/ptf/api.py`
- Create: `tests/research/ptf/test_api.py`

- [ ] **Step 1: Write the failing test**

Write `tests/research/ptf/test_api.py`:

```python
import pytest
from hydrus_research.ptf import texture_to_vg


def test_auto_picks_h2_when_bd_given():
    """rosetta3_auto = pick the highest-hierarchy ROSETTA model the inputs support."""
    pytest.importorskip("rosetta")
    r = texture_to_vg(sand_pct=45, silt_pct=35, clay_pct=20,
                      bulk_density_g_cm3=1.4)
    assert r.method == "rosetta3_h2"


def test_carsel_method_explicit():
    """Caller can force Carsel-Parrish by passing a USDA class instead of texture %."""
    from hydrus_research.ptf.api import texture_to_vg as f
    # Carsel mode is reached via usda_class_to_vg, not texture_to_vg.
    # texture_to_vg with method='carsel_parrish' should fall through to the
    # nearest USDA class center.
    r = f(sand_pct=40, silt_pct=40, clay_pct=20, method="carsel_parrish")
    assert r.method == "carsel_parrish"


def test_wosten_method_explicit():
    r = texture_to_vg(sand_pct=45, silt_pct=35, clay_pct=20,
                      bulk_density_g_cm3=1.4, organic_matter_pct=1.5,
                      method="wosten")
    assert r.method == "wosten"


def test_invalid_method_raises():
    with pytest.raises(ValueError):
        texture_to_vg(sand_pct=45, silt_pct=35, clay_pct=20, method="other_thing")
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest tests/research/ptf/test_api.py -v
```

Expected: ImportError on `texture_to_vg` (api.py).

- [ ] **Step 3: Implement**

Write `hydrus_research/ptf/api.py`:

```python
"""Public texture_to_vg() entry point — dispatches to ROSETTA-3, Carsel-Parrish,
or Wösten HYPRES based on `method`. `method='rosetta3_auto'` picks the deepest
ROSETTA hierarchy supported by the provided inputs."""
from __future__ import annotations
from typing import Literal

from .result import PTFResult


PTFMethodArg = Literal["rosetta3_auto", "carsel_parrish", "wosten",
                       "rosetta3_h1", "rosetta3_h2", "rosetta3_h3", "rosetta3_h4"]


def texture_to_vg(sand_pct: float, silt_pct: float, clay_pct: float,
                  bulk_density_g_cm3: float | None = None,
                  theta_33: float | None = None,
                  theta_1500: float | None = None,
                  organic_matter_pct: float | None = None,
                  organic_carbon_pct: float | None = None,
                  topsoil: bool = True,
                  method: PTFMethodArg = "rosetta3_auto") -> PTFResult:
    """Pedotransfer dispatcher. See package docstring for backend semantics."""
    if method.startswith("rosetta3"):
        from .rosetta import rosetta3_predict
        # `rosetta3_auto` lets rosetta3_predict pick model by which inputs are non-None
        return rosetta3_predict(
            sand_pct=sand_pct, silt_pct=silt_pct, clay_pct=clay_pct,
            bulk_density_g_cm3=bulk_density_g_cm3,
            theta_33=theta_33, theta_1500=theta_1500,
        )
    if method == "carsel_parrish":
        from .carsel_parrish import carsel_parrish_lookup
        from .presets import USDA_TEXTURE_CENTERS
        # Snap to the nearest USDA class center
        nearest, best_d2 = "loam", float("inf")
        for cname, c in USDA_TEXTURE_CENTERS.items():
            d2 = ((c["sand_pct"] - sand_pct) ** 2
                  + (c["silt_pct"] - silt_pct) ** 2
                  + (c["clay_pct"] - clay_pct) ** 2)
            if d2 < best_d2:
                nearest, best_d2 = cname, d2
        return carsel_parrish_lookup(nearest)
    if method == "wosten":
        from .wosten_hypres import wosten_predict
        # Wösten needs OM; OC is often the available measurement (OM ≈ 1.724 * OC)
        om = organic_matter_pct
        if om is None:
            if organic_carbon_pct is not None:
                om = 1.724 * organic_carbon_pct
            else:
                raise ValueError("wosten requires organic_matter_pct or organic_carbon_pct")
        if bulk_density_g_cm3 is None:
            raise ValueError("wosten requires bulk_density_g_cm3")
        return wosten_predict(sand_pct=sand_pct, silt_pct=silt_pct, clay_pct=clay_pct,
                              bulk_density_g_cm3=bulk_density_g_cm3,
                              organic_matter_pct=om, topsoil=topsoil)
    raise ValueError(f"unknown method {method!r}")
```

- [ ] **Step 4: Run + commit**

```bash
pytest tests/research/ptf/test_api.py -v
git add hydrus_research/ptf/api.py tests/research/ptf/test_api.py
git commit -m "M2.8: texture_to_vg public API with rosetta3_auto / carsel_parrish / wosten"
```

---

### Task 9: REST router `/research/ptf/*`

**Files:**
- Create: `hydrus_port_server/routers/research_ptf.py`
- Modify: `hydrus_port_server/app.py` — register router in `build_app()` (the factory introduced in M1)
- Create: `tests/research/ptf/test_rest.py`

- [ ] **Step 1: Write the failing test**

Write `tests/research/ptf/test_rest.py`:

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


def test_ptf_usda_classes(client):
    r = client.get("/research/ptf/usda-classes")
    assert r.status_code == 200
    body = r.json()
    assert "loam" in body and "clay" in body and "sand" in body
    assert len(body) == 12
    assert "sand_pct" in body["loam"]


def test_ptf_predict_carsel_parrish(client):
    payload = {"sand_pct": 40, "silt_pct": 40, "clay_pct": 20,
               "method": "carsel_parrish"}
    r = client.post("/research/ptf/predict", json=payload)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["method"] == "carsel_parrish"
    assert 0.02 < body["alpha"] < 0.05    # loam range


def test_ptf_predict_invalid_texture(client):
    payload = {"sand_pct": 50, "silt_pct": 50, "clay_pct": 50,
               "method": "carsel_parrish"}   # sums to 150
    r = client.post("/research/ptf/predict", json=payload)
    assert r.status_code in (400, 422)
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest tests/research/ptf/test_rest.py -v
```

Expected: ImportError or 404.

- [ ] **Step 3: Implement**

Write `hydrus_port_server/routers/research_ptf.py`:

```python
"""/research/ptf/* — pedotransfer function REST endpoints."""
from __future__ import annotations
from typing import Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from hydrus_research.ptf import texture_to_vg
from hydrus_research.ptf.presets import USDA_TEXTURE_CENTERS


router = APIRouter()


class PredictRequest(BaseModel):
    sand_pct: float
    silt_pct: float
    clay_pct: float
    bulk_density_g_cm3: float | None = None
    theta_33: float | None = None
    theta_1500: float | None = None
    organic_matter_pct: float | None = None
    organic_carbon_pct: float | None = None
    topsoil: bool = True
    method: Literal["rosetta3_auto", "carsel_parrish", "wosten",
                    "rosetta3_h1", "rosetta3_h2", "rosetta3_h3", "rosetta3_h4"
                    ] = "rosetta3_auto"


@router.post("/predict")
def predict(req: PredictRequest):
    try:
        r = texture_to_vg(
            sand_pct=req.sand_pct, silt_pct=req.silt_pct, clay_pct=req.clay_pct,
            bulk_density_g_cm3=req.bulk_density_g_cm3,
            theta_33=req.theta_33, theta_1500=req.theta_1500,
            organic_matter_pct=req.organic_matter_pct,
            organic_carbon_pct=req.organic_carbon_pct,
            topsoil=req.topsoil, method=req.method,
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    return r.model_dump()


@router.get("/usda-classes")
def usda_classes():
    return USDA_TEXTURE_CENTERS
```

In `hydrus_port_server/app.py` `build_app()`, add (next to the M1 dndc include_router):

```python
try:
    from .routers.research_ptf import router as ptf_router
    app.include_router(ptf_router, prefix="/research/ptf", tags=["research", "ptf"])
except ImportError:
    pass
```

- [ ] **Step 4: Run + commit**

```bash
pytest tests/research/ptf/test_rest.py -v
git add hydrus_port_server/routers/research_ptf.py hydrus_port_server/app.py tests/research/ptf/test_rest.py
git commit -m "M2.9: /research/ptf/{predict,usda-classes} REST routes"
```

---

### Task 10: CLI `hydrus research soil ptf`

**Files:**
- Modify: `hydrus_port/cli.py` — append a `soil ptf` subcommand under the `research` group introduced in M1
- Create: `tests/research/ptf/test_cli.py`

- [ ] **Step 1: Write the failing test**

Write `tests/research/ptf/test_cli.py`:

```python
import subprocess


def test_cli_soil_ptf_carsel():
    r = subprocess.run(
        ["hydrus", "research", "soil", "ptf",
         "--texture", "sand=40,silt=40,clay=20",
         "--method", "carsel_parrish"],
        capture_output=True, text=True,
    )
    assert r.returncode == 0, r.stderr
    out = r.stdout.lower()
    for tag in ("theta_r", "theta_s", "alpha", "n", "ks"):
        assert tag in out
    assert "carsel_parrish" in out
```

- [ ] **Step 2: Implement the subcommand in `cli.py`**

In `hydrus_port/cli.py`, inside the `_build_research_subparser(sub)` function (added in M1), append:

```python
    p_soil = rsub.add_parser("soil", help="soil-library / PTF")
    ssub = p_soil.add_subparsers(dest="soil_cmd", required=True)

    p_ptf = ssub.add_parser("ptf", help="texture → van Genuchten parameters")
    p_ptf.add_argument("--texture", required=True,
                       help="comma-separated sand=N,silt=M,clay=K (percent)")
    p_ptf.add_argument("--bd", type=float, default=None,
                       help="bulk density g/cm³ (optional; enables ROSETTA H2)")
    p_ptf.add_argument("--theta-33", type=float, default=None)
    p_ptf.add_argument("--theta-1500", type=float, default=None)
    p_ptf.add_argument("--om", type=float, default=None, help="organic matter %")
    p_ptf.add_argument("--method", default="rosetta3_auto",
                       choices=["rosetta3_auto", "carsel_parrish", "wosten",
                                "rosetta3_h1", "rosetta3_h2", "rosetta3_h3", "rosetta3_h4"])
    p_ptf.set_defaults(_cmd=_cmd_soil_ptf)


def _cmd_soil_ptf(args):
    from hydrus_research.ptf import texture_to_vg
    parts = dict(p.split("=") for p in args.texture.split(","))
    r = texture_to_vg(
        sand_pct=float(parts["sand"]), silt_pct=float(parts["silt"]),
        clay_pct=float(parts["clay"]),
        bulk_density_g_cm3=args.bd, theta_33=args.theta_33,
        theta_1500=args.theta_1500, organic_matter_pct=args.om,
        method=args.method,
    )
    print(f"{'theta_r':12s} {'theta_s':12s} {'alpha':12s} {'n':12s} {'Ks':12s} method")
    print(f"{r.theta_r:<12.4f} {r.theta_s:<12.4f} {r.alpha:<12.4f} "
          f"{r.n:<12.4f} {r.Ks:<12.4f} {r.method}")
    return 0
```

- [ ] **Step 3: Run + commit**

```bash
pytest tests/research/ptf/test_cli.py -v
git add hydrus_port/cli.py tests/research/ptf/test_cli.py
git commit -m "M2.10: hydrus research soil ptf CLI"
```

---

### Task 11: GUI — TextureTriangle.vue reusable component

**Files:**
- Create: `desktop/src/components/TextureTriangle.vue`

This is a reusable ternary (triangle) plot built on `plotly.js-dist-min` (already a dep). Click emits `(sand_pct, silt_pct, clay_pct)`. Visual: USDA classification polygons in the background.

- [ ] **Step 1: Implement**

Write `desktop/src/components/TextureTriangle.vue`:

```vue
<template>
  <div ref="plotEl" class="texture-triangle"></div>
</template>

<script setup lang="ts">
import { onMounted, ref, watch } from "vue";
import Plotly from "plotly.js-dist-min";

const props = defineProps<{ sand?: number; silt?: number; clay?: number }>();
const emit = defineEmits<{ (e: "update", v: { sand_pct: number; silt_pct: number; clay_pct: number }): void }>();
const plotEl = ref<HTMLDivElement | null>(null);

function _draw() {
  if (!plotEl.value) return;
  const marker = (props.sand !== undefined && props.silt !== undefined && props.clay !== undefined)
    ? [{ type: "scatterternary", mode: "markers",
         a: [props.clay], b: [props.sand], c: [props.silt],
         marker: { size: 16, color: "#c00" } }]
    : [];
  Plotly.newPlot(plotEl.value, marker, {
    ternary: {
      sum: 100,
      aaxis: { title: "clay %", ticksuffix: "%" },
      baxis: { title: "sand %", ticksuffix: "%" },
      caxis: { title: "silt %", ticksuffix: "%" },
    },
    margin: { t: 20, l: 60, r: 60, b: 60 }, showlegend: false,
  }, { responsive: true, displayModeBar: false });
  plotEl.value.on("plotly_click", (ev: any) => {
    const pt = ev.points[0];
    emit("update", {
      clay_pct: pt.a, sand_pct: pt.b, silt_pct: pt.c,
    });
  });
}

onMounted(_draw);
watch(() => [props.sand, props.silt, props.clay], _draw);
</script>

<style scoped>
.texture-triangle { width: 100%; height: 480px; }
</style>
```

- [ ] **Step 2: Commit**

```bash
git add desktop/src/components/TextureTriangle.vue
git commit -m "M2.11: TextureTriangle.vue reusable ternary picker (Plotly)"
```

---

### Task 12: GUI — SoilLibrary.vue page + REST wrapper + nav entry

**Files:**
- Modify: `desktop/src/api.ts` — add `ptf.*` wrappers
- Create: `desktop/src/pages/research/SoilLibrary.vue`
- Modify: `desktop/src/App.vue` — add nav entry under Research menu

- [ ] **Step 1: Append `ptf` wrappers to api.ts**

In `desktop/src/api.ts`, append after the `dndc` block (introduced in M1):

```ts
export interface PTFRequest {
  sand_pct: number;
  silt_pct: number;
  clay_pct: number;
  bulk_density_g_cm3?: number;
  theta_33?: number;
  theta_1500?: number;
  organic_matter_pct?: number;
  organic_carbon_pct?: number;
  topsoil?: boolean;
  method?: "rosetta3_auto" | "carsel_parrish" | "wosten"
         | "rosetta3_h1" | "rosetta3_h2" | "rosetta3_h3" | "rosetta3_h4";
}

export interface PTFResult {
  theta_r: number; theta_s: number; alpha: number; n: number; Ks: number;
  L: number; method: string; covariance: number[][] | null;
}

const PTF_BASE = (import.meta.env.VITE_DNDC_BASE as string)
                 ?? "http://127.0.0.1:8765";

export const ptf = {
  async predict(req: PTFRequest): Promise<PTFResult> {
    const r = await fetch(`${PTF_BASE}/research/ptf/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(req),
    });
    if (!r.ok) throw new Error(`HTTP ${r.status}: ${await r.text()}`);
    return r.json();
  },
  async usdaClasses(): Promise<Record<string, { sand_pct: number; silt_pct: number; clay_pct: number }>> {
    const r = await fetch(`${PTF_BASE}/research/ptf/usda-classes`);
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    return r.json();
  },
};
```

- [ ] **Step 2: Implement SoilLibrary.vue**

Write `desktop/src/pages/research/SoilLibrary.vue`:

```vue
<template>
  <div class="soil-library">
    <h2>Soil Library — F1 Pedotransfer</h2>
    <div class="row">
      <div class="col">
        <h3>Texture triangle</h3>
        <TextureTriangle :sand="sand" :silt="silt" :clay="clay" @update="onPick" />
        <label>USDA class shortcut:
          <select v-model="selectedClass" @change="onClassPick">
            <option value="">— pick a class —</option>
            <option v-for="(c, name) in usda" :key="name" :value="name">{{ name }}</option>
          </select>
        </label>
      </div>
      <div class="col">
        <h3>Inputs</h3>
        <label>sand% <input type="number" v-model.number="sand" min="0" max="100" /></label>
        <label>silt% <input type="number" v-model.number="silt" min="0" max="100" /></label>
        <label>clay% <input type="number" v-model.number="clay" min="0" max="100" /></label>
        <label>BD g/cm³ <input type="number" v-model.number="bd" step="0.01" /></label>
        <label>OM % <input type="number" v-model.number="om" step="0.1" /></label>
        <label>Method:
          <select v-model="method">
            <option value="rosetta3_auto">ROSETTA-3 (auto hierarchy)</option>
            <option value="carsel_parrish">Carsel-Parrish 1988</option>
            <option value="wosten">Wösten HYPRES 1999</option>
          </select>
        </label>
        <button @click="predict">Compute VG params</button>
        <pre v-if="result" class="result">{{ resultText }}</pre>
        <p v-if="error" class="err">{{ error }}</p>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted } from "vue";
import TextureTriangle from "../../components/TextureTriangle.vue";
import { ptf, type PTFResult } from "../../api";

const sand = ref(40), silt = ref(40), clay = ref(20);
const bd = ref<number | null>(1.4), om = ref<number | null>(1.5);
const method = ref<"rosetta3_auto" | "carsel_parrish" | "wosten">("rosetta3_auto");
const usda = ref<Record<string, { sand_pct: number; silt_pct: number; clay_pct: number }>>({});
const selectedClass = ref("");
const result = ref<PTFResult | null>(null);
const error = ref<string | null>(null);

onMounted(async () => { usda.value = await ptf.usdaClasses(); });

function onPick(v: { sand_pct: number; silt_pct: number; clay_pct: number }) {
  sand.value = Math.round(v.sand_pct); silt.value = Math.round(v.silt_pct);
  clay.value = Math.round(v.clay_pct);
}
function onClassPick() {
  if (selectedClass.value && usda.value[selectedClass.value]) {
    const c = usda.value[selectedClass.value];
    sand.value = c.sand_pct; silt.value = c.silt_pct; clay.value = c.clay_pct;
  }
}
async function predict() {
  error.value = null; result.value = null;
  try {
    result.value = await ptf.predict({
      sand_pct: sand.value, silt_pct: silt.value, clay_pct: clay.value,
      bulk_density_g_cm3: bd.value ?? undefined,
      organic_matter_pct: om.value ?? undefined,
      method: method.value,
    });
  } catch (e: any) { error.value = e.message ?? String(e); }
}
const resultText = computed(() => result.value
  ? `theta_r = ${result.value.theta_r.toFixed(4)}
theta_s = ${result.value.theta_s.toFixed(4)}
alpha   = ${result.value.alpha.toFixed(4)}  1/cm
n       = ${result.value.n.toFixed(3)}
Ks      = ${result.value.Ks.toFixed(2)}  cm/day
L       = ${result.value.L.toFixed(2)}
method  = ${result.value.method}`
  : "");
</script>

<style scoped>
.soil-library { padding: 16px; max-width: 1200px; }
.row { display: flex; gap: 16px; }
.col { flex: 1; }
label { display: block; margin: 6px 0; }
input, select { margin-left: 6px; }
.result { background: #f4f4f4; padding: 12px; }
.err { color: #c00; }
</style>
```

- [ ] **Step 3: Add nav entry to App.vue**

In `desktop/src/App.vue`, locate the existing nav/route structure (added during M1's DNDC page wiring). Add a sibling entry "Soil Library" pointing to `SoilLibrary.vue`.

- [ ] **Step 4: Smoke + commit**

```bash
cd desktop && npm run dev &
sleep 4
curl -sf http://localhost:1420 | head -1
# Open in browser, navigate to Research → Soil Library, click on the triangle,
# verify VG params populate after "Compute" with the FastAPI server running.

git add desktop/src/api.ts desktop/src/pages/research/SoilLibrary.vue desktop/src/App.vue
git commit -m "M2.12: SoilLibrary.vue page + ptf REST wrapper + nav entry"
```

---

### Task 13: End-to-end + regression + M2-complete marker

**Files:** none (verification only)

- [ ] **Step 1: Full research suite**

```bash
pytest tests/research/ -v 2>&1 | tail -10
```

Expected: all tests pass (M0 + M1 if merged + M2).

- [ ] **Step 2: No regression in existing CLI smokes**

```bash
hydrus test 1d 2>&1 | tail -3
hydrus test roundtrip 2>&1 | tail -3
hydrus research soil ptf --texture sand=40,silt=40,clay=20 --method carsel_parrish 2>&1 | tail -3
```

Expected: all PASS + a tidy 5-column PTF table.

- [ ] **Step 3: Empty marker commit**

```bash
git commit --allow-empty -m "M2 complete: PTF (ROSETTA-3 + Carsel-Parrish + Wösten) + SoilLibrary GUI"
git log --oneline | head -20
```

---

## Definition of Done for M2

1. `pytest tests/research/ptf/ -v` — all green.
2. `pytest tests/research/ -v` — no regression in M0 (and M1 if merged).
3. `python -c "from hydrus_research.ptf import texture_to_vg, usda_class_to_vg, vg_to_prior; print('OK')"` prints `OK`.
4. `hydrus research soil ptf --texture sand=40,silt=40,clay=20 --method carsel_parrish` prints a tidy table.
5. `hydrus-port-serve` running → `POST /research/ptf/predict` works; `GET /research/ptf/usda-classes` returns 12 classes.
6. Tauri dev → `Research → Soil Library` renders the texture triangle; clicking updates inputs; "Compute" populates VG params.
7. `hydrus test 1d/2d/3d/roundtrip` still PASS.
