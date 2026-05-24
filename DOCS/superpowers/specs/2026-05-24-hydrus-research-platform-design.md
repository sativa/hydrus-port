# HYDRUS Research Platform — Design Spec

**Status:** Approved 2026-05-24
**Authors:** zhangfeng (decisions), Claude (drafting)
**Scope:** Redesign of `H1D_Src/` from a HYDRUS port + visualization GUI into a
research platform for water-and-solute movement under different soil types,
with a clean seam for future tight coupling to the user's Python DNDC model.

---

## 0. Context and decisions

### 0.1 Current state (locked, do not duplicate)

- `hydrus1d/` — 1:1 Python port of HYDRUS-1D 4.08 / 7.0 Fortran (Richards +
  heat + solute). 15 of 16 fixtures bit-equal to Fortran.
- `swms2d/` — 1:1 Python port of SWMS_2D v1.22. EX.1 bit-equal; EX.2/3/4
  within Fortran's own mass-balance error budget.
- `swms2d/richards3d.py` — 3D Richards solver on scikit-fem (MeshTet / MeshHex,
  lumped + consistent mass options).
- `hydrus_port/` — unified `hydrus` CLI with canonical scenario JSON schema
  (commit `e84271a`) and SELECTOR.IN round-trip pipeline.
- `hydrus_port_server/` — FastAPI sidecar for the GUI.
- `desktop/` — Tauri 2 + Vue 3 GUI with parameter editor, 1D/2D/3D viz,
  CSV export, auto-polling output files.

The solver core is research-grade and verified. The redesign sits **on top**
of it. Solver files are unchanged.

### 0.2 Approved scope

| Dimension | Decision |
|---|---|
| Overall positioning | **A** (research platform) primary + **D** (GUI enhancement) secondary |
| DNDC ↔ HYDRUS coupling (long term) | **B2** — in-process Python tight coupling |
| Current phase | Manual GUI forms for DNDC inputs; data contract is the future B2 API |
| DNDC → HYDRUS data contract | 11 items (see §3.1) |
| #7 N transformation | Pre-wired "external source term injection" interface |
| #9 State exchange | Bidirectional (HYDRUS final_state → DNDC restart) |
| Research features P0 | F1 PTF + F2 sensitivity + F3 inversion (LM + PyEMU) + F4 batch + F5 UQ + Surrogate (sklearn GP + PCK) |
| Research features P1 | F3-Bayesian (PyMC / emcee) |
| Research features P2 | F6 decision optimization |
| F3 inversion backends (P0) | PEST/PyEMU (main) + scipy LM (fast) |
| F3 inversion backends (P1) | PyMC / emcee |
| Dimension coverage | **D3** — F2/F3/F5 cover 1D + 2D + 3D; 3D uses GP/PCK surrogate |
| Architecture | **Layered Plugin** — sibling `hydrus_research/` package, abstraction-layer-centric |
| Optional extras | `research`, `research-uq`, `research-3d`, `research-opt` |
| F4 backends | joblib + pyemu_tcp only (dask dropped) |
| F2 SALib methods | All four — Morris / Sobol / FAST / PAWN |
| F3 PyEMU default | IES (not GLM) |
| Surrogate libs | sklearn GP (default) + PCK (best). SMT / chaospy as transitive only |
| F6 optimization | P2, out of P0 |

### 0.3 Tools referenced from online research

| Tool | Package | Role |
|---|---|---|
| ROSETTA-3 PTF | `rosetta-soil` (USDA-ARS official) | F1 — neural net PTF, 4 hierarchical models, predicts 7 VG params from sand/silt/clay/BD/θ |
| PEST / PEST++ / PyEMU | `pyemu` v1.3.7+ (pypest/pyemu) | F3 main, F2, F5 — model-independent uncertainty/calibration. Provides PESTPP-IES (Iterative Ensemble Smoother), TCP workers |
| SALib | `SALib` v1.5.3+ | F2 — ProblemSpec chainable API, Sobol/Morris/FAST/PAWN/HDMR |
| scipy.optimize | `scipy.optimize.least_squares` (trust-region LM variant) | F3 fast path |
| PCSE / WOFOST | `pcse` (Wageningen) | Reference for "agronomic driver + water engine" pattern only — not imported |
| Surrogates | `scikit-learn` GP, `smt`, `chaospy` | F2/F3/F5 3D path. PCK (PC + Kriging) is hydrology SOTA per Schöbi et al. |
| Multi-objective opt | `pymoo` v0.6+ | F6 (P2) |

---

## 1. Top-level architecture and package layout

### 1.1 File tree (★ = new; ● = modified)

```
H1D_Src/
├── hydrus1d/                    # unchanged
├── swms2d/                      # unchanged (incl. richards3d.py)
├── hydrus_port/                 # ● add `hydrus research <subcmd>` CLI group
├── hydrus_research/             # ★ new package — research platform
│   ├── simulator/               # Simulator ABC + 3 adapters (1d/2d/3d)
│   ├── parameters/              # ParameterSpec, ParameterMap, transforms, priors
│   ├── observations/            # ObservationSpec, loaders (CSV, NetCDF, HYDRUS .OUT)
│   ├── ptf/                     # rosetta-soil wrap + Carsel-Parrish + Wösten HYPRES
│   ├── batch/                   # joblib runner + PyEMU TCP worker mode
│   ├── sensitivity/             # SALib wrappers (Morris/Sobol/FAST/PAWN)
│   ├── inversion/               # lm_scipy / pyemu_pestpp / pymc_bayes
│   ├── uq/                      # Monte Carlo + posterior propagation + GLUE
│   ├── surrogate/               # sklearn GP + PCK (SMT/chaospy as internal deps)
│   ├── optimization/            # pymoo + Optuna (P2; stubbed in P0)
│   └── dndc_seam/               # Pydantic 11-item contract + GUI manual / B2 live adapters
├── hydrus_port_server/          # ● new /research/* REST routers
│   └── routers/research/        #   soil_library, batch, sensitivity, inversion, uq, surrogate, dndc_seam
├── desktop/                     # ● GUI enhancement (D route)
│   └── src/pages/research/      #   SoilLibrary / BatchSweep / Sensitivity / InversionStudio / UQ / Surrogate / DNDCForms
├── tests/research/              # ★ new test tree
├── docs/research/               # ★ tutorials 01–08
└── pyproject.toml               # ● new optional extras
```

### 1.1.1 GUI ↔ engine decoupling (load-bearing discipline)

The GUI (`desktop/`) is **strictly optional**. The simulation engine
(`hydrus1d/`, `swms2d/`, `hydrus_research/`, `hydrus_port/` CLI) must be
fully usable without it — in headless servers, CI, batch jobs, notebooks,
and (most importantly) inside the future B2 DNDC live-coupling loop where
no GUI exists.

Concretely:
- **Allowed dependency direction:** GUI → REST (`hydrus_port_server`) → engine
  (`hydrus_research`, `hydrus_port`, solvers). Engine never imports
  anything from the GUI or REST layer.
- **Every research workflow ships with a CLI subcommand AND a Python API.**
  The GUI page is a third, optional consumer. If a feature is only
  reachable from the GUI, that's a design smell — refactor.
- **Optional extras enforce this at install time:** `pip install -e
  '.[research]'` (no `[gui]`) must yield a fully functional engine. The
  GUI's REST server lives in the `[gui]` extras group.
- **Tests target the CLI / Python API first.** GUI / REST tests are wrappers.

This discipline is what makes the future B2 coupling (DNDC drives the
engine in-process, no GUI in the loop) tractable.

### 1.2 Dependency graph (strict; arrows go inward only)

```
                     hydrus1d   swms2d
                         │         │
                         └────┬────┘
                              │
                  hydrus_research.simulator
                              │
            ┌─────────────────┼────────────────┐
            │                 │                │
       parameters       observations         (PTF is independent — no deps)
            │                 │
            └─────────┬───────┘
                      │
   ┌──────────┬───────┼────────┬──────────┬───────────┐
 batch    sensitivity  inversion    uq       surrogate
                                                  │
                                            optimization

dndc_seam ── independent Pydantic module; consumed by server / GUI / future B2
```

**Discipline:** research modules MUST NOT import each other. Solvers MUST NOT
import anything from `hydrus_research`. Violations break testability.

### 1.3 Optional extras

```toml
[project.optional-dependencies]
research      = ["pyemu>=1.3", "SALib>=1.5", "rosetta-soil",
                 "scikit-learn", "joblib", "pydantic>=2"]
research-uq   = ["pymc>=5", "arviz"]                  # P1 Bayesian
research-3d   = ["smt>=2", "chaospy"]                 # D3 surrogate (PCK internals; sklearn GP lives in base [research])
research-opt  = ["pymoo>=0.6", "optuna"]              # P2
```

Install paths:
- `pip install -e '.[research]'` → P0 (F1+F2+F3-LM+F3-PyEMU+F4)
- `pip install -e '.[research,research-3d]'` → adds D3 3D surrogate
- `pip install -e '.[research,research-uq,research-3d,research-opt]'` → everything

### 1.4 CLI surface

```
hydrus research soil ptf --texture sand=45,silt=35,clay=20 --bd 1.4
hydrus research sweep <scenario.json> --param ksat:0.1:10:log --n 32 --workers 8
hydrus research sensitize <scenario.json> --method morris --params alpha,n,ksat --n 100
hydrus research invert <scenario.json> --obs obs.csv --backend pyemu_ies --n_real 200
hydrus research surrogate train <results.parquet> --type pck
hydrus research worker --master <host>:<port>          # PEST++ TCP worker
```

CLI is a thin wrapper: parses args → calls `hydrus_research` API → writes
results to disk. No business logic in CLI.

---

## 2. Core abstractions (the system spine)

All research modules consume the same triad: `Simulator + ParameterMap + ObservationSet`.

### 2.1 Simulator ABC

```python
# hydrus_research/simulator/base.py
@dataclass(frozen=True)
class Forcing:
    """Time-varying drivers; populated by dndc_seam.to_forcing()."""
    times_days: np.ndarray
    precip_cm_per_day: np.ndarray
    pet_cm_per_day: np.ndarray
    lai: np.ndarray
    root_depth_cm: np.ndarray
    root_density_fn: Callable[[np.ndarray, float], np.ndarray]    # (z, t) → β
    irrigation_events: list[Event]
    fert_events: list[Event]
    n_source_terms: Callable[..., tuple[float, float]]            # B2 hook
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
    times: np.ndarray
    z: np.ndarray                          # 1D depth; 2D/3D use mesh field
    theta: np.ndarray
    h: np.ndarray
    c: np.ndarray | None
    fluxes: dict[str, np.ndarray]
    mass_balance: dict[str, float]
    final_state: InitialState              # for #9 DNDC writeback
    meta: dict                             # solver name, dt, wall_s, convergence

class Simulator(ABC):
    name: str
    dimension: int                         # 1, 2, 3, or -1 for surrogate

    @abstractmethod
    def run(self, params: ParameterMap, forcing: Forcing,
            ic: InitialState) -> SimResult: ...

    @abstractmethod
    def observable_at(self, result: SimResult,
                      spec: ObservationSpec) -> float: ...

    def batch_observables(self, result, specs):
        return np.array([self.observable_at(result, s) for s in specs])
```

### 2.2 ParameterSpec / ParameterMap

```python
class ParameterSpec(BaseModel):
    name: str
    target: str           # path into canonical scenario JSON, e.g. "material[0].alpha"
    bounds: tuple[float, float]
    transform: Literal["linear", "log", "logit"] = "linear"
    prior_mean: float | None = None
    prior_std: float | None = None
    group: str = "default"

class ParameterMap:
    def __init__(self, specs: list[ParameterSpec]): ...
    def to_vector(self, named: dict) -> np.ndarray: ...
    def from_vector(self, theta: np.ndarray) -> dict: ...
    def bounds_array(self) -> np.ndarray: ...
    def apply_to_scenario(self, scenario_json: dict, named: dict) -> dict: ...
```

`target` strings reuse the canonical-JSON scenario schema (commit `e84271a`).
Parameter layer never touches solver internals.

### 2.3 ObservationSpec / ObservationSet

```python
class ObservationSpec(BaseModel):
    name: str
    kind: Literal["theta", "h", "c", "flux", "cumulative_flux",
                  "concentration_flux"]
    location: dict                # 1D: {"z_cm": 20}; 2D/3D: {"node": N} or {"xyz": [..]}
    time_day: float
    weight: float = 1.0
    species: str | None = None

class ObservationSet:
    def __init__(self, specs, values, sigmas): ...

    @classmethod
    def from_csv(cls, path, schema): ...
    @classmethod
    def from_hydrus_obsnod(cls, obs_out_path): ...

    def residuals(self, sim: np.ndarray) -> np.ndarray:
        """(sim - obs) / sigma"""

    def objective_l2(self, sim: np.ndarray) -> float:
        """sum((sim - obs)**2 / sigma**2)"""
```

### 2.4 The narrow waist: unified forward callable

```python
def make_forward(simulator, param_map, forcing, ic, obs_specs):
    def forward(theta: np.ndarray) -> np.ndarray:
        named = param_map.from_vector(theta)
        sim_result = simulator.run(named, forcing, ic)
        return simulator.batch_observables(sim_result, obs_specs)
    return forward
```

Every research tool consumes only `forward`. Surrogates implement `Simulator`
so they are drop-in. This is the entire architectural commitment.

---

## 3. DNDC seam — data contract / GUI forms / B2 hook

### 3.1 Pydantic schema (11 items)

```python
# hydrus_research/dndc_seam/schema.py

class AtmDaily(BaseModel):                # #1 atmospheric forcing
    dates: list[date]
    precip_cm: list[float]
    pet_cm: list[float] | None = None     # if None, server computes FAO-56
    t_min_c / t_max_c / rh_pct / wind_m_s / solar_mj_m2: list[float] | None

class EtPartition(BaseModel):             # #2 E/T split
    mode: Literal["lai_beer", "explicit_split", "kc_dual"]
    lai: list[float] | None = None
    extinction_k: float = 0.6
    explicit_e_frac: list[float] | None = None
    kcb: list[float] | None = None

class RootGrowth(BaseModel):              # #3 root dynamics
    z_max_cm: float
    growth_curve: Literal["linear", "logistic", "table"]
    days_to_zmax: float | None = None
    table: list[tuple[date, float]] | None = None
    density_profile: Literal["uniform", "linear_decline", "exponential", "raats"]
    density_param: float | None = None

class FeddesParams(BaseModel):            # #4 water stress
    h1, h2, h3_high, h3_low, h4: float    # cm pressure head
    pet_high_cm_d, pet_low_cm_d: float

class FertEvent(BaseModel):               # #5 fertilizer event
    date: date
    depth_cm: float                       # 0 = surface; >0 = injected/banded
    mass_kg_n_ha: float
    form: Literal["NH4", "NO3", "urea", "NH4NO3", "compound"]
    composition: dict[str, float] | None = None

class IrrigEvent(BaseModel):              # #6 irrigation event
    date: date
    method: Literal["flood", "sprinkler", "drip", "subsurface"]
    amount_cm: float
    duration_h: float
    solute_concs_mg_l: dict[str, float] = {}
    drip_emitter_xyz: tuple[float, float, float] | None = None

class NTransformation(BaseModel):         # #7 N transformation — B2 hook
    mode: Literal["constant_rates", "external_callable", "lookup_table"]
    # constant_rates: GUI fills k_mineralization_d, k_nitrification_d,
    #                 k_denitrification_d, k_volatilization_d
    # external_callable: callable_ref="dndc.n_module:compute_rates" (B2)
    # lookup_table: table_path=NetCDF with rates(z, t, theta, c)

class PlantNUptake(BaseModel):            # #8 plant N uptake
    mode: Literal["passive_with_water", "michaelis_menten",
                  "demand_driven", "external"]
    km_mg_l / vmax_mg_per_day_per_root_cm: float | None
    daily_demand_kg_n_ha: list[float] | None = None
    callable_ref: str | None = None       # B2

class StateExchange(BaseModel):           # #9 bidirectional state
    initial_theta / initial_h / initial_c / initial_t: ... | None
    z_grid_cm: list[float]
    writeback_daily: bool = False         # True in B2; False in manual mode
    writeback_path: Path | None = None    # NetCDF

class SoilTemp(BaseModel):                # #10 soil temperature
    enabled: bool = False
    surface_t_daily_c: list[float] | None = None

class Residue(BaseModel):                 # #11 residue / mulch
    mulch_fraction: float = 0.0
    residue_kg_ha: float = 0.0
    e_reduction_factor: float = 1.0

class DndcSeamInputs(BaseModel):
    atm: AtmDaily
    et: EtPartition
    root: RootGrowth
    feddes: FeddesParams
    fert_events: list[FertEvent] = []
    irrig_events: list[IrrigEvent] = []
    n_transform: NTransformation
    plant_n_uptake: PlantNUptake
    state: StateExchange
    soil_temp: SoilTemp = SoilTemp()
    residue: Residue = Residue()
    extras: dict = {}                     # forward-compat buffer for B2 surprises

    def to_forcing(self, simulation_times: np.ndarray) -> Forcing:
        """Single conversion point: DNDC vocabulary → HYDRUS vocabulary."""
```

### 3.2 Adapter ABC

```python
class DndcSeamAdapter(ABC):
    @abstractmethod
    def produce(self, scenario_id: str,
                day_range: tuple[date, date]) -> DndcSeamInputs: ...

class ManualGuiAdapter(DndcSeamAdapter):  # current
    def __init__(self, form_state_path: Path): ...

class CsvNetcdfAdapter(DndcSeamAdapter):  # batch research
    def __init__(self, atm_csv, mgmt_csv): ...

class DndcLiveAdapter(DndcSeamAdapter):   # future B2 — only new file needed
    def __init__(self, dndc_session): ...
    def produce(self, scenario_id, day_range):
        if self.state_path.exists():
            last = load_netcdf(self.state_path)
            self.dndc.ingest_soil_state(last.theta, last.c_no3, last.c_nh4, last.t)
        self.dndc.step_day(day_range[0])
        return self.dndc.export_hydrus_forcing(day_range)
```

All research workflows accept `DndcSeamAdapter`, never concrete classes.
This is the B2 hand-off guarantee.

### 3.3 GUI forms

`desktop/src/pages/research/DNDCForms.vue` with 11 collapsible sections,
two-way bound to a Pinia store holding the reactive `DndcSeamInputs` object.
Backend validation via Pydantic. Crop presets (~15 crops: maize, wheat, rice,
soybean, cotton, tomato, potato, grass, etc.) hardcoded in
`hydrus_research/dndc_seam/crop_presets.yaml`; values from HYDRUS docs + FAO-56.

### 3.4 REST endpoints (current)

```
POST  /research/dndc/validate        → 422 if invalid, 200 + warnings
POST  /research/dndc/to-forcing      → Forcing as NetCDF (debug)
GET   /research/dndc/crop-presets
POST  /research/dndc/save-preset
GET   /research/dndc/presets
```

Future B2 additions:

```
POST  /research/dndc/live/start      → start in-process DndcLiveAdapter
POST  /research/dndc/live/step       → step one day, preview produced inputs
```

---

## 4. Research modules

### 4.1 F1 — Pedotransfer Functions

```
ptf/
├── rosetta.py              # rosetta-soil wrapper (4 hierarchical models)
├── carsel_parrish.py       # 1988 USDA 12-class lookup with covariance
├── wosten_hypres.py        # European HYPRES PTF
├── presets.py              # USDA centers + Vereecken w/ OC
├── uncertainty.py          # PTF → ParameterSpec priors (feeds F3/F5)
└── api.py
```

API:
```python
class PTFResult(BaseModel):
    theta_r, theta_s, alpha, n, Ks: float; L: float = 0.5
    method: Literal["rosetta3_h1", ..., "carsel_parrish", "wosten"]
    covariance: np.ndarray | None = None     # 5×5 for UQ

def texture_to_vg(sand_pct, silt_pct, clay_pct,
                  bulk_density_g_cm3=None, theta_33=None, theta_1500=None,
                  organic_carbon_pct=None,
                  method="rosetta3_auto") -> PTFResult: ...

def usda_class_to_vg(class_name) -> PTFResult: ...
def vg_to_prior(ptf: PTFResult) -> list[ParameterSpec]: ...
```

### 4.2 F4 — Batch runner

```
batch/
├── runner.py               # BatchRunner (joblib + tqdm)
├── pyemu_worker.py         # CLI: `hydrus research worker --master host:port`
├── result_store.py         # parquet/zarr storage of (θ, y_sim) for reuse
└── api.py
```

API:
```python
class BatchRunner:
    def __init__(self, simulator, forcing, ic, obs_specs,
                 n_workers="auto", backend="joblib"): ...
    def run(self, thetas: np.ndarray) -> BatchResult: ...
    def stream(self, thetas): ...

class BatchResult:
    thetas: np.ndarray         # (N, D)
    ys: np.ndarray             # (N, M_obs)
    wall_s: np.ndarray
    converged: np.ndarray
    meta: dict
    def to_parquet(self, path): ...
```

Backends: `joblib` (default, single-host parallelism) and `pyemu_tcp` (acts as
PEST++ worker pool). Dask explicitly dropped.

### 4.3 F2 — Sensitivity analysis

```
sensitivity/
├── morris.py / sobol.py / fast.py / pawn.py
├── grouped.py              # group by ParameterSpec.group
└── api.py
```

All four SALib methods exposed:
```python
def morris_screen(forward, param_map,
                  n_trajectories=20, num_levels=4) -> SensitivityResult: ...
def sobol_decompose(forward, param_map,
                    n_base=1024, calc_second_order=False) -> SensitivityResult: ...
def fast_indices(forward, param_map, n=1000) -> SensitivityResult: ...
def pawn_kde(forward, param_map, n=2000, s=10) -> SensitivityResult: ...

class SensitivityResult(BaseModel):
    method: str
    param_names: list[str]
    indices: dict[str, np.ndarray]   # e.g. {"S1": ..., "ST": ..., "S1_conf": ...}
    sample_size: int
    forward_cost_s: float
```

Implementation goes through `BatchRunner` (not SALib's built-in `.evaluate()`)
so dask / pyemu_tcp / progress bars work uniformly.

### 4.4 F3 — Inversion

```
inversion/
├── base.py                 # InversionResult schema + backend ABC
├── lm_scipy.py             # P0 fast path
├── pyemu_pestpp.py         # P0 main: GLM + IES exposed; IES is default
├── pymc_bayes.py           # P1; lazy-import pymc
└── api.py
```

Unified result schema:
```python
class InversionResult(BaseModel):
    backend: str
    best_params: dict[str, float]
    parameter_ci_lo / parameter_ci_hi: dict[str, float]
    posterior_ensemble: np.ndarray | None       # (N_real, D); LM = None
    objective_history: list[float]
    n_forward_calls: int
    wall_s: float
    jacobian_path: Path | None = None
    pest_workspace: Path | None = None
    diagnostics: dict
```

LM:
```python
def fit_lm(forward, param_map, obs, x0=None, max_nfev=200) -> InversionResult:
    res = scipy.optimize.least_squares(
        lambda t: obs.residuals(forward(t)),
        x0 or param_map.midpoints(),
        bounds=param_map.bounds_array().T,
        method="trf", jac="2-point", x_scale="jac", max_nfev=max_nfev,
    )
    # CI from res.jac SVD
```

PyEMU:
```python
def fit_pyemu(forward, param_map, obs, method="ies",
              n_real=200, n_iter=4, workspace=None) -> InversionResult:
    # 1) wrap forward as PEST++ model_command (Python subprocess script)
    # 2) start workers via BatchRunner.pyemu_worker
    # 3) pyemu.utils.os_utils.start_workers + run PESTPP-IES
    # 4) parse .par.csv / .obs.csv ensembles → InversionResult
```

Auto backend selection (used by GUI default):
- Params < 10 AND 1D simulator → LM
- Params ≥ 10 OR 2D/3D simulator → PyEMU IES
- User explicitly requests posterior → PyMC (if installed)

### 4.5 F5 — Uncertainty quantification

```
uq/
├── monte_carlo.py          # propagate PTF covariance
├── posterior_predict.py    # reuse F3 posterior_ensemble
├── glue.py                 # GLUE (Beven & Binley 1992) — filter existing batch
└── api.py
```

Most F5 calls reuse existing runs (posterior from F3, batch from F2/F4) — few
new forward evaluations needed. This is why `BatchResult.to_parquet()` is
central in §4.2.

### 4.6 Surrogate

```
surrogate/
├── base.py                 # SurrogateSimulator(Simulator) — drop-in replacement
├── gp_sklearn.py           # default, lightweight
├── pck.py                  # PC-Kriging (hydrology SOTA, uses smt + chaospy internally)
├── trainer.py              # fit/validate/save/load + k-fold CV
└── api.py
```

```python
class SurrogateSimulator(Simulator):
    name = "<surrogate-type>"
    dimension = -1
    def run(self, params, forcing, ic):
        y, sigma = self.model.predict(theta, return_std=True)
        return SimResult(..., meta={"is_surrogate": True, "predict_std": sigma})

def train_gp(batch_result, kernel="matern52", validate_kfold=5) -> SurrogateSimulator: ...
def train_pck(batch_result, pce_degree=3) -> SurrogateSimulator: ...
def evaluate(surrogate, holdout: BatchResult) -> dict:    # NSE, RMSE, coverage
```

3D inversion workflow:
1. LHS 200 thetas → `BatchRunner.run(thetas, Richards3DSimulator)` → parquet (hours, one-time).
2. `surrogate = train_pck(batch_result)` (minutes).
3. `fit_pyemu(surrogate.forward, ..., method="ies", n_real=500)` (minutes).
4. `predict_with_posterior(surrogate.forward, inv_result)` (minutes).
5. Re-run real 3D simulator at best_params for final validation.

### 4.7 F6 — Optimization (P2 — stubbed in P0)

```
optimization/
├── pymoo_nsga.py           # multi-objective: min N leaching, max yield proxy, min water
├── optuna_single.py        # single objective: WUE, NUE
├── decision_vars.py        # encode irrigation / fertilizer schedules as θ
├── constraints.py          # field capacity, dry threshold, regulatory N caps
└── api.py
```

---

## 5. GUI / REST / phasing / testing / risks

### 5.1 GUI pages (D route)

```
desktop/src/pages/research/
├── DNDCForms.vue              # §3.3
├── SoilLibrary.vue            # F1: texture triangle + USDA 12 + ROSETTA + BD slider
├── BatchSweep.vue             # F4: pick params × ranges × grid/LHS → progress + table
├── SensitivityReport.vue      # F2: Morris EE / Sobol bars / FAST spectra / PAWN KDE
├── InversionStudio.vue        # F3: obs upload → backend select → run → residuals / param PDFs / Jacobian heatmap
├── UQExplorer.vue             # F5: prediction bands + GLUE + PTF MC
├── SurrogateBench.vue         # train GP / PCK + k-fold CV + true-vs-surrogate scatter
└── components/
    ├── TextureTriangle.vue
    ├── ParamSpecEditor.vue
    ├── ObsCsvImporter.vue
    └── EnsembleViz.vue        # N curves + 95% band (LTTB downsampled for >200)
```

Navigation menu adds a `Research ▾` group with seven items.

### 5.2 REST endpoints

```
GET   /research/info
POST  /research/ptf/predict
GET   /research/ptf/usda-classes
POST  /research/batch/start              → job_id
GET   /research/batch/{job_id}/status    (SSE-streamable)
GET   /research/batch/{job_id}/result    → parquet
POST  /research/sensitivity/{method}     method ∈ morris/sobol/fast/pawn
POST  /research/inversion/{backend}      backend ∈ lm/pyemu_ies/pyemu_glm/pymc
GET   /research/inversion/{job_id}/posterior
POST  /research/uq/posterior-predict
POST  /research/uq/ptf-monte-carlo
POST  /research/surrogate/train
POST  /research/surrogate/{model_id}/predict
POST  /research/dndc/*                   # §3.4
```

Async model: FastAPI BackgroundTasks; job state in SQLite (`~/.hydrus/jobs.db`);
GUI consumes status via Server-Sent Events.

### 5.3 Implementation phasing

| M# | Content | Effort | Verification |
|---|---|---|---|
| M0 | §2 abstraction layer + 1D adapter + 1 end-to-end test | 1.5 wk | `forward(θ)` runs; `apply_to_scenario` round-trip OK |
| M1 | §3 DNDC seam + 11-section Vue forms + crop_presets.yaml + REST | 2 wk | Rebuild EX fixtures from DndcSeamInputs and reproduce results |
| M2 | §4.1 F1 PTF + SoilLibrary.vue + TextureTriangle | 1 wk | rosetta-soil reproduces ROSETTA paper Table 2; Carsel-Parrish matches 1988 |
| M3 | §4.2 F4 BatchRunner (joblib + pyemu_tcp) + BatchSweep.vue + parquet | 1.5 wk | 64-core LHS=1024 on 1D fixture; pyemu worker mode passes minimal PEST++ test |
| M4 | §4.3 F2 (4 methods) + SensitivityReport.vue | 1 wk | Ishigami analytical reproduced (S1/ST within SALib docs) |
| M5 | §4.4 F3 LM + InversionStudio.vue basic | 1 wk | 1D fixture with injected perturbation → LM recovers ≤ 1% |
| M6 | §4.4 F3 PyEMU (IES + GLM) + InversionStudio.vue advanced | 2.5 wk | 2D fixture with perturbation → IES converges with n_real=100 |
| M7 | swms2d_adapter.py + richards3d_adapter.py + observable_at spatial interp | 1.5 wk | EX.1–4 + 3D demo all run through abstraction layer |
| M8 | §4.5 F5 + §4.6 surrogate (sklearn GP + PCK) + UQExplorer + SurrogateBench | 2.5 wk | 3D end-to-end: LHS 200 → PCK → IES → best_params matches true model |
| M9 (P1) | §4.4 F3 PyMC backend (lazy-imported) | 1 wk | 1D NUTS posterior vs IES posterior KS test |
| M10 (P2) | §4.7 F6 + pymoo + UI | 2 wk | NSGA-II Pareto front on synthetic fertigation problem |

P0 total: **M0–M8 ≈ 14 weeks**. Parallelism opportunities: M1/M2/M3 parallel;
M5/M7 parallel. Fastest realistic schedule ≈ 9–10 weeks.

### 5.4 Testing strategy

| Layer | Tool | Coverage target | Content |
|---|---|---|---|
| Unit | pytest | 90% | every public function; mock Simulator for research modules |
| Integration | pytest + tmp_path | 80% | fixture → abstraction → real Simulator → result |
| Regression | golden parquet | 100% of key fixtures | Sobol indices, IES posterior moments vs baseline (5% tolerance) |
| Cross-impl | explicit | — | LM vs PyEMU GLM agree on best_params; GP vs real NSE > 0.95 |
| Scientific | open benchmarks | — | PTF: ROSETTA paper Table 2; F2: Ishigami; F3: HYDRUS docs inverse case; F5: GLUE Beven 1992 toy |
| GUI | Playwright | key paths | 7 research pages: load → submit → result render |

### 5.5 Risks and mitigations

| Risk | P | Impact | Mitigation |
|---|---|---|---|
| PyEMU TCP worker unstable on Win/macOS | M | H | M3 early spike (2–3 days); fallback: joblib + custom IES (algorithm is straightforward) |
| Pydantic v2 incompat with existing schema | L | M | Audit canonical scenario schema in M0; upgrade in one pass |
| ROSETTA-3 pulls TF/Keras | M | M | rosetta-soil migrated to ONNX runtime — verify; fallback to H1 + Carsel-Parrish |
| Tauri webview chokes on >1000-curve ensemble | M | M | EnsembleViz uses Canvas + LTTB downsample; >200 curves → alpha=0.05 + quantile bands |
| 3D simulator minutes-per-run → even LHS 200 slow | H | M | Start with LHS 50, expand adaptively if GP fit poor |
| 11-item DNDC contract insufficient long-term | M | H | `DndcSeamInputs.extras: dict` forward-compat buffer; Pydantic `model_config = ConfigDict(extra="allow")` |
| 22 existing scenarios break during abstraction migration | L | H | M0 runs all 22 through abstraction layer; CI gate |

### 5.6 Documentation

`docs/research/`:
- `01_concepts.md` — Simulator / Parameter / Observation
- `02_tutorial_ptf.md` — texture → VG (10 min)
- `03_tutorial_sensitivity.md` — Morris + Sobol
- `04_tutorial_inversion.md` — LM fast / PyEMU serious
- `05_tutorial_uq.md` — PTF MC + posterior predict
- `06_tutorial_3d_surrogate.md` — 3D inversion via GP
- `07_dndc_seam.md` — manual now / B2 later
- `08_extending.md` — add a new research tool (e.g. plug Optuna)

---

## 6. Out of scope (explicit)

To keep this spec finite, the following are explicitly **not** in this design:

- Re-porting any solver (the HYDRUS / SWMS_2D / 3D Richards code is unchanged).
- Implementing the DNDC model itself (lives in the user's separate Python package).
- F3-Bayesian (PyMC) implementation — interface reserved, implementation in P1.
- F6 Optimization implementation — package stub created, real work in P2.
- Distributed (multi-host) batch runs via Dask — joblib + pyemu_tcp cover P0/P1.
- Heat-transport-driven research workflows — solver supports it, but no
  dedicated research wrappers (covered as "future").
- Multi-user concurrent jobs / RBAC on the server — single-user local app.
- Experiment-tracking integrations (MLflow / DVC) — parquet on disk is enough
  for P0; integrations are future work.
- Mobile / web-hosted GUI — Tauri desktop remains the target.

---

## 7. Glossary

- **B2** — coupling option chosen for DNDC ↔ HYDRUS: in-process Python tight
  coupling, day-step orchestrator, bidirectional state exchange.
- **D3** — dimension-coverage option chosen: F2/F3/F5 cover 1D + 2D + 3D, with
  3D using surrogate models for tractability.
- **F1–F6** — research feature categories: F1 PTF, F2 sensitivity, F3 inversion,
  F4 batch, F5 UQ, F6 optimization.
- **IES** — Iterative Ensemble Smoother, the default PESTPP / PyEMU inversion.
- **PCK** — Polynomial Chaos + Kriging hybrid surrogate (Schöbi et al., shown
  to outperform pure PCE / GP for hydrological response surfaces).
- **PTF** — Pedotransfer Function: maps soil texture / bulk density to van
  Genuchten hydraulic parameters.
- **Narrow waist** — the single shared interface (`forward(θ) → y`) through
  which all research tools talk to all simulators.
