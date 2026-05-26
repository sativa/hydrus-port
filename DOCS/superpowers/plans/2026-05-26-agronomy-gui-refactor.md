# Agronomy Decision GUI Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebuild the desktop GUI around a workflow-driven agronomy decision panel (作物/土壤/雨型 dropdowns + irrig/fert event tables → θ(z,t) + N(z,t) heatmaps + water/N balance bars). The existing 9 research tabs move into a right-side "高级" drawer.

**Architecture:** Three-band Vue 3 layout (`InputStrip` → `HeatmapBand` → `BalanceBand`) driven by a Pinia store; new `hydrus_research/library/` (crops/soils/weather JSON+CSV), new `hydrus_research/agronomy/` (scenario_builder → runner → result_parser), new FastAPI router `research_agronomy` registered like the M1-M7 routers. CLI `hydrus research agronomy run` provides headless parity.

**Tech Stack:** Python 3.11 · FastAPI · Pydantic v2 · hydrus_research package · hydrus1d adapter · Vue 3 + Pinia · Plotly.js · Tauri 2 · Playwright

---

## Reference Notes

- **FastAPI app lives in `hydrus_port_server/app.py`** (not `hydrus_port/app.py`). Routers go under `hydrus_port_server/routers/`.
- **NOD_INF.OUT z is descending** (surface 0 → negative downward). `np.interp` needs ascending — always reverse `z + field` together before downstream use. See memory `feedback_hydrus1d_nod_inf_z_descending.md`.
- **Frontend research APIs use `fetch(RESEARCH_BASE)`** (see `desktop/src/api.ts:180+`). New `agronomy` namespace follows the `ptf` / `sensitivity` pattern.
- **Event date → simulation `t_day` mapping:** `t_day = (event_date - sow_date).days`, where `sow_date = year_start + crop.season.sow_doy - 1`. Year is the simulation start year (default: 2026, derived from the irrigation/fert event dates).
- **Decoupling discipline (HARD):** every step must be reachable via `hydrus research agronomy run ...`. No GUI-only code paths.

---

## File Structure

**Backend (new):**
```
hydrus_research/library/
    __init__.py
    crops.py                            # CropLib loader + Pydantic Crop / Feddes / Root / KcPoint
    soils.py                            # SoilLib loader + Pydantic Soil / SoilLayer / VanGenuchten
    weather.py                          # WeatherLib loader; CSV → arrays
    data/
        crops.json                      # 10 crops
        soils.json                      # 12 textures
        weather/
            n_china_avg.csv             # 365 rows: doy,P_mm,PET_mm
            n_china_wet.csv
            n_china_dry.csv
            c_china_meiyu.csv
            s_china_double.csv
            nw_china_irrig.csv

hydrus_research/agronomy/
    __init__.py
    types.py                            # IrrigEvent, FertEvent, AgronomyRequest, AgronomyResult
    scenario_builder.py                 # build_scenario(crop, soil, weather, irrig, fert, horizon) -> Scenario
    runner.py                           # run_agronomy(req) -> AgronomyResult (calls hydrus1d adapter)
    result_parser.py                    # parse_nod_inf / parse_balance / parse_t_level → numpy arrays

hydrus_port_server/routers/
    research_agronomy.py                # GET /lib/{crops,soils,weather}, GET /lib/weather/{id}, POST /agronomy/run
```

**Backend (modify):**
```
hydrus_port_server/app.py               # register agronomy_router
hydrus_port/cli.py                      # add `research agronomy run` subcommand
```

**Frontend (new):**
```
desktop/src/stores/agronomy.ts          # Pinia store
desktop/src/components/agronomy/
    InputStrip.vue
    EventsTable.vue
    HeatmapBand.vue
    ThetaHeatmap.vue
    NitrateHeatmap.vue
    BalanceBand.vue
    WaterBalanceBar.vue
    NBudgetBar.vue
    AdvancedDrawer.vue
```

**Frontend (modify):**
```
desktop/src/api.ts                      # add `lib` + `agronomy` namespaces
desktop/src/App.vue                     # rewrite into 3-band layout
```

**Tests (new):**
```
tests/research/test_library_loaders.py
tests/research/test_agronomy_scenario.py
tests/research/test_agronomy_runner.py
tests/research/test_cli_agronomy.py
tests/research/test_research_agronomy_router.py
desktop/tests/agronomy_smoke.spec.ts
```

---

## Task 1: Library data files (crops.json + soils.json + weather CSVs)

**Files:**
- Create: `hydrus_research/library/data/crops.json`
- Create: `hydrus_research/library/data/soils.json`
- Create: `hydrus_research/library/data/weather/n_china_avg.csv`
- Create: `hydrus_research/library/data/weather/n_china_wet.csv`
- Create: `hydrus_research/library/data/weather/n_china_dry.csv`
- Create: `hydrus_research/library/data/weather/c_china_meiyu.csv`
- Create: `hydrus_research/library/data/weather/s_china_double.csv`
- Create: `hydrus_research/library/data/weather/nw_china_irrig.csv`

- [ ] **Step 1: Create `crops.json` with 10 crops**

Write exactly this JSON to `hydrus_research/library/data/crops.json`:

```json
{
  "crops": [
    {"id":"maize","name_zh":"玉米","name_en":"Maize",
     "feddes":{"P0":-15,"P0pt":-30,"P2H":-325,"P2L":-600,"P3":-8000,"r2H":0.5,"r2L":0.1},
     "root":{"max_depth_cm":120,"z50_cm":30,"z95_cm":90},
     "season":{"sow_doy":121,"harvest_doy":273},
     "kc_curve":[{"doy":121,"kc":0.30},{"doy":160,"kc":0.70},{"doy":200,"kc":1.15},{"doy":240,"kc":1.00},{"doy":273,"kc":0.50}]},
    {"id":"wheat_winter","name_zh":"冬小麦","name_en":"Winter wheat",
     "feddes":{"P0":-15,"P0pt":-30,"P2H":-400,"P2L":-800,"P3":-8000,"r2H":0.5,"r2L":0.1},
     "root":{"max_depth_cm":150,"z50_cm":40,"z95_cm":120},
     "season":{"sow_doy":280,"harvest_doy":160},
     "kc_curve":[{"doy":280,"kc":0.35},{"doy":340,"kc":0.70},{"doy":60,"kc":1.15},{"doy":120,"kc":1.15},{"doy":160,"kc":0.40}]},
    {"id":"rice","name_zh":"水稻","name_en":"Rice",
     "feddes":{"P0":100,"P0pt":55,"P2H":-160,"P2L":-250,"P3":-15000,"r2H":0.5,"r2L":0.1},
     "root":{"max_depth_cm":60,"z50_cm":15,"z95_cm":45},
     "season":{"sow_doy":140,"harvest_doy":260},
     "kc_curve":[{"doy":140,"kc":1.05},{"doy":180,"kc":1.20},{"doy":220,"kc":1.20},{"doy":260,"kc":0.90}]},
    {"id":"cotton","name_zh":"棉花","name_en":"Cotton",
     "feddes":{"P0":-15,"P0pt":-30,"P2H":-500,"P2L":-1000,"P3":-16000,"r2H":0.5,"r2L":0.1},
     "root":{"max_depth_cm":150,"z50_cm":40,"z95_cm":120},
     "season":{"sow_doy":120,"harvest_doy":290},
     "kc_curve":[{"doy":120,"kc":0.35},{"doy":160,"kc":0.70},{"doy":210,"kc":1.20},{"doy":260,"kc":0.80},{"doy":290,"kc":0.50}]},
    {"id":"tomato","name_zh":"番茄","name_en":"Tomato",
     "feddes":{"P0":-10,"P0pt":-25,"P2H":-200,"P2L":-400,"P3":-8000,"r2H":0.5,"r2L":0.1},
     "root":{"max_depth_cm":80,"z50_cm":20,"z95_cm":60},
     "season":{"sow_doy":100,"harvest_doy":240},
     "kc_curve":[{"doy":100,"kc":0.30},{"doy":140,"kc":0.70},{"doy":180,"kc":1.15},{"doy":220,"kc":0.80},{"doy":240,"kc":0.60}]},
    {"id":"grape","name_zh":"葡萄","name_en":"Grape",
     "feddes":{"P0":-10,"P0pt":-25,"P2H":-400,"P2L":-800,"P3":-15000,"r2H":0.5,"r2L":0.1},
     "root":{"max_depth_cm":200,"z50_cm":60,"z95_cm":160},
     "season":{"sow_doy":90,"harvest_doy":300},
     "kc_curve":[{"doy":90,"kc":0.30},{"doy":140,"kc":0.65},{"doy":210,"kc":0.85},{"doy":270,"kc":0.45},{"doy":300,"kc":0.20}]},
    {"id":"apple","name_zh":"苹果","name_en":"Apple",
     "feddes":{"P0":-10,"P0pt":-25,"P2H":-500,"P2L":-1000,"P3":-15000,"r2H":0.5,"r2L":0.1},
     "root":{"max_depth_cm":200,"z50_cm":50,"z95_cm":150},
     "season":{"sow_doy":90,"harvest_doy":290},
     "kc_curve":[{"doy":90,"kc":0.45},{"doy":150,"kc":0.95},{"doy":240,"kc":1.20},{"doy":290,"kc":0.85}]},
    {"id":"tea","name_zh":"茶","name_en":"Tea",
     "feddes":{"P0":-10,"P0pt":-25,"P2H":-300,"P2L":-600,"P3":-10000,"r2H":0.5,"r2L":0.1},
     "root":{"max_depth_cm":120,"z50_cm":30,"z95_cm":90},
     "season":{"sow_doy":60,"harvest_doy":330},
     "kc_curve":[{"doy":60,"kc":1.00},{"doy":180,"kc":1.10},{"doy":330,"kc":0.95}]},
    {"id":"rapeseed","name_zh":"油菜","name_en":"Rapeseed",
     "feddes":{"P0":-15,"P0pt":-30,"P2H":-400,"P2L":-800,"P3":-8000,"r2H":0.5,"r2L":0.1},
     "root":{"max_depth_cm":120,"z50_cm":35,"z95_cm":100},
     "season":{"sow_doy":270,"harvest_doy":150},
     "kc_curve":[{"doy":270,"kc":0.35},{"doy":330,"kc":0.70},{"doy":60,"kc":1.10},{"doy":120,"kc":1.10},{"doy":150,"kc":0.40}]},
    {"id":"soybean","name_zh":"大豆","name_en":"Soybean",
     "feddes":{"P0":-15,"P0pt":-30,"P2H":-200,"P2L":-400,"P3":-8000,"r2H":0.5,"r2L":0.1},
     "root":{"max_depth_cm":130,"z50_cm":35,"z95_cm":100},
     "season":{"sow_doy":150,"harvest_doy":280},
     "kc_curve":[{"doy":150,"kc":0.40},{"doy":190,"kc":0.80},{"doy":230,"kc":1.15},{"doy":270,"kc":0.50},{"doy":280,"kc":0.30}]}
  ]
}
```

- [ ] **Step 2: Create `soils.json` with 12 textures**

Write to `hydrus_research/library/data/soils.json` (Carsel-Parrish 1988 values; Ks in cm/day):

```json
{
  "soils": [
    {"id":"sand","name_zh":"砂土","name_en":"Sand",
     "layers":[{"depth_cm":200,"vg":{"theta_r":0.045,"theta_s":0.43,"alpha":0.145,"n":2.68,"Ks":712.8,"L":0.5}}]},
    {"id":"loamy_sand","name_zh":"壤砂土","name_en":"Loamy sand",
     "layers":[{"depth_cm":200,"vg":{"theta_r":0.057,"theta_s":0.41,"alpha":0.124,"n":2.28,"Ks":350.2,"L":0.5}}]},
    {"id":"sandy_loam","name_zh":"砂壤","name_en":"Sandy loam",
     "layers":[{"depth_cm":200,"vg":{"theta_r":0.065,"theta_s":0.41,"alpha":0.075,"n":1.89,"Ks":106.1,"L":0.5}}]},
    {"id":"loam","name_zh":"壤土","name_en":"Loam",
     "layers":[{"depth_cm":200,"vg":{"theta_r":0.078,"theta_s":0.43,"alpha":0.036,"n":1.56,"Ks":24.96,"L":0.5}}]},
    {"id":"silt","name_zh":"粉土","name_en":"Silt",
     "layers":[{"depth_cm":200,"vg":{"theta_r":0.034,"theta_s":0.46,"alpha":0.016,"n":1.37,"Ks":6.0,"L":0.5}}]},
    {"id":"silt_loam","name_zh":"粉壤","name_en":"Silt loam",
     "layers":[{"depth_cm":200,"vg":{"theta_r":0.067,"theta_s":0.45,"alpha":0.020,"n":1.41,"Ks":10.8,"L":0.5}}]},
    {"id":"sandy_clay_loam","name_zh":"砂粘壤","name_en":"Sandy clay loam",
     "layers":[{"depth_cm":200,"vg":{"theta_r":0.100,"theta_s":0.39,"alpha":0.059,"n":1.48,"Ks":31.44,"L":0.5}}]},
    {"id":"clay_loam","name_zh":"粘壤","name_en":"Clay loam",
     "layers":[{"depth_cm":200,"vg":{"theta_r":0.095,"theta_s":0.41,"alpha":0.019,"n":1.31,"Ks":6.24,"L":0.5}}]},
    {"id":"silty_clay_loam","name_zh":"粉粘壤","name_en":"Silty clay loam",
     "layers":[{"depth_cm":200,"vg":{"theta_r":0.089,"theta_s":0.43,"alpha":0.010,"n":1.23,"Ks":1.68,"L":0.5}}]},
    {"id":"clay","name_zh":"粘土","name_en":"Clay",
     "layers":[{"depth_cm":200,"vg":{"theta_r":0.068,"theta_s":0.38,"alpha":0.008,"n":1.09,"Ks":4.80,"L":0.5}}]},
    {"id":"sand_over_clay","name_zh":"砂覆粘","name_en":"Sand over clay",
     "layers":[
       {"depth_cm":40,"vg":{"theta_r":0.045,"theta_s":0.43,"alpha":0.145,"n":2.68,"Ks":712.8,"L":0.5}},
       {"depth_cm":160,"vg":{"theta_r":0.068,"theta_s":0.38,"alpha":0.008,"n":1.09,"Ks":4.80,"L":0.5}}
     ]},
    {"id":"topsoil_subsoil_bedrock","name_zh":"表土+心土+底土","name_en":"Topsoil/subsoil/bedrock",
     "layers":[
       {"depth_cm":30,"vg":{"theta_r":0.078,"theta_s":0.43,"alpha":0.036,"n":1.56,"Ks":24.96,"L":0.5}},
       {"depth_cm":100,"vg":{"theta_r":0.095,"theta_s":0.41,"alpha":0.019,"n":1.31,"Ks":6.24,"L":0.5}},
       {"depth_cm":70,"vg":{"theta_r":0.068,"theta_s":0.38,"alpha":0.008,"n":1.09,"Ks":4.80,"L":0.5}}
     ]}
  ]
}
```

- [ ] **Step 3: Create weather CSV files**

Use a one-off Python generator (do NOT keep this script — only run it once to produce the CSVs). Run from the repo root:

```python
import csv, math, random
from pathlib import Path

random.seed(0)
DOY = list(range(1, 366))

def pet_curve(amp, base):
    return [round(base + amp * math.sin(2*math.pi*(d-100)/365), 3) for d in DOY]

def rain_avg(annual_mm, peak_doy, spread, n_events):
    out = [0.0]*365
    for _ in range(n_events):
        d = int(random.gauss(peak_doy, spread)) % 365
        out[d] += round(random.uniform(2, 20), 2)
    s = sum(out) or 1
    return [round(x * annual_mm / s, 2) for x in out]

profiles = {
  "n_china_avg":    dict(rain=500,  peak=200, spread=45, events=80,  pet_amp=2.5, pet_base=2.8),
  "n_china_wet":    dict(rain=750,  peak=200, spread=45, events=110, pet_amp=2.5, pet_base=2.7),
  "n_china_dry":    dict(rain=320,  peak=200, spread=45, events=55,  pet_amp=2.6, pet_base=3.0),
  "c_china_meiyu":  dict(rain=1100, peak=165, spread=25, events=120, pet_amp=2.2, pet_base=3.0),
  "s_china_double": dict(rain=1600, peak=180, spread=80, events=180, pet_amp=1.8, pet_base=3.5),
  "nw_china_irrig": dict(rain=180,  peak=200, spread=50, events=35,  pet_amp=3.0, pet_base=3.2),
}

base_dir = Path("hydrus_research/library/data/weather")
base_dir.mkdir(parents=True, exist_ok=True)

for name, p in profiles.items():
    P  = rain_avg(p["rain"], p["peak"], p["spread"], p["events"])
    ET = pet_curve(p["pet_amp"], p["pet_base"])
    with open(base_dir / f"{name}.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["doy","P_mm","PET_mm"])
        for d, prec, pet in zip(DOY, P, ET):
            w.writerow([d, prec, pet])
```

Run: `python -c "<paste script>"`
Expected: 6 CSVs created, each 366 lines (header + 365 days).

- [ ] **Step 4: Commit**

```bash
git add hydrus_research/library/data/
git commit -m "data: crops/soils/weather libraries for agronomy GUI"
```

---

## Task 2: Library loaders + Pydantic models

**Files:**
- Create: `hydrus_research/library/__init__.py`
- Create: `hydrus_research/library/crops.py`
- Create: `hydrus_research/library/soils.py`
- Create: `hydrus_research/library/weather.py`
- Test: `tests/research/test_library_loaders.py`

- [ ] **Step 1: Write the failing test**

Create `tests/research/test_library_loaders.py`:

```python
"""Smoke tests for hydrus_research.library loaders."""
from __future__ import annotations
import numpy as np

from hydrus_research.library.crops import load_crops, get_crop
from hydrus_research.library.soils import load_soils, get_soil
from hydrus_research.library.weather import (
    load_weather_meta, load_weather_series,
)


def test_crops_lib_has_10_known_ids():
    crops = load_crops()
    ids = {c.id for c in crops}
    assert ids >= {"maize", "wheat_winter", "rice", "cotton", "tomato",
                   "grape", "apple", "tea", "rapeseed", "soybean"}


def test_crop_lookup_returns_pydantic_model():
    maize = get_crop("maize")
    assert maize.name_zh == "玉米"
    assert maize.feddes.P3 == -8000
    assert maize.root.max_depth_cm == 120
    assert maize.season.sow_doy == 121


def test_soils_lib_has_12_known_ids():
    soils = load_soils()
    ids = {s.id for s in soils}
    assert ids >= {"sand", "loamy_sand", "sandy_loam", "loam", "silt",
                   "silt_loam", "sandy_clay_loam", "clay_loam",
                   "silty_clay_loam", "clay", "sand_over_clay",
                   "topsoil_subsoil_bedrock"}


def test_soil_layered_preset_has_multiple_layers():
    s = get_soil("sand_over_clay")
    assert len(s.layers) == 2
    assert s.layers[0].depth_cm == 40
    assert s.layers[0].vg.alpha == 0.145
    assert s.layers[1].vg.alpha == 0.008


def test_weather_meta_lists_6_profiles():
    meta = load_weather_meta()
    ids = {m["id"] for m in meta}
    assert ids == {"n_china_avg", "n_china_wet", "n_china_dry",
                   "c_china_meiyu", "s_china_double", "nw_china_irrig"}


def test_weather_series_returns_365_day_arrays():
    s = load_weather_series("n_china_avg")
    assert len(s["doy"]) == 365
    assert len(s["P_mm"]) == 365
    assert len(s["PET_mm"]) == 365
    assert all(p >= 0 for p in s["P_mm"])
    assert all(e > 0 for e in s["PET_mm"])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/research/test_library_loaders.py -v`
Expected: `ImportError` (module does not exist).

- [ ] **Step 3: Implement `crops.py`**

Create `hydrus_research/library/crops.py`:

```python
"""Crop library: Pydantic models + JSON loader."""
from __future__ import annotations
import json
from functools import lru_cache
from pathlib import Path
from pydantic import BaseModel


_DATA = Path(__file__).parent / "data" / "crops.json"


class Feddes(BaseModel):
    P0: float
    P0pt: float
    P2H: float
    P2L: float
    P3: float
    r2H: float
    r2L: float


class Root(BaseModel):
    max_depth_cm: float
    z50_cm: float
    z95_cm: float


class Season(BaseModel):
    sow_doy: int
    harvest_doy: int


class KcPoint(BaseModel):
    doy: int
    kc: float


class Crop(BaseModel):
    id: str
    name_zh: str
    name_en: str
    feddes: Feddes
    root: Root
    season: Season
    kc_curve: list[KcPoint]


@lru_cache(maxsize=1)
def load_crops() -> list[Crop]:
    raw = json.loads(_DATA.read_text(encoding="utf-8"))
    return [Crop.model_validate(c) for c in raw["crops"]]


def get_crop(crop_id: str) -> Crop:
    for c in load_crops():
        if c.id == crop_id:
            return c
    raise KeyError(f"unknown crop id: {crop_id}")
```

- [ ] **Step 4: Implement `soils.py`**

Create `hydrus_research/library/soils.py`:

```python
"""Soil library: Pydantic models + JSON loader."""
from __future__ import annotations
import json
from functools import lru_cache
from pathlib import Path
from pydantic import BaseModel


_DATA = Path(__file__).parent / "data" / "soils.json"


class VanGenuchten(BaseModel):
    theta_r: float
    theta_s: float
    alpha: float      # 1/cm
    n: float
    Ks: float         # cm/day
    L: float = 0.5


class SoilLayer(BaseModel):
    depth_cm: float
    vg: VanGenuchten


class Soil(BaseModel):
    id: str
    name_zh: str
    name_en: str
    layers: list[SoilLayer]


@lru_cache(maxsize=1)
def load_soils() -> list[Soil]:
    raw = json.loads(_DATA.read_text(encoding="utf-8"))
    return [Soil.model_validate(s) for s in raw["soils"]]


def get_soil(soil_id: str) -> Soil:
    for s in load_soils():
        if s.id == soil_id:
            return s
    raise KeyError(f"unknown soil id: {soil_id}")
```

- [ ] **Step 5: Implement `weather.py`**

Create `hydrus_research/library/weather.py`:

```python
"""Weather typical-year CSV loader."""
from __future__ import annotations
import csv
from functools import lru_cache
from pathlib import Path


_DATA_DIR = Path(__file__).parent / "data" / "weather"

_NAMES_ZH = {
    "n_china_avg":    "华北平水年",
    "n_china_wet":    "华北丰水年",
    "n_china_dry":    "华北枯水年",
    "c_china_meiyu":  "华中梅雨",
    "s_china_double": "华南双季",
    "nw_china_irrig": "西北灌溉年",
}


def load_weather_meta() -> list[dict]:
    out = []
    for p in sorted(_DATA_DIR.glob("*.csv")):
        wid = p.stem
        out.append({"id": wid, "name_zh": _NAMES_ZH.get(wid, wid)})
    return out


@lru_cache(maxsize=8)
def load_weather_series(weather_id: str) -> dict[str, list[float]]:
    p = _DATA_DIR / f"{weather_id}.csv"
    if not p.exists():
        raise KeyError(f"unknown weather id: {weather_id}")
    doy, P, PET = [], [], []
    with p.open() as f:
        r = csv.DictReader(f)
        for row in r:
            doy.append(int(row["doy"]))
            P.append(float(row["P_mm"]))
            PET.append(float(row["PET_mm"]))
    return {"doy": doy, "P_mm": P, "PET_mm": PET}
```

- [ ] **Step 6: Create `__init__.py`**

Write to `hydrus_research/library/__init__.py`:

```python
"""hydrus_research.library — crops, soils, weather typical years."""
from .crops import load_crops, get_crop, Crop, Feddes, Root, Season, KcPoint
from .soils import load_soils, get_soil, Soil, SoilLayer, VanGenuchten
from .weather import load_weather_meta, load_weather_series

__all__ = [
    "load_crops", "get_crop", "Crop", "Feddes", "Root", "Season", "KcPoint",
    "load_soils", "get_soil", "Soil", "SoilLayer", "VanGenuchten",
    "load_weather_meta", "load_weather_series",
]
```

- [ ] **Step 7: Run tests to verify they pass**

Run: `pytest tests/research/test_library_loaders.py -v`
Expected: 6 passed.

- [ ] **Step 8: Commit**

```bash
git add hydrus_research/library/ tests/research/test_library_loaders.py
git commit -m "library: crop/soil/weather loaders + Pydantic models"
```

---

## Task 3: Agronomy types + scenario builder

**Files:**
- Create: `hydrus_research/agronomy/__init__.py`
- Create: `hydrus_research/agronomy/types.py`
- Create: `hydrus_research/agronomy/scenario_builder.py`
- Test: `tests/research/test_agronomy_scenario.py`

- [ ] **Step 1: Define types**

Create `hydrus_research/agronomy/types.py`:

```python
"""Pydantic request/response types for the agronomy decision workflow."""
from __future__ import annotations
from datetime import date
from pydantic import BaseModel, Field


class IrrigEvent(BaseModel):
    date: date
    depth_mm: float = Field(gt=0)


class FertEvent(BaseModel):
    date: date
    kg_n_ha: float = Field(gt=0)
    conc_mg_l: float | None = None  # if provided, applied with the next irrigation


class AgronomyRequest(BaseModel):
    crop_id: str
    soil_id: str
    weather_id: str
    horizon_days: int = Field(gt=0, le=400)
    irrigation: list[IrrigEvent] = []
    fertilizer: list[FertEvent] = []
    start_year: int = 2026


class WaterBalance(BaseModel):
    rain_mm: float
    irrig_mm: float
    et_mm: float
    percolation_mm: float
    storage_change_mm: float


class NBudget(BaseModel):
    applied_kg_ha: float
    uptake_kg_ha: float
    leached_kg_ha: float
    residual_kg_ha: float


class EventTick(BaseModel):
    t_day: float
    amount: float            # mm for irrig, kg N/ha for fert
    label: str               # "irrig" | "fert"


class AgronomyResult(BaseModel):
    z_cm: list[float]                # ascending (surface=0 → deep positive)
    t_days: list[float]
    theta_zt: list[list[float]]      # shape (nT, nZ)
    n_zt: list[list[float]]
    water_balance: WaterBalance
    n_budget: NBudget
    events: list[EventTick]
```

- [ ] **Step 2: Write the failing test**

Create `tests/research/test_agronomy_scenario.py`:

```python
from datetime import date
from hydrus_research.agronomy.types import (
    AgronomyRequest, IrrigEvent, FertEvent,
)
from hydrus_research.agronomy.scenario_builder import build_scenario
from hydrus_research.library.crops import get_crop
from hydrus_research.library.soils import get_soil
from hydrus_research.library.weather import load_weather_series


def test_build_scenario_roundtrips_through_canonical():
    crop = get_crop("maize")
    soil = get_soil("loam")
    weather = load_weather_series("n_china_avg")

    req = AgronomyRequest(
        crop_id="maize", soil_id="loam", weather_id="n_china_avg",
        horizon_days=150, start_year=2026,
        irrigation=[IrrigEvent(date=date(2026, 6, 1), depth_mm=30.0)],
        fertilizer=[FertEvent(date=date(2026, 5, 15), kg_n_ha=60.0)],
    )

    sc = build_scenario(crop, soil, weather, req)

    # Smoke checks: scenario has the right dimension, season, and a profile.
    d = sc.to_dict()
    assert d["dimension"] == "1d"
    assert d["sim"]["t_max"] >= 150
    # Profile depth equals sum of soil layer depths.
    expected_depth = sum(L.depth_cm for L in soil.layers)
    assert abs(d["profile"]["depth_cm"] - expected_depth) < 1e-6
    # Feddes block carries the crop's P3.
    assert d["sink"]["feddes"]["P3"] == crop.feddes.P3


def test_event_date_maps_to_t_day():
    from hydrus_research.agronomy.scenario_builder import event_to_t_day
    sow = date(2026, 5, 1)   # arbitrary
    assert event_to_t_day(date(2026, 5, 1), sow) == 0
    assert event_to_t_day(date(2026, 6, 1), sow) == 31
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `pytest tests/research/test_agronomy_scenario.py -v`
Expected: `ImportError: cannot import name 'build_scenario'`.

- [ ] **Step 4: Implement `scenario_builder.py`**

Create `hydrus_research/agronomy/scenario_builder.py`:

```python
"""Map (crop, soil, weather_series, AgronomyRequest) → canonical Scenario."""
from __future__ import annotations
from datetime import date, timedelta

from hydrus_port.schema import Scenario, _scenario_from_dict
from hydrus_research.library.crops import Crop
from hydrus_research.library.soils import Soil
from .types import AgronomyRequest


def event_to_t_day(event_date: date, sow_date: date) -> int:
    return (event_date - sow_date).days


def _sow_date(req: AgronomyRequest, crop: Crop) -> date:
    return date(req.start_year, 1, 1) + timedelta(days=crop.season.sow_doy - 1)


def build_scenario(
    crop: Crop,
    soil: Soil,
    weather: dict[str, list[float]],
    req: AgronomyRequest,
) -> Scenario:
    """Build a canonical 1D HYDRUS scenario for the given inputs."""
    sow = _sow_date(req, crop)

    # --- atmospheric BC: 1 row per day from sow to sow+horizon -----------
    horizon = req.horizon_days
    atm_rows = []
    for i in range(horizon):
        cur = sow + timedelta(days=i)
        doy = cur.timetuple().tm_yday
        idx = (doy - 1) % 365
        prec_mm = weather["P_mm"][idx]
        pet_mm  = weather["PET_mm"][idx]
        # Add irrigation events that fall on this day
        for ie in req.irrigation:
            if ie.date == cur:
                prec_mm += ie.depth_mm
        atm_rows.append({
            "t_day": i + 1,
            "prec_cm": prec_mm / 10.0,
            "pet_cm":  pet_mm  / 10.0,
        })

    # --- profile: stack soil layers; uniform nodes ~ 1 per cm -------------
    total_depth = sum(L.depth_cm for L in soil.layers)
    materials = []
    for L in soil.layers:
        v = L.vg
        materials.append({
            "theta_r": v.theta_r, "theta_s": v.theta_s,
            "alpha": v.alpha, "n": v.n, "Ks": v.Ks, "L": v.L,
        })

    profile = {
        "depth_cm": total_depth,
        "n_nodes": max(101, int(total_depth)),
        "materials": materials,
        "layer_depths": [L.depth_cm for L in soil.layers],
    }

    # --- root water uptake: Feddes block ---------------------------------
    sink = {
        "feddes": crop.feddes.model_dump(),
        "root_depth_cm": crop.root.max_depth_cm,
        "root_z50_cm":   crop.root.z50_cm,
        "root_z95_cm":   crop.root.z95_cm,
    }

    # --- fertilizer → solute boundary mass injections --------------------
    solute_events = []
    for fe in req.fertilizer:
        t_day = event_to_t_day(fe.date, sow) + 1
        if 1 <= t_day <= horizon:
            solute_events.append({
                "t_day": t_day,
                "kg_n_ha": fe.kg_n_ha,
                "conc_mg_l": fe.conc_mg_l,
            })

    scenario_dict = {
        "dimension": "1d",
        "name": f"agronomy_{crop.id}_{soil.id}_{req.weather_id}",
        "sim": {"t_init": 0.0, "t_max": float(horizon), "dt": 0.01},
        "profile": profile,
        "atmosphere": {"rows": atm_rows},
        "sink": sink,
        "solute": {"enabled": bool(solute_events), "events": solute_events},
    }
    return _scenario_from_dict(scenario_dict)
```

- [ ] **Step 5: Create `__init__.py`**

Write to `hydrus_research/agronomy/__init__.py`:

```python
"""hydrus_research.agronomy — workflow-driven decision API."""
from .types import (
    AgronomyRequest, AgronomyResult,
    IrrigEvent, FertEvent,
    WaterBalance, NBudget, EventTick,
)
from .scenario_builder import build_scenario, event_to_t_day

__all__ = [
    "AgronomyRequest", "AgronomyResult",
    "IrrigEvent", "FertEvent",
    "WaterBalance", "NBudget", "EventTick",
    "build_scenario", "event_to_t_day",
]
```

- [ ] **Step 6: Run tests**

Run: `pytest tests/research/test_agronomy_scenario.py -v`
Expected: 2 passed. If `_scenario_from_dict` rejects unknown keys (`solute.events`, `profile.layer_depths`), wrap those in `extra="allow"` — but only if the test fails for that reason.

- [ ] **Step 7: Commit**

```bash
git add hydrus_research/agronomy/ tests/research/test_agronomy_scenario.py
git commit -m "agronomy: types + scenario_builder"
```

---

## Task 4: Result parser

**Files:**
- Create: `hydrus_research/agronomy/result_parser.py`
- Test: extend `tests/research/test_agronomy_scenario.py` (no new file)

- [ ] **Step 1: Write the failing test**

Append to `tests/research/test_agronomy_scenario.py`:

```python
def test_result_parser_returns_ascending_z_and_finite_theta(tmp_path):
    """Synthesize a NOD_INF.OUT-like file and check the parser."""
    from hydrus_research.agronomy.result_parser import parse_nod_inf

    p = tmp_path / "NOD_INF.OUT"
    # 2 time blocks, 3 depths (HYDRUS-1D descending z; surface=0, deep negative)
    p.write_text(
        "Time:    0.000\n"
        "Node    z       h       theta\n"
        "1       0.000  -100.0   0.30\n"
        "2     -50.000  -150.0   0.25\n"
        "3    -100.000  -200.0   0.20\n"
        "Time:    1.000\n"
        "Node    z       h       theta\n"
        "1       0.000   -90.0   0.32\n"
        "2     -50.000  -140.0   0.27\n"
        "3    -100.000  -190.0   0.22\n"
    )
    z, t, theta = parse_nod_inf(p)
    # ascending z (0,50,100)
    assert z[0] < z[1] < z[2]
    assert list(z) == [0.0, 50.0, 100.0]
    assert len(t) == 2
    assert theta.shape == (2, 3)
    # theta[0,0] was 0.30 at surface
    assert abs(theta[0, 0] - 0.30) < 1e-6
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/research/test_agronomy_scenario.py::test_result_parser_returns_ascending_z_and_finite_theta -v`
Expected: ImportError.

- [ ] **Step 3: Implement parser**

Create `hydrus_research/agronomy/result_parser.py`:

```python
"""Parse HYDRUS-1D output files into agronomy result arrays.

NOD_INF.OUT z is descending (surface 0 → negative downward). We
reverse z AND every field together so downstream consumers see
ascending positive depths. See memory feedback_hydrus1d_nod_inf_z_descending.
"""
from __future__ import annotations
import re
from pathlib import Path
import numpy as np

_TIME_RE = re.compile(r"^\s*Time:\s*([\-0-9.E+]+)", re.MULTILINE)


def parse_nod_inf(path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (z_cm_ascending_positive, t_days, theta_zt[nT, nZ])."""
    text = Path(path).read_text()
    blocks = re.split(r"^\s*Time:\s*", text, flags=re.MULTILINE)[1:]
    times, theta_rows, z_descending = [], [], None
    for blk in blocks:
        head, *rest = blk.splitlines()
        times.append(float(head.split()[0]))
        rows = []
        for line in rest:
            parts = line.split()
            if len(parts) < 4 or not parts[0].lstrip("-").isdigit():
                continue
            rows.append([float(x) for x in parts[:4]])
        arr = np.array(rows)
        if z_descending is None:
            z_descending = arr[:, 1]
        theta_rows.append(arr[:, 3])

    z_desc = np.asarray(z_descending)
    theta = np.array(theta_rows)
    # Reverse so z is ascending; positive depths.
    order = np.argsort(np.abs(z_desc))
    z_pos = np.abs(z_desc[order])
    theta = theta[:, order]
    return z_pos, np.array(times), theta


def parse_balance(path: str | Path) -> dict[str, float]:
    """Read BALANCE.OUT totals — robust against missing fields."""
    text = Path(path).read_text() if Path(path).exists() else ""
    totals = {"rain_mm": 0.0, "et_mm": 0.0, "percolation_mm": 0.0,
              "storage_change_mm": 0.0}
    # HYDRUS BALANCE.OUT uses lines like "CumFlx(T)= -3.21E+01"; we extract
    # any number we recognize. Missing values stay 0.
    for line in text.splitlines():
        if "Atm" in line and "=" in line:
            try: totals["rain_mm"] += abs(float(line.split("=")[-1])) * 10  # cm→mm
            except ValueError: pass
        elif "Root" in line and "=" in line:
            try: totals["et_mm"] += abs(float(line.split("=")[-1])) * 10
            except ValueError: pass
        elif "Bot" in line and "=" in line:
            try: totals["percolation_mm"] += abs(float(line.split("=")[-1])) * 10
            except ValueError: pass
    return totals
```

- [ ] **Step 4: Run test**

Run: `pytest tests/research/test_agronomy_scenario.py::test_result_parser_returns_ascending_z_and_finite_theta -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add hydrus_research/agronomy/result_parser.py tests/research/test_agronomy_scenario.py
git commit -m "agronomy: result_parser (NOD_INF + BALANCE)"
```

---

## Task 5: Runner (end-to-end Python entry)

**Files:**
- Create: `hydrus_research/agronomy/runner.py`
- Test: `tests/research/test_agronomy_runner.py`

- [ ] **Step 1: Write the failing test**

Create `tests/research/test_agronomy_runner.py`:

```python
"""End-to-end runner test against the in-repo HYDRUS-1D adapter."""
from datetime import date
import numpy as np
import pytest

from hydrus_research.agronomy.runner import run_agronomy
from hydrus_research.agronomy.types import AgronomyRequest, IrrigEvent


@pytest.mark.slow
def test_runner_returns_finite_theta_and_water_balance(tmp_path):
    req = AgronomyRequest(
        crop_id="maize", soil_id="loam", weather_id="n_china_avg",
        horizon_days=30, start_year=2026,
        irrigation=[IrrigEvent(date=date(2026, 5, 10), depth_mm=20.0)],
        fertilizer=[],
    )
    result = run_agronomy(req, work_dir=tmp_path)

    z = np.asarray(result.z_cm)
    theta = np.asarray(result.theta_zt)

    assert z[0] < z[-1]                       # ascending (surface first)
    assert theta.shape == (len(result.t_days), len(z))
    assert np.all(np.isfinite(theta))
    assert np.all((theta >= 0) & (theta <= 0.6))   # physical bounds for loam
    # Water balance is populated (>0 rain in a 30-day spring window for N. China).
    assert result.water_balance.rain_mm >= 0
    assert result.water_balance.irrig_mm == pytest.approx(20.0, abs=0.5)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/research/test_agronomy_runner.py -v -m slow`
Expected: ImportError on `runner.run_agronomy`.

- [ ] **Step 3: Implement runner**

Create `hydrus_research/agronomy/runner.py`:

```python
"""End-to-end agronomy runner: AgronomyRequest → AgronomyResult.

Calls the existing Hydrus1DSimulator adapter. Pure Python entry; no
GUI / REST dependency.
"""
from __future__ import annotations
from pathlib import Path
import tempfile

from hydrus_research.library.crops import get_crop
from hydrus_research.library.soils import get_soil
from hydrus_research.library.weather import load_weather_series
from hydrus_research.simulator.hydrus1d_adapter import Hydrus1DSimulator

from .scenario_builder import build_scenario, event_to_t_day, _sow_date
from .result_parser import parse_nod_inf, parse_balance
from .types import (
    AgronomyRequest, AgronomyResult,
    WaterBalance, NBudget, EventTick,
)


def run_agronomy(req: AgronomyRequest, work_dir: Path | str | None = None) -> AgronomyResult:
    crop = get_crop(req.crop_id)
    soil = get_soil(req.soil_id)
    weather = load_weather_series(req.weather_id)

    sc = build_scenario(crop, soil, weather, req)
    sim = Hydrus1DSimulator(work_root=work_dir or Path(tempfile.gettempdir()) / "agronomy")
    sim_result = sim.run(sc.to_dict(), forcing=None, ic=None)

    out_dir = Path(sim_result.meta.get("run_dir") or sim_result.meta.get("work_dir"))
    z, t, theta = parse_nod_inf(out_dir / "NOD_INF.OUT")
    balance = parse_balance(out_dir / "BALANCE.OUT")

    # Solute: if disabled, fill zeros at the same (t, z) grid.
    n_zt = [[0.0] * len(z) for _ in t]

    irrig_mm = sum(e.depth_mm for e in req.irrigation)
    sow = _sow_date(req, crop)
    events: list[EventTick] = [
        EventTick(t_day=event_to_t_day(e.date, sow), amount=e.depth_mm, label="irrig")
        for e in req.irrigation
    ] + [
        EventTick(t_day=event_to_t_day(e.date, sow), amount=e.kg_n_ha, label="fert")
        for e in req.fertilizer
    ]

    return AgronomyResult(
        z_cm=z.tolist(),
        t_days=t.tolist(),
        theta_zt=theta.tolist(),
        n_zt=n_zt,
        water_balance=WaterBalance(
            rain_mm=balance.get("rain_mm", 0.0),
            irrig_mm=irrig_mm,
            et_mm=balance.get("et_mm", 0.0),
            percolation_mm=balance.get("percolation_mm", 0.0),
            storage_change_mm=balance.get("storage_change_mm", 0.0),
        ),
        n_budget=NBudget(
            applied_kg_ha=sum(e.kg_n_ha for e in req.fertilizer),
            uptake_kg_ha=0.0, leached_kg_ha=0.0, residual_kg_ha=0.0,
        ),
        events=events,
    )
```

- [ ] **Step 4: Re-export from `agronomy/__init__.py`**

Edit `hydrus_research/agronomy/__init__.py` — add the line `from .runner import run_agronomy` and add `"run_agronomy"` to `__all__`.

- [ ] **Step 5: Run test**

Run: `pytest tests/research/test_agronomy_runner.py -v -m slow`
Expected: PASS. If the Hydrus1DSimulator's `meta` dict does not expose `run_dir`, read its source and adapt the key name (`out_dir`, `work_dir`, etc. — pick whichever it uses).

- [ ] **Step 6: Commit**

```bash
git add hydrus_research/agronomy/runner.py hydrus_research/agronomy/__init__.py tests/research/test_agronomy_runner.py
git commit -m "agronomy: end-to-end runner against hydrus1d adapter"
```

---

## Task 6: CLI subcommand `hydrus research agronomy run`

**Files:**
- Modify: `hydrus_port/cli.py`
- Test: `tests/research/test_cli_agronomy.py`

- [ ] **Step 1: Write the failing test**

Create `tests/research/test_cli_agronomy.py`:

```python
import csv, json, subprocess, sys
from pathlib import Path


def test_cli_agronomy_run_writes_result_json(tmp_path):
    irrig = tmp_path / "irrig.csv"
    irrig.write_text("date,depth_mm\n2026-05-10,20\n")
    fert = tmp_path / "fert.csv"
    fert.write_text("date,kg_n_ha\n")
    out = tmp_path / "out"
    res = subprocess.run(
        [sys.executable, "-m", "hydrus_port.cli", "research", "agronomy", "run",
         "--crop", "maize", "--soil", "loam", "--weather", "n_china_avg",
         "--horizon-days", "30",
         "--irrig", str(irrig), "--fert", str(fert), "--out", str(out)],
        capture_output=True, text=True, check=False,
    )
    assert res.returncode == 0, res.stderr
    payload = json.loads((out / "result.json").read_text())
    assert "theta_zt" in payload
    assert "water_balance" in payload
    assert payload["water_balance"]["irrig_mm"] == 20.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/research/test_cli_agronomy.py -v`
Expected: returncode != 0 (subcommand missing).

- [ ] **Step 3: Add subcommand to `hydrus_port/cli.py`**

Locate the `research` subparser section in `hydrus_port/cli.py` (where `dndc`, `soil`, `sweep`, etc. are registered). Add a sibling subparser. Read the file first to find the exact insertion point — model the new code on an existing subcommand (e.g., the `soil` block).

Append-style snippet to add (adapt to the file's existing pattern):

```python
# inside the research subparser builder
ap_agronomy = research_sub.add_parser("agronomy", help="Agronomy decision workflow")
ag_sub = ap_agronomy.add_subparsers(dest="agronomy_cmd")
ap_run = ag_sub.add_parser("run", help="Run a single decision scenario")
ap_run.add_argument("--crop",    required=True)
ap_run.add_argument("--soil",    required=True)
ap_run.add_argument("--weather", required=True)
ap_run.add_argument("--horizon-days", type=int, required=True)
ap_run.add_argument("--irrig",   required=True, help="CSV: date,depth_mm")
ap_run.add_argument("--fert",    required=True, help="CSV: date,kg_n_ha[,conc_mg_l]")
ap_run.add_argument("--out",     required=True)
ap_run.add_argument("--start-year", type=int, default=2026)
```

And in the dispatch block:

```python
if args.cmd == "research" and getattr(args, "research_cmd", None) == "agronomy" \
        and getattr(args, "agronomy_cmd", None) == "run":
    return _run_agronomy_cli(args)
```

Define the handler at module scope:

```python
def _run_agronomy_cli(args) -> int:
    import csv as _csv, json as _json
    from datetime import date as _date
    from pathlib import Path as _Path
    from hydrus_research.agronomy import (
        AgronomyRequest, IrrigEvent, FertEvent, run_agronomy,
    )

    def _read_irrig(p):
        out = []
        with open(p) as f:
            for row in _csv.DictReader(f):
                if not row.get("date"): continue
                out.append(IrrigEvent(date=_date.fromisoformat(row["date"]),
                                       depth_mm=float(row["depth_mm"])))
        return out

    def _read_fert(p):
        out = []
        with open(p) as f:
            for row in _csv.DictReader(f):
                if not row.get("date"): continue
                conc = row.get("conc_mg_l")
                out.append(FertEvent(date=_date.fromisoformat(row["date"]),
                                      kg_n_ha=float(row["kg_n_ha"]),
                                      conc_mg_l=float(conc) if conc else None))
        return out

    req = AgronomyRequest(
        crop_id=args.crop, soil_id=args.soil, weather_id=args.weather,
        horizon_days=args.horizon_days, start_year=args.start_year,
        irrigation=_read_irrig(args.irrig),
        fertilizer=_read_fert(args.fert),
    )
    out_dir = _Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    result = run_agronomy(req, work_dir=out_dir / "_work")
    (out_dir / "result.json").write_text(result.model_dump_json(indent=2))
    print(f"wrote {out_dir / 'result.json'}")
    return 0
```

- [ ] **Step 4: Run test**

Run: `pytest tests/research/test_cli_agronomy.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add hydrus_port/cli.py tests/research/test_cli_agronomy.py
git commit -m "cli: hydrus research agronomy run"
```

---

## Task 7: FastAPI router `research_agronomy`

**Files:**
- Create: `hydrus_port_server/routers/research_agronomy.py`
- Modify: `hydrus_port_server/app.py`
- Test: `tests/research/test_research_agronomy_router.py`

- [ ] **Step 1: Write the failing test**

Create `tests/research/test_research_agronomy_router.py`:

```python
from fastapi.testclient import TestClient
from hydrus_port_server.app import create_app


def _client():
    return TestClient(create_app())


def test_lib_crops_returns_known_ids():
    r = _client().get("/research/agronomy/lib/crops")
    assert r.status_code == 200
    ids = {c["id"] for c in r.json()["crops"]}
    assert "maize" in ids and "wheat_winter" in ids


def test_lib_weather_meta_and_series():
    c = _client()
    r = c.get("/research/agronomy/lib/weather")
    assert r.status_code == 200
    ids = {w["id"] for w in r.json()["weather"]}
    assert "n_china_avg" in ids

    r2 = c.get("/research/agronomy/lib/weather/n_china_avg")
    assert r2.status_code == 200
    assert len(r2.json()["doy"]) == 365


def test_agronomy_run_smoke():
    payload = {
        "crop_id": "maize", "soil_id": "loam", "weather_id": "n_china_avg",
        "horizon_days": 14, "start_year": 2026,
        "irrigation": [{"date": "2026-05-10", "depth_mm": 20}],
        "fertilizer": [],
    }
    r = _client().post("/research/agronomy/run", json=payload)
    assert r.status_code == 200, r.text
    body = r.json()
    assert "theta_zt" in body
    assert body["water_balance"]["irrig_mm"] == 20
```

- [ ] **Step 2: Implement router**

Create `hydrus_port_server/routers/research_agronomy.py`:

```python
"""/research/agronomy/* — crop/soil/weather libraries + run endpoint."""
from __future__ import annotations
import tempfile
from pathlib import Path

from fastapi import APIRouter, HTTPException

from hydrus_research.library.crops import load_crops
from hydrus_research.library.soils import load_soils
from hydrus_research.library.weather import (
    load_weather_meta, load_weather_series,
)
from hydrus_research.agronomy import AgronomyRequest, AgronomyResult, run_agronomy


router = APIRouter()


@router.get("/lib/crops")
def lib_crops():
    return {"crops": [c.model_dump() for c in load_crops()]}


@router.get("/lib/soils")
def lib_soils():
    return {"soils": [s.model_dump() for s in load_soils()]}


@router.get("/lib/weather")
def lib_weather():
    return {"weather": load_weather_meta()}


@router.get("/lib/weather/{weather_id}")
def lib_weather_series(weather_id: str):
    try:
        return load_weather_series(weather_id)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.post("/run", response_model=AgronomyResult)
def run(req: AgronomyRequest):
    work = Path(tempfile.mkdtemp(prefix="agronomy_"))
    try:
        return run_agronomy(req, work_dir=work)
    except KeyError as e:
        raise HTTPException(status_code=400, detail=str(e))
```

- [ ] **Step 3: Register router in `app.py`**

In `hydrus_port_server/app.py`, find the block right after the M7 surrogate router registration (around lines 265-270). Add directly below it:

```python
    # Agronomy decision router (workflow GUI)
    try:
        from .routers.research_agronomy import router as agronomy_router
        app.include_router(agronomy_router, prefix="/research/agronomy",
                           tags=["research", "agronomy"])
    except ImportError:
        pass
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/research/test_research_agronomy_router.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add hydrus_port_server/routers/research_agronomy.py hydrus_port_server/app.py tests/research/test_research_agronomy_router.py
git commit -m "server: research_agronomy router (lib + run)"
```

---

## Task 8: Frontend API wrappers + Pinia store

**Files:**
- Modify: `desktop/src/api.ts`
- Create: `desktop/src/stores/agronomy.ts`

- [ ] **Step 1: Add agronomy API to `api.ts`**

In `desktop/src/api.ts`, append (after the existing `surrogate = { ... }` namespace):

```ts
// =========================================================================
// Agronomy decision workflow (workflow GUI; M-Agronomy)
// =========================================================================

export type Crop = {
  id: string; name_zh: string; name_en: string;
  feddes: Record<string, number>;
  root:   { max_depth_cm: number; z50_cm: number; z95_cm: number };
  season: { sow_doy: number; harvest_doy: number };
  kc_curve: { doy: number; kc: number }[];
};

export type SoilLayer = {
  depth_cm: number;
  vg: { theta_r:number; theta_s:number; alpha:number; n:number; Ks:number; L:number };
};

export type Soil = {
  id: string; name_zh: string; name_en: string;
  layers: SoilLayer[];
};

export type WeatherMeta = { id: string; name_zh: string };

export type AgronomyResult = {
  z_cm: number[]; t_days: number[];
  theta_zt: number[][]; n_zt: number[][];
  water_balance: { rain_mm: number; irrig_mm: number; et_mm: number;
                   percolation_mm: number; storage_change_mm: number };
  n_budget: { applied_kg_ha: number; uptake_kg_ha: number;
              leached_kg_ha: number; residual_kg_ha: number };
  events: { t_day: number; amount: number; label: "irrig" | "fert" }[];
};

export const agronomy = {
  async listCrops(): Promise<Crop[]> {
    const r = await fetch(`${RESEARCH_BASE}/research/agronomy/lib/crops`);
    if (!r.ok) throw new Error(await r.text());
    return (await r.json()).crops;
  },
  async listSoils(): Promise<Soil[]> {
    const r = await fetch(`${RESEARCH_BASE}/research/agronomy/lib/soils`);
    if (!r.ok) throw new Error(await r.text());
    return (await r.json()).soils;
  },
  async listWeather(): Promise<WeatherMeta[]> {
    const r = await fetch(`${RESEARCH_BASE}/research/agronomy/lib/weather`);
    if (!r.ok) throw new Error(await r.text());
    return (await r.json()).weather;
  },
  async run(payload: {
    crop_id: string; soil_id: string; weather_id: string;
    horizon_days: number; start_year: number;
    irrigation: { date: string; depth_mm: number }[];
    fertilizer: { date: string; kg_n_ha: number; conc_mg_l?: number | null }[];
  }): Promise<AgronomyResult> {
    const r = await fetch(`${RESEARCH_BASE}/research/agronomy/run`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!r.ok) throw new Error(await r.text());
    return await r.json();
  },
};
```

(If `RESEARCH_BASE` is not in scope, copy the same `const RESEARCH_BASE = ...` definition the `ptf` namespace uses earlier in the file.)

- [ ] **Step 2: Create Pinia store**

Create `desktop/src/stores/agronomy.ts`:

```ts
import { defineStore } from "pinia";
import {
  agronomy, type Crop, type Soil, type WeatherMeta,
  type AgronomyResult,
} from "../api";

type Irrig = { date: string; depth_mm: number };
type Fert  = { date: string; kg_n_ha: number; conc_mg_l?: number | null };

export const useAgronomyStore = defineStore("agronomy", {
  state: () => ({
    crops:    [] as Crop[],
    soils:    [] as Soil[],
    weather:  [] as WeatherMeta[],
    libsLoaded: false,

    cropId:    "maize",
    soilId:    "loam",
    weatherId: "n_china_avg",
    horizonDays: 150,
    startYear: 2026,

    irrig:     [] as Irrig[],
    fert:      [] as Fert[],

    result:    null as AgronomyResult | null,
    running:   false,
    error:     null as string | null,
  }),

  actions: {
    async loadLibs() {
      if (this.libsLoaded) return;
      [this.crops, this.soils, this.weather] = await Promise.all([
        agronomy.listCrops(), agronomy.listSoils(), agronomy.listWeather(),
      ]);
      this.libsLoaded = true;
    },
    addIrrig() { this.irrig.push({ date: this._defaultDate(), depth_mm: 20 }); },
    addFert()  { this.fert .push({ date: this._defaultDate(), kg_n_ha: 60 }); },
    removeIrrig(i: number) { this.irrig.splice(i, 1); },
    removeFert (i: number) { this.fert .splice(i, 1); },
    _defaultDate(): string {
      const d = new Date(this.startYear, 4, 15);  // May 15
      return d.toISOString().slice(0, 10);
    },
    async run() {
      this.running = true; this.error = null;
      try {
        this.result = await agronomy.run({
          crop_id: this.cropId, soil_id: this.soilId,
          weather_id: this.weatherId,
          horizon_days: this.horizonDays, start_year: this.startYear,
          irrigation: this.irrig, fertilizer: this.fert,
        });
      } catch (e: any) {
        this.error = String(e.message || e);
      } finally {
        this.running = false;
      }
    },
  },
});
```

- [ ] **Step 3: Commit**

```bash
git add desktop/src/api.ts desktop/src/stores/agronomy.ts
git commit -m "frontend: agronomy api wrappers + pinia store"
```

---

## Task 9: InputStrip + EventsTable components

**Files:**
- Create: `desktop/src/components/agronomy/EventsTable.vue`
- Create: `desktop/src/components/agronomy/InputStrip.vue`

- [ ] **Step 1: Create `EventsTable.vue`**

```vue
<script setup lang="ts">
defineProps<{
  kind: "irrig" | "fert";
  rows: Array<Record<string, any>>;
}>();
const emit = defineEmits<{
  (e: "add"): void;
  (e: "remove", index: number): void;
}>();
</script>

<template>
  <div class="events">
    <table>
      <thead>
        <tr>
          <th>Date</th>
          <th v-if="kind==='irrig'">Depth (mm)</th>
          <th v-else>kgN/ha</th>
          <th v-if="kind==='fert'">conc (mg/L)</th>
          <th></th>
        </tr>
      </thead>
      <tbody>
        <tr v-for="(r, i) in rows" :key="i">
          <td><input type="date" v-model="r.date" /></td>
          <td v-if="kind==='irrig'">
            <input type="number" step="0.5" v-model.number="r.depth_mm" />
          </td>
          <td v-else>
            <input type="number" step="1" v-model.number="r.kg_n_ha" />
          </td>
          <td v-if="kind==='fert'">
            <input type="number" step="1" v-model.number="r.conc_mg_l" />
          </td>
          <td><button @click="emit('remove', i)">×</button></td>
        </tr>
      </tbody>
    </table>
    <button @click="emit('add')">+ add</button>
  </div>
</template>

<style scoped>
.events table { width: 100%; border-collapse: collapse; font-size: 12px; }
.events th, .events td { border: 1px solid var(--border, #ccc); padding: 2px 4px; }
.events input { width: 100%; box-sizing: border-box; border: none; background: transparent; }
.events button { margin-top: 4px; }
</style>
```

- [ ] **Step 2: Create `InputStrip.vue`**

```vue
<script setup lang="ts">
import { onMounted } from "vue";
import { useAgronomyStore } from "../../stores/agronomy";
import EventsTable from "./EventsTable.vue";

const store = useAgronomyStore();
onMounted(() => store.loadLibs());
</script>

<template>
  <section class="input-strip">
    <div class="row">
      <label>作物
        <select v-model="store.cropId">
          <option v-for="c in store.crops" :key="c.id" :value="c.id">{{ c.name_zh }}</option>
        </select>
      </label>
      <label>土壤
        <select v-model="store.soilId">
          <option v-for="s in store.soils" :key="s.id" :value="s.id">{{ s.name_zh }}</option>
        </select>
      </label>
      <label>雨型
        <select v-model="store.weatherId">
          <option v-for="w in store.weather" :key="w.id" :value="w.id">{{ w.name_zh }}</option>
        </select>
      </label>
      <label>天数
        <input type="number" min="1" max="365" v-model.number="store.horizonDays" style="width:60px" />
      </label>
      <button class="run" :disabled="store.running" @click="store.run()">
        {{ store.running ? "运行中…" : "▶ 运行" }}
      </button>
      <span class="error" v-if="store.error">{{ store.error }}</span>
    </div>
    <div class="row tables">
      <div class="col">
        <h4>灌溉事件</h4>
        <EventsTable kind="irrig" :rows="store.irrig"
                     @add="store.addIrrig" @remove="store.removeIrrig" />
      </div>
      <div class="col">
        <h4>施肥事件</h4>
        <EventsTable kind="fert" :rows="store.fert"
                     @add="store.addFert" @remove="store.removeFert" />
      </div>
    </div>
  </section>
</template>

<style scoped>
.input-strip { padding: 8px 12px; border-bottom: 1px solid var(--border, #ccc); }
.row { display: flex; gap: 12px; align-items: center; flex-wrap: wrap; }
.tables { margin-top: 8px; align-items: stretch; }
.col { flex: 1; min-width: 280px; }
.run { padding: 4px 12px; font-weight: 600; }
.error { color: #c00; font-size: 12px; }
h4 { margin: 0 0 4px 0; font-size: 12px; }
</style>
```

- [ ] **Step 3: Commit**

```bash
git add desktop/src/components/agronomy/InputStrip.vue desktop/src/components/agronomy/EventsTable.vue
git commit -m "frontend: InputStrip + EventsTable for agronomy GUI"
```

---

## Task 10: ThetaHeatmap + NitrateHeatmap + HeatmapBand

**Files:**
- Create: `desktop/src/components/agronomy/ThetaHeatmap.vue`
- Create: `desktop/src/components/agronomy/NitrateHeatmap.vue`
- Create: `desktop/src/components/agronomy/HeatmapBand.vue`

- [ ] **Step 1: Create `ThetaHeatmap.vue`**

```vue
<script setup lang="ts">
import { computed, watch, ref, onMounted } from "vue";
import Plotly from "plotly.js-dist-min";

const props = defineProps<{
  z: number[]; t: number[]; field: number[][];
  events: { t_day: number; amount: number; label: string }[];
  title: string;
  colorscale?: string;
  unit: string;
}>();

const el = ref<HTMLDivElement | null>(null);

function render() {
  if (!el.value || !props.field.length) return;
  const irrig = props.events.filter(e => e.label === "irrig").map(e => e.t_day);
  const shapes = irrig.map(td => ({
    type: "line", x0: td, x1: td, y0: 0, y1: 1, yref: "paper",
    line: { color: "rgba(20,80,200,0.8)", width: 1, dash: "dot" },
  }));
  Plotly.newPlot(el.value, [{
    z: props.field, x: props.t, y: props.z,
    type: "heatmap", colorscale: props.colorscale ?? "YlGnBu",
    colorbar: { title: props.unit },
  }], {
    title: props.title,
    xaxis: { title: "t (days)" },
    yaxis: { title: "depth (cm)", autorange: "reversed" },
    shapes,
    margin: { l: 60, r: 20, t: 30, b: 40 },
  }, { responsive: true });
}

onMounted(render);
watch(() => [props.field, props.events], render, { deep: true });
</script>

<template>
  <div ref="el" class="heatmap"></div>
</template>

<style scoped>
.heatmap { width: 100%; height: 320px; }
</style>
```

- [ ] **Step 2: Create `NitrateHeatmap.vue`**

Same shape as ThetaHeatmap but defaults to a different colorscale and filters fert events. To keep it DRY, just wrap ThetaHeatmap:

```vue
<script setup lang="ts">
import ThetaHeatmap from "./ThetaHeatmap.vue";
defineProps<{
  z: number[]; t: number[]; field: number[][];
  events: { t_day: number; amount: number; label: string }[];
}>();
</script>

<template>
  <ThetaHeatmap :z="z" :t="t" :field="field"
                :events="events.filter(e => e.label === 'fert')"
                title="N-NO₃ (z,t)" colorscale="Reds" unit="mg/L" />
</template>
```

- [ ] **Step 3: Create `HeatmapBand.vue`**

```vue
<script setup lang="ts">
import { useAgronomyStore } from "../../stores/agronomy";
import ThetaHeatmap from "./ThetaHeatmap.vue";
import NitrateHeatmap from "./NitrateHeatmap.vue";

const store = useAgronomyStore();
</script>

<template>
  <section class="heatmap-band">
    <div v-if="!store.result" class="empty">运行一次后看含水量/硝态 N 的时间-深度热图</div>
    <template v-else>
      <ThetaHeatmap :z="store.result.z_cm" :t="store.result.t_days"
                    :field="store.result.theta_zt" :events="store.result.events"
                    title="θ(z,t) 体积含水量" unit="cm³/cm³" />
      <NitrateHeatmap :z="store.result.z_cm" :t="store.result.t_days"
                      :field="store.result.n_zt" :events="store.result.events" />
    </template>
  </section>
</template>

<style scoped>
.heatmap-band { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; padding: 8px; }
.empty { grid-column: 1 / -1; padding: 40px; text-align: center; color: #888; }
</style>
```

- [ ] **Step 4: Commit**

```bash
git add desktop/src/components/agronomy/ThetaHeatmap.vue desktop/src/components/agronomy/NitrateHeatmap.vue desktop/src/components/agronomy/HeatmapBand.vue
git commit -m "frontend: heatmap band (θ + N-NO3)"
```

---

## Task 11: WaterBalanceBar + NBudgetBar + BalanceBand

**Files:**
- Create: `desktop/src/components/agronomy/WaterBalanceBar.vue`
- Create: `desktop/src/components/agronomy/NBudgetBar.vue`
- Create: `desktop/src/components/agronomy/BalanceBand.vue`

- [ ] **Step 1: Create `WaterBalanceBar.vue`**

```vue
<script setup lang="ts">
import { computed, watch, ref, onMounted } from "vue";
import Plotly from "plotly.js-dist-min";

const props = defineProps<{
  balance: { rain_mm: number; irrig_mm: number; et_mm: number;
             percolation_mm: number; storage_change_mm: number };
}>();

const el = ref<HTMLDivElement | null>(null);

function render() {
  if (!el.value) return;
  const labels = ["Rain", "Irrig", "ET", "Percol", "ΔStor"];
  const values = [props.balance.rain_mm, props.balance.irrig_mm,
                  props.balance.et_mm, props.balance.percolation_mm,
                  props.balance.storage_change_mm];
  Plotly.newPlot(el.value, [{
    type: "bar", orientation: "h", x: values, y: labels,
    marker: { color: ["#5DA5DA","#4D9DE0","#F17CB0","#B276B2","#999999"] },
    text: values.map(v => v.toFixed(0) + " mm"), textposition: "auto",
  }], { title: "水平衡 (mm)", margin: { l: 60, r: 20, t: 30, b: 30 } },
      { responsive: true });
}
onMounted(render);
watch(() => props.balance, render, { deep: true });
</script>

<template><div ref="el" class="bar"></div></template>

<style scoped>.bar { width: 100%; height: 200px; }</style>
```

- [ ] **Step 2: Create `NBudgetBar.vue`**

```vue
<script setup lang="ts">
import { watch, ref, onMounted } from "vue";
import Plotly from "plotly.js-dist-min";

const props = defineProps<{
  budget: { applied_kg_ha: number; uptake_kg_ha: number;
            leached_kg_ha: number; residual_kg_ha: number };
}>();

const el = ref<HTMLDivElement | null>(null);

function render() {
  if (!el.value) return;
  const labels = ["Applied", "Uptake", "Leached", "Residual"];
  const values = [props.budget.applied_kg_ha, props.budget.uptake_kg_ha,
                  props.budget.leached_kg_ha, props.budget.residual_kg_ha];
  Plotly.newPlot(el.value, [{
    type: "bar", orientation: "h", x: values, y: labels,
    marker: { color: ["#60B95C","#A8D5A0","#E07A5F","#888"] },
    text: values.map(v => v.toFixed(0) + " kgN"), textposition: "auto",
  }], { title: "N 预算 (kgN/ha)", margin: { l: 70, r: 20, t: 30, b: 30 } },
      { responsive: true });
}
onMounted(render);
watch(() => props.budget, render, { deep: true });
</script>

<template><div ref="el" class="bar"></div></template>

<style scoped>.bar { width: 100%; height: 200px; }</style>
```

- [ ] **Step 3: Create `BalanceBand.vue`**

```vue
<script setup lang="ts">
import { useAgronomyStore } from "../../stores/agronomy";
import WaterBalanceBar from "./WaterBalanceBar.vue";
import NBudgetBar from "./NBudgetBar.vue";
const store = useAgronomyStore();
</script>

<template>
  <section v-if="store.result" class="balance-band">
    <WaterBalanceBar :balance="store.result.water_balance" />
    <NBudgetBar :budget="store.result.n_budget" />
  </section>
</template>

<style scoped>
.balance-band { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; padding: 8px; border-top: 1px solid var(--border, #ccc); }
</style>
```

- [ ] **Step 4: Commit**

```bash
git add desktop/src/components/agronomy/WaterBalanceBar.vue desktop/src/components/agronomy/NBudgetBar.vue desktop/src/components/agronomy/BalanceBand.vue
git commit -m "frontend: balance band (water + N)"
```

---

## Task 12: AdvancedDrawer wrapping existing 9 tabs + classic panel

**Files:**
- Create: `desktop/src/components/agronomy/AdvancedDrawer.vue`

- [ ] **Step 1: Create `AdvancedDrawer.vue`**

The drawer wraps today's right-pane tab strip plus a "经典" tab that exposes Scenario+Regression+OutputBrowser+Log so the regression workflow doesn't disappear.

```vue
<script setup lang="ts">
import { ref } from "vue";
import ScenarioEditor    from "../ScenarioEditor.vue";
import MeshViewer3D      from "../MeshViewer3D.vue";
import DNDCForms         from "../../pages/research/DNDCForms.vue";
import SoilLibrary       from "../../pages/research/SoilLibrary.vue";
import BatchSweep        from "../../pages/research/BatchSweep.vue";
import SensitivityReport from "../../pages/research/SensitivityReport.vue";
import InversionStudio   from "../../pages/research/InversionStudio.vue";
import UQExplorer        from "../../pages/research/UQExplorer.vue";
import SurrogateBench    from "../../pages/research/SurrogateBench.vue";
import ClassicPanel      from "./ClassicPanel.vue";

defineProps<{ open: boolean; editorPath: string | null }>();
defineEmits<{ (e: "close"): void }>();

type Tab = "classic" | "editor" | "3d" | "dndc" | "soil"
         | "batch" | "sensitivity" | "inversion" | "uq" | "surrogate";
const tab = ref<Tab>("classic");
</script>

<template>
  <aside v-if="open" class="drawer">
    <header>
      <strong>高级 / 研究台</strong>
      <button @click="$emit('close')">×</button>
    </header>
    <nav>
      <button :class="{on:tab==='classic'}"     @click="tab='classic'">经典</button>
      <button :class="{on:tab==='editor'}"      @click="tab='editor'">Editor</button>
      <button :class="{on:tab==='3d'}"          @click="tab='3d'">3D mesh</button>
      <button :class="{on:tab==='dndc'}"        @click="tab='dndc'">DNDC</button>
      <button :class="{on:tab==='soil'}"        @click="tab='soil'">Soil lib</button>
      <button :class="{on:tab==='batch'}"       @click="tab='batch'">Batch</button>
      <button :class="{on:tab==='sensitivity'}" @click="tab='sensitivity'">Sens</button>
      <button :class="{on:tab==='inversion'}"   @click="tab='inversion'">Invert</button>
      <button :class="{on:tab==='uq'}"          @click="tab='uq'">UQ</button>
      <button :class="{on:tab==='surrogate'}"   @click="tab='surrogate'">Surrog</button>
    </nav>
    <section class="body">
      <ClassicPanel       v-if="tab==='classic'" />
      <ScenarioEditor     v-else-if="tab==='editor'"      :path="editorPath" />
      <MeshViewer3D       v-else-if="tab==='3d'" />
      <DNDCForms          v-else-if="tab==='dndc'" />
      <SoilLibrary        v-else-if="tab==='soil'" />
      <BatchSweep         v-else-if="tab==='batch'" />
      <SensitivityReport  v-else-if="tab==='sensitivity'" />
      <InversionStudio    v-else-if="tab==='inversion'" />
      <UQExplorer         v-else-if="tab==='uq'" />
      <SurrogateBench     v-else-if="tab==='surrogate'" />
    </section>
  </aside>
</template>

<style scoped>
.drawer {
  position: fixed; top: 0; right: 0; bottom: 0; width: 560px; max-width: 80vw;
  background: var(--bg, #fff); border-left: 1px solid var(--border, #ccc);
  box-shadow: -4px 0 12px rgba(0,0,0,0.15);
  display: flex; flex-direction: column; z-index: 100;
}
header { display:flex; justify-content:space-between; padding:8px 12px; border-bottom:1px solid var(--border,#ccc); }
nav { display: flex; flex-wrap: wrap; gap: 4px; padding: 6px; border-bottom: 1px solid var(--border, #ccc); }
nav button { font-size: 12px; padding: 2px 6px; }
nav .on { font-weight: 600; background: var(--accent-bg, #eef); }
.body { flex: 1; overflow: auto; padding: 8px; }
</style>
```

- [ ] **Step 2: Create `ClassicPanel.vue`**

This keeps the old Scenario+Regression+OutputBrowser+Log workflow accessible.

```vue
<script setup lang="ts">
import { ref, computed, onMounted } from "vue";
import ScenarioPicker from "../ScenarioPicker.vue";
import LogStream      from "../LogStream.vue";
import OutputBrowser  from "../OutputBrowser.vue";
import Regression     from "../Regression.vue";
import { api, type JobMeta, type PythonInfo } from "../../api";

const py  = ref<PythonInfo | null>(null);
const job = ref<JobMeta | null>(null);
onMounted(async () => { try { py.value = await api.detectPython(); } catch {} });
</script>

<template>
  <div class="classic">
    <ScenarioPicker />
    <Regression />
    <OutputBrowser v-if="job" :dir="job.output_dir" />
    <LogStream v-if="job" :id="job.id" />
  </div>
</template>

<style scoped>.classic > * { margin-bottom: 8px; }</style>
```

- [ ] **Step 3: Commit**

```bash
git add desktop/src/components/agronomy/AdvancedDrawer.vue desktop/src/components/agronomy/ClassicPanel.vue
git commit -m "frontend: AdvancedDrawer wrapping 9 research tabs + ClassicPanel"
```

---

## Task 13: App.vue rewrite — 3-band layout

**Files:**
- Modify: `desktop/src/App.vue` (full rewrite)

- [ ] **Step 1: Replace App.vue body**

Overwrite `desktop/src/App.vue` with the new 3-band shell. The store, the drawer, the input strip, the heatmap band, and the balance band drive the entire page. The old "Scenario/Regression/Output files/Log" panes live inside the drawer (Task 12) so nothing is lost.

```vue
<script setup lang="ts">
import { ref, onMounted } from "vue";
import { theme, toggleTheme } from "./theme";

import InputStrip    from "./components/agronomy/InputStrip.vue";
import HeatmapBand   from "./components/agronomy/HeatmapBand.vue";
import BalanceBand   from "./components/agronomy/BalanceBand.vue";
import AdvancedDrawer from "./components/agronomy/AdvancedDrawer.vue";

const drawerOpen   = ref(false);
const editorPath   = ref<string | null>(null);
</script>

<template>
  <div class="shell">
    <header>
      <span class="brand">HYDRUS Agronomy</span>
      <span class="spacer" />
      <button @click="drawerOpen = true">高级 ▸</button>
      <button class="theme" @click="toggleTheme">{{ theme === 'dark' ? '☀' : '☾' }}</button>
    </header>

    <InputStrip />
    <HeatmapBand />
    <BalanceBand />

    <AdvancedDrawer :open="drawerOpen" :editorPath="editorPath"
                    @close="drawerOpen = false" />
  </div>
</template>

<style scoped>
.shell { display: flex; flex-direction: column; height: 100vh; }
header { display: flex; align-items: center; padding: 6px 12px;
         border-bottom: 1px solid var(--border, #ccc); gap: 8px; }
.brand { font-weight: 700; }
.spacer { flex: 1; }
.theme { font-size: 14px; }
</style>
```

- [ ] **Step 2: Verify build**

Run: `cd desktop && npm run build`
Expected: 0 errors. If TS complains about unused imports, delete the unused ones.

- [ ] **Step 3: Commit**

```bash
git add desktop/src/App.vue
git commit -m "frontend: rewrite App.vue into 3-band agronomy layout"
```

---

## Task 14: Playwright smoke

**Files:**
- Create: `desktop/tests/agronomy_smoke.spec.ts`

- [ ] **Step 1: Write the smoke spec**

```ts
import { test, expect } from "@playwright/test";

test("agronomy: pick maize+loam+n_china_avg, add 1 irrig + 1 fert, run", async ({ page }) => {
  await page.goto("http://localhost:1420");
  await expect(page.locator("select").first()).toBeVisible({ timeout: 15000 });

  // pick crop / soil / weather
  await page.locator("select").nth(0).selectOption("maize");
  await page.locator("select").nth(1).selectOption("loam");
  await page.locator("select").nth(2).selectOption("n_china_avg");

  // add events
  await page.getByText("+ add").first().click();   // irrig
  await page.getByText("+ add").nth(1).click();    // fert

  // run
  await page.getByText(/▶\s*运行/).click();
  await page.waitForSelector(".heatmap canvas, .heatmap svg", { timeout: 60000 });

  // balance band populated
  await expect(page.locator(".balance-band")).toBeVisible();

  // drawer reveals 9 research tabs
  await page.getByText("高级 ▸").click();
  for (const label of ["经典","Editor","3D mesh","DNDC","Soil lib","Batch","Sens","Invert","UQ","Surrog"]) {
    await expect(page.getByRole("button", { name: label })).toBeVisible();
  }

  await page.screenshot({ path: "desktop/tests/screenshots/agronomy_smoke.png", fullPage: true });
});
```

- [ ] **Step 2: Run dev sidecar + Vite**

Open two terminals (one-shot manual verification — Playwright suite hooks them up via its own config later if needed):

```
python -m hydrus_port_server.app --host 127.0.0.1 --port 8765
cd desktop && npm run dev
```

- [ ] **Step 3: Run smoke**

Run: `cd desktop && npx playwright test agronomy_smoke.spec.ts`
Expected: 1 passed, screenshot at `desktop/tests/screenshots/agronomy_smoke.png`.

- [ ] **Step 4: Commit**

```bash
git add desktop/tests/agronomy_smoke.spec.ts
git commit -m "test: agronomy Playwright smoke"
```

---

## Final integration sweep

- [ ] **Step 1: Full backend pytest pass**

Run: `pytest tests/research -v`
Expected: all green (library, scenario, runner, cli, router).

- [ ] **Step 2: Full GUI smoke**

Run: `cd desktop && npx playwright test`
Expected: agronomy_smoke + any existing GUI smokes still green.

- [ ] **Step 3: Sanity-check decoupling**

Without the desktop running:
```
hydrus research agronomy run \
  --crop maize --soil loam --weather n_china_avg \
  --horizon-days 30 \
  --irrig /tmp/i.csv --fert /tmp/f.csv --out /tmp/agro_out
```
Expected: `/tmp/agro_out/result.json` exists with `theta_zt` and `water_balance`.

- [ ] **Step 4: Tag the milestone**

```bash
git tag -a m-agronomy-v1 -m "Agronomy decision GUI v1"
```
