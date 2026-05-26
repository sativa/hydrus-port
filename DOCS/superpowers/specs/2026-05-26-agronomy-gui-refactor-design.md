# Agronomy Decision GUI Refactor — Design

> Refactors the HYDRUS Port desktop GUI from a 9-research-tab layout into a workflow-driven agronomy decision panel. The user picks a crop, soil, and rain year from dropdowns, adds irrigation + fertilizer events in two tables, runs the simulation, and sees soil-water and N profiles as time-depth heatmaps plus water/N balance bars. The existing 9 research tabs (Parameters, 3D mesh, DNDC Inputs, Soil Library, Batch Sweep, Sensitivity, Inversion, UQ, Surrogate) survive untouched behind a right-side "高级" drawer.

**Status:** Approved 2026-05-26.

---

## 1. Goal

Make the GUI directly answer: *given this crop on this soil under this rain year, what does my irrigation + fertilizer schedule do to soil-water and N in the root zone?*

Current GUI fails this because:
- Crop selection does not exist (no crop library at all).
- Soil selection is a per-parameter VG form, not a one-click texture pick.
- Result rendering is "Output files" file list + per-file Plotly chart — not a decision-grade θ(z,t) heatmap.
- 9 research tabs (Sensitivity, Inversion, UQ, Surrogate, …) dominate the right pane and bury the everyday workflow.

## 2. Non-Goals

- No new physics. Engine is the existing `hydrus1d.hydrus.run_simulation` adapter.
- No B2 DNDC coupling work — DNDC seam stays as the 11 Pydantic types in `hydrus_research/dndc_seam/`, surfaced only in the advanced drawer.
- No multi-scenario A/B compare in this iteration (kept as future P2).
- No 2D/3D rewrite — 2D drip/flood and 3D box stay reachable via the existing "Regression" panel only.

## 3. Decoupling Discipline (HARD)

Every workflow the GUI exposes MUST be reproducible from the CLI / Python API without the desktop running. Concretely:

```
hydrus research agronomy run \
  --crop maize --soil loam --weather n_china_avg \
  --irrig irrig.csv --fert fert.csv --horizon-days 150 \
  --out /tmp/out
```

This MUST produce the same `theta_zt`, `n_zt`, `water_balance`, `n_budget` arrays the GUI consumes. Allowed dependency direction: **GUI → REST → engine**, never reverse.

## 4. Architecture

### 4.1 Single-page 3-band Vue layout

```
┌─ HYDRUS Agronomy ─────────────────────────────── [高级 ▸] ┐
│ Band 1 — Input Strip                                      │
│   Crop[玉米 ▾]  Soil[壤土 ▾]  Weather[华北平水年 ▾]  ▶Run│
│   ┌ Irrigation events ┐  ┌ Fertilizer events ┐           │
│   │ date  depth_mm   │  │ date  kgN/ha  conc │           │
│   │ 2026-05-12  30   │  │ 2026-05-10  60     │           │
│   └──────────────────┘  └────────────────────┘           │
├───────────────────────────────────────────────────────────┤
│ Band 2 — Main Heatmaps                                    │
│   θ(z,t)  [Plotly heatmap, irrig ticks]                   │
│   N-NO₃(z,t)  [Plotly heatmap, fert ticks]                │
├───────────────────────────────────────────────────────────┤
│ Band 3 — Balance Bars                                     │
│   Water mm: [Rain 230 | Irrig 180 | ET 360 | Perc 50]    │
│   N kgN/ha: [Applied 120 | Uptake 80 | Leach 15 | Res 25]│
└───────────────────────────────────────────────────────────┘
```

### 4.2 Right-side Advanced Drawer

A `<AdvancedDrawer>` toggled by the top-right `[高级 ▸]` button. Its contents are *exactly* today's 9 right-panel tabs, lifted into the drawer with no logic changes. Default state: closed.

### 4.3 Component tree

```
App.vue
├── InputStrip.vue            (crop/soil/weather dropdowns + Run)
│   └── EventsTable.vue × 2   (irrigation, fertilizer)
├── HeatmapBand.vue
│   ├── ThetaHeatmap.vue      (Plotly heatmap + irrig vlines)
│   └── NitrateHeatmap.vue    (Plotly heatmap + fert vlines)
├── BalanceBand.vue
│   ├── WaterBalanceBar.vue
│   └── NBudgetBar.vue
├── AdvancedDrawer.vue        (holds existing 9 tabs)
└── stores/agronomy.ts        (Pinia: libs cache, selection, events, results)
```

## 5. Predefined Libraries

### 5.1 Crops (≈10)

`hydrus_research/library/data/crops.json` keys: `maize`, `wheat_winter`, `rice`, `cotton`, `tomato`, `grape`, `apple`, `tea`, `rapeseed`, `soybean`.

Each entry:
```json
{
  "id": "maize",
  "name_zh": "玉米",
  "feddes": {"P0":-15, "P0pt":-30, "P2H":-325, "P2L":-600, "P3":-8000, "r2H":0.5, "r2L":0.1},
  "root": {"max_depth_cm": 120, "shape": "s_curve", "z50_cm": 30, "z95_cm": 90},
  "season": {"sow_doy": 121, "harvest_doy": 273},
  "kc_curve": [{"doy":121,"kc":0.3},{"doy":160,"kc":0.7},{"doy":200,"kc":1.15},{"doy":240,"kc":1.0},{"doy":273,"kc":0.5}]
}
```

### 5.2 Soils (≈12)

`hydrus_research/library/data/soils.json` keys: `sand`, `sandy_loam`, `loam`, `silt_loam`, `silt`, `clay_loam`, `sandy_clay_loam`, `silty_clay_loam`, `clay`, `sand_over_clay`, `loam_over_sand`, `topsoil_subsoil_bedrock`.

Single-layer entry (Carsel-Parrish 1988):
```json
{
  "id": "loam",
  "name_zh": "壤土",
  "layers": [
    {"depth_cm": 200, "vg": {"theta_r":0.078, "theta_s":0.43, "alpha":0.036, "n":1.56, "Ks":24.96, "L":0.5}}
  ]
}
```

Layered entry has `layers: [...]` with multiple `{depth_cm, vg}` blocks summing to profile depth (200 cm default).

### 5.3 Weather typical years (6)

`hydrus_research/library/data/weather/*.csv` IDs: `n_china_avg`, `n_china_wet`, `n_china_dry`, `c_china_meiyu`, `s_china_double`, `nw_china_irrig`.

Each CSV has 365 rows: `doy, P_mm, PET_mm`. Default profile horizon = full season of the selected crop (sow_doy → harvest_doy), but can be overridden.

## 6. REST API

All new endpoints added to `hydrus_port/app.py`:

| Method | Path | Returns |
|---|---|---|
| GET | `/lib/crops` | `{crops: [Crop, ...]}` (full library) |
| GET | `/lib/soils` | `{soils: [Soil, ...]}` |
| GET | `/lib/weather` | `{weather: [WeatherMeta, ...]}` (id + name only) |
| GET | `/lib/weather/{id}` | `{doy: [...], P_mm: [...], PET_mm: [...]}` |
| POST | `/agronomy/run` | runs simulation, returns parsed result |

`/agronomy/run` request:
```json
{
  "crop_id": "maize",
  "soil_id": "loam",
  "weather_id": "n_china_avg",
  "horizon_days": 150,
  "irrigation": [{"date":"2026-05-12","depth_mm":30}, ...],
  "fertilizer":  [{"date":"2026-05-10","kg_n_ha":60,"conc_mg_l":null}, ...]
}
```

Response:
```json
{
  "z_cm": [...],          // ascending (surface→deep, positive cm)
  "t_days": [...],        // day-of-simulation indices
  "theta_zt": [[...]],    // shape (nT, nZ)
  "n_zt":     [[...]],    // shape (nT, nZ); zero if no fert events
  "water_balance_mm": {"rain":230, "irrig":180, "et":360, "percolation":50, "storage_change":0},
  "n_budget_kg_ha":    {"applied":120, "uptake":80, "leached":15, "residual":25},
  "irrig_events": [{"t_day":7,"depth_mm":30}, ...],
  "fert_events":  [{"t_day":5,"kg_n_ha":60}, ...]
}
```

## 7. Backend Modules

### 7.1 `hydrus_research/library/`

- `crops.py` — `load_crops()` → list[Crop]; `get_crop(id)` → Crop. Crop is a Pydantic model.
- `soils.py` — same shape, with `Soil`, `SoilLayer`, `VanGenuchten`.
- `weather.py` — `load_weather_meta()`, `load_weather_series(id)`.

### 7.2 `hydrus_research/agronomy/`

- `scenario_builder.py` — `build_scenario(crop, soil, weather_series, irrig_events, fert_events, horizon_days)` → `Scenario` (the existing Pydantic Scenario from `hydrus_research/simulator/`).
  - Maps crop.feddes → Feddes block.
  - Maps soil.layers → MAT blocks + VG params.
  - Maps weather P/PET → atmospheric BC arrays (one row per day).
  - Maps irrig events → adds to atmospheric P or surface flux on the event day.
  - Maps fert events → solute BC (mg/L on irrig day, or surface deposition mg/cm²).
- `runner.py` — `run_agronomy(scenario, work_dir)` → calls `hydrus1d.hydrus.run_simulation`.
- `result_parser.py` — reads `NOD_INF.OUT` (z descending — reverse for ascending; cf. `feedback_hydrus1d_nod_inf_z_descending.md`), `T_LEVEL.OUT`, `BALANCE.OUT`, returns the response payload above.

### 7.3 `hydrus_port/cli.py`

New subcommand:
```
hydrus research agronomy run \
  --crop maize --soil loam --weather n_china_avg \
  --irrig irrig.csv --fert fert.csv --horizon-days 150 \
  --out /path/out
```
Writes `result.json` (same schema as the REST response) into `--out`.

## 8. Frontend Modules

### 8.1 Pinia store `stores/agronomy.ts`

State:
```ts
{
  libs:      { crops: Crop[], soils: Soil[], weather: WeatherMeta[] },
  selected:  { cropId: string, soilId: string, weatherId: string, horizonDays: number },
  events:    { irrig: IrrigEvent[], fert: FertEvent[] },
  result:    AgronomyResult | null,
  running:   boolean,
  error:     string | null,
}
```

Actions: `loadLibs()`, `run()` (POST /agronomy/run), `addIrrigRow()`, `addFertRow()`, `removeRow()`.

### 8.2 New components

- `InputStrip.vue` — emits selection changes to store; Run button disabled when `running` or no crop chosen.
- `EventsTable.vue` — props: `kind` ("irrig"|"fert"), `rows`, `onAdd`, `onRemove`. Plain `<table>` with editable cells; date input uses native `<input type="date">`.
- `ThetaHeatmap.vue` — Plotly heatmap, x=t_days, y=z_cm (ascending so surface at top), z=theta_zt. Overlay vertical lines at each irrig event t_day, label depth_mm.
- `NitrateHeatmap.vue` — same shape with `n_zt`, fert event lines.
- `WaterBalanceBar.vue` — Plotly stacked horizontal bar with 5 segments (rain/irrig/et/percolation/storage_change).
- `NBudgetBar.vue` — same with 4 segments (applied/uptake/leached/residual).
- `AdvancedDrawer.vue` — wraps today's 9-tab strip; slide-in from right; closed by default.

### 8.3 App.vue rewrite

Discards today's 3-column "Scenario / Profile plot / 9-tab pane" layout. New body:
```
<header>HYDRUS Port  <button>[高级 ▸]</button></header>
<InputStrip />
<HeatmapBand />
<BalanceBand />
<AdvancedDrawer v-if="open" />
```

The existing "Scenario / Regression / Output files / Log" panes from old App.vue stay reachable from inside `AdvancedDrawer` under a "经典" tab so the regression workflow doesn't disappear.

## 9. Decoupling: CLI parity tests

Each backend module ships with a pytest that exercises it via the Python API, not the REST layer:

- `tests/test_library_loaders.py` — `load_crops()`, `load_soils()`, `load_weather_meta()`, `load_weather_series("n_china_avg")` all return non-empty Pydantic objects.
- `tests/test_agronomy_scenario.py` — `build_scenario(maize, loam, n_china_avg, [], [], 150)` returns a Scenario that round-trips through `Scenario.from_dict(s.to_dict())`.
- `tests/test_agronomy_runner.py` — full `run_agronomy(...)` against the in-repo HYDRUS-1D binary, asserts result.theta_zt shape == (nT, nZ), all finite, and within plausible bounds (0 ≤ θ ≤ θs).
- `tests/test_cli_agronomy.py` — invokes `hydrus research agronomy run ...` as a subprocess, asserts `result.json` is written and matches the in-process call.

## 10. GUI smoke test

`desktop/tests/agronomy_smoke.spec.ts` (Playwright):
1. Launch FastAPI sidecar + Vite dev.
2. Open `localhost:1420`.
3. Wait for `InputStrip` to render with crop options populated.
4. Select 玉米, 壤土, 华北平水年.
5. Add 1 irrigation row (`2026-06-01`, `30` mm).
6. Add 1 fertilizer row (`2026-05-15`, `60` kgN/ha).
7. Click Run, wait for `ThetaHeatmap` to render (poll for non-empty z trace).
8. Assert `WaterBalanceBar` shows numeric `rain` segment > 0.
9. Open `AdvancedDrawer`, assert all 9 tab headers are present.
10. Screenshot all assertions to `desktop/tests/screenshots/`.

## 11. Out of scope (future)

- Multi-scenario A/B compare (Layout-A would extend to scenario tabs in Band 1).
- Crop-specific Kc curve editor in main panel (today: drawer only).
- DNDC seam direct binding in main panel (today: drawer only; awaits B2).
- 2D drip / flood — kept under classic Regression panel inside the drawer.
