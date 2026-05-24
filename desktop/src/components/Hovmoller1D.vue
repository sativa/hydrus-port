<script setup lang="ts">
import { ref, watch, onMounted, onBeforeUnmount, computed, nextTick } from "vue";
// @ts-ignore
import Plotly from "plotly.js-dist-min";
import { api, type JobMeta, type NodInfSeries, type OutputFile } from "../api";
import { theme, plotlyLayout } from "../theme";

const props = defineProps<{ job: JobMeta | null }>();

const heatEl = ref<HTMLDivElement | null>(null);
const profileEl = ref<HTMLDivElement | null>(null);
const fluxEl = ref<HTMLDivElement | null>(null);

const series = ref<NodInfSeries | null>(null);
// Default to Moisture (θ, water content) — most agronomically meaningful
// for irrigation analysis. Head (h) is still selectable.
const varName = ref<string>("Moisture");
const colormap = ref<string>("Viridis");
const tIndex = ref<number>(0);
const err = ref<string | null>(null);
const fluxFile = ref<OutputFile | null>(null);
const units = ref<{length: string; time: string} | null>(null);
// Root zone depth (positive cm into the soil). 0..rootDepth from the
// surface is the band a crop's roots can actually use.
const rootDepth = ref<number>(30);
// Parsed T_LEVEL.OUT time series for water-budget integrations
const fluxRows = ref<{t: number; values: number[]}[]>([]);
const fluxHeaders = ref<string[]>([]);

// Best-effort: read the scenario's units so axes can show "time [day]"
// rather than just "time". Hits api.readScenario when a 1D job runs.
async function loadUnits() {
  if (!props.job) { units.value = null; return; }
  try {
    const s = await api.readScenario(props.job.input_dir);
    units.value = { length: s.units.length, time: s.units.time };
  } catch { units.value = null; }
}

// ----- water budget --------------------------------------------------
// Integrate θ over depth at the current time:
//   profile water = ∫₀^L θ(z) dz   ≈   Σ θᵢ · Δz_node
//   root water    = ∫_root_band θ(z) dz
// Returns cm (or whatever length unit the scenario uses).

const profileWater = computed(() => integrateTheta(false));
const rootWater    = computed(() => integrateTheta(true));

function integrateTheta(rootOnly: boolean): number | null {
  const s = series.value;
  if (!s) return null;
  const th = s.vars["Moisture"];
  if (!th) return null;
  const depths = s.depths;
  const ti = tIndex.value;
  if (ti >= th.length) return null;
  const row = th[ti];
  if (!row || row.length !== depths.length) return null;
  // depths are stored ascending; surface = max(z), bottom = min(z).
  // For an "in soil" depth d (>=0), absolute z = surface - d.
  const zSurface = Math.max(...depths);
  let total = 0;
  for (let i = 0; i < depths.length; i++) {
    const depth_into_soil = zSurface - depths[i];  // 0 at surface, +X cm deep
    if (rootOnly && depth_into_soil > rootDepth.value) continue;
    // Δz: trapezoid using neighbour spacing
    const zLo = i > 0 ? (depths[i] + depths[i - 1]) / 2 : depths[i];
    const zHi = i < depths.length - 1
      ? (depths[i] + depths[i + 1]) / 2 : depths[i];
    const dz = zHi - zLo;
    total += row[i] * dz;
  }
  return total;
}

// Cumulative infiltration / drainage at the current time from T_LEVEL.OUT
const cumInfil = computed(() => fluxValueAtT("sum_Infil") ?? fluxValueAtT("sum(Infil)"));
const cumDrain = computed(() => {
  // sum_vBot is negative when water leaves the bottom — flip to positive
  const v = fluxValueAtT("sum_vBot") ?? fluxValueAtT("sum(vBot)");
  return v === null ? null : -v;
});

function fluxValueAtT(name: string): number | null {
  if (!fluxRows.value.length) return null;
  const idx = fluxHeaders.value.findIndex((h) => h === name);
  if (idx < 0) return null;
  const t = series.value?.times[tIndex.value] ?? 0;
  // pick nearest row
  let best = 0, bestDt = Infinity;
  for (let i = 0; i < fluxRows.value.length; i++) {
    const dt = Math.abs(fluxRows.value[i].t - t);
    if (dt < bestDt) { bestDt = dt; best = i; }
  }
  const v = fluxRows.value[best].values[idx];
  return Number.isFinite(v) ? v : null;
}

function fmtW(v: number | null): string {
  if (v === null) return "—";
  const u = units.value?.length ?? "";
  return `${v.toFixed(2)} ${u}`;
}

// HYDRUS-1D NOD_INF column → unit string (in terms of L, T from units)
function _varUnit(name: string): string {
  if (!units.value) return "";
  const L = units.value.length, T = units.value.time;
  const u: Record<string, string> = {
    "Head":      `[${L}]`,
    "Moisture":  `[-]`,
    "K":         `[${L}/${T}]`,
    "C":         `[1/${L}]`,
    "Flux":      `[${L}/${T}]`,
    "Sink":      `[1/${T}]`,
    "Kappa":     `[-]`,
    "vOverKsTop":`[-]`,
    "Temp":      `[°C]`,
  };
  return u[name] ? ` ${u[name]}` : "";
}

const colormaps = ["Viridis", "Cividis", "Plasma", "Turbo", "RdBu"];

const currentTime = computed(
  () => series.value?.times?.[tIndex.value] ?? 0,
);

watch(
  () => [props.job?.id, props.job?.status, props.job?.finished_at_ms],
  refresh, { immediate: true },
);

async function refresh() {
  err.value = null;
  series.value = null;
  fluxFile.value = null;
  fluxRows.value = [];
  fluxHeaders.value = [];
  if (!props.job) return;
  loadUnits();
  try {
    const files = await api.listOutputFiles(props.job.output_dir);
    const nodInf = files.find((f) =>
      /^nod[_]?inf\.(out|OUT)$/i.test(f.name),
    );
    const tLevel = files.find((f) =>
      /^t[_]?level\.(out|OUT)$/i.test(f.name),
    );
    fluxFile.value = tLevel ?? null;
    if (!nodInf) {
      err.value =
        "no NOD_INF.OUT in this run — Hovmöller view only renders 1D HYDRUS outputs";
      return;
    }
    series.value = await api.parseNodInf(nodInf.path);
    if (series.value.times.length) tIndex.value = series.value.times.length - 1;
    if (!series.value.var_names.includes(varName.value)) {
      varName.value = series.value.var_names[0];
    }
    // The heat / profile / flux divs are gated behind v-if="series", so
    // they only mount AFTER `series.value` is assigned. Wait one tick.
    await nextTick();
    drawHeat();
    drawProfile();
    if (tLevel) drawFlux(tLevel.path);
  } catch (e: any) {
    err.value = String(e);
  }
}

watch([varName, colormap, rootDepth], () => {
  drawHeat();
  drawProfile();
});

watch(tIndex, () => {
  drawProfile();
  // Update vertical scrub line on heat-map
  updateScrubLine();
});

const darkLayout = computed(() => plotlyLayout());
// Re-render charts when theme flips so paper/grid colors track.
watch(theme, () => { drawHeat(); drawProfile(); });

function drawHeat() {
  if (!heatEl.value || !series.value) return;
  const s = series.value;
  const z2d = s.vars[varName.value];
  if (!z2d) return;
  const heat: any = {
    type: "heatmap",
    x: s.times,
    y: s.depths,
    z: transpose(z2d), // plotly wants z[y][x] = z[depth_idx][time_idx]
    colorscale: colormap.value,
    colorbar: { thickness: 8, len: 0.8, x: 1.0, ticks: "outside" },
    hovertemplate: "t=%{x:.3f}<br>z=%{y:.2f}<br>v=%{z:.4f}<extra></extra>",
  };
  const scrub = {
    type: "scatter",
    mode: "lines",
    x: [s.times[tIndex.value], s.times[tIndex.value]],
    y: [Math.min(...s.depths), Math.max(...s.depths)],
    line: { color: "#f85149", width: 2, dash: "dash" },
    name: "t",
    hoverinfo: "skip",
    showlegend: false,
  };
  // Root zone band — horizontal band at z ∈ [z_surface - rootDepth, z_surface]
  const zSurface = Math.max(...s.depths);
  const rootBottom = zSurface - rootDepth.value;
  const rootBand = {
    type: "scatter",
    mode: "lines",
    x: [s.times[0], s.times[s.times.length - 1], s.times[s.times.length - 1],
        s.times[0], s.times[0]],
    y: [rootBottom, rootBottom, zSurface, zSurface, rootBottom],
    fill: "toself",
    fillcolor: "rgba(63, 185, 80, 0.10)",
    line: { color: "rgba(63, 185, 80, 0.55)", width: 1, dash: "dot" },
    name: "root zone",
    hoverinfo: "skip",
    showlegend: false,
  };
  Plotly.react(
    heatEl.value,
    [heat, rootBand, scrub],
    {
      ...darkLayout.value,
      xaxis: { ...darkLayout.value.xaxis,
               title: `time${units.value ? ' [' + units.value.time + ']' : ''}` },
      yaxis: { ...darkLayout.value.yaxis,
               title: `depth${units.value ? ' [' + units.value.length + ']' : ''}` },
      title: { text: `${varName.value}(z, t)  ·  root zone shaded`, font: { size: 13 } },
      showlegend: false,
    },
    { responsive: true, displaylogo: false },
  );
  if (heatEl.value && (heatEl.value as any).removeAllListeners) {
    (heatEl.value as any).removeAllListeners("plotly_click");
  }
  (heatEl.value as any).on?.("plotly_click", (ev: any) => {
    const tx = ev.points?.[0]?.x;
    if (typeof tx !== "number") return;
    const i = nearestIndex(series.value!.times, tx);
    tIndex.value = i;
  });
}

function updateScrubLine() {
  if (!heatEl.value || !series.value) return;
  const t = series.value.times[tIndex.value];
  // Traces: [0]=heatmap, [1]=root band, [2]=scrub line. Update [2].
  Plotly.restyle(
    heatEl.value,
    { x: [[t, t]] },
    [2],
  );
}

function drawProfile() {
  if (!profileEl.value || !series.value) return;
  const s = series.value;
  const ti = tIndex.value;
  const z2d = s.vars[varName.value];
  if (!z2d) return;
  const trace = {
    type: "scatter",
    mode: "lines+markers",
    x: z2d[ti],
    y: s.depths,
    marker: { size: 4, color: "#58a6ff" },
    line: { color: "#58a6ff", width: 2 },
    hovertemplate: "z=%{y:.2f}<br>v=%{x:.4f}<extra></extra>",
  };
  Plotly.react(
    profileEl.value,
    [trace],
    {
      ...darkLayout.value,
      xaxis: { ...darkLayout.value.xaxis,
               title: `${varName.value}${_varUnit(varName.value)}` },
      yaxis: { ...darkLayout.value.yaxis,
               title: `depth${units.value ? ' [' + units.value.length + ']' : ''}` },
      title: {
        text: `profile @ t=${s.times[ti].toFixed(3)}${units.value ? ' ' + units.value.time : ''}`,
        font: { size: 13 },
      },
      showlegend: false,
    },
    { responsive: true, displaylogo: false },
  );
}

async function drawFlux(path: string) {
  if (!fluxEl.value) return;
  try {
    const ser = await api.parseTable(path);
    // Cache for water-budget readout
    fluxHeaders.value = ser.headers;
    fluxRows.value = ser.rows.map((r) => ({ t: r[0], values: r }));
    // Find columns
    const idx = (name: string) =>
      ser.headers.findIndex((h) => h === name);
    const tIdx = Math.max(0, idx("Time"));
    const candidates = [
      ["rTop", "#58a6ff"],
      ["vTop", "#3fb950"],
      ["vBot", "#d29922"],
      ["Volume", "#f85149"],
    ];
    const traces: any[] = [];
    for (const [name, color] of candidates) {
      const c = idx(name);
      if (c < 0) continue;
      traces.push({
        type: "scatter",
        mode: "lines",
        x: ser.rows.map((r) => r[tIdx]),
        y: ser.rows.map((r) => r[c]),
        name,
        line: { color, width: 1.4 },
      });
    }
    Plotly.react(
      fluxEl.value,
      traces,
      {
        ...darkLayout.value,
        xaxis: { ...darkLayout.value.xaxis,
                 title: `time${units.value ? ' [' + units.value.time + ']' : ''}` },
        yaxis: { ...darkLayout.value.yaxis, title: "" },
        title: { text: "boundary fluxes", font: { size: 12 } },
        showlegend: true,
        legend: { font: { size: 10 }, orientation: "h", y: 1.15 },
        margin: { l: 50, r: 16, t: 28, b: 30 },
      },
      { responsive: true, displaylogo: false },
    );
  } catch (e: any) {
    /* swallow; flux strip is optional */
  }
}

function nearestIndex(arr: number[], v: number): number {
  let best = 0;
  let bd = Infinity;
  for (let i = 0; i < arr.length; i++) {
    const d = Math.abs(arr[i] - v);
    if (d < bd) { bd = d; best = i; }
  }
  return best;
}

function transpose(m: number[][]): number[][] {
  if (!m.length) return [];
  const r = m.length, c = m[0].length;
  const t: number[][] = Array.from({ length: c }, () => new Array(r));
  for (let i = 0; i < r; i++)
    for (let j = 0; j < c; j++)
      t[j][i] = m[i][j];
  return t;
}

onMounted(() => {
  window.addEventListener("resize", onResize);
});
onBeforeUnmount(() => {
  window.removeEventListener("resize", onResize);
});
function onResize() {
  for (const el of [heatEl, profileEl, fluxEl]) {
    if (el.value) Plotly.Plots.resize(el.value);
  }
}
</script>

<template>
  <div class="panel hov-panel">
    <div class="row" style="justify-content: space-between; flex-wrap: wrap; gap: 6px">
      <div class="title">1D Hovmöller</div>
      <div class="row small" style="gap: 8px; flex-wrap: wrap">
        <span class="muted">var:</span>
        <select v-model="varName" :disabled="!series">
          <option v-for="n in series?.var_names ?? []" :key="n" :value="n">{{ n }}</option>
        </select>
        <span class="muted">cmap:</span>
        <select v-model="colormap">
          <option v-for="c in colormaps" :key="c" :value="c">{{ c }}</option>
        </select>
        <span class="muted">root&nbsp;zone:</span>
        <input type="number" min="0" step="1" v-model.number="rootDepth"
               style="width: 50px" :title="`root depth, ${units?.length ?? 'cm'}`" />
        <span class="muted">{{ units?.length ?? 'cm' }}</span>
        <span class="muted">·</span>
        <span class="muted">t:</span>
        <span class="mono">{{ currentTime.toFixed(3) }}</span>
      </div>
    </div>
    <div v-if="!series" class="muted small mono" style="padding: 10px">
      {{ err ?? "waiting for NOD_INF.OUT…" }}
    </div>
    <div v-else class="hov-grid">
      <div ref="heatEl" class="heat"></div>
      <div ref="profileEl" class="profile"></div>
      <div class="row scrub-row" style="grid-column: 1 / -1">
        <input
          type="range"
          :min="0" :max="series.times.length - 1"
          v-model.number="tIndex"
          style="flex: 1"
        />
      </div>
      <div class="budget" style="grid-column: 1 / -1">
        <div class="budget-cell">
          <div class="bk">Profile water</div>
          <div class="bv">{{ fmtW(profileWater) }}</div>
        </div>
        <div class="budget-cell">
          <div class="bk">Root zone (0–{{ rootDepth }})</div>
          <div class="bv">{{ fmtW(rootWater) }}</div>
        </div>
        <div class="budget-cell">
          <div class="bk">Cumulative infil ↓</div>
          <div class="bv">{{ fmtW(cumInfil) }}</div>
        </div>
        <div class="budget-cell">
          <div class="bk">Cumulative drain ↓</div>
          <div class="bv">{{ fmtW(cumDrain) }}</div>
        </div>
      </div>
      <div ref="fluxEl" class="flux" style="grid-column: 1 / -1"></div>
    </div>
  </div>
</template>

<style scoped>
.hov-panel {
  flex: 1;
  min-height: 400px;
  display: flex;
  flex-direction: column;
}
.title { font-weight: 600; }
.small { font-size: 11px; }
.hov-grid {
  flex: 1;
  display: grid;
  grid-template-columns: 2fr 1fr;
  grid-template-rows: minmax(180px, 1fr) auto minmax(110px, 0.6fr);
  gap: 6px;
  min-height: 0;
  margin-top: 8px;
}
.heat { min-height: 0; }
.profile { min-height: 0; }
.scrub-row {
  align-items: center;
  padding: 0 2px;
}
.flux { min-height: 0; }
.budget {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 6px;
  padding: 6px 2px;
  border: 1px solid var(--border);
  border-radius: 4px;
  background: var(--panel-2);
}
.budget-cell { padding: 2px 6px; }
.bk { font-size: 10px; color: var(--muted); }
.bv { font-size: 13px; font-weight: 600; color: var(--accent-2);
      font-family: ui-monospace, Menlo, monospace; }
</style>
