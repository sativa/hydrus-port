<script setup lang="ts">
import { ref, watch, onMounted, onBeforeUnmount, computed, nextTick } from "vue";
// @ts-ignore
import Plotly from "plotly.js-dist-min";
import { api, type JobMeta, type NodInfSeries, type OutputFile } from "../api";

const props = defineProps<{ job: JobMeta | null }>();

const heatEl = ref<HTMLDivElement | null>(null);
const profileEl = ref<HTMLDivElement | null>(null);
const fluxEl = ref<HTMLDivElement | null>(null);

const series = ref<NodInfSeries | null>(null);
const varName = ref<string>("Head");
const colormap = ref<string>("Viridis");
const tIndex = ref<number>(0);
const err = ref<string | null>(null);
const fluxFile = ref<OutputFile | null>(null);

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
  if (!props.job) return;
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

watch([varName, colormap], () => {
  drawHeat();
  drawProfile();
});

watch(tIndex, () => {
  drawProfile();
  // Update vertical scrub line on heat-map
  updateScrubLine();
});

const darkLayout = {
  paper_bgcolor: "#161b22",
  plot_bgcolor: "#0d1117",
  font: { color: "#c9d1d9", size: 11 },
  xaxis: { gridcolor: "#30363d", zerolinecolor: "#30363d" },
  yaxis: { gridcolor: "#30363d", zerolinecolor: "#30363d" },
  margin: { l: 50, r: 16, t: 24, b: 36 },
};

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
  Plotly.react(
    heatEl.value,
    [heat, scrub],
    {
      ...darkLayout,
      xaxis: { ...darkLayout.xaxis, title: "time" },
      yaxis: { ...darkLayout.yaxis, title: "depth" },
      title: { text: `${varName.value}(z, t)`, font: { size: 13 } },
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
  Plotly.restyle(
    heatEl.value,
    { x: [[t, t]] },
    [1],
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
      ...darkLayout,
      xaxis: { ...darkLayout.xaxis, title: varName.value },
      yaxis: { ...darkLayout.yaxis, title: "depth" },
      title: {
        text: `profile @ t=${s.times[ti].toFixed(3)}`,
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
        ...darkLayout,
        xaxis: { ...darkLayout.xaxis, title: "time" },
        yaxis: { ...darkLayout.yaxis, title: "" },
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
      <div class="row small" style="gap: 8px">
        <span class="muted">var:</span>
        <select v-model="varName" :disabled="!series">
          <option v-for="n in series?.var_names ?? []" :key="n" :value="n">{{ n }}</option>
        </select>
        <span class="muted">cmap:</span>
        <select v-model="colormap">
          <option v-for="c in colormaps" :key="c" :value="c">{{ c }}</option>
        </select>
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
</style>
