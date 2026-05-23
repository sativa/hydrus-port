<script setup lang="ts">
import { ref, watch, onMounted, onBeforeUnmount } from "vue";
// @ts-ignore — plotly.js-dist-min ships JS only
import Plotly from "plotly.js-dist-min";
import { api, type JobMeta, type OutputFile, type Series } from "../api";

const props = defineProps<{ job: JobMeta | null }>();
const plotDiv = ref<HTMLDivElement | null>(null);
const files = ref<OutputFile[]>([]);
const selected = ref<string | null>(null);
const series = ref<Series | null>(null);
const xCol = ref<number>(0);
const yCols = ref<number[]>([]);
const err = ref<string | null>(null);

const dark = {
  paper_bgcolor: "#161b22",
  plot_bgcolor: "#0d1117",
  font: { color: "#c9d1d9", size: 11 },
  xaxis: { gridcolor: "#30363d", zerolinecolor: "#30363d" },
  yaxis: { gridcolor: "#30363d", zerolinecolor: "#30363d" },
  margin: { l: 50, r: 20, t: 24, b: 40 },
};

watch(
  () => [props.job?.id, props.job?.status, props.job?.finished_at_ms],
  refreshFiles,
  { immediate: true },
);

async function refreshFiles() {
  files.value = [];
  selected.value = null;
  series.value = null;
  if (!props.job) return;
  try {
    const all = await api.listOutputFiles(props.job.output_dir);
    files.value = all.filter(
      (f) =>
        /\.(out|OUT|dat|DAT|txt|TXT)$/.test(f.name) ||
        /ObsNod|Nod_Inf|A_Level|Run_Inf|Balance/i.test(f.name),
    );
    // Auto-pick the most useful file if available
    const obs = files.value.find((f) => /ObsNod/i.test(f.name));
    selected.value = obs?.path ?? files.value[0]?.path ?? null;
    if (selected.value) await loadSeries(selected.value);
  } catch (e: any) {
    err.value = String(e);
  }
}

async function loadSeries(path: string) {
  err.value = null;
  try {
    const s = await api.parseTable(path);
    series.value = s;
    xCol.value = 0;
    // Default: plot every column after the first
    yCols.value = s.headers.slice(1).map((_, i) => i + 1).slice(0, 6);
    draw();
  } catch (e: any) {
    err.value = String(e);
    series.value = null;
  }
}

function draw() {
  if (!plotDiv.value || !series.value) return;
  const s = series.value;
  const x = s.rows.map((r) => r[xCol.value] ?? NaN);
  const traces = yCols.value.map((c) => ({
    x,
    y: s.rows.map((r) => r[c] ?? NaN),
    name: s.headers[c] ?? `col${c}`,
    mode: "lines",
    type: "scattergl",
  }));
  Plotly.react(plotDiv.value, traces, {
    ...dark,
    xaxis: { ...dark.xaxis, title: s.headers[xCol.value] ?? "x" },
    yaxis: { ...dark.yaxis, title: "value" },
    showlegend: true,
    legend: { font: { size: 10 } },
    autosize: true,
  }, { displaylogo: false, responsive: true });
}

watch([xCol, yCols], draw, { deep: true });

onMounted(() => {
  window.addEventListener("resize", onResize);
});
onBeforeUnmount(() => {
  window.removeEventListener("resize", onResize);
});
function onResize() {
  if (plotDiv.value) Plotly.Plots.resize(plotDiv.value);
}
</script>

<template>
  <div class="panel plot-panel">
    <div class="row" style="justify-content: space-between; flex-wrap: wrap">
      <div class="title">Profile plot</div>
      <select
        v-if="files.length"
        v-model="selected"
        @change="selected && loadSeries(selected)"
      >
        <option v-for="f in files" :key="f.path" :value="f.path">
          {{ f.name }}
        </option>
      </select>
    </div>
    <div v-if="series" class="row small" style="gap: 12px; flex-wrap: wrap">
      <div class="row">
        <span class="muted">x:</span>
        <select v-model.number="xCol">
          <option v-for="(h, i) in series.headers" :key="i" :value="i">
            {{ h }}
          </option>
        </select>
      </div>
      <div class="row" style="gap: 6px; flex-wrap: wrap">
        <span class="muted">y:</span>
        <label v-for="(h, i) in series.headers" :key="i" class="chip">
          <input
            type="checkbox"
            :value="i"
            v-model="yCols"
            :disabled="i === xCol"
          />
          {{ h }}
        </label>
      </div>
    </div>
    <div ref="plotDiv" class="plot"></div>
    <div v-if="err" class="status-failed small mono">{{ err }}</div>
  </div>
</template>

<style scoped>
.plot-panel {
  flex: 1;
  min-height: 280px;
  display: flex;
  flex-direction: column;
}
.title { font-weight: 600; }
.small { font-size: 11px; }
.chip {
  display: inline-flex;
  align-items: center;
  gap: 3px;
  padding: 2px 6px;
  border: 1px solid var(--border);
  border-radius: 999px;
}
.plot { flex: 1; min-height: 200px; }
</style>
