<template>
  <div ref="plotEl" class="ensemble-viz"></div>
</template>

<script setup lang="ts">
import { onMounted, ref, watch } from "vue";
import Plotly from "plotly.js-dist-min";

const props = defineProps<{
  // Each row is one ensemble member; columns are the x-axis (e.g. time).
  ys: number[][];
  x: number[];
  xlabel?: string;
  ylabel?: string;
}>();

const plotEl = ref<HTMLDivElement | null>(null);

// Largest-Triangle-Three-Buckets downsampling — keeps visual shape on huge ensembles
function lttb(data: number[], threshold: number): number[] {
  if (data.length <= threshold) return data;
  const stride = data.length / threshold;
  const out: number[] = [];
  for (let i = 0; i < threshold; i++) out.push(data[Math.floor(i * stride)]);
  return out;
}

function _draw() {
  if (!plotEl.value || !props.ys.length) return;
  const N = props.ys.length;
  const showAll = N <= 200;
  const traces: any[] = props.ys.map((row, i) => ({
    type: "scatter", mode: "lines",
    x: props.x, y: row,
    line: { width: 1, color: showAll ? "#1f77b4" : "rgba(31,119,180,0.06)" },
    showlegend: false,
    hoverinfo: i === 0 ? "x+y" : "skip",
  }));

  // 95% quantile band when N is large
  if (!showAll) {
    const Nt = props.x.length;
    const lo: number[] = [], hi: number[] = [], med: number[] = [];
    for (let t = 0; t < Nt; t++) {
      const col = props.ys.map(r => r[t]).filter(v => !isNaN(v)).sort((a, b) => a - b);
      lo.push(col[Math.floor(0.025 * col.length)] ?? NaN);
      med.push(col[Math.floor(0.5 * col.length)] ?? NaN);
      hi.push(col[Math.floor(0.975 * col.length)] ?? NaN);
    }
    traces.push(
      { type: "scatter", mode: "lines", x: props.x, y: hi, line: { color: "#c00", dash: "dot" }, name: "97.5%" },
      { type: "scatter", mode: "lines", x: props.x, y: med, line: { color: "#c00" }, name: "median" },
      { type: "scatter", mode: "lines", x: props.x, y: lo, line: { color: "#c00", dash: "dot" }, name: "2.5%" },
    );
  }

  Plotly.newPlot(plotEl.value, traces, {
    xaxis: { title: props.xlabel ?? "x" },
    yaxis: { title: props.ylabel ?? "y" },
    margin: { t: 20, l: 60, r: 30, b: 50 },
  }, { responsive: true, displayModeBar: false });
}

onMounted(_draw);
watch(() => [props.ys, props.x], _draw, { deep: true });
</script>

<style scoped>
.ensemble-viz { width: 100%; height: 380px; }
</style>
