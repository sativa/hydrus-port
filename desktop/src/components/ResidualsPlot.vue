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
