<template>
  <div ref="el" class="bars"></div>
</template>

<script setup lang="ts">
import { onMounted, ref, watch } from "vue";
import Plotly from "plotly.js-dist-min";

const props = defineProps<{
  paramNames: string[];
  bestParams: Record<string, number>;
  ciLo?: Record<string, number>;
  ciHi?: Record<string, number>;
}>();

const el = ref<HTMLDivElement | null>(null);

function _draw() {
  if (!el.value) return;
  const best = props.paramNames.map(n => props.bestParams[n]);
  const lo = props.ciLo ? props.paramNames.map(n => props.bestParams[n] - props.ciLo![n]) : null;
  const hi = props.ciHi ? props.paramNames.map(n => props.ciHi![n] - props.bestParams[n]) : null;

  Plotly.newPlot(el.value, [{
    type: "bar", x: props.paramNames, y: best,
    error_y: lo && hi ? { type: "data", symmetric: false, array: hi, arrayminus: lo } : undefined,
    marker: { color: "#1f77b4" },
  }], {
    yaxis: { title: "best-fit value" },
    margin: { t: 20, l: 60, r: 20, b: 50 },
  }, { responsive: true, displayModeBar: false });
}

onMounted(_draw);
watch(() => [props.paramNames, props.bestParams, props.ciLo, props.ciHi], _draw, { deep: true });
</script>

<style scoped>.bars { width: 100%; height: 340px; }</style>
