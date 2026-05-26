<script setup lang="ts">
import { watch, ref, onMounted } from "vue";
// @ts-ignore
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
