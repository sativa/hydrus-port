<template>
  <div ref="el" class="ee"></div>
</template>

<script setup lang="ts">
import { onMounted, ref, watch } from "vue";
import Plotly from "plotly.js-dist-min";

const props = defineProps<{
  paramNames: string[];
  muStar: number[];
  sigma: number[];
}>();

const el = ref<HTMLDivElement | null>(null);

function _draw() {
  if (!el.value) return;
  Plotly.newPlot(
    el.value,
    [
      {
        type: "scatter" as const,
        mode: "markers+text",
        x: props.muStar,
        y: props.sigma,
        text: props.paramNames,
        textposition: "top center",
        marker: { size: 12, color: "#1f77b4" },
      },
    ],
    {
      xaxis: { title: "μ* (overall effect magnitude)" },
      yaxis: { title: "σ (interaction / nonlinearity)" },
      margin: { t: 20, l: 60, r: 20, b: 50 },
    },
    { responsive: true, displayModeBar: false },
  );
}

onMounted(_draw);
watch(
  () => [props.paramNames, props.muStar, props.sigma],
  _draw,
  { deep: true },
);
</script>

<style scoped>
.ee {
  width: 100%;
  height: 380px;
}
</style>
