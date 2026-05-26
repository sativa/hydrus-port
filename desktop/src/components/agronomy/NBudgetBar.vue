<script setup lang="ts">
import { watch, ref, onMounted } from "vue";
// @ts-ignore
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
