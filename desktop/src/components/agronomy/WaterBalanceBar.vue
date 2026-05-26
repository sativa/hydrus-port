<script setup lang="ts">
import { watch, ref, onMounted } from "vue";
// @ts-ignore
import Plotly from "plotly.js-dist-min";

const props = defineProps<{
  balance: { rain_mm: number; irrig_mm: number; et_mm: number;
             percolation_mm: number; storage_change_mm: number };
}>();

const el = ref<HTMLDivElement | null>(null);

function render() {
  if (!el.value) return;
  const labels = ["Rain", "Irrig", "ET", "Percol", "ΔStor"];
  const values = [props.balance.rain_mm, props.balance.irrig_mm,
                  props.balance.et_mm, props.balance.percolation_mm,
                  props.balance.storage_change_mm];
  Plotly.newPlot(el.value, [{
    type: "bar", orientation: "h", x: values, y: labels,
    marker: { color: ["#5DA5DA","#4D9DE0","#F17CB0","#B276B2","#999999"] },
    text: values.map(v => v.toFixed(0) + " mm"), textposition: "auto",
  }], { title: "水平衡 (mm)", margin: { l: 60, r: 20, t: 30, b: 30 } },
      { responsive: true });
}
onMounted(render);
watch(() => props.balance, render, { deep: true });
</script>

<template><div ref="el" class="bar"></div></template>

<style scoped>.bar { width: 100%; height: 200px; }</style>
