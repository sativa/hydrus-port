<template>
  <div ref="plotEl" class="texture-triangle"></div>
</template>

<script setup lang="ts">
import { onMounted, ref, watch } from "vue";
import Plotly from "plotly.js-dist-min";

const props = defineProps<{ sand?: number; silt?: number; clay?: number }>();
const emit = defineEmits<{ (e: "update", v: { sand_pct: number; silt_pct: number; clay_pct: number }): void }>();
const plotEl = ref<HTMLDivElement | null>(null);

function _draw() {
  if (!plotEl.value) return;
  const marker = (props.sand !== undefined && props.silt !== undefined && props.clay !== undefined)
    ? [{ type: "scatterternary", mode: "markers",
         a: [props.clay], b: [props.sand], c: [props.silt],
         marker: { size: 16, color: "#c00" } }]
    : [];
  Plotly.newPlot(plotEl.value, marker, {
    ternary: {
      sum: 100,
      aaxis: { title: "clay %", ticksuffix: "%" },
      baxis: { title: "sand %", ticksuffix: "%" },
      caxis: { title: "silt %", ticksuffix: "%" },
    },
    margin: { t: 20, l: 60, r: 60, b: 60 }, showlegend: false,
  }, { responsive: true, displayModeBar: false });
  (plotEl.value as any).on("plotly_click", (ev: any) => {
    const pt = ev.points[0];
    emit("update", {
      clay_pct: pt.a, sand_pct: pt.b, silt_pct: pt.c,
    });
  });
}

onMounted(_draw);
watch(() => [props.sand, props.silt, props.clay], _draw);
</script>

<style scoped>
.texture-triangle { width: 100%; height: 480px; }
</style>
