<template>
  <div ref="el" class="bars"></div>
</template>

<script setup lang="ts">
import { onMounted, ref, watch } from "vue";
import Plotly from "plotly.js-dist-min";

const props = defineProps<{
  paramNames: string[];
  indices: Record<string, number[]>;
  title?: string;
}>();

const el = ref<HTMLDivElement | null>(null);

function _draw() {
  if (!el.value) return;
  const traces = Object.entries(props.indices).map(([key, vals]) => ({
    type: "bar" as const,
    x: props.paramNames,
    y: vals,
    name: key,
  }));
  Plotly.newPlot(
    el.value,
    traces,
    {
      title: props.title ?? "",
      barmode: "group",
      yaxis: { title: "index value" },
      margin: { t: 40, l: 60, r: 20, b: 60 },
    },
    { responsive: true, displayModeBar: false },
  );
}

onMounted(_draw);
watch(() => [props.paramNames, props.indices], _draw, { deep: true });
</script>

<style scoped>
.bars {
  width: 100%;
  height: 360px;
}
</style>
