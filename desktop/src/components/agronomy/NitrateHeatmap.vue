<script setup lang="ts">
import ThetaHeatmap from "./ThetaHeatmap.vue";
defineProps<{
  z: number[]; t: number[]; field: number[][];
  events: { t_day: number; amount: number; label: string }[];
}>();
</script>

<template>
  <ThetaHeatmap :z="z" :t="t" :field="field"
                :events="events.filter(e => e.label === 'fert')"
                title="N-NO₃ (z,t)" colorscale="Reds" unit="mg/L" />
</template>
