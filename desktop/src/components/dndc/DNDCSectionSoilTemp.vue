<template>
  <div class="section">
    <p class="hint">Soil temperature coupling (optional). When enabled, provide a daily surface temperature series.</p>
    <div class="row">
      <label class="inline">
        <input type="checkbox" v-model="soilTemp.enabled" />
        Enable soil temperature
      </label>
    </div>

    <template v-if="soilTemp.enabled">
      <label>Surface temperature (&#xB0;C, comma-separated, one value per day)
        <textarea v-model="tempText" rows="4" placeholder="15.0,16.5,18.0,20.2,21.5" @change="parseTemp"></textarea>
      </label>
      <p class="status">{{ (soilTemp.surface_t_daily_c ?? []).length }} values loaded</p>
    </template>
    <p v-else class="muted">Soil temperature is disabled — heat transport ignored.</p>
  </div>
</template>

<script setup lang="ts">
import { ref, computed } from "vue";
import { useDndcSeamStore } from "../../stores/dndcSeam";

const store = useDndcSeamStore();
const soilTemp = computed(() => store.inputs.soil_temp);
const tempText = ref((soilTemp.value.surface_t_daily_c ?? []).join(","));

function parseTemp() {
  const v = tempText.value.split(",").map((s) => parseFloat(s.trim())).filter((n) => !isNaN(n));
  store.inputs.soil_temp.surface_t_daily_c = v.length ? v : undefined;
}
</script>

<style scoped>
.section { padding: 6px 0; }
.hint { color: #666; font-size: 0.9em; margin: 0 0 8px; }
.muted { color: #888; font-size: 12px; font-style: italic; }
.row { margin-bottom: 8px; }
label { display: flex; flex-direction: column; gap: 3px; font-size: 12px; }
label.inline { flex-direction: row; align-items: center; gap: 6px; }
textarea { width: 100%; font-family: monospace; font-size: 12px; margin-top: 4px; }
.status { font-size: 12px; color: var(--muted, #888); margin: 4px 0 0; }
</style>
