<template>
  <div class="section">
    <p class="hint">Surface residue / mulch effects on soil evaporation and energy balance.</p>
    <div class="grid">
      <label>Mulch fraction (0 - 1)
        <input type="number" v-model.number="residue.mulch_fraction" step="0.05" min="0" max="1" />
      </label>
      <label>Residue mass (kg/ha)
        <input type="number" v-model.number="residue.residue_kg_ha" step="100" min="0" />
      </label>
      <label>Evaporation reduction factor (0 - 1)
        <input type="number" v-model.number="residue.e_reduction_factor" step="0.05" min="0" max="1" />
      </label>
    </div>
    <p class="hint">mulch_fraction = 0 means no mulch; e_reduction_factor = 1 means no reduction.</p>
  </div>
</template>

<script setup lang="ts">
import { computed } from "vue";
import { useDndcSeamStore } from "../../stores/dndcSeam";

const store = useDndcSeamStore();
const residue = computed(() => store.inputs.residue);
</script>

<style scoped>
.section { padding: 6px 0; }
.hint { color: #666; font-size: 0.9em; margin: 0 0 8px; }
.grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 12px; margin-bottom: 8px; }
label { display: flex; flex-direction: column; gap: 3px; font-size: 12px; }
input[type="number"] { font-size: 12px; padding: 3px 6px; }
</style>
