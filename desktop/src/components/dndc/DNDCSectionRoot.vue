<template>
  <div class="section">
    <p class="hint">Root depth growth curve and vertical density profile used by the Feddes sink term.</p>
    <div class="grid">
      <label>Max root depth (cm)
        <input type="number" v-model.number="root.z_max_cm" min="0" step="5" />
      </label>
      <label>Growth curve
        <select v-model="root.growth_curve">
          <option value="logistic">Logistic (default)</option>
          <option value="linear">Linear</option>
          <option value="table">Table (date, depth)</option>
        </select>
      </label>
      <label v-if="root.growth_curve !== 'table'">Days to z_max
        <input type="number" v-model.number="root.days_to_zmax" min="1" step="1" />
      </label>
      <label>Density profile
        <select v-model="root.density_profile">
          <option value="linear_decline">Linear decline (default)</option>
          <option value="uniform">Uniform</option>
          <option value="exponential">Exponential</option>
          <option value="raats">Raats (1974)</option>
        </select>
      </label>
      <label v-if="root.density_profile === 'exponential' || root.density_profile === 'raats'">
        Density param (cm&#x207B;&#xB9;)
        <input type="number" v-model.number="root.density_param" step="0.001" min="0" />
      </label>
    </div>

    <template v-if="root.growth_curve === 'table'">
      <p class="hint">Table: paste as CSV — date,depth_cm — one row per date.</p>
      <label>Root depth table
        <textarea v-model="tableText" rows="5" placeholder="2026-05-01,0.0&#10;2026-06-01,30.0"></textarea>
      </label>
    </template>
  </div>
</template>

<script setup lang="ts">
import { ref, computed } from "vue";
import { useDndcSeamStore } from "../../stores/dndcSeam";

const store = useDndcSeamStore();
const root = computed(() => store.inputs.root);
const tableText = ref("");
</script>

<style scoped>
.section { padding: 6px 0; }
.hint { color: #666; font-size: 0.9em; margin: 0 0 8px; }
.grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(180px, 1fr)); gap: 10px; margin-bottom: 8px; }
label { display: flex; flex-direction: column; gap: 3px; font-size: 12px; }
select, input[type="number"] { font-size: 12px; padding: 3px 6px; }
textarea { width: 100%; font-family: monospace; font-size: 12px; margin-top: 4px; }
</style>
