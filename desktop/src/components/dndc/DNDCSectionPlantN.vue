<template>
  <div class="section">
    <p class="hint">Plant N uptake model. Passive-with-water is the default; Michaelis-Menten adds saturation kinetics.</p>
    <div class="row">
      <label>Mode
        <select v-model="pu.mode">
          <option value="passive_with_water">Passive with water flux</option>
          <option value="michaelis_menten">Michaelis-Menten</option>
          <option value="demand_driven">Demand-driven (daily schedule)</option>
          <option value="external">External callable (B2)</option>
        </select>
      </label>
    </div>

    <template v-if="pu.mode === 'michaelis_menten'">
      <div class="grid">
        <label>Km (mg/L)
          <input type="number" v-model.number="pu.km_mg_l" step="0.5" min="0" placeholder="10.0" />
        </label>
        <label>Vmax (mg N / d / root-cm)
          <input type="number" v-model.number="pu.vmax_mg_per_day_per_root_cm" step="0.001" min="0" placeholder="0.05" />
        </label>
      </div>
    </template>

    <template v-else-if="pu.mode === 'demand_driven'">
      <label>Daily N demand (kg N/ha, comma-separated, one per day)
        <textarea v-model="demandText" rows="3" placeholder="2.0,2.2,2.5,2.4" @change="parseDemand"></textarea>
      </label>
    </template>

    <template v-else-if="pu.mode === 'external'">
      <p class="b2-note">B2 hook — callable that returns daily uptake per node.</p>
      <label>Callable ref (module:attr)
        <input type="text" v-model="pu.callable_ref" placeholder="dndc.n_uptake:compute" class="wide" />
      </label>
    </template>
  </div>
</template>

<script setup lang="ts">
import { ref, computed } from "vue";
import { useDndcSeamStore } from "../../stores/dndcSeam";

const store = useDndcSeamStore();
const pu = computed(() => store.inputs.plant_n_uptake);
const demandText = ref((pu.value.daily_demand_kg_n_ha ?? []).join(","));

function parseDemand() {
  store.inputs.plant_n_uptake.daily_demand_kg_n_ha = demandText.value
    .split(",").map((s) => parseFloat(s.trim())).filter((n) => !isNaN(n));
}
</script>

<style scoped>
.section { padding: 6px 0; }
.hint { color: #666; font-size: 0.9em; margin: 0 0 8px; }
.b2-note { color: #884400; font-size: 11.5px; margin: 4px 0 8px; font-style: italic; }
.row { margin-bottom: 8px; }
.grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(220px, 1fr)); gap: 10px; margin-bottom: 8px; }
label { display: flex; flex-direction: column; gap: 3px; font-size: 12px; }
select, input { font-size: 12px; padding: 3px 6px; }
textarea { width: 100%; font-family: monospace; font-size: 12px; margin-top: 4px; }
.wide { width: 360px; }
</style>
