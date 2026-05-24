<template>
  <div class="section">
    <p class="hint">N transformation kinetics. When DNDC live coupling is enabled (B2), this switches to live mode automatically.</p>
    <div class="row">
      <label>Mode
        <select v-model="nt.mode">
          <option value="constant_rates">Constant first-order rates</option>
          <option value="external_callable">External callable (B2)</option>
          <option value="lookup_table">Lookup table</option>
        </select>
      </label>
    </div>

    <template v-if="nt.mode === 'constant_rates'">
      <div class="grid">
        <label>Mineralization k (d&#x207B;&#xB9;)
          <input type="number" v-model.number="nt.k_mineralization_d" step="0.001" min="0" placeholder="optional" />
        </label>
        <label>Nitrification k (d&#x207B;&#xB9;)
          <input type="number" v-model.number="nt.k_nitrification_d" step="0.01" min="0" placeholder="0.1" />
        </label>
        <label>Denitrification k (d&#x207B;&#xB9;)
          <input type="number" v-model.number="nt.k_denitrification_d" step="0.001" min="0" placeholder="optional" />
        </label>
        <label>Volatilization k (d&#x207B;&#xB9;)
          <input type="number" v-model.number="nt.k_volatilization_d" step="0.001" min="0" placeholder="optional" />
        </label>
      </div>
    </template>

    <template v-else-if="nt.mode === 'external_callable'">
      <p class="b2-note">B2 hook: provide a Python dotted path to the callable that computes N rates.</p>
      <label>Callable ref (module:attr)
        <input type="text" v-model="nt.callable_ref" placeholder="dndc.n_module:compute_rates" class="wide" />
      </label>
    </template>

    <template v-else-if="nt.mode === 'lookup_table'">
      <p class="b2-note">Lookup table mode lands in M2.x. Provide a path to a NetCDF/CSV table.</p>
    </template>
  </div>
</template>

<script setup lang="ts">
import { computed } from "vue";
import { useDndcSeamStore } from "../../stores/dndcSeam";

const store = useDndcSeamStore();
const nt = computed(() => store.inputs.n_transform);
</script>

<style scoped>
.section { padding: 6px 0; }
.hint { color: #666; font-size: 0.9em; margin: 0 0 8px; }
.b2-note { color: #884400; font-size: 11.5px; margin: 4px 0 8px; font-style: italic; }
.row { margin-bottom: 8px; }
.grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 10px; }
label { display: flex; flex-direction: column; gap: 3px; font-size: 12px; }
select, input { font-size: 12px; padding: 3px 6px; }
.wide { width: 360px; }
</style>
