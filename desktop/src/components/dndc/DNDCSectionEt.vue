<template>
  <div class="section">
    <p class="hint">Controls how total ET is split between soil evaporation and canopy transpiration.</p>
    <div class="row">
      <label>Mode
        <select v-model="et.mode">
          <option value="lai_beer">LAI-Beer (default)</option>
          <option value="explicit_split">Explicit E fraction</option>
          <option value="kc_dual">Kc dual (FAO-56)</option>
        </select>
      </label>
      <label v-if="et.mode !== 'kc_dual'">
        Extinction k
        <input type="number" v-model.number="et.extinction_k" step="0.01" min="0" max="2" />
      </label>
    </div>

    <template v-if="et.mode === 'lai_beer'">
      <label>LAI series (comma-separated, one value per day)
        <textarea v-model="laiText" rows="3" placeholder="2.0,2.2,2.5,3.0" @change="parseLai"></textarea>
      </label>
    </template>

    <template v-else-if="et.mode === 'explicit_split'">
      <label>Explicit E fraction per day (0-1, comma-separated)
        <textarea v-model="eFracText" rows="3" placeholder="0.4,0.35,0.3" @change="parseEFrac"></textarea>
      </label>
    </template>

    <template v-else-if="et.mode === 'kc_dual'">
      <label>Kcb series (basal crop coefficient, comma-separated)
        <textarea v-model="kcbText" rows="3" placeholder="0.15,0.5,1.1" @change="parseKcb"></textarea>
      </label>
    </template>
  </div>
</template>

<script setup lang="ts">
import { ref, computed } from "vue";
import { useDndcSeamStore } from "../../stores/dndcSeam";

const store = useDndcSeamStore();
const et = computed(() => store.inputs.et);

function parseNumbers(text: string): number[] {
  return text.split(",").map((s) => parseFloat(s.trim())).filter((n) => !isNaN(n));
}

const laiText = ref((et.value.lai ?? []).join(","));
const eFracText = ref((et.value.explicit_e_frac ?? []).join(","));
const kcbText = ref((et.value.kcb ?? []).join(","));

function parseLai() { store.inputs.et.lai = parseNumbers(laiText.value); }
function parseEFrac() { store.inputs.et.explicit_e_frac = parseNumbers(eFracText.value); }
function parseKcb() { store.inputs.et.kcb = parseNumbers(kcbText.value); }
</script>

<style scoped>
.section { padding: 6px 0; }
.hint { color: #666; font-size: 0.9em; margin: 0 0 8px; }
.row { display: flex; gap: 16px; flex-wrap: wrap; margin-bottom: 8px; }
label { display: flex; flex-direction: column; gap: 3px; font-size: 12px; }
select, input[type="number"] { font-size: 12px; padding: 3px 6px; }
textarea { width: 100%; font-family: monospace; font-size: 12px; margin-top: 4px; }
</style>
