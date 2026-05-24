<template>
  <div class="section">
    <p class="hint">Soil profile grid and optional initial-state profiles (theta, h, solute, temperature). Writeback is a B2 feature.</p>

    <label>Z grid (cm, comma-separated; 0 = surface, negative = depth)
      <textarea v-model="zGridText" rows="2" placeholder="0.0,-5,-10,-25,-50,-100" @change="parseZGrid"></textarea>
    </label>

    <div class="grid">
      <label>Initial theta (cm&#xB3;/cm&#xB3;, comma-separated)
        <textarea v-model="thetaText" rows="2" placeholder="0.30,0.31,0.32,0.33,0.34,0.35" @change="parseTheta"></textarea>
      </label>
      <label>Initial h (cm, comma-separated)
        <textarea v-model="hText" rows="2" placeholder="-100,-120,-130,-150,-200,-300" @change="parseH"></textarea>
      </label>
    </div>

    <div class="row">
      <label class="inline">
        <input type="checkbox" v-model="state.writeback_daily" />
        Writeback daily (B2 only)
      </label>
      <label v-if="state.writeback_daily">Writeback path
        <input type="text" v-model="state.writeback_path" placeholder="/tmp/dndc_writeback" class="wide" />
      </label>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed } from "vue";
import { useDndcSeamStore } from "../../stores/dndcSeam";

const store = useDndcSeamStore();
const state = computed(() => store.inputs.state);

function parseArr(text: string): number[] {
  return text.split(",").map((s) => parseFloat(s.trim())).filter((n) => !isNaN(n));
}

const zGridText = ref((state.value.z_grid_cm ?? []).join(","));
const thetaText = ref((state.value.initial_theta ?? []).join(","));
const hText = ref((state.value.initial_h ?? []).join(","));

function parseZGrid() { store.inputs.state.z_grid_cm = parseArr(zGridText.value); }
function parseTheta() {
  const v = parseArr(thetaText.value);
  store.inputs.state.initial_theta = v.length ? v : undefined;
}
function parseH() {
  const v = parseArr(hText.value);
  store.inputs.state.initial_h = v.length ? v : undefined;
}
</script>

<style scoped>
.section { padding: 6px 0; }
.hint { color: #666; font-size: 0.9em; margin: 0 0 8px; }
.grid { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin: 8px 0; }
.row { display: flex; gap: 16px; align-items: flex-end; margin-top: 8px; }
label { display: flex; flex-direction: column; gap: 3px; font-size: 12px; }
label.inline { flex-direction: row; align-items: center; gap: 6px; font-size: 12px; }
textarea { width: 100%; font-family: monospace; font-size: 12px; margin-top: 4px; }
input[type="text"] { font-size: 12px; padding: 3px 6px; }
.wide { width: 280px; }
</style>
