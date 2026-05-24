<template>
  <div class="soil-library">
    <h2>Soil Library — F1 Pedotransfer</h2>
    <div class="row">
      <div class="col">
        <h3>Texture triangle</h3>
        <TextureTriangle :sand="sand" :silt="silt" :clay="clay" @update="onPick" />
        <label>USDA class shortcut:
          <select v-model="selectedClass" @change="onClassPick">
            <option value="">— pick a class —</option>
            <option v-for="(c, name) in usda" :key="name" :value="name">{{ name }}</option>
          </select>
        </label>
      </div>
      <div class="col">
        <h3>Inputs</h3>
        <label>sand% <input type="number" v-model.number="sand" min="0" max="100" /></label>
        <label>silt% <input type="number" v-model.number="silt" min="0" max="100" /></label>
        <label>clay% <input type="number" v-model.number="clay" min="0" max="100" /></label>
        <label>BD g/cm³ <input type="number" v-model.number="bd" step="0.01" /></label>
        <label>OM pct <input type="number" v-model.number="om" step="0.1" /></label>
        <label>Method:
          <select v-model="method">
            <option value="rosetta3_auto">ROSETTA-3 (auto hierarchy)</option>
            <option value="carsel_parrish">Carsel-Parrish 1988</option>
            <option value="wosten">Wösten HYPRES 1999</option>
          </select>
        </label>
        <button @click="predict">Compute VG params</button>
        <pre v-if="result" class="result">{{ resultText }}</pre>
        <p v-if="error" class="err">{{ error }}</p>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted } from "vue";
import TextureTriangle from "../../components/TextureTriangle.vue";
import { ptf, type PTFResult } from "../../api";

const sand = ref(40), silt = ref(40), clay = ref(20);
const bd = ref<number | null>(1.4), om = ref<number | null>(1.5);
const method = ref<"rosetta3_auto" | "carsel_parrish" | "wosten">("rosetta3_auto");
const usda = ref<Record<string, { sand_pct: number; silt_pct: number; clay_pct: number }>>({});
const selectedClass = ref("");
const result = ref<PTFResult | null>(null);
const error = ref<string | null>(null);

onMounted(async () => { usda.value = await ptf.usdaClasses(); });

function onPick(v: { sand_pct: number; silt_pct: number; clay_pct: number }) {
  sand.value = Math.round(v.sand_pct); silt.value = Math.round(v.silt_pct);
  clay.value = Math.round(v.clay_pct);
}
function onClassPick() {
  if (selectedClass.value && usda.value[selectedClass.value]) {
    const c = usda.value[selectedClass.value];
    sand.value = c.sand_pct; silt.value = c.silt_pct; clay.value = c.clay_pct;
  }
}
async function predict() {
  error.value = null; result.value = null;
  try {
    result.value = await ptf.predict({
      sand_pct: sand.value, silt_pct: silt.value, clay_pct: clay.value,
      bulk_density_g_cm3: bd.value ?? undefined,
      organic_matter_pct: om.value ?? undefined,
      method: method.value,
    });
  } catch (e: any) { error.value = e.message ?? String(e); }
}
const resultText = computed(() => result.value
  ? `theta_r = ${result.value.theta_r.toFixed(4)}
theta_s = ${result.value.theta_s.toFixed(4)}
alpha   = ${result.value.alpha.toFixed(4)}  1/cm
n       = ${result.value.n.toFixed(3)}
Ks      = ${result.value.Ks.toFixed(2)}  cm/day
L       = ${result.value.L.toFixed(2)}
method  = ${result.value.method}`
  : "");
</script>

<style scoped>
.soil-library { padding: 16px; max-width: 1200px; }
.row { display: flex; gap: 16px; }
.col { flex: 1; }
label { display: block; margin: 6px 0; }
input, select { margin-left: 6px; }
.result { background: #f4f4f4; padding: 12px; }
.err { color: #c00; }
</style>
