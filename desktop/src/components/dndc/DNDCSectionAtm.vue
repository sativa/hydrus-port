<template>
  <div class="section">
    <p class="hint">Daily atmospheric series. Either provide pet_cm directly or
       leave it blank and supply t_min/t_max/rh/wind/solar for server-side FAO-56.</p>
    <label>Paste CSV (date,precip_cm[,pet_cm])
      <textarea v-model="csvText" rows="6" placeholder="2026-05-01,0.0,0.4&#10;2026-05-02,0.3,0.5"></textarea>
    </label>
    <button @click="parse">Parse</button>
    <p class="status">{{ store.inputs.atm.dates.length }} rows loaded</p>
  </div>
</template>

<script setup lang="ts">
import { ref } from "vue";
import { useDndcSeamStore } from "../../stores/dndcSeam";

const store = useDndcSeamStore();
const csvText = ref("");

function parse() {
  const lines = csvText.value.trim().split(/\r?\n/).filter(Boolean);
  const dates: string[] = [];
  const precip: number[] = [];
  const pet: number[] = [];
  let havePet = false;
  for (const ln of lines) {
    const [d, p, e] = ln.split(",").map((s) => s.trim());
    dates.push(d);
    precip.push(parseFloat(p));
    if (e !== undefined && e !== "") {
      havePet = true;
      pet.push(parseFloat(e));
    }
  }
  store.inputs.atm.dates = dates;
  store.inputs.atm.precip_cm = precip;
  store.inputs.atm.pet_cm = havePet ? pet : null;
}
</script>

<style scoped>
.section { padding: 6px 0; }
textarea { width: 100%; font-family: monospace; font-size: 12px; margin-top: 4px; }
.hint { color: #666; font-size: 0.9em; margin: 0 0 8px; }
.status { font-size: 12px; color: var(--muted, #888); margin: 4px 0 0; }
</style>
