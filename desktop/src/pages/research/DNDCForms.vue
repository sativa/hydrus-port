<template>
  <div class="dndc-forms">
    <header>
      <h2>DNDC Inputs</h2>
      <div class="controls">
        <button @click="store.reset">Reset</button>
        <button @click="validate">Validate</button>
        <span v-if="store.serverError" class="err">{{ store.serverError }}</span>
        <span v-else-if="lastOk" class="ok">&#x2713; valid</span>
      </div>
    </header>
    <details open><summary>1. Atmospheric (daily P, ET&#x2080; or weather)</summary>
      <DNDCSectionAtm /></details>
    <details><summary>2. ET partition (LAI / split / Kc dual)</summary>
      <DNDCSectionEt /></details>
    <details><summary>3. Root growth (depth + density profile)</summary>
      <DNDCSectionRoot /></details>
    <details><summary>4. Feddes water-stress parameters</summary>
      <DNDCSectionFeddes /></details>
    <details><summary>5. Fertilizer events</summary>
      <DNDCSectionFertEvents /></details>
    <details><summary>6. Irrigation events</summary>
      <DNDCSectionIrrigEvents /></details>
    <details><summary>7. N transformation</summary>
      <DNDCSectionNTransform /></details>
    <details><summary>8. Plant N uptake</summary>
      <DNDCSectionPlantN /></details>
    <details><summary>9. State exchange (initial profile + writeback)</summary>
      <DNDCSectionState /></details>
    <details><summary>10. Soil temperature</summary>
      <DNDCSectionSoilTemp /></details>
    <details><summary>11. Residue / mulch</summary>
      <DNDCSectionResidue /></details>
  </div>
</template>

<script setup lang="ts">
import { ref } from "vue";
import { useDndcSeamStore } from "../../stores/dndcSeam";
import { dndc } from "../../api";
import DNDCSectionAtm from "../../components/dndc/DNDCSectionAtm.vue";
import DNDCSectionEt from "../../components/dndc/DNDCSectionEt.vue";
import DNDCSectionRoot from "../../components/dndc/DNDCSectionRoot.vue";
import DNDCSectionFeddes from "../../components/dndc/DNDCSectionFeddes.vue";
import DNDCSectionFertEvents from "../../components/dndc/DNDCSectionFertEvents.vue";
import DNDCSectionIrrigEvents from "../../components/dndc/DNDCSectionIrrigEvents.vue";
import DNDCSectionNTransform from "../../components/dndc/DNDCSectionNTransform.vue";
import DNDCSectionPlantN from "../../components/dndc/DNDCSectionPlantN.vue";
import DNDCSectionState from "../../components/dndc/DNDCSectionState.vue";
import DNDCSectionSoilTemp from "../../components/dndc/DNDCSectionSoilTemp.vue";
import DNDCSectionResidue from "../../components/dndc/DNDCSectionResidue.vue";

const store = useDndcSeamStore();
const lastOk = ref(false);

async function validate() {
  store.serverError = null;
  lastOk.value = false;
  try {
    await dndc.validate(store.inputs);
    lastOk.value = true;
  } catch (e: unknown) {
    store.serverError = e instanceof Error ? e.message : String(e);
  }
}
</script>

<style scoped>
.dndc-forms { padding: 16px; max-width: 920px; }
header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px; }
h2 { margin: 0; font-size: 15px; color: var(--accent, #3a6ea5); }
.controls { display: flex; gap: 8px; align-items: center; }
.err { color: #c00; font-size: 12px; }
.ok  { color: #060; font-size: 12px; }
details { margin: 6px 0; border: 1px solid var(--border, #ddd); border-radius: 4px; padding: 6px 12px; }
summary { cursor: pointer; font-weight: 600; font-size: 12.5px; }
</style>
