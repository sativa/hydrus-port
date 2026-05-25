<template>
  <div class="uq">
    <h2>UQ Explorer — F5</h2>
    <p class="hint">Upload a batch parquet (from M3 sweep) and provide observations
       to run a GLUE behavioral filter.</p>
    <label>Batch parquet path
      <input v-model="parquetPath" placeholder="/tmp/sweep.parquet" />
    </label>
    <label>Obs values (comma-separated)
      <input v-model="obsValues" placeholder="0.31, 0.30, 0.33" /></label>
    <label>Obs sigmas (comma-separated)
      <input v-model="obsSigmas" placeholder="0.02, 0.02, 0.02" /></label>
    <label>Likelihood cutoff
      <input v-model.number="cutoff" type="number" step="0.05" min="0" max="1" /></label>
    <button @click="run" :disabled="running">Run GLUE</button>
    <p v-if="error" class="err">{{ error }}</p>
    <pre v-if="result">{{ JSON.stringify(result.quantiles, null, 2) }}</pre>
  </div>
</template>

<script setup lang="ts">
import { ref } from "vue";

const parquetPath = ref("");
const obsValues = ref("");
const obsSigmas = ref("");
const cutoff = ref(0.5);
const error = ref<string | null>(null);
const result = ref<any | null>(null);
const running = ref(false);

async function run() {
  // Note: the GUI version requires reading the parquet client-side OR
  // sending the parquet path to a future server-side endpoint. For now
  // this page is a placeholder explaining the API; full GUI integration
  // is a follow-up.
  error.value = "UQ GUI requires either parquet upload (TBD) or use of the CLI: \n" +
                "hydrus research uq glue " + parquetPath.value +
                " --obs " + obsValues.value +
                " --sigmas " + obsSigmas.value +
                " --cutoff " + cutoff.value + " --out result.json";
}
</script>

<style scoped>
.uq { padding: 16px; max-width: 720px; }
label { display: block; margin: 6px 0; }
input { width: 100%; padding: 4px; }
.err { color: #c00; white-space: pre-wrap; }
pre { background: #f4f4f4; padding: 12px; }
.hint { color: #666; font-size: 0.9em; }
</style>
