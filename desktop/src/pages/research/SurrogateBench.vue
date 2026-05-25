<template>
  <div class="surr">
    <h2>Surrogate Bench — F8</h2>
    <label>Batch parquet path <input v-model="parquet" placeholder="/tmp/sweep.parquet" /></label>
    <label>Type
      <select v-model="type">
        <option value="gp">sklearn GP</option>
        <option value="pck">PC-Kriging</option>
      </select>
    </label>
    <button @click="train" :disabled="busy">Train</button>
    <p v-if="modelId">Trained: {{ modelId }} <button @click="evaluate">Evaluate</button></p>
    <p v-if="error" class="err">{{ error }}</p>
    <pre v-if="metrics">{{ JSON.stringify(metrics, null, 2) }}</pre>
  </div>
</template>

<script setup lang="ts">
import { ref } from "vue";
import { surrogate } from "../../api";
const parquet = ref(""); const type = ref<"gp" | "pck">("gp");
const modelId = ref<string | null>(null);
const error = ref<string | null>(null);
const metrics = ref<any | null>(null);
const busy = ref(false);
async function train() {
  busy.value = true; error.value = null; metrics.value = null;
  try {
    const r = await surrogate.train({ batch_parquet: parquet.value, type: type.value });
    modelId.value = r.model_id;
  } catch (e: any) { error.value = e.message; }
  finally { busy.value = false; }
}
async function evaluate() {
  if (!modelId.value) return;
  try { metrics.value = await surrogate.evaluate(modelId.value); }
  catch (e: any) { error.value = e.message; }
}
</script>

<style scoped>
.surr { padding: 16px; max-width: 720px; }
label { display: block; margin: 6px 0; }
input, select { padding: 4px; width: 100%; }
.err { color: #c00; } pre { background: #f4f4f4; padding: 12px; }
</style>
