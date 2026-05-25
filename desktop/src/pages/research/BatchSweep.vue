<template>
  <div class="batch-sweep">
    <h2>Batch Sweep — F4</h2>
    <div class="form">
      <label>Scenario directory
        <input v-model="scenarioDir" placeholder="tests/fixtures/infiltr_v1/inputs" />
      </label>
      <label>Param spec (target:lo:hi:transform)
        <input v-model="paramSpec" placeholder="materials[0].alpha:0.005:0.05:log" />
      </label>
      <label>Obs depth (cm, negative = below surface)
        <input v-model.number="obsDepth" type="number" />
      </label>
      <label>Obs time (days)
        <input v-model.number="obsTime" type="number" step="0.1" />
      </label>
      <label>N samples <input v-model.number="n" type="number" min="1" /></label>
      <label>Sampler
        <select v-model="sampler">
          <option value="lhs">Latin Hypercube</option>
          <option value="grid">Full-factorial grid</option>
          <option value="uniform">Uniform random</option>
        </select>
      </label>
      <label>Workers <input v-model.number="workers" type="number" min="1" /></label>
      <button @click="run" :disabled="store.polling">Start sweep</button>
      <button @click="store.stop" :disabled="!store.polling">Stop polling</button>
    </div>

    <div v-if="store.status" class="status">
      <p>State: <b>{{ store.status.state }}</b></p>
      <p>Progress: {{ store.status.n_done ?? 0 }} / {{ store.status.n_total }}</p>
      <p v-if="store.status.n_failed">Failed: {{ store.status.n_failed }}</p>
    </div>
    <p v-if="store.error" class="err">{{ store.error }}</p>

    <p v-if="store.status?.state === 'done' && store.jobId">
      <a :href="resultLink">Download parquet</a>
    </p>
  </div>
</template>

<script setup lang="ts">
import { ref, computed } from "vue";
import { useBatchStore } from "../../stores/batch";
import { batch as batchApi } from "../../api";

const store = useBatchStore();

const scenarioDir = ref("tests/fixtures/infiltr_v1/inputs");
const paramSpec = ref("materials[0].alpha:0.005:0.05:log");
const obsDepth = ref(-30);
const obsTime = ref(1.0);
const n = ref(8);
const sampler = ref<"lhs" | "grid" | "uniform">("lhs");
const workers = ref(2);

function _parseSpec(s: string) {
  const parts = s.split(":");
  const target = parts[0];
  const lo = parseFloat(parts[1]); const hi = parseFloat(parts[2]);
  const transform = (parts[3] as any) ?? "linear";
  const name = target.split(".").pop() ?? target;
  return { name, target, bounds: [lo, hi] as [number, number], transform };
}

async function run() {
  await store.start({
    scenario_dir: scenarioDir.value,
    params: [_parseSpec(paramSpec.value)],
    obs: [{
      name: `theta_${obsDepth.value}cm_d${obsTime.value}`,
      kind: "theta", location: { z_cm: obsDepth.value }, time_day: obsTime.value,
    }],
    n: n.value, sampler: sampler.value, workers: workers.value,
  });
}

const resultLink = computed(() =>
  store.jobId ? batchApi.resultUrl(store.jobId) : "");
</script>

<style scoped>
.batch-sweep { padding: 16px; max-width: 720px; }
.form { display: grid; grid-template-columns: 200px 1fr; gap: 6px; align-items: center; margin-bottom: 12px; }
.form input, .form select { padding: 4px; }
.form button { grid-column: 1 / 3; padding: 8px; margin-top: 8px; }
.status { background: #f4f4f4; padding: 12px; border-radius: 4px; }
.err { color: #c00; }
</style>
