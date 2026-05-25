<template>
  <div class="inversion">
    <h2>Inversion Studio — F3</h2>
    <div class="form">
      <label>Scenario dir <input v-model="scenarioDir" /></label>
      <label>Params (target:lo:hi[:transform], one per line)
        <textarea v-model="paramText" rows="3"></textarea>
      </label>
      <label>Obs: name, z_cm, time_day, value, sigma (one per line)
        <textarea v-model="obsText" rows="4"></textarea>
      </label>
      <label>Backend
        <select v-model="backend">
          <option value="auto">auto</option>
          <option value="lm">scipy LM (fast)</option>
          <option value="ies">PEST/PyEMU IES</option>
        </select>
      </label>
      <label>Max nfev (LM) / N real (IES)
        <input v-model.number="nIter" type="number" min="1" /></label>
      <button @click="run" :disabled="running">Calibrate</button>
    </div>

    <p v-if="running">Running… ({{ Math.round(elapsed) }}s)</p>
    <p v-if="error" class="err">{{ error }}</p>

    <div v-if="result">
      <p>{{ result.backend }} — {{ result.n_forward_calls }} forward calls, {{ result.wall_s.toFixed(1) }}s</p>
      <h3>Best-fit parameters</h3>
      <ParamPosteriorBar :param-names="paramNames"
                         :best-params="result.best_params"
                         :ci-lo="result.parameter_ci_lo"
                         :ci-hi="result.parameter_ci_hi" />
      <h3>Convergence</h3>
      <ResidualsPlot :history="result.objective_history" />
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed } from "vue";
import { inversion, type InversionResult } from "../../api";
import ParamPosteriorBar from "../../components/ParamPosteriorBar.vue";
import ResidualsPlot from "../../components/ResidualsPlot.vue";

const scenarioDir = ref("tests/fixtures/infiltr_v1/inputs");
const paramText = ref("materials[0].alpha:0.005:0.05:log");
const obsText = ref("theta_z30_d1, -30, 1.0, 0.31, 0.02");
const backend = ref<"auto" | "lm" | "ies">("lm");
const nIter = ref(50);
const result = ref<InversionResult | null>(null);
const error = ref<string | null>(null);
const running = ref(false);
const elapsed = ref(0);

const paramNames = computed(() =>
  paramText.value.trim().split("\n").map(l => {
    const target = l.split(":")[0].trim();
    return target.split(".").pop() ?? target;
  }));

function _parseParams() {
  return paramText.value.trim().split("\n").map(l => {
    const parts = l.split(":");
    const target = parts[0].trim();
    const lo = parseFloat(parts[1]); const hi = parseFloat(parts[2]);
    const transform = (parts[3]?.trim() as any) ?? "linear";
    const name = target.split(".").pop() ?? target;
    return { name, target, bounds: [lo, hi] as [number, number], transform };
  });
}

function _parseObs() {
  const lines = obsText.value.trim().split("\n");
  const specs = lines.map(l => {
    const [name, z, t] = l.split(",").map(x => x.trim());
    return { name, kind: "theta" as const, location: { z_cm: parseFloat(z) }, time_day: parseFloat(t) };
  });
  const values = lines.map(l => parseFloat(l.split(",")[3].trim()));
  const sigmas = lines.map(l => parseFloat(l.split(",")[4].trim()));
  return { specs, values, sigmas };
}

async function run() {
  running.value = true; error.value = null; result.value = null;
  const t0 = Date.now();
  const tick = setInterval(() => { elapsed.value = (Date.now() - t0) / 1000; }, 200);
  try {
    result.value = await inversion.run(backend.value, {
      scenario_dir: scenarioDir.value,
      params: _parseParams(),
      obs_inline: _parseObs(),
      max_nfev: nIter.value,
      n_real: nIter.value, n_iter: 3,
    });
  } catch (e: any) {
    error.value = e.message ?? String(e);
  } finally {
    clearInterval(tick); running.value = false;
  }
}
</script>

<style scoped>
.inversion { padding: 16px; max-width: 900px; }
.form { display: grid; grid-template-columns: 200px 1fr; gap: 6px; align-items: start; margin-bottom: 12px; }
.form input, .form select, .form textarea { padding: 4px; font-family: inherit; }
.form button { grid-column: 1 / 3; padding: 8px; margin-top: 8px; }
.err { color: #c00; }
h3 { margin-top: 16px; font-size: 13px; }
</style>
