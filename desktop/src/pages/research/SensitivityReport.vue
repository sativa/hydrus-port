<template>
  <div class="sens-report">
    <h2>Sensitivity Report — F2</h2>

    <div class="form">
      <label>Scenario dir
        <input v-model="scenarioDir" />
      </label>
      <label>Params (one per line, target:lo:hi[:transform])
        <textarea v-model="paramText" rows="3"></textarea>
      </label>
      <label>Obs kind
        <select v-model="obsKind">
          <option value="theta">theta (water content)</option>
          <option value="h">h (pressure head)</option>
          <option value="flux">flux</option>
        </select>
      </label>
      <label>Obs depth (cm)
        <input v-model.number="obsDepth" type="number" />
      </label>
      <label>Obs time (days)
        <input v-model.number="obsTime" type="number" step="0.1" />
      </label>
      <label>Method
        <select v-model="method">
          <option value="morris">Morris EE (screening)</option>
          <option value="sobol">Sobol (variance)</option>
          <option value="fast">FAST</option>
          <option value="pawn">PAWN (distribution)</option>
        </select>
      </label>
      <label>N
        <input v-model.number="n" type="number" min="1" />
      </label>
      <label>Workers
        <input v-model.number="workers" type="number" min="1" />
      </label>
      <label>Seed (optional)
        <input v-model.number="seed" type="number" placeholder="leave blank for random" />
      </label>
      <button class="run-btn" @click="run" :disabled="running">
        {{ running ? "Running…" : "Run sensitivity analysis" }}
      </button>
    </div>

    <p v-if="running" class="status">
      Running… {{ Math.round(elapsed) }}s elapsed
    </p>
    <p v-if="error" class="err">{{ error }}</p>

    <div v-if="result" class="result-panel">
      <p class="summary">
        <strong>{{ result.method }}</strong> —
        {{ result.sample_size }} samples,
        {{ result.forward_cost_s.toFixed(1) }}s forward cost
      </p>

      <MorrisEEPlot
        v-if="result.method === 'morris'"
        :param-names="result.param_names"
        :mu-star="(result.indices['mu_star'] as number[][])[0]"
        :sigma="(result.indices['sigma'] as number[][])[0]"
      />

      <SensitivityIndicesBar
        v-else
        :param-names="result.param_names"
        :indices="firstObsIndices"
        :title="`${result.method} indices — ${result.obs_names[0] ?? ''}`"
      />

      <details class="diag">
        <summary>Diagnostics</summary>
        <pre>{{ JSON.stringify(result.diagnostics, null, 2) }}</pre>
      </details>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed } from "vue";
import { sensitivity, type SensitivityResult } from "../../api";
import SensitivityIndicesBar from "../../components/SensitivityIndicesBar.vue";
import MorrisEEPlot from "../../components/MorrisEEPlot.vue";

const scenarioDir = ref("tests/fixtures/infiltr_v1/inputs");
const paramText = ref(
  "materials[0].alpha:0.005:0.05:log\nmaterials[0].n:1.1:2.0:linear",
);
const obsKind = ref<"theta" | "h" | "flux">("theta");
const obsDepth = ref(-30);
const obsTime = ref(1.0);
const method = ref<"morris" | "sobol" | "fast" | "pawn">("morris");
const n = ref(20);
const workers = ref(2);
const seed = ref<number | undefined>(undefined);

const result = ref<SensitivityResult | null>(null);
const error = ref<string | null>(null);
const running = ref(false);
const elapsed = ref(0);

function _parseParams() {
  return paramText.value
    .trim()
    .split("\n")
    .filter((l) => l.trim().length > 0)
    .map((line) => {
      const parts = line.trim().split(":");
      const target = parts[0].trim();
      const lo = parseFloat(parts[1]);
      const hi = parseFloat(parts[2]);
      const transform = (parts[3]?.trim() ?? "linear") as
        | "linear"
        | "log"
        | "logit";
      const name = target.split(".").pop() ?? target;
      return { name, target, bounds: [lo, hi] as [number, number], transform };
    });
}

async function run() {
  running.value = true;
  error.value = null;
  result.value = null;
  elapsed.value = 0;

  const t0 = Date.now();
  const tick = setInterval(() => {
    elapsed.value = (Date.now() - t0) / 1000;
  }, 200);

  try {
    result.value = await sensitivity.run(method.value, {
      scenario_dir: scenarioDir.value,
      params: _parseParams(),
      obs: [
        {
          name: `${obsKind.value}_${obsDepth.value}_${obsTime.value}`,
          kind: obsKind.value,
          location: { z_cm: obsDepth.value },
          time_day: obsTime.value,
        },
      ],
      n: n.value,
      workers: workers.value,
      seed: seed.value,
    });
  } catch (e: any) {
    error.value = e.message ?? String(e);
  } finally {
    clearInterval(tick);
    running.value = false;
  }
}

/** For Sobol/FAST/PAWN: extract the first observable's index values,
 *  filtering out *_conf keys so only primary indices appear as bars. */
const firstObsIndices = computed((): Record<string, number[]> => {
  if (!result.value) return {};
  const out: Record<string, number[]> = {};
  for (const [k, v] of Object.entries(result.value.indices)) {
    if (k.endsWith("_conf")) continue;
    const row = (v as number[][])[0];
    if (Array.isArray(row) && row.every((x) => typeof x === "number")) {
      out[k] = row;
    }
  }
  return out;
});
</script>

<style scoped>
.sens-report {
  padding: 16px;
  overflow-y: auto;
  height: 100%;
  box-sizing: border-box;
}
.sens-report h2 {
  margin: 0 0 12px;
  font-size: 14px;
  color: var(--accent);
}
.form {
  display: grid;
  grid-template-columns: 160px 1fr;
  gap: 5px 8px;
  align-items: start;
  margin-bottom: 14px;
}
.form label {
  display: contents;
  font-size: 12px;
  color: var(--muted);
}
.form input,
.form select,
.form textarea {
  padding: 4px 6px;
  font-family: inherit;
  font-size: 12px;
  background: var(--panel-2, #1e1e1e);
  color: var(--text);
  border: 1px solid var(--border);
  border-radius: 3px;
  width: 100%;
  box-sizing: border-box;
}
.run-btn {
  grid-column: 1 / 3;
  padding: 8px;
  margin-top: 6px;
  font-size: 13px;
  font-weight: 600;
  background: var(--accent);
  color: #fff;
  border: none;
  border-radius: 4px;
  cursor: pointer;
}
.run-btn:disabled {
  opacity: 0.5;
  cursor: default;
}
.status {
  font-size: 12px;
  color: var(--muted);
}
.err {
  color: #c00;
  font-size: 12px;
}
.result-panel {
  margin-top: 10px;
}
.summary {
  font-size: 12px;
  margin-bottom: 8px;
}
.diag {
  margin-top: 10px;
  font-size: 11px;
}
.diag pre {
  background: var(--panel-2, #1e1e1e);
  border: 1px solid var(--border);
  border-radius: 3px;
  padding: 8px;
  overflow-x: auto;
}
</style>
