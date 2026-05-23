<script setup lang="ts">
import { ref, watch, computed } from "vue";
import { api, type Scenario_JSON, type Material } from "../api";

const props = defineProps<{ scenarioPath: string | null }>();
const emit = defineEmits<{ (e: "scenario-saved", path: string): void }>();

const scen = ref<Scenario_JSON | null>(null);
const dirty = ref(false);
const busy = ref(false);
const err = ref<string | null>(null);
const saveMsg = ref<string | null>(null);

watch(() => props.scenarioPath, async (p) => {
  scen.value = null; dirty.value = false; err.value = null;
  if (!p) return;
  try {
    scen.value = await api.readScenario(p);
  } catch (e: any) {
    err.value = String(e);
  }
}, { immediate: true });

function markDirty() { dirty.value = true; saveMsg.value = null; }

async function save() {
  if (!scen.value || !props.scenarioPath) return;
  busy.value = true; err.value = null; saveMsg.value = null;
  try {
    await api.writeScenario(props.scenarioPath, scen.value);
    dirty.value = false;
    saveMsg.value = `saved ${new Date().toLocaleTimeString()}`;
    emit("scenario-saved", props.scenarioPath);
  } catch (e: any) {
    err.value = String(e);
  } finally {
    busy.value = false;
  }
}

async function reload() {
  if (!props.scenarioPath) return;
  busy.value = true; err.value = null; saveMsg.value = null;
  try {
    scen.value = await api.readScenario(props.scenarioPath);
    dirty.value = false;
  } catch (e: any) {
    err.value = String(e);
  } finally {
    busy.value = false;
  }
}

function addMaterial() {
  if (!scen.value) return;
  const cur = scen.value.materials;
  const tmpl: Material = cur.length
    ? { ...cur[cur.length - 1] }
    : { thr: 0.05, ths: 0.40, tha: 0.05, thm: 0.40,
        alpha: 0.04, n: 1.5, Ks: 0.01, Kk: 0.01, thk: 0.40 };
  cur.push(tmpl);
  markDirty();
}

function removeMaterial(idx: number) {
  if (!scen.value || scen.value.materials.length <= 1) return;
  scen.value.materials.splice(idx, 1);
  markDirty();
}

// Soil texture presets (Rosetta H1 USDA classes — abbreviated):
// θr, θs, α (1/cm), n, Ks (cm/d). Stored in cm/d; expose as-is.
const presets: Record<string, Partial<Material>> = {
  "sand":        { thr: 0.045, ths: 0.43, alpha: 0.145,  n: 2.68, Ks: 712.8 },
  "loamy sand":  { thr: 0.057, ths: 0.41, alpha: 0.124,  n: 2.28, Ks: 350.2 },
  "sandy loam":  { thr: 0.065, ths: 0.41, alpha: 0.075,  n: 1.89, Ks: 106.1 },
  "loam":        { thr: 0.078, ths: 0.43, alpha: 0.036,  n: 1.56, Ks: 24.96 },
  "silt":        { thr: 0.034, ths: 0.46, alpha: 0.016,  n: 1.37, Ks: 6.00 },
  "silt loam":   { thr: 0.067, ths: 0.45, alpha: 0.020,  n: 1.41, Ks: 10.80 },
  "sandy clay loam":{ thr: 0.100, ths: 0.39, alpha: 0.059, n: 1.48, Ks: 31.44 },
  "clay loam":   { thr: 0.095, ths: 0.41, alpha: 0.019,  n: 1.31, Ks: 6.24 },
  "silty clay loam":{ thr: 0.089, ths: 0.43, alpha: 0.010, n: 1.23, Ks: 1.68 },
  "sandy clay":  { thr: 0.100, ths: 0.38, alpha: 0.027,  n: 1.23, Ks: 2.88 },
  "silty clay":  { thr: 0.070, ths: 0.36, alpha: 0.005,  n: 1.09, Ks: 0.48 },
  "clay":        { thr: 0.068, ths: 0.38, alpha: 0.008,  n: 1.09, Ks: 4.80 },
};
const presetNames = Object.keys(presets);

function applyPreset(idx: number, name: string) {
  if (!scen.value) return;
  const p = presets[name];
  if (!p) return;
  Object.assign(scen.value.materials[idx], p);
  // Sync hysteresis bounds: tha/thm default to thr/ths
  scen.value.materials[idx].tha = scen.value.materials[idx].thr;
  scen.value.materials[idx].thm = scen.value.materials[idx].ths;
  // Kk defaults to Ks
  scen.value.materials[idx].Kk  = scen.value.materials[idx].Ks;
  scen.value.materials[idx].thk = scen.value.materials[idx].ths;
  markDirty();
}

const tprintStr = computed({
  get: () => scen.value?.TPrint.join(" ") ?? "",
  set: (s: string) => {
    if (!scen.value) return;
    const xs = s.split(/[\s,]+/).filter(Boolean).map(Number).filter(n => !isNaN(n));
    scen.value.TPrint = xs;
    markDirty();
  },
});

// Validation errors per field
const problems = computed(() => {
  const ps: string[] = [];
  const s = scen.value;
  if (!s) return ps;
  for (let i = 0; i < s.materials.length; i++) {
    const m = s.materials[i];
    if (m.thr >= m.ths) ps.push(`Material ${i+1}: θr ≥ θs (${m.thr} ≥ ${m.ths})`);
    if (m.n <= 1)        ps.push(`Material ${i+1}: n must be > 1 (got ${m.n})`);
    if (m.alpha <= 0)    ps.push(`Material ${i+1}: α must be > 0`);
    if (m.Ks <= 0)       ps.push(`Material ${i+1}: Ks must be > 0`);
  }
  if (s.time.dtMin >= s.time.dt)      ps.push(`time: dtMin ≥ dt`);
  if (s.time.dt > s.time.dtMaxW)       ps.push(`time: dt > dtMaxW`);
  if (s.time.dMul <= 1)               ps.push(`time: dMul should be > 1`);
  if (s.time.dMul2 >= 1)               ps.push(`time: dMul2 should be < 1`);
  return ps;
});
</script>

<template>
  <div class="panel editor-panel">
    <div class="row" style="justify-content: space-between">
      <div class="title">Parameter editor</div>
      <div class="row small" style="gap: 6px">
        <button class="secondary" @click="reload" :disabled="!scen || busy">Reload</button>
        <button @click="save" :disabled="!scen || !dirty || busy || problems.length > 0">
          {{ busy ? "saving…" : (dirty ? "Save" : "Saved") }}
        </button>
      </div>
    </div>
    <div v-if="!scen" class="muted small mono" style="padding: 8px">
      {{ err ?? (props.scenarioPath ? "loading…" : "no scenario selected") }}
    </div>
    <div v-else class="form-scroll">
      <section class="grp">
        <div class="grp-h">Heading</div>
        <input type="text" v-model="scen.heading" @input="markDirty" class="wide" />
        <div class="row small" style="gap: 4px; margin-top: 4px">
          <span class="muted">units:</span>
          <input v-for="(_, i) in scen.units" :key="i"
                 v-model="scen.units[i]" @input="markDirty"
                 class="unit" />
        </div>
      </section>

      <section class="grp">
        <div class="grp-h">Flow geometry &amp; solver</div>
        <div class="grid2">
          <label>Kat
            <select v-model.number="scen.config.KAT" @change="markDirty">
              <option :value="0">0 — horizontal plane</option>
              <option :value="1">1 — axisymmetric vertical</option>
              <option :value="2">2 — vertical plane</option>
            </select>
          </label>
          <label>MaxIt
            <input type="number" v-model.number="scen.config.MaxIt" @input="markDirty" min="1" />
          </label>
          <label>TolTh
            <input type="number" v-model.number="scen.config.TolTh" @input="markDirty" step="0.0001" />
          </label>
          <label>TolH
            <input type="number" v-model.number="scen.config.TolH" @input="markDirty" step="0.001" />
          </label>
        </div>
        <div class="flag-grid">
          <label v-for="key in ['lWat','lChem','CheckF','ShortF','FluxF','AtmInF','SeepF','FreeD','DrainF'] as const"
                 :key="key" class="flag">
            <input type="checkbox" :checked="(scen!.config as any)[key]"
                   @change="(e) => { (scen!.config as any)[key] = (e.target as HTMLInputElement).checked; markDirty(); }" />
            {{ key }}
          </label>
        </div>
      </section>

      <section class="grp">
        <div class="row grp-h" style="justify-content: space-between">
          <span>Materials ({{ scen.materials.length }})</span>
          <button class="secondary tiny" @click="addMaterial">+ add</button>
        </div>
        <table class="mat-tbl mono">
          <thead>
            <tr>
              <th>#</th>
              <th>θr</th><th>θs</th><th>tha</th><th>thm</th>
              <th>α (1/cm)</th><th>n</th><th>Ks (cm/T)</th><th>Kk</th><th>thk</th>
              <th>preset</th>
              <th></th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="(m, i) in scen.materials" :key="i">
              <td>{{ i + 1 }}</td>
              <td v-for="k in (['thr','ths','tha','thm','alpha','n','Ks','Kk','thk'] as const)" :key="k">
                <input type="number" step="any" v-model.number="m[k]" @input="markDirty" />
              </td>
              <td>
                <select @change="(e) => applyPreset(i, (e.target as HTMLSelectElement).value)">
                  <option value="">choose…</option>
                  <option v-for="n in presetNames" :key="n" :value="n">{{ n }}</option>
                </select>
              </td>
              <td>
                <button class="tiny danger" @click="removeMaterial(i)"
                        :disabled="scen.materials.length <= 1">−</button>
              </td>
            </tr>
          </tbody>
        </table>
      </section>

      <section class="grp">
        <div class="grp-h">Time control</div>
        <div class="grid5">
          <label>dt
            <input type="number" step="any" v-model.number="scen.time.dt" @input="markDirty" />
          </label>
          <label>dtMin
            <input type="number" step="any" v-model.number="scen.time.dtMin" @input="markDirty" />
          </label>
          <label>dtMaxW
            <input type="number" step="any" v-model.number="scen.time.dtMaxW" @input="markDirty" />
          </label>
          <label>dMul
            <input type="number" step="any" v-model.number="scen.time.dMul" @input="markDirty" />
          </label>
          <label>dMul2
            <input type="number" step="any" v-model.number="scen.time.dMul2" @input="markDirty" />
          </label>
        </div>
        <div class="row" style="margin-top: 6px">
          <label class="wide">
            TPrint (space- or comma-separated print times)
            <input type="text" v-model="tprintStr" class="mono wide" />
          </label>
        </div>
      </section>

      <section v-if="problems.length" class="problems">
        <div class="grp-h">Problems</div>
        <ul>
          <li v-for="p in problems" :key="p">{{ p }}</li>
        </ul>
      </section>
    </div>
    <div v-if="err" class="status-failed small mono">{{ err }}</div>
    <div v-if="saveMsg" class="status-done small mono" style="padding: 4px">{{ saveMsg }}</div>
  </div>
</template>

<style scoped>
.editor-panel {
  flex: 1;
  min-height: 360px;
  display: flex;
  flex-direction: column;
  min-width: 0;
}
.title { font-weight: 600; }
.small { font-size: 11px; }
.form-scroll {
  flex: 1;
  overflow-y: auto;
  padding: 6px 2px 6px 0;
  min-height: 0;
}
.grp { margin-bottom: 10px; }
.grp-h {
  font-size: 11px;
  font-weight: 600;
  color: var(--muted);
  text-transform: uppercase;
  letter-spacing: 0.4px;
  margin-bottom: 4px;
}
.wide { width: 100%; }
.unit { width: 50px; font-family: ui-monospace, Menlo, monospace; font-size: 11px; padding: 2px 4px; }
.grid2 {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 6px 8px;
}
.grid5 {
  display: grid;
  grid-template-columns: repeat(5, 1fr);
  gap: 6px;
}
.grid2 label, .grid5 label {
  display: flex; flex-direction: column; gap: 2px;
  font-size: 11px; color: var(--muted);
}
.grid2 label input, .grid5 label input, .grid2 label select {
  font-size: 12px; color: var(--text); padding: 3px 5px;
}
.flag-grid {
  display: grid;
  grid-template-columns: repeat(5, 1fr);
  gap: 3px 6px;
  margin-top: 6px;
  font-size: 11px;
}
.flag { display: flex; align-items: center; gap: 3px; }
.mat-tbl {
  width: 100%; border-collapse: collapse;
  font-size: 10.5px; margin-top: 4px;
}
.mat-tbl th, .mat-tbl td {
  border: 1px solid var(--border);
  padding: 1px 3px; text-align: center;
}
.mat-tbl th { background: var(--panel-2); color: var(--muted); font-weight: 600; }
.mat-tbl input {
  width: 55px; font-family: ui-monospace, Menlo, monospace;
  font-size: 10.5px; padding: 2px 3px; border: none; background: transparent;
  color: var(--text); text-align: right;
}
.mat-tbl input:focus { outline: 1px solid var(--accent); background: var(--bg); }
.mat-tbl select { font-size: 10px; }
.tiny { font-size: 11px; padding: 2px 6px; }
.problems {
  background: rgba(248, 81, 73, 0.08);
  border: 1px solid var(--err);
  border-radius: 4px;
  padding: 6px 8px;
  font-size: 11px;
}
.problems ul { margin: 4px 0 0 18px; padding: 0; color: var(--err); }
</style>
