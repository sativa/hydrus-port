<script setup lang="ts">
import { ref, watch, computed } from "vue";
import { api, type CanonicalScenario, type CanonicalMaterial } from "../api";

const props = defineProps<{ scenarioPath: string | null }>();
const emit = defineEmits<{ (e: "scenario-saved", path: string): void }>();

const scen = ref<CanonicalScenario | null>(null);
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
  const tmpl: CanonicalMaterial = cur.length
    ? { ...cur[cur.length - 1] }
    : {
        theta_r: 0.05, theta_s: 0.40,
        alpha: 0.04, n: 1.5, Ks: 0.01,
        l: 0.5,
        theta_a: null, theta_m: null, theta_k: null, Kk: null,
      };
  cur.push(tmpl);
  markDirty();
}

function removeMaterial(idx: number) {
  if (!scen.value || scen.value.materials.length <= 1) return;
  scen.value.materials.splice(idx, 1);
  markDirty();
}

// Soil texture presets (Rosetta H1 USDA classes, cm and day units).
// θr / θs / α (1/cm) / n / Ks (cm/d).
const presets: Record<string, Partial<CanonicalMaterial>> = {
  "sand":           { theta_r: 0.045, theta_s: 0.43, alpha: 0.145, n: 2.68, Ks: 712.8 },
  "loamy sand":     { theta_r: 0.057, theta_s: 0.41, alpha: 0.124, n: 2.28, Ks: 350.2 },
  "sandy loam":     { theta_r: 0.065, theta_s: 0.41, alpha: 0.075, n: 1.89, Ks: 106.1 },
  "loam":           { theta_r: 0.078, theta_s: 0.43, alpha: 0.036, n: 1.56, Ks: 24.96 },
  "silt":           { theta_r: 0.034, theta_s: 0.46, alpha: 0.016, n: 1.37, Ks: 6.00 },
  "silt loam":      { theta_r: 0.067, theta_s: 0.45, alpha: 0.020, n: 1.41, Ks: 10.80 },
  "sandy clay loam":{ theta_r: 0.100, theta_s: 0.39, alpha: 0.059, n: 1.48, Ks: 31.44 },
  "clay loam":      { theta_r: 0.095, theta_s: 0.41, alpha: 0.019, n: 1.31, Ks: 6.24 },
  "silty clay loam":{ theta_r: 0.089, theta_s: 0.43, alpha: 0.010, n: 1.23, Ks: 1.68 },
  "sandy clay":     { theta_r: 0.100, theta_s: 0.38, alpha: 0.027, n: 1.23, Ks: 2.88 },
  "silty clay":     { theta_r: 0.070, theta_s: 0.36, alpha: 0.005, n: 1.09, Ks: 0.48 },
  "clay":           { theta_r: 0.068, theta_s: 0.38, alpha: 0.008, n: 1.09, Ks: 4.80 },
};
const presetNames = Object.keys(presets);

function applyPreset(idx: number, name: string) {
  if (!scen.value || !name) return;
  const p = presets[name];
  if (!p) return;
  Object.assign(scen.value.materials[idx], p);
  markDirty();
}

const tprintStr = computed({
  get: () => scen.value?.time.print_times.map(t => formatNum(t)).join(" ") ?? "",
  set: (s: string) => {
    if (!scen.value) return;
    const xs = s.split(/[\s,]+/).filter(Boolean).map(Number).filter(n => !isNaN(n));
    scen.value.time.print_times = xs;
    markDirty();
  },
});

const nPrint = ref<number>(6);

function formatNum(v: number): string {
  // Round to ~5 significant digits to avoid float-noise tails ("0.30000000000000004")
  if (v === 0) return "0";
  const a = Math.abs(v);
  if (a >= 1e-3 && a < 1e6) {
    // Use up to 6 sig digits, strip trailing zeros after decimal
    return Number(v.toPrecision(6)).toString();
  }
  return v.toExponential(3);
}

function genLinear() {
  if (!scen.value) return;
  const t0 = scen.value.time.t_init, t1 = scen.value.time.t_max;
  const n = Math.max(1, Math.floor(nPrint.value));
  if (!(t1 > t0)) return;
  const out: number[] = [];
  for (let i = 1; i <= n; i++) out.push(t0 + (t1 - t0) * (i / n));
  scen.value.time.print_times = out.map(v => Number(v.toPrecision(6)));
  markDirty();
}

function genLog() {
  if (!scen.value) return;
  const t0 = scen.value.time.t_init, t1 = scen.value.time.t_max;
  const n = Math.max(1, Math.floor(nPrint.value));
  if (!(t1 > Math.max(t0, 0))) return;
  // Log space from max(t0, t1/10000) to t1 — early bias is useful for
  // sharp infiltration fronts when most of the action is near t_init.
  const start = Math.max(t0, t1 / 10000);
  const out: number[] = [];
  for (let i = 1; i <= n; i++) {
    const f = i / n;
    out.push(start * Math.pow(t1 / start, f));
  }
  scen.value.time.print_times = out.map(v => Number(v.toPrecision(6)));
  markDirty();
}

function clearTPrint() {
  if (!scen.value) return;
  scen.value.time.print_times = [];
  markDirty();
}

const dim = computed(() => scen.value?.geometry.kind ?? "1d");

const numNodes = computed(() => {
  if (!scen.value) return 0;
  const g: any = scen.value.geometry;
  if (g.kind === "1d") return (g.z ?? []).length;
  if (g.kind === "2d" || g.kind === "3d") return (g.x ?? []).length;
  return 0;
});

const numElements = computed(() => {
  if (!scen.value) return 0;
  const g: any = scen.value.geometry;
  if (g.kind === "2d") return (g.elements ?? []).length;
  if (g.kind === "3d") return (g.cells ?? []).length;
  return 0;
});

// Cross-field validation in the unified shape
const problems = computed(() => {
  const ps: string[] = [];
  const s = scen.value;
  if (!s) return ps;
  for (let i = 0; i < s.materials.length; i++) {
    const m = s.materials[i];
    if (m.theta_r >= m.theta_s) ps.push(`Material ${i+1}: θr ≥ θs (${m.theta_r} ≥ ${m.theta_s})`);
    if (m.n <= 1)                ps.push(`Material ${i+1}: n must be > 1 (got ${m.n})`);
    if (m.alpha <= 0)            ps.push(`Material ${i+1}: α must be > 0`);
    if (m.Ks <= 0)               ps.push(`Material ${i+1}: Ks must be > 0`);
  }
  const t = s.time;
  if (t.dt_min >= t.dt)   ps.push(`time: dt_min ≥ dt`);
  if (t.dt > t.dt_max)    ps.push(`time: dt > dt_max`);
  if (t.dt_mul <= 1)      ps.push(`time: dt_mul should be > 1`);
  if (t.dt_mul2 >= 1)     ps.push(`time: dt_mul2 should be < 1`);
  return ps;
});

// Material extras (Vogel-Cislerova) shown only when materially different
// from defaults. Avoids cluttering the row with redundant fields.
function showVC(m: CanonicalMaterial): boolean {
  return m.theta_a !== null || m.theta_m !== null
      || m.theta_k !== null || m.Kk !== null;
}
</script>

<template>
  <div class="panel editor-panel">
    <div class="row" style="justify-content: space-between">
      <div class="title">
        Parameter editor
        <span v-if="scen" class="muted small mono" style="margin-left: 8px">
          [{{ dim.toUpperCase() }}] {{ numNodes }} nodes<span v-if="numElements"> · {{ numElements }} elems</span>
        </span>
      </div>
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
      <!-- Meta + units shared across all dimensions -->
      <section class="grp">
        <div class="grp-h">Heading</div>
        <input type="text" v-model="scen.meta.name" @input="markDirty" class="wide" />
        <div class="row small" style="gap: 4px; margin-top: 4px">
          <span class="muted">L</span>
          <input v-model="scen.units.length" @input="markDirty" class="unit" />
          <span class="muted">T</span>
          <input v-model="scen.units.time"   @input="markDirty" class="unit" />
          <span class="muted">M</span>
          <input v-model="scen.units.mass"   @input="markDirty" class="unit" />
        </div>
      </section>

      <!-- Flow geometry + solver -->
      <section class="grp">
        <div class="grp-h">Flow geometry &amp; solver</div>
        <div class="grid2">
          <label>Geometry
            <select v-model="scen.solver.geometry_kind" @change="markDirty">
              <option value="horizontal">horizontal plane</option>
              <option value="axisymmetric">axisymmetric vertical</option>
              <option value="vertical">vertical plane</option>
            </select>
          </label>
          <label>MaxIt
            <input type="number" v-model.number="scen.solver.max_picard" @input="markDirty" min="1" />
          </label>
          <label>TolTh
            <input type="number" v-model.number="scen.solver.tol_theta" @input="markDirty" step="0.0001" />
          </label>
          <label>TolH
            <input type="number" v-model.number="scen.solver.tol_h" @input="markDirty" step="0.001" />
          </label>
        </div>
        <div class="flag-grid">
          <label v-for="key in ([
            'water_flow','solute_transport','heat_transport','root_uptake',
            'atmospheric_bc','free_drainage','seepage_face','subsurface_drain',
            'short_output','flux_output','check_output','hysteresis',
          ] as const)" :key="key" class="flag">
            <input type="checkbox" :checked="(scen!.solver as any)[key]"
                   @change="(e) => { (scen!.solver as any)[key] = (e.target as HTMLInputElement).checked; markDirty(); }" />
            {{ key.replace(/_/g, ' ') }}
          </label>
        </div>
      </section>

      <!-- Materials table -->
      <section class="grp">
        <div class="row grp-h" style="justify-content: space-between">
          <span>Materials ({{ scen.materials.length }})</span>
          <button class="secondary tiny" @click="addMaterial">+ add</button>
        </div>
        <table class="mat-tbl mono">
          <thead>
            <tr>
              <th>#</th>
              <th>θr</th><th>θs</th>
              <th>α (1/L)</th><th>n</th><th>Ks (L/T)</th><th>l</th>
              <th>preset</th>
              <th></th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="(m, i) in scen.materials" :key="i">
              <td>{{ i + 1 }}</td>
              <td v-for="k in (['theta_r','theta_s','alpha','n','Ks','l'] as const)" :key="k">
                <input type="number" step="any" v-model.number="m[k]" @input="markDirty" />
              </td>
              <td>
                <select :value="''"
                        @change="(e) => applyPreset(i, (e.target as HTMLSelectElement).value)">
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

      <!-- Time control -->
      <section class="grp">
        <div class="grp-h">Time control</div>
        <div class="grid5">
          <label>dt
            <input type="number" step="any" v-model.number="scen.time.dt" @input="markDirty" />
          </label>
          <label>dt_min
            <input type="number" step="any" v-model.number="scen.time.dt_min" @input="markDirty" />
          </label>
          <label>dt_max
            <input type="number" step="any" v-model.number="scen.time.dt_max" @input="markDirty" />
          </label>
          <label>dt_mul
            <input type="number" step="any" v-model.number="scen.time.dt_mul" @input="markDirty" />
          </label>
          <label>dt_mul2
            <input type="number" step="any" v-model.number="scen.time.dt_mul2" @input="markDirty" />
          </label>
        </div>
        <div class="grid2" style="margin-top: 6px">
          <label>t_init
            <input type="number" step="any" v-model.number="scen.time.t_init" @input="markDirty" />
          </label>
          <label>t_max
            <input type="number" step="any" v-model.number="scen.time.t_max" @input="markDirty" />
          </label>
        </div>
        <div style="margin-top: 6px">
          <div class="row small" style="gap: 6px; flex-wrap: wrap; margin-bottom: 4px">
            <span class="muted">Print times ({{ scen.time.print_times.length }})</span>
            <span class="muted">·</span>
            <span class="muted">generate</span>
            <input type="number" min="1" max="999" v-model.number="nPrint"
                   style="width: 50px" />
            <button class="secondary tiny" @click="genLinear"
                    :title="`${nPrint} evenly-spaced times from t_init to t_max`">
              Linear
            </button>
            <button class="secondary tiny" @click="genLog"
                    :title="`${nPrint} log-spaced times (early-bias) from t_init to t_max`">
              Log
            </button>
            <button class="secondary tiny" @click="clearTPrint">Clear</button>
          </div>
          <input type="text" v-model="tprintStr" class="mono wide"
                 :placeholder="`e.g. ${scen.time.t_init} … ${scen.time.t_max}  (or use the generators above)`" />
        </div>
      </section>

      <!-- Geometry summary (read-only for now) -->
      <section class="grp" v-if="scen.geometry">
        <div class="grp-h">Geometry summary</div>
        <div class="muted small mono">
          <div v-if="dim === '1d'">
            1D profile · {{ numNodes }} nodes,
            depth {{ (scen.geometry as any).z?.[0] }}
            → {{ (scen.geometry as any).z?.[(scen.geometry as any).z?.length - 1] }}
          </div>
          <div v-else-if="dim === '2d'">
            2D FE mesh · {{ numNodes }} nodes · {{ numElements }} cells ·
            {{ ((scen.geometry as any).boundary_nodes ?? []).length }} boundary nodes
          </div>
          <div v-else>
            3D mesh · {{ numNodes }} nodes · {{ numElements }} cells
          </div>
          <div style="margin-top: 4px; color: var(--muted)">
            (mesh editing not yet exposed — edit through the original
             GRID.IN / Profile.dat and re-import)
          </div>
        </div>
      </section>

      <!-- Validation -->
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
  grid-template-columns: repeat(4, 1fr);
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
  width: 60px; font-family: ui-monospace, Menlo, monospace;
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
