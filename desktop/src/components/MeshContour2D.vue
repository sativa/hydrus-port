<script setup lang="ts">
import { ref, watch, computed, onMounted, onBeforeUnmount, nextTick } from "vue";
// @ts-ignore
import Plotly from "plotly.js-dist-min";
import { api, type JobMeta, type Swms2dMesh, type Swms2dField } from "../api";
import { theme, plotlyLayout } from "../theme";

const props = defineProps<{ job: JobMeta | null }>();

const heatEl = ref<HTMLDivElement | null>(null);
const mesh = ref<Swms2dMesh | null>(null);
const fields = ref<Record<string, Swms2dField>>({});
// Default to θ (water content) — what matters for irrigation analysis.
const varName = ref<"h" | "th">("th");
const tIndex = ref(0);
const colormap = ref<string>("Viridis");
const showStreamlines = ref<boolean>(false);
const showRootZone = ref<boolean>(true);
const rootDepth = ref<number>(30);
const err = ref<string | null>(null);
const units = ref<{length: string; time: string} | null>(null);

async function loadUnits() {
  if (!props.job) { units.value = null; return; }
  try {
    const s = await api.readScenario(props.job.input_dir);
    units.value = { length: s.units.length, time: s.units.time };
  } catch { units.value = null; }
}

const colormaps = ["Viridis", "Cividis", "Plasma", "Turbo", "RdBu"];

watch(
  () => [props.job?.id, props.job?.status, props.job?.finished_at_ms],
  refresh, { immediate: true },
);

async function refresh() {
  err.value = null;
  mesh.value = null;
  fields.value = {};
  if (!props.job) return;
  loadUnits();
  try {
    const gridPath = `${props.job.input_dir}/GRID.IN`;
    mesh.value = await api.parseSwms2dGrid(gridPath);
    const outs = await api.listOutputFiles(props.job.output_dir);
    for (const v of ["h", "th", "vx", "vz"] as const) {
      const f = outs.find((o) => o.name.toLowerCase() === `${v}.out`);
      if (!f) continue;
      fields.value[v] = await api.parseSwms2dField(f.path, mesh.value.num_np);
    }
    if (!fields.value["h"] && !fields.value["th"]) {
      err.value = "no h.out / th.out — 2D viz needs SWMS_2D output";
      return;
    }
    if (!fields.value[varName.value]) {
      varName.value = fields.value["h"] ? "h" : "th";
    }
    tIndex.value = (fields.value[varName.value]?.times.length ?? 1) - 1;
    // Default streamlines on whenever velocity fields are available
    if (fields.value["vx"] && fields.value["vz"]) {
      showStreamlines.value = true;
    }
    await nextTick();
    drawHeat();
  } catch (e: any) {
    err.value = String(e);
  }
}

watch([varName, tIndex, colormap, showStreamlines,
       showRootZone, rootDepth], drawHeat);

const currentField = computed(() => {
  const f = fields.value[varName.value];
  if (!f || !f.values.length) return null;
  return f.values[tIndex.value];
});

const times = computed(() => fields.value[varName.value]?.times ?? []);

// ----- resample FE triangle mesh onto a regular grid via barycentric ----

const GRID_SIZE = 220;

type Bary = {
  triIdx: number;
  l0: number; l1: number; l2: number;
};

function buildGrid(m: Swms2dMesh): { xs: number[]; zs: number[]; lookup: (Bary | null)[][] } {
  let minX = Infinity, maxX = -Infinity, minZ = Infinity, maxZ = -Infinity;
  for (let i = 0; i < m.nodes_x.length; i++) {
    if (m.nodes_x[i] < minX) minX = m.nodes_x[i];
    if (m.nodes_x[i] > maxX) maxX = m.nodes_x[i];
    if (m.nodes_z[i] < minZ) minZ = m.nodes_z[i];
    if (m.nodes_z[i] > maxZ) maxZ = m.nodes_z[i];
  }
  // Slight pad
  const dx = (maxX - minX) || 1;
  const dz = (maxZ - minZ) || 1;
  const x0 = minX - dx * 0.01, x1 = maxX + dx * 0.01;
  const z0 = minZ - dz * 0.01, z1 = maxZ + dz * 0.01;
  // For thin tall meshes (e.g. EX1's 1 × 61 column), use a non-square
  // grid so we don't waste 99% of the cells outside the mesh.
  const ratio = (z1 - z0) / (x1 - x0);
  const nx = Math.max(20, Math.round(GRID_SIZE / Math.max(1, ratio)));
  const nz = Math.max(20, Math.round(GRID_SIZE * Math.min(1, ratio)) || GRID_SIZE);
  const xs: number[] = new Array(nx);
  const zs: number[] = new Array(nz);
  for (let i = 0; i < nx; i++) xs[i] = x0 + (x1 - x0) * (i / (nx - 1));
  for (let j = 0; j < nz; j++) zs[j] = z0 + (z1 - z0) * (j / (nz - 1));

  // Build axis-aligned bounding box per triangle for a O(N) cell search per pixel.
  const N = m.triangles.length;
  const tbx0 = new Float64Array(N), tbx1 = new Float64Array(N);
  const tbz0 = new Float64Array(N), tbz1 = new Float64Array(N);
  for (let t = 0; t < N; t++) {
    const [a, b, c] = m.triangles[t];
    const ax = m.nodes_x[a], ay = m.nodes_z[a];
    const bx = m.nodes_x[b], by = m.nodes_z[b];
    const cx = m.nodes_x[c], cy = m.nodes_z[c];
    tbx0[t] = Math.min(ax, bx, cx); tbx1[t] = Math.max(ax, bx, cx);
    tbz0[t] = Math.min(ay, by, cy); tbz1[t] = Math.max(ay, by, cy);
  }

  const lookup: (Bary | null)[][] = new Array(nz);
  for (let j = 0; j < nz; j++) {
    const row: (Bary | null)[] = new Array(nx);
    const z = zs[j];
    for (let i = 0; i < nx; i++) {
      const x = xs[i];
      row[i] = locateTriangle(x, z, m, tbx0, tbx1, tbz0, tbz1);
    }
    lookup[j] = row;
  }
  return { xs, zs, lookup };
}

function locateTriangle(
  x: number, z: number,
  m: Swms2dMesh,
  tbx0: Float64Array, tbx1: Float64Array,
  tbz0: Float64Array, tbz1: Float64Array,
): Bary | null {
  const N = m.triangles.length;
  for (let t = 0; t < N; t++) {
    if (x < tbx0[t] || x > tbx1[t] || z < tbz0[t] || z > tbz1[t]) continue;
    const [a, b, c] = m.triangles[t];
    const ax = m.nodes_x[a], ay = m.nodes_z[a];
    const bx = m.nodes_x[b], by = m.nodes_z[b];
    const cx = m.nodes_x[c], cy = m.nodes_z[c];
    const denom = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy);
    if (Math.abs(denom) < 1e-15) continue;
    const l0 = ((by - cy) * (x - cx) + (cx - bx) * (z - cy)) / denom;
    const l1 = ((cy - ay) * (x - cx) + (ax - cx) * (z - cy)) / denom;
    const l2 = 1 - l0 - l1;
    const eps = -1e-9;
    if (l0 >= eps && l1 >= eps && l2 >= eps) {
      return { triIdx: t, l0, l1, l2 };
    }
  }
  return null;
}

// Cache the lookup table — it depends only on the mesh, not on time/var.
let cachedLookup: { xs: number[]; zs: number[]; lookup: (Bary | null)[][] } | null = null;
let cachedMeshRef: Swms2dMesh | null = null;

function getLookup(m: Swms2dMesh) {
  if (cachedMeshRef === m && cachedLookup) return cachedLookup;
  cachedLookup = buildGrid(m);
  cachedMeshRef = m;
  return cachedLookup;
}

const darkLayout = computed(() => plotlyLayout());
watch(theme, () => drawHeat());

function drawHeat() {
  if (!heatEl.value || !mesh.value || !currentField.value) return;
  const m = mesh.value;
  const { xs, zs, lookup } = getLookup(m);
  const field = currentField.value;
  // Interpolate
  const z2d: (number | null)[][] = new Array(zs.length);
  for (let j = 0; j < zs.length; j++) {
    const row: (number | null)[] = new Array(xs.length);
    const lkRow = lookup[j];
    for (let i = 0; i < xs.length; i++) {
      const b = lkRow[i];
      if (!b) { row[i] = null; continue; }
      const [a, c, d] = m.triangles[b.triIdx];
      row[i] = b.l0 * field[a] + b.l1 * field[c] + b.l2 * field[d];
    }
    z2d[j] = row;
  }
  // Compute range from valid values
  let lo = Infinity, hi = -Infinity;
  for (const v of field) {
    if (Number.isFinite(v)) {
      if (v < lo) lo = v; if (v > hi) hi = v;
    }
  }
  const trace: any = {
    type: "heatmap",
    x: xs, y: zs, z: z2d,
    zmin: lo, zmax: hi,
    colorscale: colormap.value,
    colorbar: { thickness: 8, len: 0.85, ticks: "outside" },
    hovertemplate: "x=%{x:.2f}<br>z=%{y:.2f}<br>v=%{z:.3f}<extra></extra>",
    connectgaps: false,
  };
  const traces: any[] = [trace];
  if (showStreamlines.value) {
    const sl = computeStreamlines(m, lookup, xs, zs);
    if (sl) traces.push(sl);
  }
  // Root-zone band: a green dashed rectangle covering depths from
  // the surface down to rootDepth. SWMS uses z positive upward, so
  // root zone is z ∈ [z_surface - rootDepth, z_surface].
  if (showRootZone.value) {
    const zSurface = Math.max(...m.nodes_z);
    const rootBot  = zSurface - rootDepth.value;
    const x0 = xs[0], x1 = xs[xs.length - 1];
    traces.push({
      type: "scatter",
      mode: "lines",
      x: [x0, x1, x1, x0, x0],
      y: [rootBot, rootBot, zSurface, zSurface, rootBot],
      fill: "toself",
      fillcolor: "rgba(63, 185, 80, 0.10)",
      line: { color: "rgba(63, 185, 80, 0.6)", width: 1, dash: "dot" },
      name: "root zone",
      hoverinfo: "skip",
      showlegend: false,
    });
  }
  const layout: any = {
    ...darkLayout.value,
    xaxis: { ...darkLayout.value.xaxis,
             title: `x${units.value ? ' [' + units.value.length + ']' : ''}` },
    yaxis: { ...darkLayout.value.yaxis,
             title: `z${units.value ? ' [' + units.value.length + ']' : ''}` },
    title: { text: `${varName.value} @ t=${times.value[tIndex.value]?.toFixed(2)}${units.value ? ' ' + units.value.time : ''}`, font: { size: 12 } },
  };
  Plotly.react(heatEl.value, traces, layout, { responsive: true, displaylogo: false });
}

// --- Streamline integration --------------------------------------------
// Sample (vx, vz) at the current time from fields.vx / fields.vz, seed
// particles on a coarse grid inside the mesh, integrate trajectories
// forward and backward via RK2 (Heun's method), render as a single
// Plotly scatter trace with `null`-separated polylines.

function sampleVector(
  m: Swms2dMesh,
  vx: number[], vz: number[],
  x: number, z: number,
  tbx0: Float64Array, tbx1: Float64Array,
  tbz0: Float64Array, tbz1: Float64Array,
): [number, number] | null {
  const N = m.triangles.length;
  for (let t = 0; t < N; t++) {
    if (x < tbx0[t] || x > tbx1[t] || z < tbz0[t] || z > tbz1[t]) continue;
    const [a, b, c] = m.triangles[t];
    const ax = m.nodes_x[a], ay = m.nodes_z[a];
    const bx = m.nodes_x[b], by = m.nodes_z[b];
    const cx = m.nodes_x[c], cy = m.nodes_z[c];
    const denom = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy);
    if (Math.abs(denom) < 1e-15) continue;
    const l0 = ((by - cy) * (x - cx) + (cx - bx) * (z - cy)) / denom;
    const l1 = ((cy - ay) * (x - cx) + (ax - cx) * (z - cy)) / denom;
    const l2 = 1 - l0 - l1;
    if (l0 >= -1e-9 && l1 >= -1e-9 && l2 >= -1e-9) {
      return [l0 * vx[a] + l1 * vx[b] + l2 * vx[c],
              l0 * vz[a] + l1 * vz[b] + l2 * vz[c]];
    }
  }
  return null;
}

function computeStreamlines(
  m: Swms2dMesh,
  lookup: (Bary | null)[][],
  xs: number[], zs: number[],
): any | null {
  const vxField = fields.value["vx"];
  const vzField = fields.value["vz"];
  if (!vxField || !vzField) return null;
  // Pick a velocity snapshot near the current time index; the field's
  // time grid may not match the heatmap's, so use the same index modulo
  // length (a more careful match would linearly interpolate).
  const ti = Math.min(tIndex.value, vxField.values.length - 1);
  const vx = vxField.values[ti];
  const vz = vzField.values[ti];

  // Precompute triangle AABBs.
  const N = m.triangles.length;
  const tbx0 = new Float64Array(N), tbx1 = new Float64Array(N);
  const tbz0 = new Float64Array(N), tbz1 = new Float64Array(N);
  for (let t = 0; t < N; t++) {
    const [a, b, c] = m.triangles[t];
    const ax = m.nodes_x[a], ay = m.nodes_z[a];
    const bx = m.nodes_x[b], by = m.nodes_z[b];
    const cx = m.nodes_x[c], cy = m.nodes_z[c];
    tbx0[t] = Math.min(ax, bx, cx); tbx1[t] = Math.max(ax, bx, cx);
    tbz0[t] = Math.min(ay, by, cy); tbz1[t] = Math.max(ay, by, cy);
  }

  // Estimate step size from mesh extent
  const x0 = xs[0], xN = xs[xs.length - 1];
  const z0 = zs[0], zN = zs[zs.length - 1];
  const ext = Math.max(xN - x0, zN - z0);
  const dt = ext * 0.005;        // ~200 steps to cross the longer extent
  const maxSteps = 200;

  // Seed grid: coarser than viz grid (~12 × 8 for typical aspect ratios)
  const nSeedZ = Math.max(8, Math.min(20, Math.round(zs.length / 14)));
  const nSeedX = Math.max(6, Math.min(20, Math.round(xs.length / 14)));
  const xLine: (number | null)[] = [];
  const zLine: (number | null)[] = [];

  // Find a characteristic velocity magnitude for normalization
  let vmag = 0;
  for (let i = 0; i < vx.length; i++) {
    const m2 = vx[i] * vx[i] + vz[i] * vz[i];
    if (m2 > vmag) vmag = m2;
  }
  vmag = Math.sqrt(vmag) || 1;
  const stepLen = ext * 0.012; // arc-length-per-step in physical units

  for (let i = 0; i < nSeedX; i++) {
    for (let j = 0; j < nSeedZ; j++) {
      const sx = x0 + (xN - x0) * ((i + 0.5) / nSeedX);
      const sz = z0 + (zN - z0) * ((j + 0.5) / nSeedZ);
      // Only seed if inside the mesh
      const v0 = sampleVector(m, vx, vz, sx, sz, tbx0, tbx1, tbz0, tbz1);
      if (!v0) continue;
      const traj: [number, number][] = [[sx, sz]];
      // Forward integration
      let cx = sx, cz = sz;
      for (let s = 0; s < maxSteps; s++) {
        const v = sampleVector(m, vx, vz, cx, cz, tbx0, tbx1, tbz0, tbz1);
        if (!v) break;
        const mg = Math.sqrt(v[0]*v[0] + v[1]*v[1]) || 1e-12;
        const dx = (v[0] / mg) * stepLen, dz_ = (v[1] / mg) * stepLen;
        // Heun corrector
        const mx = cx + dx, mz = cz + dz_;
        const v2 = sampleVector(m, vx, vz, mx, mz, tbx0, tbx1, tbz0, tbz1) ?? v;
        const mg2 = Math.sqrt(v2[0]*v2[0] + v2[1]*v2[1]) || 1e-12;
        const ddx = 0.5 * (dx + (v2[0] / mg2) * stepLen);
        const ddz = 0.5 * (dz_ + (v2[1] / mg2) * stepLen);
        cx += ddx; cz += ddz;
        traj.push([cx, cz]);
      }
      // Backward integration from seed
      cx = sx; cz = sz;
      const back: [number, number][] = [];
      for (let s = 0; s < maxSteps; s++) {
        const v = sampleVector(m, vx, vz, cx, cz, tbx0, tbx1, tbz0, tbz1);
        if (!v) break;
        const mg = Math.sqrt(v[0]*v[0] + v[1]*v[1]) || 1e-12;
        const dx = (v[0] / mg) * stepLen, dz_ = (v[1] / mg) * stepLen;
        cx -= dx; cz -= dz_;
        back.unshift([cx, cz]);
      }
      const full = [...back, ...traj];
      for (const [px, pz] of full) {
        xLine.push(px); zLine.push(pz);
      }
      xLine.push(null); zLine.push(null);
    }
  }
  // Avoid `void` dimension reads
  void lookup;
  void vmag;
  return {
    type: "scatter",
    mode: "lines",
    x: xLine, y: zLine,
    line: { color: "rgba(255,255,255,0.55)", width: 0.8 },
    hoverinfo: "skip",
    showlegend: false,
  };
}

onMounted(() => window.addEventListener("resize", onResize));
onBeforeUnmount(() => {
  window.removeEventListener("resize", onResize);
  cachedLookup = null; cachedMeshRef = null;
});
function onResize() {
  if (heatEl.value) Plotly.Plots.resize(heatEl.value);
}
</script>

<template>
  <div class="panel mc-panel">
    <div class="row" style="justify-content: space-between; flex-wrap: wrap; gap: 6px">
      <div class="title">2D contour</div>
      <div class="row small" style="gap: 8px">
        <span class="muted">var:</span>
        <select v-model="varName" :disabled="!mesh">
          <option value="h">h (head)</option>
          <option value="th">θ (water content)</option>
        </select>
        <span class="muted">cmap:</span>
        <select v-model="colormap">
          <option v-for="c in colormaps" :key="c" :value="c">{{ c }}</option>
        </select>
        <label class="row small" style="gap: 4px"
               v-if="fields.vx && fields.vz">
          <input type="checkbox" v-model="showStreamlines" />
          streamlines
        </label>
        <label class="row small" style="gap: 4px">
          <input type="checkbox" v-model="showRootZone" />
          root&nbsp;zone
        </label>
        <input v-if="showRootZone" type="number" min="0" step="1"
               v-model.number="rootDepth" style="width: 50px"
               :title="`root depth (${units?.length ?? 'cm'})`" />
        <span v-if="mesh" class="muted mono small">
          {{ mesh.num_np }} nodes · {{ mesh.triangles.length }} tris
        </span>
      </div>
    </div>
    <div v-if="!mesh" class="muted small mono" style="padding: 10px">
      {{ err ?? "waiting for GRID.IN + h.out / th.out…" }}
    </div>
    <template v-else>
      <div ref="heatEl" class="heat"></div>
      <div class="scrub-row" v-if="times.length">
        <input
          type="range"
          :min="0" :max="times.length - 1"
          v-model.number="tIndex" style="flex: 1"
        />
        <span class="mono small">t={{ times[tIndex]?.toFixed(2) }}</span>
      </div>
    </template>
  </div>
</template>

<style scoped>
.mc-panel {
  flex: 1;
  min-height: 400px;
  display: flex;
  flex-direction: column;
}
.title { font-weight: 600; }
.small { font-size: 11px; }
.heat { flex: 1; min-height: 240px; margin-top: 8px; }
.scrub-row {
  display: flex; align-items: center; gap: 8px; margin-top: 4px;
}
</style>
