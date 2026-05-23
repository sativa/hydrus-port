<script setup lang="ts">
import { ref, watch, computed, onMounted, onBeforeUnmount, nextTick } from "vue";
import * as THREE from "three";
import { api, type JobMeta, type OutputFile,
         type Swms2dMesh, type Swms2dField } from "../api";

const props = defineProps<{ job: JobMeta | null }>();

const canvasWrap = ref<HTMLDivElement | null>(null);
const colorbarEl = ref<HTMLCanvasElement | null>(null);
const mesh = ref<Swms2dMesh | null>(null);
const fields = ref<Record<string, Swms2dField>>({});
const varName = ref<"h" | "th">("h");
const tIndex = ref(0);
const err = ref<string | null>(null);

// Discover GRID.IN + h.out / th.out for the current job.
async function refresh() {
  err.value = null;
  mesh.value = null;
  fields.value = {};
  if (!props.job) return;
  try {
    // GRID.IN lives under <input_dir>, the rest under <output_dir>.
    const gridPath = `${props.job.input_dir}/GRID.IN`;
    mesh.value = await api.parseSwms2dGrid(gridPath);
    const outs = await api.listOutputFiles(props.job.output_dir);
    for (const v of ["h", "th"] as const) {
      const f = outs.find((o) => o.name.toLowerCase() === `${v}.out`);
      if (!f) continue;
      fields.value[v] = await api.parseSwms2dField(f.path, mesh.value.num_np);
    }
    if (!fields.value["h"] && !fields.value["th"]) {
      err.value = "no h.out / th.out — 2D viz needs SWMS_2D output";
      return;
    }
    if (!fields.value[varName.value]) varName.value = "th";
    tIndex.value = (fields.value[varName.value]?.times.length ?? 1) - 1;
    // canvasWrap is behind v-if="mesh" — wait for the DOM update so
    // the WebGL renderer attaches into a sized container.
    await nextTick();
    ensureScene();
    redraw();
    drawColorbar();
  } catch (e: any) {
    err.value = String(e);
  }
}

watch(
  () => [props.job?.id, props.job?.status, props.job?.finished_at_ms],
  refresh, { immediate: true },
);

watch([varName, tIndex], () => { redraw(); drawColorbar(); });

const currentField = computed(() => {
  const f = fields.value[varName.value];
  if (!f || !f.values.length) return null;
  return f.values[tIndex.value];
});

// ---------- three.js scene ----------

let renderer: THREE.WebGLRenderer | null = null;
let scene: THREE.Scene | null = null;
let camera: THREE.OrthographicCamera | null = null;
let meshObj: THREE.Mesh | null = null;
let wireObj: THREE.LineSegments | null = null;
let rafId = 0;

const VERT_SHADER = `
  attribute float scalar;
  varying float vScalar;
  void main() {
    vScalar = scalar;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
  }
`;

const FRAG_SHADER = `
  precision mediump float;
  varying float vScalar;
  uniform float uMin;
  uniform float uMax;
  // viridis 4-stop ramp (cheap inline approximation, good enough at fragment scale)
  vec3 viridis(float t) {
    vec3 c0 = vec3(0.267, 0.005, 0.329);
    vec3 c1 = vec3(0.127, 0.567, 0.551);
    vec3 c2 = vec3(0.369, 0.788, 0.382);
    vec3 c3 = vec3(0.993, 0.906, 0.144);
    if (t < 0.33)      return mix(c0, c1, t / 0.33);
    else if (t < 0.66) return mix(c1, c2, (t - 0.33) / 0.33);
    else               return mix(c2, c3, (t - 0.66) / 0.34);
  }
  void main() {
    float t = clamp((vScalar - uMin) / (uMax - uMin + 1e-12), 0.0, 1.0);
    gl_FragColor = vec4(viridis(t), 1.0);
  }
`;

let resizeObs: ResizeObserver | null = null;

function ensureScene() {
  if (renderer && scene && camera) return;
  if (!canvasWrap.value) {
    api.debugLog("[mc2d] ensureScene: canvasWrap is null").catch(()=>{});
    return;
  }
  const w = canvasWrap.value.clientWidth;
  const h = canvasWrap.value.clientHeight;
  api.debugLog(`[mc2d] ensureScene: container ${w} x ${h}`).catch(()=>{});
  if (w === 0 || h === 0) {
    // The container has not laid out yet; defer scene creation until
    // ResizeObserver tells us it has real dimensions.
    if (!resizeObs) {
      resizeObs = new ResizeObserver((entries) => {
        const e = entries[0];
        if (e && e.contentRect.width > 0 && e.contentRect.height > 0) {
          resizeObs?.disconnect(); resizeObs = null;
          ensureScene();
          if (mesh.value) redraw();
        }
      });
      resizeObs.observe(canvasWrap.value);
    }
    return;
  }
  renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(window.devicePixelRatio);
  renderer.setSize(w, h);
  renderer.setClearColor(0x0d1117);
  canvasWrap.value.appendChild(renderer.domElement);
  scene = new THREE.Scene();
  camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0.1, 10);
  camera.position.set(0, 0, 5);
  camera.lookAt(0, 0, 0);
  animate();
  window.addEventListener("resize", onResize);
}

function fitCamera(m: Swms2dMesh) {
  if (!camera || !canvasWrap.value || !renderer) return;
  let minX = Infinity, maxX = -Infinity, minZ = Infinity, maxZ = -Infinity;
  for (let i = 0; i < m.nodes_x.length; i++) {
    minX = Math.min(minX, m.nodes_x[i]); maxX = Math.max(maxX, m.nodes_x[i]);
    minZ = Math.min(minZ, m.nodes_z[i]); maxZ = Math.max(maxZ, m.nodes_z[i]);
  }
  const dx = Math.max(1e-9, maxX - minX);
  const dz = Math.max(1e-9, maxZ - minZ);
  // For column problems (dx << dz) preserving physical aspect makes the
  // mesh a single pixel wide. Instead stretch non-uniformly to fill the
  // canvas — soil-physics users care about the front, not the aspect.
  // Only re-preserve aspect when dx and dz are within 4× of each other.
  const ratio = Math.max(dx, dz) / Math.min(dx, dz);
  const preserve = ratio < 4;
  const aspect = renderer.domElement.width / Math.max(1, renderer.domElement.height);
  let viewW = dx, viewH = dz;
  if (preserve) {
    if (viewW / aspect < viewH) viewW = viewH * aspect;
    else                       viewH = viewW / aspect;
  }
  const cx = (maxX + minX) / 2, cz = (maxZ + minZ) / 2;
  const pad = 1.06;
  camera.left = cx - viewW / 2 * pad;
  camera.right = cx + viewW / 2 * pad;
  camera.top = cz + viewH / 2 * pad;
  camera.bottom = cz - viewH / 2 * pad;
  camera.near = 0.1; camera.far = 10;
  // Move camera to the mesh centre in XY so it looks straight at the mesh.
  camera.position.set(cx, cz, 5);
  camera.lookAt(cx, cz, 0);
  camera.updateProjectionMatrix();
}

function redraw() {
  ensureScene();
  api.debugLog(`[mc2d] redraw: scene=${!!scene} mesh=${!!mesh.value} field=${!!currentField.value}`).catch(()=>{});
  if (!scene || !mesh.value || !currentField.value) return;
  // Clean previous
  if (meshObj) { scene.remove(meshObj); (meshObj.geometry as any).dispose(); }
  if (wireObj) { scene.remove(wireObj); (wireObj.geometry as any).dispose(); }
  meshObj = null; wireObj = null;

  const m = mesh.value;
  const N = m.nodes_x.length;
  const pos = new Float32Array(N * 3);
  const scalar = new Float32Array(N);
  for (let i = 0; i < N; i++) {
    pos[i*3]   = m.nodes_x[i];
    pos[i*3+1] = m.nodes_z[i];
    pos[i*3+2] = 0.0;
    scalar[i]  = currentField.value[i];
  }
  const indices = new Uint32Array(m.triangles.length * 3);
  for (let i = 0; i < m.triangles.length; i++) {
    indices[i*3]   = m.triangles[i][0];
    indices[i*3+1] = m.triangles[i][1];
    indices[i*3+2] = m.triangles[i][2];
  }
  const geom = new THREE.BufferGeometry();
  geom.setAttribute("position", new THREE.BufferAttribute(pos, 3));
  geom.setAttribute("scalar",   new THREE.BufferAttribute(scalar, 1));
  geom.setIndex(new THREE.BufferAttribute(indices, 1));

  const [sMin, sMax] = minMaxFinite(scalar);
  // Per-vertex RGB colors from viridis LUT (CPU-side; ~simple, robust).
  const colors = new Float32Array(N * 3);
  const range = Math.max(1e-12, sMax - sMin);
  for (let i = 0; i < N; i++) {
    const t = (scalar[i] - sMin) / range;
    const [r, g, b] = viridisFloat(t);
    colors[i*3] = r; colors[i*3+1] = g; colors[i*3+2] = b;
  }
  geom.setAttribute("color", new THREE.BufferAttribute(colors, 3));
  const mat = new THREE.MeshBasicMaterial({
    vertexColors: true,
    side: THREE.DoubleSide,
  });
  meshObj = new THREE.Mesh(geom, mat);
  scene.add(meshObj);
  api.debugLog(`[mc2d] added mesh: ${N} verts, ${m.triangles.length} tris, scalar=[${sMin.toFixed(2)}, ${sMax.toFixed(2)}]`).catch(()=>{});

  // Wireframe overlay (semi-transparent) so user sees the FE mesh
  const wireGeom = new THREE.BufferGeometry();
  const edges = trianglesToEdgeIndex(m.triangles);
  wireGeom.setAttribute("position", new THREE.BufferAttribute(pos, 3));
  wireGeom.setIndex(new THREE.BufferAttribute(new Uint32Array(edges), 1));
  const wireMat = new THREE.LineBasicMaterial({
    color: 0xffffff, transparent: true, opacity: 0.12,
  });
  wireObj = new THREE.LineSegments(wireGeom, wireMat);
  scene.add(wireObj);

  fitCamera(m);
}

function animate() {
  rafId = requestAnimationFrame(animate);
  if (renderer && scene && camera) renderer.render(scene, camera);
}

function onResize() {
  if (!renderer || !canvasWrap.value || !mesh.value) return;
  const w = canvasWrap.value.clientWidth, h = canvasWrap.value.clientHeight;
  renderer.setSize(w, h);
  fitCamera(mesh.value);
}

function minMaxFinite(a: ArrayLike<number>): [number, number] {
  let lo = Infinity, hi = -Infinity;
  for (let i = 0; i < a.length; i++) {
    const v = a[i];
    if (Number.isFinite(v)) {
      if (v < lo) lo = v;
      if (v > hi) hi = v;
    }
  }
  return [lo, hi];
}

function trianglesToEdgeIndex(tris: [number, number, number][]): number[] {
  const seen = new Set<string>();
  const edges: number[] = [];
  for (const [a, b, c] of tris) {
    for (const [u, v] of [[a, b], [b, c], [c, a]] as [number, number][]) {
      const key = u < v ? `${u}:${v}` : `${v}:${u}`;
      if (!seen.has(key)) { seen.add(key); edges.push(u, v); }
    }
  }
  return edges;
}

function drawColorbar() {
  if (!colorbarEl.value || !currentField.value) return;
  const c = colorbarEl.value;
  const ctx = c.getContext("2d")!;
  const W = c.width, H = c.height;
  ctx.clearRect(0, 0, W, H);
  const [lo, hi] = minMaxFinite(currentField.value);
  const grad = ctx.createLinearGradient(0, H, 0, 0);
  for (let i = 0; i <= 10; i++) {
    const t = i / 10;
    grad.addColorStop(t, viridisCss(t));
  }
  ctx.fillStyle = grad;
  ctx.fillRect(W - 16, 4, 12, H - 8);
  ctx.fillStyle = "#c9d1d9";
  ctx.font = "11px ui-monospace, Menlo, monospace";
  ctx.textBaseline = "middle";
  ctx.textAlign = "right";
  ctx.fillText(hi.toFixed(2), W - 20, 8);
  ctx.fillText(((lo + hi) / 2).toFixed(2), W - 20, H / 2);
  ctx.fillText(lo.toFixed(2), W - 20, H - 8);
}

function viridisFloat(t: number): [number, number, number] {
  t = Math.max(0, Math.min(1, t));
  const lerp = (a: number, b: number, k: number) => a + (b - a) * k;
  let r: number, g: number, b: number;
  if (t < 0.33) {
    const k = t / 0.33;
    r = lerp(0.267, 0.127, k); g = lerp(0.005, 0.567, k); b = lerp(0.329, 0.551, k);
  } else if (t < 0.66) {
    const k = (t - 0.33) / 0.33;
    r = lerp(0.127, 0.369, k); g = lerp(0.567, 0.788, k); b = lerp(0.551, 0.382, k);
  } else {
    const k = (t - 0.66) / 0.34;
    r = lerp(0.369, 0.993, k); g = lerp(0.788, 0.906, k); b = lerp(0.382, 0.144, k);
  }
  return [r, g, b];
}

function viridisCss(t: number): string {
  t = Math.max(0, Math.min(1, t));
  let r: number, g: number, b: number;
  const lerp = (a: number, b: number, k: number) => a + (b - a) * k;
  if (t < 0.33) {
    const k = t / 0.33;
    r = lerp(0.267, 0.127, k); g = lerp(0.005, 0.567, k); b = lerp(0.329, 0.551, k);
  } else if (t < 0.66) {
    const k = (t - 0.33) / 0.33;
    r = lerp(0.127, 0.369, k); g = lerp(0.567, 0.788, k); b = lerp(0.551, 0.382, k);
  } else {
    const k = (t - 0.66) / 0.34;
    r = lerp(0.369, 0.993, k); g = lerp(0.788, 0.906, k); b = lerp(0.382, 0.144, k);
  }
  const to255 = (v: number) => Math.round(v * 255);
  return `rgb(${to255(r)}, ${to255(g)}, ${to255(b)})`;
}

const times = computed(() => fields.value[varName.value]?.times ?? []);

onMounted(() => { /* canvasWrap may not be sized yet — ensureScene is lazy */ });
onBeforeUnmount(() => {
  if (rafId) cancelAnimationFrame(rafId);
  renderer?.dispose();
  window.removeEventListener("resize", onResize);
});
</script>

<template>
  <div class="panel mc-panel">
    <div class="row" style="justify-content: space-between; flex-wrap: wrap">
      <div class="title">2D contour</div>
      <div class="row small" style="gap: 8px">
        <span class="muted">var:</span>
        <select v-model="varName" :disabled="!mesh">
          <option value="h">h (pressure head)</option>
          <option value="th">θ (water content)</option>
        </select>
        <span v-if="mesh" class="muted mono">
          {{ mesh.num_np }} nodes · {{ mesh.triangles.length }} tris
        </span>
      </div>
    </div>
    <div v-if="!mesh" class="muted small mono" style="padding: 10px">
      {{ err ?? "waiting for GRID.IN + h.out / th.out…" }}
    </div>
    <div v-else class="mc-grid">
      <div ref="canvasWrap" class="canvas-wrap"></div>
      <canvas ref="colorbarEl" class="colorbar" width="80" height="200"></canvas>
      <div class="scrub-row" v-if="times.length">
        <input
          type="range"
          :min="0" :max="times.length - 1"
          v-model.number="tIndex" style="flex: 1"
        />
        <span class="mono small">t={{ times[tIndex]?.toFixed(2) }}</span>
      </div>
    </div>
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
.mc-grid {
  flex: 1;
  display: grid;
  grid-template-columns: 1fr 60px;
  grid-template-rows: 1fr auto;
  gap: 6px;
  margin-top: 8px;
  min-height: 0;
}
.canvas-wrap {
  min-height: 0;
  position: relative;
  border: 1px solid var(--border);
  border-radius: 4px;
  overflow: hidden;
  background: #0d1117;
}
.colorbar { background: var(--panel-2); border-radius: 4px; }
.scrub-row {
  grid-column: 1 / -1;
  display: flex; align-items: center; gap: 8px;
}
</style>
