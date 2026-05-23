<script setup lang="ts">
import { ref, watch, onMounted, onBeforeUnmount, computed } from "vue";
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { api, type JobMeta } from "../api";
import { parseVtu, type VtuMesh } from "../vtu";
import { theme, cssVar } from "../theme";

const props = defineProps<{ job: JobMeta | null }>();
const wrap = ref<HTMLDivElement | null>(null);
const vtus = ref<string[]>([]);
const tIndex = ref(0);
const field = ref<string>("h");
const err = ref<string | null>(null);
const current = ref<VtuMesh | null>(null);

let renderer: THREE.WebGLRenderer | null = null;
let scene: THREE.Scene | null = null;
let camera: THREE.PerspectiveCamera | null = null;
let controls: OrbitControls | null = null;
let meshObj: THREE.Object3D | null = null;
let rafId = 0;

const currentPath = computed(() => vtus.value[tIndex.value] ?? null);

watch(
  () => [props.job?.id, props.job?.status, props.job?.finished_at_ms],
  refresh,
  { immediate: true },
);

watch(currentPath, async (p) => {
  if (!p) return;
  await loadFile(p);
});

watch(field, () => {
  if (current.value) renderMesh(current.value);
});

watch(theme, () => {
  if (renderer) renderer.setClearColor(theme.value === "light" ? 0xffffff : 0x0d1117);
  void cssVar; // referenced for potential future shader uniforms
});

async function refresh() {
  err.value = null;
  vtus.value = [];
  current.value = null;
  if (!props.job) return;
  try {
    const all = await api.listVtuSeries(props.job.output_dir);
    vtus.value = all.filter((p) => p.endsWith(".vtu"));
    tIndex.value = 0;
    if (vtus.value[0]) await loadFile(vtus.value[0]);
  } catch (e: any) {
    err.value = String(e);
  }
}

async function loadFile(path: string) {
  try {
    const bytesArr = await api.readBytes(path);
    const u8 = new Uint8Array(bytesArr);
    const text = new TextDecoder("utf-8").decode(u8);
    const mesh = parseVtu(text);
    current.value = mesh;
    // Prefer 'h' if present, else first scalar
    const names = Object.keys(mesh.scalars);
    if (!names.includes(field.value)) {
      field.value = names[0] ?? "";
    }
    renderMesh(mesh);
  } catch (e: any) {
    err.value = `${path}: ${e}`;
  }
}

function ensureScene() {
  if (renderer && scene && camera) return;
  if (!wrap.value) return;
  const w = wrap.value.clientWidth;
  const h = wrap.value.clientHeight;
  renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(window.devicePixelRatio);
  renderer.setSize(w, h);
  renderer.setClearColor(theme.value === "light" ? 0xffffff : 0x0d1117);
  wrap.value.appendChild(renderer.domElement);
  scene = new THREE.Scene();
  camera = new THREE.PerspectiveCamera(50, w / h, 0.001, 1000);
  camera.position.set(2, 2, 2);
  controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  scene.add(new THREE.AmbientLight(0xffffff, 0.4));
  const dir = new THREE.DirectionalLight(0xffffff, 0.9);
  dir.position.set(2, 4, 3);
  scene.add(dir);
  animate();
  window.addEventListener("resize", onResize);
}

function animate() {
  rafId = requestAnimationFrame(animate);
  controls?.update();
  if (renderer && scene && camera) renderer.render(scene, camera);
}

function onResize() {
  if (!renderer || !camera || !wrap.value) return;
  const w = wrap.value.clientWidth;
  const h = wrap.value.clientHeight;
  renderer.setSize(w, h);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
}

function renderMesh(mesh: VtuMesh) {
  ensureScene();
  if (!scene) return;
  if (meshObj) {
    scene.remove(meshObj);
    meshObj.traverse((o: any) => {
      o.geometry?.dispose();
      o.material?.dispose();
    });
    meshObj = null;
  }

  const positions = new Float32Array(mesh.points);
  const scalar = mesh.scalars[field.value] ?? null;
  const [smin, smax] = scalar
    ? minmax(scalar)
    : [0, 1];

  const geom = new THREE.BufferGeometry();
  geom.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  if (scalar) {
    const colors = scalarToColors(scalar, smin, smax);
    geom.setAttribute("color", new THREE.BufferAttribute(colors, 3));
  }

  // Wireframe + colored vertices: edges of the cells via index buffer
  const edgeIndex = cellsToEdges(mesh.cells, mesh.cellTypes);
  if (edgeIndex.length > 0) {
    const eg = geom.clone();
    eg.setIndex(new THREE.BufferAttribute(new Uint32Array(edgeIndex), 1));
    const mat = new THREE.LineBasicMaterial({
      vertexColors: !!scalar,
      color: scalar ? 0xffffff : 0x58a6ff,
      transparent: true,
      opacity: 0.6,
    });
    const lines = new THREE.LineSegments(eg, mat);
    scene.add(lines);
    meshObj = lines;
  } else {
    const mat = new THREE.PointsMaterial({
      size: 0.02,
      vertexColors: !!scalar,
      color: scalar ? 0xffffff : 0x58a6ff,
      sizeAttenuation: true,
    });
    const pts = new THREE.Points(geom, mat);
    scene!.add(pts);
    meshObj = pts;
  }
  fitCamera(mesh.points);
}

function fitCamera(points: number[]) {
  if (!camera || !controls) return;
  let minX = Infinity, minY = Infinity, minZ = Infinity;
  let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;
  for (let i = 0; i < points.length; i += 3) {
    minX = Math.min(minX, points[i]);     maxX = Math.max(maxX, points[i]);
    minY = Math.min(minY, points[i + 1]); maxY = Math.max(maxY, points[i + 1]);
    minZ = Math.min(minZ, points[i + 2]); maxZ = Math.max(maxZ, points[i + 2]);
  }
  const cx = (minX + maxX) / 2;
  const cy = (minY + maxY) / 2;
  const cz = (minZ + maxZ) / 2;
  const size = Math.max(maxX - minX, maxY - minY, maxZ - minZ);
  controls.target.set(cx, cy, cz);
  camera.position.set(cx + size * 1.5, cy + size * 1.5, cz + size * 1.5);
  camera.near = size / 1000;
  camera.far = size * 100;
  camera.updateProjectionMatrix();
  controls.update();
}

function minmax(a: number[]): [number, number] {
  let lo = Infinity, hi = -Infinity;
  for (const v of a) {
    if (v < lo) lo = v;
    if (v > hi) hi = v;
  }
  return [lo, hi];
}

function scalarToColors(s: number[], lo: number, hi: number): Float32Array {
  const out = new Float32Array(s.length * 3);
  const range = Math.max(hi - lo, 1e-12);
  for (let i = 0; i < s.length; i++) {
    const t = Math.max(0, Math.min(1, (s[i] - lo) / range));
    // Viridis-like ramp
    out[i * 3]     = 0.267 + t * (0.993 - 0.267);
    out[i * 3 + 1] = 0.005 + t * (0.906 - 0.005);
    out[i * 3 + 2] = 0.329 + t * (0.144 - 0.329);
  }
  return out;
}

function cellsToEdges(cells: number[][], types: number[]): number[] {
  // VTK cell types: 10=tetra (6 edges), 12=hex (12 edges).
  // We deduplicate edges via a Set of sorted "a:b" keys.
  const seen = new Set<string>();
  const edges: number[] = [];
  const TETRA: [number, number][] = [
    [0,1],[1,2],[2,0],[0,3],[1,3],[2,3],
  ];
  const HEX: [number, number][] = [
    [0,1],[1,2],[2,3],[3,0],
    [4,5],[5,6],[6,7],[7,4],
    [0,4],[1,5],[2,6],[3,7],
  ];
  for (let i = 0; i < cells.length; i++) {
    const c = cells[i];
    const t = types[i];
    const tbl = t === 10 ? TETRA : t === 12 ? HEX : null;
    if (!tbl) continue;
    for (const [a, b] of tbl) {
      const u = c[a], v = c[b];
      const key = u < v ? `${u}:${v}` : `${v}:${u}`;
      if (!seen.has(key)) {
        seen.add(key);
        edges.push(u, v);
      }
    }
  }
  return edges;
}

onMounted(() => {
  // Defer scene creation until container has size
  setTimeout(() => current.value && renderMesh(current.value), 80);
});

onBeforeUnmount(() => {
  if (rafId) cancelAnimationFrame(rafId);
  renderer?.dispose();
  window.removeEventListener("resize", onResize);
});

function next() {
  if (tIndex.value < vtus.value.length - 1) tIndex.value++;
}
function prev() {
  if (tIndex.value > 0) tIndex.value--;
}

const fieldOptions = computed(() =>
  current.value ? Object.keys(current.value.scalars) : [],
);
</script>

<template>
  <div class="panel mesh-panel">
    <div class="row" style="justify-content: space-between">
      <div class="title">3D mesh</div>
      <div class="row small">
        <span class="muted" v-if="vtus.length">
          {{ tIndex + 1 }} / {{ vtus.length }}
        </span>
        <button class="secondary" @click="prev" :disabled="tIndex <= 0">‹</button>
        <input
          v-if="vtus.length"
          type="range"
          :min="0"
          :max="vtus.length - 1"
          v-model.number="tIndex"
        />
        <button class="secondary" @click="next" :disabled="tIndex >= vtus.length - 1">›</button>
      </div>
    </div>
    <div class="row small" v-if="fieldOptions.length">
      <span class="muted">field:</span>
      <select v-model="field">
        <option v-for="n in fieldOptions" :key="n" :value="n">{{ n }}</option>
      </select>
    </div>
    <div ref="wrap" class="canvas-wrap">
      <div v-if="!vtus.length" class="overlay muted small">
        no .vtu files in {{ job?.output_dir ?? "(no job)" }}
      </div>
    </div>
    <div v-if="err" class="status-failed small mono">{{ err }}</div>
  </div>
</template>

<style scoped>
.mesh-panel {
  flex: 1;
  min-height: 280px;
  display: flex;
  flex-direction: column;
}
.title { font-weight: 600; }
.small { font-size: 11px; }
.canvas-wrap {
  flex: 1;
  min-height: 240px;
  position: relative;
  border: 1px solid var(--border);
  border-radius: 4px;
  overflow: hidden;
  background: #0d1117;
}
.overlay {
  position: absolute;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
}
</style>
