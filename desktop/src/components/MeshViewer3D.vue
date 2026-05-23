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
const renderMode = ref<"wire" | "iso">("iso");
const isovalueNorm = ref<number>(0.5);   // 0..1 in scalar range

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

watch(field, () => { if (current.value) renderMesh(current.value); });
watch(renderMode, () => { if (current.value) renderMesh(current.value); });
watch(isovalueNorm, () => {
  if (current.value && renderMode.value === "iso") renderMesh(current.value);
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
  const [smin, smax] = scalar ? minmax(scalar) : [0, 1];

  // Isosurface mode: extract a triangulated surface at the chosen
  // scalar value via marching tetrahedra (hex cells are split into
  // 5 tets first). Falls back to wireframe if no scalar.
  if (renderMode.value === "iso" && scalar) {
    const iso = smin + (smax - smin) * isovalueNorm.value;
    const tris = isosurfaceMarchingTets(mesh, scalar, iso);
    if (tris && tris.length > 0) {
      const triPos = new Float32Array(tris.length * 9);
      for (let i = 0; i < tris.length; i++) {
        for (let v = 0; v < 3; v++) {
          triPos[i*9 + v*3]     = tris[i][v][0];
          triPos[i*9 + v*3 + 1] = tris[i][v][1];
          triPos[i*9 + v*3 + 2] = tris[i][v][2];
        }
      }
      const ig = new THREE.BufferGeometry();
      ig.setAttribute("position", new THREE.BufferAttribute(triPos, 3));
      ig.computeVertexNormals();
      const isoColor = scalarColor(isovalueNorm.value);
      const im = new THREE.MeshPhongMaterial({
        color: new THREE.Color(isoColor[0], isoColor[1], isoColor[2]),
        side: THREE.DoubleSide,
        flatShading: true,
        shininess: 30,
        transparent: true,
        opacity: 0.85,
      });
      const isoMesh = new THREE.Mesh(ig, im);
      const group = new THREE.Group();
      group.add(isoMesh);
      // Also show a faint wire frame so the user gets context
      const edgeIndex = cellsToEdges(mesh.cells, mesh.cellTypes);
      if (edgeIndex.length) {
        const eg = new THREE.BufferGeometry();
        eg.setAttribute("position", new THREE.BufferAttribute(positions, 3));
        eg.setIndex(new THREE.BufferAttribute(new Uint32Array(edgeIndex), 1));
        const wm = new THREE.LineBasicMaterial({
          color: theme.value === "light" ? 0x999999 : 0x444444,
          transparent: true, opacity: 0.18,
        });
        group.add(new THREE.LineSegments(eg, wm));
      }
      scene.add(group);
      meshObj = group;
      fitCamera(mesh.points);
      return;
    }
    // fall through to wire mode if iso produced no triangles
  }

  // Wire / coloured-vertex mode (original)
  const geom = new THREE.BufferGeometry();
  geom.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  if (scalar) {
    const colors = scalarToColors(scalar, smin, smax);
    geom.setAttribute("color", new THREE.BufferAttribute(colors, 3));
  }
  const edgeIndex = cellsToEdges(mesh.cells, mesh.cellTypes);
  if (edgeIndex.length > 0) {
    const eg = geom.clone();
    eg.setIndex(new THREE.BufferAttribute(new Uint32Array(edgeIndex), 1));
    const mat = new THREE.LineBasicMaterial({
      vertexColors: !!scalar,
      color: scalar ? 0xffffff : 0x58a6ff,
      transparent: true, opacity: 0.6,
    });
    const lines = new THREE.LineSegments(eg, mat);
    scene.add(lines);
    meshObj = lines;
  } else {
    const mat = new THREE.PointsMaterial({
      size: 0.02, vertexColors: !!scalar,
      color: scalar ? 0xffffff : 0x58a6ff,
      sizeAttenuation: true,
    });
    const pts = new THREE.Points(geom, mat);
    scene!.add(pts);
    meshObj = pts;
  }
  fitCamera(mesh.points);
}

// ----- Marching tetrahedra ---------------------------------------------
// For each cell: split into tets (hex → 5 tets, tet → 1 tet), then for
// each tet check the 4 vertex scalars against `iso`. Output 0, 1, or 2
// triangles depending on the 4-bit "above/below" mask.

type V3 = [number, number, number];

// Pre-computed: which edges (vertex-index pairs in a tet) the iso-surface
// crosses for each 4-bit mask. Each entry is a list of 3- or 6-vertex
// triangle strips (3 numbers per vertex = edge-endpoint indices).
const TET_EDGES: [number, number][] = [[0,1],[0,2],[0,3],[1,2],[1,3],[2,3]];
// Edge bitmap per case → which edges are crossed (as flat list, length is
// a multiple of 3). Symmetric cases share the inverse mask.
const TET_TRI_TABLE: number[][] = [
  /* 0  0000 */ [],
  /* 1  0001 */ [0, 1, 2],
  /* 2  0010 */ [0, 3, 4],
  /* 3  0011 */ [1, 2, 3,   1, 3, 4],
  /* 4  0100 */ [1, 3, 5],
  /* 5  0101 */ [0, 2, 5,   0, 5, 3],
  /* 6  0110 */ [0, 1, 4,   0, 4, 5],
  /* 7  0111 */ [2, 4, 5],
  /* 8  1000 */ [2, 4, 5],
  /* 9  1001 */ [0, 1, 4,   0, 4, 5],
  /*10  1010 */ [0, 2, 5,   0, 5, 3],
  /*11  1011 */ [1, 3, 5],
  /*12  1100 */ [1, 2, 3,   1, 3, 4],
  /*13  1101 */ [0, 3, 4],
  /*14  1110 */ [0, 1, 2],
  /*15  1111 */ [],
];

function lerpVec(p: V3, q: V3, t: number): V3 {
  return [p[0] + (q[0] - p[0]) * t,
          p[1] + (q[1] - p[1]) * t,
          p[2] + (q[2] - p[2]) * t];
}

function tetIso(p: V3[], s: number[], iso: number, out: V3[][]) {
  // Mask: bit i = 1 if s[i] > iso
  let mask = 0;
  for (let i = 0; i < 4; i++) if (s[i] > iso) mask |= (1 << i);
  const tbl = TET_TRI_TABLE[mask];
  for (let k = 0; k < tbl.length; k += 3) {
    const tri: V3[] = [];
    for (let j = 0; j < 3; j++) {
      const [a, b] = TET_EDGES[tbl[k + j]];
      const sa = s[a], sb = s[b];
      // Crossing parameter on edge a-b
      const t = (sa === sb) ? 0.5 : (iso - sa) / (sb - sa);
      tri.push(lerpVec(p[a], p[b], t));
    }
    out.push(tri);
  }
}

// Hex (8 verts) split into 5 tets that fully tile it.
const HEX_TO_TETS: number[][] = [
  [0, 1, 2, 5],
  [0, 2, 3, 7],
  [0, 5, 7, 4],
  [2, 5, 7, 6],
  [0, 2, 5, 7],
];

function isosurfaceMarchingTets(
  mesh: VtuMesh,
  scalar: number[],
  iso: number,
): V3[][] {
  const out: V3[][] = [];
  const xyz = (i: number): V3 => [
    mesh.points[i*3], mesh.points[i*3 + 1], mesh.points[i*3 + 2],
  ];
  for (let c = 0; c < mesh.cells.length; c++) {
    const cell = mesh.cells[c];
    const ct = mesh.cellTypes[c];
    if (ct === 10 && cell.length >= 4) {
      // Tetrahedron
      const p: V3[] = [xyz(cell[0]), xyz(cell[1]), xyz(cell[2]), xyz(cell[3])];
      const s = [scalar[cell[0]], scalar[cell[1]], scalar[cell[2]], scalar[cell[3]]];
      tetIso(p, s, iso, out);
    } else if (ct === 12 && cell.length >= 8) {
      // Hexahedron: 5-tet decomposition
      for (const t of HEX_TO_TETS) {
        const p: V3[] = [xyz(cell[t[0]]), xyz(cell[t[1]]),
                         xyz(cell[t[2]]), xyz(cell[t[3]])];
        const s = [scalar[cell[t[0]]], scalar[cell[t[1]]],
                   scalar[cell[t[2]]], scalar[cell[t[3]]]];
        tetIso(p, s, iso, out);
      }
    }
  }
  return out;
}

function scalarColor(tNorm: number): [number, number, number] {
  // viridis approximation
  tNorm = Math.max(0, Math.min(1, tNorm));
  const lerp = (a: number, b: number, k: number) => a + (b - a) * k;
  if (tNorm < 0.33) {
    const k = tNorm / 0.33;
    return [lerp(0.267, 0.127, k), lerp(0.005, 0.567, k), lerp(0.329, 0.551, k)];
  } else if (tNorm < 0.66) {
    const k = (tNorm - 0.33) / 0.33;
    return [lerp(0.127, 0.369, k), lerp(0.567, 0.788, k), lerp(0.551, 0.382, k)];
  } else {
    const k = (tNorm - 0.66) / 0.34;
    return [lerp(0.369, 0.993, k), lerp(0.788, 0.906, k), lerp(0.382, 0.144, k)];
  }
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
    <div class="row small" v-if="fieldOptions.length" style="gap: 10px; flex-wrap: wrap">
      <span class="muted">field:</span>
      <select v-model="field">
        <option v-for="n in fieldOptions" :key="n" :value="n">{{ n }}</option>
      </select>
      <span class="muted">render:</span>
      <select v-model="renderMode">
        <option value="iso">isosurface</option>
        <option value="wire">wireframe</option>
      </select>
      <template v-if="renderMode === 'iso'">
        <span class="muted">iso:</span>
        <input type="range" :min="0" :max="1" :step="0.02"
               v-model.number="isovalueNorm" style="width: 80px"
               :title="`${(isovalueNorm * 100).toFixed(0)} % of range`" />
        <span class="mono small">{{ (isovalueNorm * 100).toFixed(0) }}%</span>
      </template>
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
