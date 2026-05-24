<script setup lang="ts">
import { ref, computed, onMounted } from "vue";
import ScenarioPicker from "./components/ScenarioPicker.vue";
import LogStream from "./components/LogStream.vue";
import ProfilePlot from "./components/ProfilePlot.vue";
import MeshViewer3D from "./components/MeshViewer3D.vue";
import OutputBrowser from "./components/OutputBrowser.vue";
import Regression from "./components/Regression.vue";
import Hovmoller1D from "./components/Hovmoller1D.vue";
import MeshContour2D from "./components/MeshContour2D.vue";
import ScenarioEditor from "./components/ScenarioEditor.vue";
import { api, type JobMeta, type PythonInfo } from "./api";
import { theme, toggleTheme } from "./theme";

const py = ref<PythonInfo | null>(null);
const py_err = ref<string | null>(null);
const job = ref<JobMeta | null>(null);

const isOneDimensional = computed(() => {
  const k = job.value?.kind ?? "";
  return k === "hydrus1d" || k === "1d";
});
const isTwoDimensional = computed(() => {
  const k = job.value?.kind ?? "";
  return k === "swms2d" || k === "2d";
});

const rightTab = ref<"3d" | "editor">("editor");
const editorPath = ref<string | null>(null);
function onScenarioSelected(path: string, kind: string) {
  // Editor handles 1d / 2d / 3d through the canonical Scenario schema:
  //  - 1d:    Selector.in + Profile.dat ↔ canonical
  //  - 2d:    SELECTOR.IN + GRID.IN     ↔ canonical
  //  - 3d:    scenario.json directly     (no ASCII variant)
  if (kind === "hydrus1d" || kind === "1d"
      || kind === "swms2d"  || kind === "2d"
      || kind === "richards3d" || kind === "3d") {
    editorPath.value = path;
  } else {
    editorPath.value = null;
  }
  // Auto-switch right-panel tab to the most useful view per scenario
  if (kind === "richards3d" || kind === "3d") {
    rightTab.value = "3d";   // viewer is most informative for 3d
  } else {
    rightTab.value = "editor";
  }
}

onMounted(async () => {
  try {
    py.value = await api.detectPython();
  } catch (e: any) {
    py_err.value = String(e);
  }
  // E2E hook: VITE_AUTORUN forces an auto-run after mount + python detect.
  const auto = (import.meta as any).env?.VITE_AUTORUN as string | undefined;
  api.debugLog(`[autorun] VITE_AUTORUN raw = ${JSON.stringify(auto)}`).catch(()=>{});
  // Special sentinel: auto-fire the regression panel instead of a scenario.
  if (auto && auto.startsWith("__test_")) {
    const target = ((auto.replace("__test_", "").replace(/__$/, "") || "all") as "all" | "1d" | "2d" | "3d");
    setTimeout(async () => {
      try {
        autorun_msg.value = `AUTORUN: hydrus test ${target}`;
        await api.debugLog(`[autorun] firing test target=${target}`);
        const j = await api.startTest(target);
        autorun_msg.value = `AUTORUN: test job ${j.id}`;
        job.value = j;
      } catch (e: any) {
        autorun_msg.value = `AUTORUN: ERR ${e}`;
        try { await api.debugLog(`[autorun] test EXC ${e}`); } catch {}
      }
    }, 800);
    return;
  }
  if (auto) {
    setTimeout(async () => {
      try {
        const ss = await api.listScenarios();
        await api.debugLog(`[autorun] scenarios: ${ss.length}  names=${ss.map(s=>s.name).join(',')}`);
        const match = ss.find((s) => s.name.includes(auto)) ?? ss[0];
        if (!match) {
          autorun_msg.value = "AUTORUN: no scenario";
          await api.debugLog("[autorun] no scenario matched");
          return;
        }
        autorun_msg.value = `AUTORUN: starting ${match.name}`;
        await api.debugLog(`[autorun] picked ${match.name} kind=${match.kind} path=${match.path}`);
        const j = await api.start({ kind: match.kind, input_dir: match.path });
        autorun_msg.value = `AUTORUN: started job ${j.id}`;
        await api.debugLog(`[autorun] started job ${j.id} status=${j.status}`);
        job.value = j;
      } catch (e: any) {
        autorun_msg.value = `AUTORUN: ERR ${e}`;
        try { await api.debugLog(`[autorun] EXCEPTION: ${e}`); } catch {}
      }
    }, 800);
  }
});

const autorun_msg = ref<string>("");

function onJobStarted(j: JobMeta) {
  job.value = j;
}
function onJobUpdated(j: JobMeta) {
  job.value = j;
}
</script>

<template>
  <div class="layout">
    <header class="topbar">
      <div class="brand">HYDRUS Port</div>
      <div class="row" style="gap: 14px">
        <div class="muted mono small">
          <span v-if="py">
            {{ py.executable }} · {{ py.version }} · root {{ py.repo_root }}
          </span>
          <span v-else-if="py_err" class="status-failed">{{ py_err }}</span>
          <span v-else>detecting python…</span>
          <span v-if="autorun_msg" style="margin-left: 10px; color: var(--warn);">
            [{{ autorun_msg }}]
          </span>
        </div>
        <button class="theme-btn" @click="toggleTheme" :title="`switch to ${theme === 'dark' ? 'light' : 'dark'} theme`">
          {{ theme === "dark" ? "☀ light" : "☾ dark" }}
        </button>
      </div>
    </header>

    <main class="grid">
      <section class="col-left">
        <ScenarioPicker
          :python-ready="!!py"
          @job-started="onJobStarted"
          @scenario-selected="onScenarioSelected"
        />
        <Regression :external-job="job" @job-started="onJobStarted" />
        <OutputBrowser :job="job" />
      </section>

      <section class="col-mid">
        <Hovmoller1D v-if="isOneDimensional" :job="job" />
        <MeshContour2D v-else-if="isTwoDimensional" :job="job" />
        <ProfilePlot v-else :job="job" />
        <LogStream :job="job" @job-status="onJobUpdated" />
      </section>

      <section class="col-right">
        <div class="tab-bar">
          <button class="tab" :class="{active: rightTab === 'editor'}"
                  @click="rightTab = 'editor'">Parameters</button>
          <button class="tab" :class="{active: rightTab === '3d'}"
                  @click="rightTab = '3d'">3D mesh</button>
        </div>
        <ScenarioEditor v-if="rightTab === 'editor'"
                        :scenario-path="editorPath" />
        <MeshViewer3D v-else :job="job" />
      </section>
    </main>
  </div>
</template>

<style scoped>
.layout {
  height: 100vh;
  display: flex;
  flex-direction: column;
}
.topbar {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 10px 16px;
  background: var(--panel);
  border-bottom: 1px solid var(--border);
}
.brand {
  font-weight: 600;
  font-size: 15px;
  color: var(--accent);
}
.small { font-size: 11px; }
.grid {
  flex: 1;
  display: grid;
  grid-template-columns: 280px 1fr 1fr;
  gap: 10px;
  padding: 10px;
  min-height: 0;
}
section {
  display: flex;
  flex-direction: column;
  gap: 10px;
  min-height: 0;
}
.tab-bar {
  display: flex;
  gap: 0;
  border-bottom: 1px solid var(--border);
  margin-bottom: -2px;
}
.tab {
  background: transparent;
  color: var(--muted);
  border: none;
  border-bottom: 2px solid transparent;
  padding: 6px 12px;
  font-size: 11.5px;
  cursor: pointer;
  font-weight: 500;
  border-radius: 0;
}
.tab.active {
  color: var(--accent);
  border-bottom-color: var(--accent);
}
.tab:hover { color: var(--text); }
.theme-btn {
  background: var(--panel-2);
  color: var(--text);
  border: 1px solid var(--border);
  padding: 4px 10px;
  font-size: 11px;
  border-radius: 999px;
  font-weight: 500;
}
.theme-btn:hover { border-color: var(--accent); color: var(--accent); }
</style>
