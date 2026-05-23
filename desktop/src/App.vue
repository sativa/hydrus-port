<script setup lang="ts">
import { ref, onMounted } from "vue";
import ScenarioPicker from "./components/ScenarioPicker.vue";
import LogStream from "./components/LogStream.vue";
import ProfilePlot from "./components/ProfilePlot.vue";
import MeshViewer3D from "./components/MeshViewer3D.vue";
import OutputBrowser from "./components/OutputBrowser.vue";
import { api, type JobMeta, type PythonInfo } from "./api";

const py = ref<PythonInfo | null>(null);
const py_err = ref<string | null>(null);
const job = ref<JobMeta | null>(null);

onMounted(async () => {
  try {
    py.value = await api.detectPython();
  } catch (e: any) {
    py_err.value = String(e);
  }
});

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
      <div class="muted mono small">
        <span v-if="py">
          {{ py.executable }} · {{ py.version }} · root {{ py.repo_root }}
        </span>
        <span v-else-if="py_err" class="status-failed">{{ py_err }}</span>
        <span v-else>detecting python…</span>
      </div>
    </header>

    <main class="grid">
      <section class="col-left">
        <ScenarioPicker
          :python-ready="!!py"
          @job-started="onJobStarted"
        />
        <OutputBrowser :job="job" />
      </section>

      <section class="col-mid">
        <ProfilePlot :job="job" />
        <LogStream :job="job" @job-status="onJobUpdated" />
      </section>

      <section class="col-right">
        <MeshViewer3D :job="job" />
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
</style>
