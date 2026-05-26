<script setup lang="ts">
import { ref, onMounted } from "vue";
import ScenarioPicker from "../ScenarioPicker.vue";
import LogStream      from "../LogStream.vue";
import OutputBrowser  from "../OutputBrowser.vue";
import Regression     from "../Regression.vue";
import { api, type JobMeta, type PythonInfo } from "../../api";

const py  = ref<PythonInfo | null>(null);
const job = ref<JobMeta | null>(null);

onMounted(async () => {
  try { py.value = await api.detectPython(); } catch {}
});

function onJobStarted(j: JobMeta) { job.value = j; }
function onJobUpdated(j: JobMeta) { job.value = j; }
</script>

<template>
  <div class="classic">
    <ScenarioPicker
      :python-ready="!!py"
      @job-started="onJobStarted"
    />
    <Regression :external-job="job" @job-started="onJobStarted" />
    <OutputBrowser :job="job" />
    <LogStream :job="job" @job-status="onJobUpdated" />
  </div>
</template>

<style scoped>
.classic > * { margin-bottom: 8px; }
</style>
