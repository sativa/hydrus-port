<script setup lang="ts">
import { ref, watch, onBeforeUnmount } from "vue";
import { api, type JobMeta, type OutputFile } from "../api";

const props = defineProps<{ job: JobMeta | null }>();
const files = ref<OutputFile[]>([]);
const err = ref<string | null>(null);

function formatSize(n: number): string {
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  return `${(n / 1024 / 1024).toFixed(2)} MB`;
}

async function refresh() {
  err.value = null;
  if (!props.job) { files.value = []; return; }
  try {
    files.value = await api.listOutputFiles(props.job.output_dir);
  } catch (e: any) {
    err.value = String(e);
  }
}

// Poll while a job is running so files appear as they're written; stop
// once status flips out of "running".
let pollTimer: ReturnType<typeof setInterval> | null = null;
function startPolling() {
  stopPolling();
  pollTimer = setInterval(refresh, 1000);
}
function stopPolling() {
  if (pollTimer) { clearInterval(pollTimer); pollTimer = null; }
}

watch(
  () => [props.job?.id, props.job?.status, props.job?.finished_at_ms],
  () => {
    refresh();
    if (props.job?.status === "running") startPolling();
    else stopPolling();
  },
  { immediate: true },
);

onBeforeUnmount(stopPolling);
</script>

<template>
  <div class="panel">
    <div class="row" style="justify-content: space-between">
      <div class="title">
        Output files
        <span v-if="files.length" class="muted small">({{ files.length }})</span>
      </div>
      <button class="secondary" @click="refresh" :disabled="!job">Refresh</button>
    </div>
    <div v-if="job" class="muted mono small dir-line"
         :title="job.output_dir">
      {{ job.output_dir }}
    </div>
    <div v-if="!job" class="muted small">no job yet</div>
    <div v-else-if="!files.length" class="muted small">
      no files in this directory yet
    </div>
    <ul v-else class="files mono">
      <li v-for="f in files" :key="f.path">
        <span :title="f.path">{{ f.name }}</span>
        <span class="muted small">{{ formatSize(f.size) }}</span>
      </li>
    </ul>
    <div v-if="err" class="status-failed small mono">{{ err }}</div>
  </div>
</template>

<style scoped>
.title { font-weight: 600; }
.files {
  list-style: none;
  padding: 0;
  margin: 8px 0 0;
  max-height: 220px;
  overflow-y: auto;
  font-size: 11px;
}
.files li {
  display: flex;
  justify-content: space-between;
  padding: 3px 4px;
  border-bottom: 1px solid var(--border);
}
.small { font-size: 11px; }
.dir-line {
  margin: 2px 0 4px 0;
  word-break: break-all;
  white-space: pre-wrap;
  font-size: 10px;
  opacity: 0.7;
}
</style>
