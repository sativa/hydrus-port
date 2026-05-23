<script setup lang="ts">
import { ref, watch } from "vue";
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
  files.value = [];
  if (!props.job) return;
  try {
    files.value = await api.listOutputFiles(props.job.output_dir);
  } catch (e: any) {
    err.value = String(e);
  }
}

watch(
  () => [props.job?.id, props.job?.status, props.job?.finished_at_ms],
  refresh,
  { immediate: true },
);
</script>

<template>
  <div class="panel">
    <div class="row" style="justify-content: space-between">
      <div class="title">Output files</div>
      <button class="secondary" @click="refresh" :disabled="!job">Refresh</button>
    </div>
    <div v-if="!job" class="muted small">no job yet</div>
    <div v-else-if="!files.length" class="muted small">
      no files yet ({{ job.output_dir }})
    </div>
    <ul v-else class="files mono">
      <li v-for="f in files" :key="f.path">
        <span>{{ f.name }}</span>
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
</style>
