<script setup lang="ts">
import { ref, watch, nextTick, onBeforeUnmount } from "vue";
import { api, onJobLog, onJobStatus, type JobMeta, type LogLine } from "../api";
import type { UnlistenFn } from "@tauri-apps/api/event";

const props = defineProps<{ job: JobMeta | null }>();
const emit = defineEmits<{ (e: "job-status", j: JobMeta): void }>();

const lines = ref<LogLine[]>([]);
const scrollEl = ref<HTMLDivElement | null>(null);
let unsubLog: UnlistenFn | null = null;
let unsubStat: UnlistenFn | null = null;

watch(
  () => props.job?.id ?? null,
  async (id, oldId) => {
    if (unsubLog) { unsubLog(); unsubLog = null; }
    if (unsubStat) { unsubStat(); unsubStat = null; }
    lines.value = [];
    if (!id) return;
    unsubLog = await onJobLog(id, (l) => {
      lines.value.push(l);
      if (lines.value.length > 5000) {
        lines.value.splice(0, lines.value.length - 5000);
      }
      nextTick(() => {
        scrollEl.value?.scrollTo({
          top: scrollEl.value.scrollHeight,
        });
      });
    });
    unsubStat = await onJobStatus(id, async () => {
      // Refresh job metadata on completion
      const j = await api.getJob(id);
      if (j) emit("job-status", j);
    });
  },
);

onBeforeUnmount(() => {
  unsubLog?.();
  unsubStat?.();
});

async function stop() {
  if (props.job) await api.stop(props.job.id);
}
</script>

<template>
  <div class="panel log-panel">
    <div class="row" style="justify-content: space-between">
      <div class="title">
        Log
        <span v-if="job" class="muted small mono">
          · {{ job.id }} ·
          <span :class="`status-${job.status}`">{{ job.status }}</span>
          <span v-if="job.exit_code !== null"> · exit {{ job.exit_code }}</span>
        </span>
      </div>
      <button
        class="danger"
        @click="stop"
        :disabled="!job || job.status !== 'running'"
      >Stop</button>
    </div>
    <div ref="scrollEl" class="log-body mono">
      <div
        v-for="(l, i) in lines"
        :key="i"
        :class="l.stream === 'stderr' ? 'err' : ''"
      >{{ l.text }}</div>
      <div v-if="!lines.length" class="muted">
        {{ job ? "waiting for first line…" : "no job running" }}
      </div>
    </div>
  </div>
</template>

<style scoped>
.log-panel {
  flex: 1;
  min-height: 200px;
  display: flex;
  flex-direction: column;
}
.title {
  font-weight: 600;
  margin-bottom: 8px;
}
.small { font-size: 11px; }
.log-body {
  flex: 1;
  overflow-y: auto;
  font-size: 11px;
  background: var(--bg);
  border: 1px solid var(--border);
  border-radius: 4px;
  padding: 6px 8px;
  white-space: pre-wrap;
  word-break: break-all;
  line-height: 1.4;
}
.log-body .err { color: var(--err); }
</style>
