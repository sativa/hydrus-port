<script setup lang="ts">
import { ref, watch, onBeforeUnmount } from "vue";
import { api, onJobLog, onJobStatus,
         type JobMeta, type LogLine } from "../api";
import type { UnlistenFn } from "@tauri-apps/api/event";

const emit = defineEmits<{ (e: "job-started", j: JobMeta): void }>();
const props = defineProps<{ externalJob?: JobMeta | null }>();

const busy = ref(false);
const lastJob = ref<JobMeta | null>(null);
const err = ref<string | null>(null);

// Parsed per-target PASS/FAIL from the streamed test output.
// `[PASS] 1d` / `[FAIL] 2d` lines from hydrus_port.cli are the source of truth.
type Verdict = "pending" | "running" | "pass" | "fail";
const verdicts = ref<Record<"1d" | "2d" | "3d", Verdict>>({
  "1d": "pending", "2d": "pending", "3d": "pending",
});
const wall_s = ref<Record<"1d" | "2d" | "3d", number | null>>({
  "1d": null, "2d": null, "3d": null,
});
const overall = ref<"pending" | "pass" | "fail">("pending");

let unsubLog: UnlistenFn | null = null;
let unsubStat: UnlistenFn | null = null;
let activeKind: "1d" | "2d" | "3d" | null = null;

async function run(target: "all" | "1d" | "2d" | "3d") {
  busy.value = true;
  err.value = null;
  overall.value = "pending";
  activeKind = null;
  // Reset only the targets we're about to run
  const touched: ("1d" | "2d" | "3d")[] =
    target === "all" ? ["1d", "2d", "3d"] : [target];
  for (const k of touched) {
    verdicts.value[k] = "pending";
    wall_s.value[k] = null;
  }
  try {
    const j = await api.startTest(target);
    lastJob.value = j;
    emit("job-started", j);
    await wireEvents(j.id);
  } catch (e: any) {
    err.value = String(e);
    busy.value = false;
  }
}

async function wireEvents(id: string) {
  if (unsubLog) { unsubLog(); unsubLog = null; }
  if (unsubStat) { unsubStat(); unsubStat = null; }
  unsubLog = await onJobLog(id, parseLine);
  unsubStat = await onJobStatus(id, () => { busy.value = false; });
}

function parseLine(l: LogLine) {
  // Match: "=== hydrus test 1d ===" → mark 1d as running
  const m1 = l.text.match(/===\s*hydrus test (1d|2d|3d)\s*===/);
  if (m1) {
    activeKind = m1[1] as "1d" | "2d" | "3d";
    verdicts.value[activeKind] = "running";
    return;
  }
  // Match: "[PASS] 1d" / "[FAIL] 2d"
  const m2 = l.text.match(/\[(PASS|FAIL)\]\s+(1d|2d|3d)/);
  if (m2) {
    const v: Verdict = m2[1] === "PASS" ? "pass" : "fail";
    verdicts.value[m2[2] as "1d" | "2d" | "3d"] = v;
    return;
  }
  // Match: "  wall_s: 0.642"
  const m3 = l.text.match(/wall_s:\s*([0-9.]+)/);
  if (m3 && activeKind) {
    wall_s.value[activeKind] = parseFloat(m3[1]);
    return;
  }
  // Match: final "OVERALL: PASS" / "OVERALL: FAIL"
  const m4 = l.text.match(/OVERALL:\s*(PASS|FAIL)/);
  if (m4) {
    overall.value = m4[1] === "PASS" ? "pass" : "fail";
  }
}

onBeforeUnmount(() => {
  unsubLog?.();
  unsubStat?.();
});

// Wire onto externally-started test jobs (e.g. fired by the AUTORUN
// hook in App.vue). We only auto-attach when the kind starts with
// "test:" — running a normal scenario should not blank these tiles.
watch(
  () => props.externalJob,
  async (j) => {
    if (!j || !j.kind?.startsWith?.("test")) return;
    if (lastJob.value?.id === j.id) return;
    lastJob.value = j;
    busy.value = j.status === "running";
    // Reset all three since `test all` is the typical autorun
    verdicts.value = { "1d": "pending", "2d": "pending", "3d": "pending" };
    wall_s.value = { "1d": null, "2d": null, "3d": null };
    overall.value = "pending";
    activeKind = null;
    await wireEvents(j.id);
  },
  { immediate: true },
);

function badgeClass(v: Verdict) {
  return {
    "badge-pending": v === "pending",
    "badge-running": v === "running",
    "badge-pass":    v === "pass",
    "badge-fail":    v === "fail",
  };
}
</script>

<template>
  <div class="panel">
    <div class="row" style="justify-content: space-between">
      <div class="title">Regression</div>
      <span v-if="overall === 'pass'" class="badge-pass">OVERALL PASS</span>
      <span v-else-if="overall === 'fail'" class="badge-fail">OVERALL FAIL</span>
    </div>
    <div class="grid3">
      <div class="cell" v-for="k in (['1d','2d','3d'] as const)" :key="k">
        <div class="row" style="justify-content: space-between">
          <span class="kind">{{ k }}</span>
          <span class="badge" :class="badgeClass(verdicts[k])">
            {{ verdicts[k] === "pending" ? "—" : verdicts[k].toUpperCase() }}
          </span>
        </div>
        <div class="wall muted" v-if="wall_s[k] !== null">
          {{ wall_s[k]!.toFixed(2) }} s
        </div>
      </div>
    </div>
    <div class="row" style="margin-top: 10px; gap: 6px; flex-wrap: wrap">
      <button @click="run('all')" :disabled="busy">
        {{ busy ? "running…" : "Run all" }}
      </button>
      <button class="secondary" @click="run('1d')" :disabled="busy">1d</button>
      <button class="secondary" @click="run('2d')" :disabled="busy">2d</button>
      <button class="secondary" @click="run('3d')" :disabled="busy">3d</button>
    </div>
    <div v-if="err" class="status-failed small mono" style="margin-top: 6px">
      {{ err }}
    </div>
  </div>
</template>

<style scoped>
.title { font-weight: 600; }
.grid3 {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 6px;
  margin-top: 8px;
}
.cell {
  border: 1px solid var(--border);
  border-radius: 4px;
  padding: 6px 8px;
  background: var(--panel-2);
}
.kind {
  font-family: ui-monospace, "SF Mono", Menlo, monospace;
  font-weight: 600;
}
.wall {
  font-size: 10px;
  margin-top: 2px;
  font-family: ui-monospace, "SF Mono", Menlo, monospace;
}
.badge {
  font-size: 10px;
  font-weight: 600;
  padding: 1px 6px;
  border-radius: 999px;
  letter-spacing: 0.3px;
}
.badge-pending { background: transparent; color: var(--muted); }
.badge-running { background: rgba(210, 153, 34, 0.18); color: var(--warn); }
.badge-pass    { background: rgba(63, 185, 80, 0.18);  color: var(--accent-2); }
.badge-fail    { background: rgba(248, 81, 73, 0.18);  color: var(--err); }
.small { font-size: 11px; }
</style>
