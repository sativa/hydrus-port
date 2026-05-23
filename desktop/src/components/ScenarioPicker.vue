<script setup lang="ts">
import { ref, onMounted, computed } from "vue";
import { api, type Scenario, type JobMeta } from "../api";

const props = defineProps<{ pythonReady: boolean }>();
const emit = defineEmits<{ (e: "job-started", j: JobMeta): void }>();

const scenarios = ref<Scenario[]>([]);
const selected = ref<string | null>(null);
const busy = ref(false);
const err = ref<string | null>(null);

const current = computed(() =>
  scenarios.value.find((s) => s.path === selected.value) ?? null,
);

onMounted(async () => {
  try {
    scenarios.value = await api.listScenarios();
    if (scenarios.value.length) selected.value = scenarios.value[0].path;
  } catch (e: any) {
    err.value = String(e);
  }
  // Dev/E2E hook: VITE_AUTORUN=<name-substring> picks that scenario
  // and fires Run automatically once python is ready.
  const auto = (import.meta as any).env?.VITE_AUTORUN;
  console.log("[autorun] VITE_AUTORUN =", auto,
              "scenarios:", scenarios.value.length);
  if (auto) {
    const match =
      scenarios.value.find((s) => s.name.includes(auto)) ??
      scenarios.value[0];
    console.log("[autorun] selected:", match?.name, match?.kind, match?.path);
    if (match) {
      selected.value = match.path;
      setTimeout(() => {
        console.log("[autorun] firing run() pythonReady=", props.pythonReady);
        run();
      }, 1500);
    }
  }
});

async function run() {
  if (!current.value) return;
  busy.value = true;
  err.value = null;
  try {
    const job = await api.start({
      kind: current.value.kind,
      input_dir: current.value.path,
    });
    emit("job-started", job);
  } catch (e: any) {
    err.value = String(e);
  } finally {
    busy.value = false;
  }
}
</script>

<template>
  <div class="panel">
    <div class="title">Scenario</div>
    <select v-model="selected" class="wide">
      <option v-for="s in scenarios" :key="s.path" :value="s.path">
        {{ s.name }} — {{ s.kind }}
      </option>
    </select>
    <div v-if="current" class="muted small mono">{{ current.path }}</div>
    <div v-if="current?.description" class="muted small">
      {{ current.description }}
    </div>
    <div class="row" style="margin-top: 10px">
      <button
        @click="run"
        :disabled="!pythonReady || !current || busy"
      >
        {{ busy ? "starting…" : "Run" }}
      </button>
      <span v-if="!pythonReady" class="muted small">waiting for python</span>
    </div>
    <div v-if="err" class="status-failed small mono" style="margin-top: 6px">
      {{ err }}
    </div>
  </div>
</template>

<style scoped>
.title {
  font-weight: 600;
  margin-bottom: 8px;
}
.wide {
  width: 100%;
}
.small {
  font-size: 11px;
}
</style>
