<script setup lang="ts">
import { ref } from "vue";
import ScenarioEditor    from "../ScenarioEditor.vue";
import MeshViewer3D      from "../MeshViewer3D.vue";
import DNDCForms         from "../../pages/research/DNDCForms.vue";
import SoilLibrary       from "../../pages/research/SoilLibrary.vue";
import BatchSweep        from "../../pages/research/BatchSweep.vue";
import SensitivityReport from "../../pages/research/SensitivityReport.vue";
import InversionStudio   from "../../pages/research/InversionStudio.vue";
import UQExplorer        from "../../pages/research/UQExplorer.vue";
import SurrogateBench    from "../../pages/research/SurrogateBench.vue";
import ClassicPanel      from "./ClassicPanel.vue";
import { type JobMeta } from "../../api";

defineProps<{ open: boolean; editorPath: string | null; job?: JobMeta | null }>();
defineEmits<{ (e: "close"): void }>();

type Tab = "classic" | "editor" | "3d" | "dndc" | "soil"
         | "batch" | "sensitivity" | "inversion" | "uq" | "surrogate";
const tab = ref<Tab>("classic");
</script>

<template>
  <aside v-if="open" class="drawer">
    <header>
      <strong>高级 / 研究台</strong>
      <button @click="$emit('close')">×</button>
    </header>
    <nav>
      <button :class="{on:tab==='classic'}"     @click="tab='classic'">经典</button>
      <button :class="{on:tab==='editor'}"      @click="tab='editor'">Editor</button>
      <button :class="{on:tab==='3d'}"          @click="tab='3d'">3D mesh</button>
      <button :class="{on:tab==='dndc'}"        @click="tab='dndc'">DNDC</button>
      <button :class="{on:tab==='soil'}"        @click="tab='soil'">Soil lib</button>
      <button :class="{on:tab==='batch'}"       @click="tab='batch'">Batch</button>
      <button :class="{on:tab==='sensitivity'}" @click="tab='sensitivity'">Sens</button>
      <button :class="{on:tab==='inversion'}"   @click="tab='inversion'">Invert</button>
      <button :class="{on:tab==='uq'}"          @click="tab='uq'">UQ</button>
      <button :class="{on:tab==='surrogate'}"   @click="tab='surrogate'">Surrog</button>
    </nav>
    <section class="body">
      <ClassicPanel       v-if="tab==='classic'" />
      <ScenarioEditor     v-else-if="tab==='editor'"      :scenario-path="editorPath" />
      <MeshViewer3D       v-else-if="tab==='3d'"          :job="job ?? null" />
      <DNDCForms          v-else-if="tab==='dndc'" />
      <SoilLibrary        v-else-if="tab==='soil'" />
      <BatchSweep         v-else-if="tab==='batch'" />
      <SensitivityReport  v-else-if="tab==='sensitivity'" />
      <InversionStudio    v-else-if="tab==='inversion'" />
      <UQExplorer         v-else-if="tab==='uq'" />
      <SurrogateBench     v-else-if="tab==='surrogate'" />
    </section>
  </aside>
</template>

<style scoped>
.drawer {
  position: fixed; top: 0; right: 0; bottom: 0; width: 560px; max-width: 80vw;
  background: var(--bg, #fff); border-left: 1px solid var(--border, #ccc);
  box-shadow: -4px 0 12px rgba(0,0,0,0.15);
  display: flex; flex-direction: column; z-index: 100;
}
header { display:flex; justify-content:space-between; padding:8px 12px; border-bottom:1px solid var(--border,#ccc); }
nav { display: flex; flex-wrap: wrap; gap: 4px; padding: 6px; border-bottom: 1px solid var(--border, #ccc); }
nav button { font-size: 12px; padding: 2px 6px; }
nav .on { font-weight: 600; background: var(--accent-bg, #eef); }
.body { flex: 1; overflow: auto; padding: 8px; }
</style>
