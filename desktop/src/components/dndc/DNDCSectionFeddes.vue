<template>
  <div class="section">
    <p class="hint">Feddes piecewise-linear water-stress reduction function (all h in cm pressure head; negative = unsaturated).</p>

    <div class="preset-row">
      <label>Crop preset
        <select v-model="selectedPreset" @change="applyPreset">
          <option value="">-- select a preset --</option>
          <option v-for="(info, name) in presets" :key="name" :value="name">
            {{ name }} — {{ info.description }}
          </option>
        </select>
      </label>
      <span v-if="presetError" class="err">{{ presetError }}</span>
    </div>

    <div class="grid">
      <label>h1 — anaerobiosis onset (cm)
        <input type="number" v-model.number="feddes.h1" step="1" />
      </label>
      <label>h2 (cm)
        <input type="number" v-model.number="feddes.h2" step="1" />
      </label>
      <label>h3 high (cm) — high PET threshold
        <input type="number" v-model.number="feddes.h3_high" step="1" />
      </label>
      <label>h3 low (cm) — low PET threshold
        <input type="number" v-model.number="feddes.h3_low" step="1" />
      </label>
      <label>h4 — wilting point (cm)
        <input type="number" v-model.number="feddes.h4" step="10" />
      </label>
      <label>PET high threshold (cm/d)
        <input type="number" v-model.number="feddes.pet_high_cm_d" step="0.05" min="0" />
      </label>
      <label>PET low threshold (cm/d)
        <input type="number" v-model.number="feddes.pet_low_cm_d" step="0.05" min="0" />
      </label>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted } from "vue";
import { useDndcSeamStore } from "../../stores/dndcSeam";
import { dndc } from "../../api";

const store = useDndcSeamStore();
const feddes = computed(() => store.inputs.feddes);

type PresetEntry = { feddes: Record<string, number>; root: Record<string, unknown>; description: string };
const presets = ref<Record<string, PresetEntry>>({});
const selectedPreset = ref("");
const presetError = ref("");

onMounted(async () => {
  try {
    presets.value = await dndc.cropPresets() as Record<string, PresetEntry>;
  } catch {
    presetError.value = "Could not load crop presets (server offline?)";
  }
});

function applyPreset() {
  if (!selectedPreset.value) return;
  const p = presets.value[selectedPreset.value];
  if (!p) return;
  // Apply feddes params
  const f = p.feddes as Record<string, number>;
  store.inputs.feddes.h1 = f.h1 ?? store.inputs.feddes.h1;
  store.inputs.feddes.h2 = f.h2 ?? store.inputs.feddes.h2;
  store.inputs.feddes.h3_high = f.h3_high ?? store.inputs.feddes.h3_high;
  store.inputs.feddes.h3_low = f.h3_low ?? store.inputs.feddes.h3_low;
  store.inputs.feddes.h4 = f.h4 ?? store.inputs.feddes.h4;
  store.inputs.feddes.pet_high_cm_d = f.pet_high_cm_d ?? store.inputs.feddes.pet_high_cm_d;
  store.inputs.feddes.pet_low_cm_d = f.pet_low_cm_d ?? store.inputs.feddes.pet_low_cm_d;
  // Apply root params
  const r = p.root as Record<string, unknown>;
  if (r.z_max_cm !== undefined) store.inputs.root.z_max_cm = r.z_max_cm as number;
  if (r.growth_curve !== undefined) store.inputs.root.growth_curve = r.growth_curve as "linear" | "logistic" | "table";
  if (r.days_to_zmax !== undefined) store.inputs.root.days_to_zmax = r.days_to_zmax as number;
  if (r.density_profile !== undefined) store.inputs.root.density_profile = r.density_profile as "uniform" | "linear_decline" | "exponential" | "raats";
  if (r.density_param !== undefined) store.inputs.root.density_param = r.density_param as number;
}
</script>

<style scoped>
.section { padding: 6px 0; }
.hint { color: #666; font-size: 0.9em; margin: 0 0 8px; }
.preset-row { display: flex; align-items: flex-end; gap: 10px; margin-bottom: 10px; }
.err { color: #c00; font-size: 12px; }
.grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 10px; }
label { display: flex; flex-direction: column; gap: 3px; font-size: 12px; }
select, input[type="number"] { font-size: 12px; padding: 3px 6px; }
</style>
