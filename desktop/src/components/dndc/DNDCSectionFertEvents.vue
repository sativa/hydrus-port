<template>
  <div class="section">
    <p class="hint">Fertilizer events. Each event is applied at a specific date, depth, and rate.</p>
    <button class="add-btn" @click="addEvent">+ Add event</button>

    <div v-if="events.length === 0" class="empty">No events — click above to add.</div>

    <div v-for="(ev, i) in events" :key="i" class="event-row">
      <span class="event-idx">#{{ i + 1 }}</span>
      <label>Date
        <input type="date" v-model="ev.date" />
      </label>
      <label>Depth (cm)
        <input type="number" v-model.number="ev.depth_cm" step="0.5" min="0" />
      </label>
      <label>N rate (kg N/ha)
        <input type="number" v-model.number="ev.mass_kg_n_ha" step="1" min="0" />
      </label>
      <label>Form
        <select v-model="ev.form">
          <option value="urea">urea</option>
          <option value="NH4">NH4</option>
          <option value="NO3">NO3</option>
          <option value="NH4NO3">NH4NO3</option>
          <option value="compound">compound</option>
        </select>
      </label>
      <template v-if="ev.form === 'compound'">
        <label>Composition (NH4:NO3 fractions, e.g. 0.5,0.5)
          <input type="text" :value="compText(ev)" @change="(e) => setComp(ev, (e.target as HTMLInputElement).value)" placeholder="0.5,0.5" />
        </label>
      </template>
      <button class="del-btn" @click="removeEvent(i)" title="Remove">&#x2715;</button>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed } from "vue";
import { useDndcSeamStore, type FertEvent } from "../../stores/dndcSeam";

const store = useDndcSeamStore();
const events = computed(() => store.inputs.fert_events);

function blankEvent(): FertEvent {
  return { date: new Date().toISOString().slice(0, 10), depth_cm: 0, mass_kg_n_ha: 100, form: "urea" };
}

function addEvent() { store.inputs.fert_events.push(blankEvent()); }
function removeEvent(i: number) { store.inputs.fert_events.splice(i, 1); }

function compText(ev: FertEvent): string {
  if (!ev.composition) return "";
  return Object.values(ev.composition).join(",");
}
function setComp(ev: FertEvent, text: string) {
  const parts = text.split(",").map((s) => parseFloat(s.trim())).filter((n) => !isNaN(n));
  if (parts.length === 2) {
    ev.composition = { NH4: parts[0], NO3: parts[1] };
  }
}
</script>

<style scoped>
.section { padding: 6px 0; }
.hint { color: #666; font-size: 0.9em; margin: 0 0 8px; }
.add-btn { margin-bottom: 10px; }
.empty { color: #888; font-size: 12px; font-style: italic; }
.event-row {
  display: flex; flex-wrap: wrap; gap: 10px; align-items: flex-end;
  padding: 8px; border: 1px solid var(--border, #ddd); border-radius: 4px; margin-bottom: 6px;
}
.event-idx { font-size: 11px; color: #888; width: 24px; flex-shrink: 0; padding-top: 18px; }
label { display: flex; flex-direction: column; gap: 2px; font-size: 12px; }
select, input { font-size: 12px; padding: 3px 6px; }
.del-btn { background: transparent; color: #c00; border: 1px solid #c00; border-radius: 3px;
           padding: 2px 6px; cursor: pointer; margin-top: 16px; }
</style>
