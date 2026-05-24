<template>
  <div class="section">
    <p class="hint">Irrigation events. Drip emitter XYZ and fertigation concentrations are optional.</p>
    <button class="add-btn" @click="addEvent">+ Add event</button>

    <div v-if="events.length === 0" class="empty">No events — click above to add.</div>

    <div v-for="(ev, i) in events" :key="i" class="event-row">
      <span class="event-idx">#{{ i + 1 }}</span>
      <label>Date
        <input type="date" v-model="ev.date" />
      </label>
      <label>Method
        <select v-model="ev.method">
          <option value="flood">flood</option>
          <option value="sprinkler">sprinkler</option>
          <option value="drip">drip</option>
          <option value="subsurface">subsurface</option>
        </select>
      </label>
      <label>Amount (cm)
        <input type="number" v-model.number="ev.amount_cm" step="0.1" min="0" />
      </label>
      <label>Duration (h)
        <input type="number" v-model.number="ev.duration_h" step="0.5" min="0" />
      </label>
      <label>Fertigation NO3 (mg/L)
        <input type="number" :value="ev.solute_concs_mg_l['NO3'] ?? ''"
               @input="(e) => setSolute(ev, 'NO3', (e.target as HTMLInputElement).value)"
               step="1" min="0" placeholder="0" />
      </label>
      <template v-if="ev.method === 'drip'">
        <label>Emitter X,Y,Z (cm)
          <input type="text"
                 :value="dripText(ev)"
                 @change="(e) => setDrip(ev, (e.target as HTMLInputElement).value)"
                 placeholder="10,10,-5" />
        </label>
      </template>
      <button class="del-btn" @click="removeEvent(i)" title="Remove">&#x2715;</button>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed } from "vue";
import { useDndcSeamStore, type IrrigEvent } from "../../stores/dndcSeam";

const store = useDndcSeamStore();
const events = computed(() => store.inputs.irrig_events);

function blankEvent(): IrrigEvent {
  return {
    date: new Date().toISOString().slice(0, 10),
    method: "sprinkler", amount_cm: 2.5, duration_h: 4,
    solute_concs_mg_l: {},
  };
}
function addEvent() { store.inputs.irrig_events.push(blankEvent()); }
function removeEvent(i: number) { store.inputs.irrig_events.splice(i, 1); }

function setSolute(ev: IrrigEvent, ion: string, val: string) {
  const n = parseFloat(val);
  if (!isNaN(n) && n > 0) { ev.solute_concs_mg_l[ion] = n; }
  else { delete ev.solute_concs_mg_l[ion]; }
}

function dripText(ev: IrrigEvent): string {
  return ev.drip_emitter_xyz ? ev.drip_emitter_xyz.join(",") : "";
}
function setDrip(ev: IrrigEvent, text: string) {
  const parts = text.split(",").map((s) => parseFloat(s.trim()));
  if (parts.length === 3 && parts.every((n) => !isNaN(n))) {
    ev.drip_emitter_xyz = [parts[0], parts[1], parts[2]];
  } else {
    ev.drip_emitter_xyz = undefined;
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
