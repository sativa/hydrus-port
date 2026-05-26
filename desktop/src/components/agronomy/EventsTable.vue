<script setup lang="ts">
defineProps<{
  kind: "irrig" | "fert";
  rows: Array<Record<string, any>>;
}>();
const emit = defineEmits<{
  (e: "add"): void;
  (e: "remove", index: number): void;
}>();
</script>

<template>
  <div class="events">
    <table>
      <thead>
        <tr>
          <th>Date</th>
          <th v-if="kind==='irrig'">Depth (mm)</th>
          <th v-else>kgN/ha</th>
          <th v-if="kind==='fert'">conc (mg/L)</th>
          <th></th>
        </tr>
      </thead>
      <tbody>
        <tr v-for="(r, i) in rows" :key="i">
          <td><input type="date" v-model="r.date" /></td>
          <td v-if="kind==='irrig'">
            <input type="number" step="0.5" v-model.number="r.depth_mm" />
          </td>
          <td v-else>
            <input type="number" step="1" v-model.number="r.kg_n_ha" />
          </td>
          <td v-if="kind==='fert'">
            <input type="number" step="1" v-model.number="r.conc_mg_l" />
          </td>
          <td><button @click="emit('remove', i)">×</button></td>
        </tr>
      </tbody>
    </table>
    <button @click="emit('add')">+ add</button>
  </div>
</template>

<style scoped>
.events table { width: 100%; border-collapse: collapse; font-size: 12px; }
.events th, .events td { border: 1px solid var(--border, #ccc); padding: 2px 4px; }
.events input { width: 100%; box-sizing: border-box; border: none; background: transparent; }
.events button { margin-top: 4px; }
</style>
