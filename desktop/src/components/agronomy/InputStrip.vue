<script setup lang="ts">
import { onMounted } from "vue";
import { useAgronomyStore } from "../../stores/agronomy";
import EventsTable from "./EventsTable.vue";

const store = useAgronomyStore();
onMounted(() => store.loadLibs());
</script>

<template>
  <section class="input-strip">
    <div class="row">
      <label>作物
        <select v-model="store.cropId">
          <option v-for="c in store.crops" :key="c.id" :value="c.id">{{ c.name_zh }}</option>
        </select>
      </label>
      <label>土壤
        <select v-model="store.soilId">
          <option v-for="s in store.soils" :key="s.id" :value="s.id">{{ s.name_zh }}</option>
        </select>
      </label>
      <label>雨型
        <select v-model="store.weatherId">
          <option v-for="w in store.weather" :key="w.id" :value="w.id">{{ w.name_zh }}</option>
        </select>
      </label>
      <label>天数
        <input type="number" min="1" max="365" v-model.number="store.horizonDays" style="width:60px" />
      </label>
      <button class="run" :disabled="store.running" @click="store.run()">
        {{ store.running ? "运行中…" : "▶ 运行" }}
      </button>
      <span class="error" v-if="store.error">{{ store.error }}</span>
    </div>
    <div class="row tables">
      <div class="col">
        <h4>灌溉事件</h4>
        <EventsTable kind="irrig" :rows="store.irrig"
                     @add="store.addIrrig" @remove="store.removeIrrig" />
      </div>
      <div class="col">
        <h4>施肥事件</h4>
        <EventsTable kind="fert" :rows="store.fert"
                     @add="store.addFert" @remove="store.removeFert" />
      </div>
    </div>
  </section>
</template>

<style scoped>
.input-strip { padding: 8px 12px; border-bottom: 1px solid var(--border, #ccc); }
.row { display: flex; gap: 12px; align-items: center; flex-wrap: wrap; }
.tables { margin-top: 8px; align-items: stretch; }
.col { flex: 1; min-width: 280px; }
.run { padding: 4px 12px; font-weight: 600; }
.error { color: #c00; font-size: 12px; }
h4 { margin: 0 0 4px 0; font-size: 12px; }
</style>
