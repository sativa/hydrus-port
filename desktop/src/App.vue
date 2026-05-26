<script setup lang="ts">
import { ref } from "vue";
import { theme, toggleTheme } from "./theme";

import InputStrip     from "./components/agronomy/InputStrip.vue";
import HeatmapBand    from "./components/agronomy/HeatmapBand.vue";
import BalanceBand    from "./components/agronomy/BalanceBand.vue";
import AdvancedDrawer from "./components/agronomy/AdvancedDrawer.vue";

const drawerOpen = ref(false);
const editorPath = ref<string | null>(null);
</script>

<template>
  <div class="shell">
    <header>
      <span class="brand">HYDRUS Agronomy</span>
      <span class="spacer" />
      <button @click="drawerOpen = true">高级 ▸</button>
      <button class="theme" @click="toggleTheme">{{ theme === 'dark' ? '☀' : '☾' }}</button>
    </header>

    <InputStrip />
    <HeatmapBand />
    <BalanceBand />

    <AdvancedDrawer :open="drawerOpen" :editorPath="editorPath" :job="null"
                    @close="drawerOpen = false" />
  </div>
</template>

<style scoped>
.shell { display: flex; flex-direction: column; height: 100vh; }
header { display: flex; align-items: center; padding: 6px 12px;
         border-bottom: 1px solid var(--border, #ccc); gap: 8px; }
.brand { font-weight: 700; }
.spacer { flex: 1; }
.theme { font-size: 14px; }
</style>
