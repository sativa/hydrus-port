<script setup lang="ts">
import { useAgronomyStore } from "../../stores/agronomy";
import ThetaHeatmap from "./ThetaHeatmap.vue";
import NitrateHeatmap from "./NitrateHeatmap.vue";

const store = useAgronomyStore();
</script>

<template>
  <section class="heatmap-band">
    <div v-if="!store.result" class="empty">运行一次后看含水量/硝态 N 的时间-深度热图</div>
    <template v-else>
      <ThetaHeatmap :z="store.result.z_cm" :t="store.result.t_days"
                    :field="store.result.theta_zt" :events="store.result.events"
                    title="θ(z,t) 体积含水量" unit="cm³/cm³" />
      <NitrateHeatmap :z="store.result.z_cm" :t="store.result.t_days"
                      :field="store.result.n_zt" :events="store.result.events" />
    </template>
  </section>
</template>

<style scoped>
.heatmap-band { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; padding: 8px; }
.empty { grid-column: 1 / -1; padding: 40px; text-align: center; color: #888; }
</style>
