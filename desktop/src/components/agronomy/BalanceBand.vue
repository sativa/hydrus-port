<script setup lang="ts">
import { useAgronomyStore } from "../../stores/agronomy";
import WaterBalanceBar from "./WaterBalanceBar.vue";
import NBudgetBar from "./NBudgetBar.vue";
const store = useAgronomyStore();
</script>

<template>
  <section v-if="store.result" class="balance-band">
    <WaterBalanceBar :balance="store.result.water_balance" />
    <NBudgetBar :budget="store.result.n_budget" />
  </section>
</template>

<style scoped>
.balance-band { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; padding: 8px; border-top: 1px solid var(--border, #ccc); }
</style>
