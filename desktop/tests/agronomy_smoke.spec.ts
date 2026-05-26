import { test, expect } from "@playwright/test";

test("agronomy: pick maize+loam+n_china_avg, add 1 irrig + 1 fert, run", async ({ page }) => {
  await page.goto("/");

  // Libs load on mount — wait for at least one option to appear in the first <select>
  await expect(page.locator("select").first()).toBeVisible({ timeout: 15000 });
  await page.waitForFunction(() => {
    const sel = document.querySelector("select");
    return sel && sel.options.length > 0;
  }, { timeout: 15000 });

  // Pick crop / soil / weather (first 3 selects in the InputStrip).
  // The store already defaults to maize / loam / n_china_avg, but we
  // explicitly select them to confirm the options are present.
  await page.locator("select").nth(0).selectOption("maize");
  await page.locator("select").nth(1).selectOption("loam");
  await page.locator("select").nth(2).selectOption("n_china_avg");

  // Add events: each EventsTable renders a "+ add" button at the bottom.
  const addButtons = page.getByText("+ add");
  await addButtons.nth(0).click();   // irrig table
  await addButtons.nth(1).click();   // fert table

  // Run the simulation.
  // InputStrip renders: {{ store.running ? "运行中…" : "▶ 运行" }}
  await page.locator("button.run").click();

  // Wait for the simulation to complete and Plotly to render.
  // ThetaHeatmap and NitrateHeatmap both render a div.heatmap;
  // Plotly inserts an <svg> (and sometimes a <canvas>) inside.
  await page.waitForSelector(".heatmap svg, .heatmap canvas", { timeout: 60_000 });

  // BalanceBand (.balance-band) is v-if="store.result" — it appears once run completes.
  await expect(page.locator(".balance-band")).toBeVisible({ timeout: 10_000 });

  // Open the advanced drawer via the header button "高级 ▸".
  // App.vue: <button @click="drawerOpen = true">高级 ▸</button>
  await page.getByRole("button", { name: /高级/ }).first().click();

  // The drawer should be visible.
  await expect(page.locator(".drawer")).toBeVisible({ timeout: 5_000 });

  // Verify all 10 nav tabs are present inside the drawer nav.
  // AdvancedDrawer.vue nav buttons (exact text):
  //   经典 | Editor | 3D mesh | DNDC | Soil lib | Batch | Sens | Invert | UQ | Surrog
  const drawerNav = page.locator(".drawer nav");
  for (const label of [
    "经典", "Editor", "3D mesh", "DNDC", "Soil lib",
    "Batch", "Sens", "Invert", "UQ", "Surrog",
  ]) {
    await expect(drawerNav.getByRole("button", { name: label })).toBeVisible();
  }

  // Capture a full-page screenshot for visual reference.
  await page.screenshot({
    path: "tests/screenshots/agronomy_smoke.png",
    fullPage: true,
  });
});
