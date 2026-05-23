import { defineConfig } from "vite";
import vue from "@vitejs/plugin-vue";

// Vite + Tauri config. Tauri injects the dev server URL into the
// webview during `tauri dev`, so we just need to make sure the dev
// server binds to a stable port that tauri.conf.json knows about.
export default defineConfig({
  plugins: [vue()],
  clearScreen: false,
  server: {
    port: 1420,
    strictPort: true,
    host: false,
    hmr: { protocol: "ws", host: "localhost", port: 1421 },
  },
  envPrefix: ["VITE_", "TAURI_"],
  build: {
    target: ["es2021", "chrome105", "safari13"],
    minify: !process.env.TAURI_DEBUG ? "esbuild" : false,
    sourcemap: !!process.env.TAURI_DEBUG,
  },
});
