// Global theme switcher with localStorage persistence + a reactive
// `theme` ref so any chart that needs theme-aware colors can watch it.

import { ref, watch } from "vue";

export type Theme = "dark" | "light";

const KEY = "hydrus-port.theme";

function initialTheme(): Theme {
  // URL override: ?theme=light or ?theme=dark (useful for E2E screenshots)
  try {
    const u = new URLSearchParams(location.search);
    const q = u.get("theme");
    if (q === "light" || q === "dark") return q;
  } catch {}
  // Vite env override
  const envT = (import.meta as any).env?.VITE_THEME;
  if (envT === "light" || envT === "dark") return envT;
  try {
    const t = localStorage.getItem(KEY);
    if (t === "light" || t === "dark") return t;
  } catch {}
  // Default: respect OS preference
  if (typeof matchMedia !== "undefined" &&
      matchMedia("(prefers-color-scheme: light)").matches) {
    return "light";
  }
  return "dark";
}

export const theme = ref<Theme>(initialTheme());

function apply(t: Theme) {
  document.documentElement.setAttribute("data-theme", t);
  try { localStorage.setItem(KEY, t); } catch {}
}

apply(theme.value);
watch(theme, apply);

export function toggleTheme() {
  theme.value = theme.value === "dark" ? "light" : "dark";
}

// Read live CSS-variable values for Plotly chart layouts.
export function cssVar(name: string): string {
  const v = getComputedStyle(document.documentElement)
    .getPropertyValue(name).trim();
  return v || "#000000";
}

export function plotlyLayout() {
  return {
    paper_bgcolor: cssVar("--plot-paper"),
    plot_bgcolor:  cssVar("--plot-bg"),
    font:  { color: cssVar("--text"), size: 11 },
    xaxis: { gridcolor: cssVar("--plot-grid"),
             zerolinecolor: cssVar("--plot-grid") },
    yaxis: { gridcolor: cssVar("--plot-grid"),
             zerolinecolor: cssVar("--plot-grid") },
    margin: { l: 50, r: 18, t: 24, b: 36 },
  };
}
