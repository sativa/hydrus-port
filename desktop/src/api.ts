// Thin wrapper around Tauri invoke + event APIs so components don't
// have to import @tauri-apps/api directly.

import { invoke } from "@tauri-apps/api/core";
import { listen, type UnlistenFn } from "@tauri-apps/api/event";

export type Scenario = {
  name: string;
  kind: "hydrus1d" | "swms2d" | "richards3d" | "unknown";
  path: string;
  description: string;
};

export type JobMeta = {
  id: string;
  kind: string;
  scenario: string;
  input_dir: string;
  output_dir: string;
  status: "running" | "done" | "failed" | "cancelled";
  exit_code: number | null;
  started_at_ms: number;
  finished_at_ms: number | null;
};

export type PythonInfo = {
  executable: string;
  version: string;
  repo_root: string;
};

export type Series = {
  headers: string[];
  rows: number[][];
};

export type NodInfSeries = {
  times: number[];                            // n_t
  depths: number[];                           // n_z (ascending)
  vars: Record<string, number[][]>;           // each name -> n_t × n_z
  var_names: string[];
};

export type Swms2dMesh = {
  nodes_x: number[];
  nodes_z: number[];
  triangles: [number, number, number][];      // 0-indexed
  num_np: number;
  num_el: number;
};

export type Swms2dField = {
  times: number[];
  values: number[][];                         // n_t × num_np
  num_np: number;
};

export type CanonicalMaterial = {
  theta_r: number; theta_s: number;
  alpha: number; n: number; Ks: number;
  l: number;
  theta_a: number | null; theta_m: number | null;
  theta_k: number | null; Kk: number | null;
};

export type CanonicalScenario = {
  schema_version: string;
  meta: { name: string; description: string; source: string };
  units: { length: string; time: string; mass: string };
  solver: {
    geometry_kind: "horizontal" | "axisymmetric" | "vertical";
    max_picard: number;
    tol_theta: number;
    tol_h: number;
    water_flow: boolean;
    solute_transport: boolean;
    heat_transport: boolean;
    root_uptake: boolean;
    atmospheric_bc: boolean;
    free_drainage: boolean;
    seepage_face: boolean;
    subsurface_drain: boolean;
    gw_level: boolean;
    short_output: boolean;
    flux_output: boolean;
    check_output: boolean;
    hysteresis: boolean;
    equilibrium: boolean;
  };
  materials: CanonicalMaterial[];
  time: {
    dt: number; dt_min: number; dt_max: number;
    dt_mul: number; dt_mul2: number;
    it_min: number; it_max: number;
    t_init: number; t_max: number;
    print_times: number[];
  };
  root_uptake: any | null;
  solute: any | null;
  geometry: { kind: "1d" | "2d" | "3d"; [k: string]: any };
  atmospheric: any | null;
  seepage: any | null;
  drain: any | null;
  legacy_extras: Record<string, any>;
};

export type OutputFile = {
  name: string;
  path: string;
  size: number;
};

export type LogLine = { stream: "stdout" | "stderr"; text: string };

export const api = {
  listScenarios: () => invoke<Scenario[]>("list_scenarios"),
  detectPython: () => invoke<PythonInfo>("detect_python"),
  debugLog: (text: string) => invoke<void>("debug_log", { text }),

  start: (args: {
    kind: string;
    input_dir: string;
    output_dir?: string;
    extra_args?: string[];
  }) => invoke<JobMeta>("start_simulation", { args }),

  startTest: (target: "all" | "1d" | "2d" | "3d") =>
    invoke<JobMeta>("start_test", { args: { target } }),

  stop: (id: string) => invoke<void>("stop_simulation", { id }),
  listJobs: () => invoke<JobMeta[]>("list_jobs"),
  getJob: (id: string) => invoke<JobMeta | null>("get_job", { id }),

  listOutputFiles: (dir: string) =>
    invoke<OutputFile[]>("list_output_files", { dir }),
  readText: (path: string, max_bytes?: number) =>
    invoke<string>("read_output_text", { path, maxBytes: max_bytes }),
  parseTable: (path: string) =>
    invoke<Series>("parse_obs_node", { path }),
  parseNodInf: (path: string) =>
    invoke<NodInfSeries>("parse_nod_inf_series", { path }),
  parseSwms2dGrid: (path: string) =>
    invoke<Swms2dMesh>("parse_swms2d_grid", { path }),
  parseSwms2dField: (path: string, numNp: number) =>
    invoke<Swms2dField>("parse_swms2d_field", { path, numNp }),

  readScenario: (inputDir: string) =>
    invoke<CanonicalScenario>("read_scenario", { inputDir }),
  writeScenario: (inputDir: string, payload: CanonicalScenario) =>
    invoke<void>("write_scenario", { inputDir, payload }),
  listVtuSeries: (dir: string) =>
    invoke<string[]>("list_vtu_series", { dir }),
  readBytes: (path: string) =>
    invoke<number[]>("read_output_bytes", { path }),
};

export function onJobLog(
  id: string,
  cb: (line: LogLine) => void,
): Promise<UnlistenFn> {
  return listen<LogLine>(`job://${id}/log`, (e) => cb(e.payload));
}

export function onJobStatus(
  id: string,
  cb: (s: { status: string; exit_code: number | null }) => void,
): Promise<UnlistenFn> {
  return listen(`job://${id}/status`, (e) => cb(e.payload as any));
}

// Research-layer REST endpoints. Both DNDC (M1) and PTF (M2) point at the
// same FastAPI sidecar; override base URL via VITE_DNDC_BASE.
const RESEARCH_BASE: string =
  (import.meta.env.VITE_DNDC_BASE as string | undefined) ??
  "http://127.0.0.1:8765";

// ---- M1: DNDC seam --------------------------------------------------
export const dndc = {
  async validate(inputs: unknown): Promise<{ ok: boolean; warnings?: string[] }> {
    const r = await fetch(`${RESEARCH_BASE}/research/dndc/validate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(inputs),
    });
    if (r.status === 422) {
      const detail = await r.json();
      throw new Error(`validation failed: ${JSON.stringify(detail)}`);
    }
    if (!r.ok) throw new Error(`HTTP ${r.status}: ${await r.text()}`);
    return r.json();
  },

  async cropPresets(): Promise<Record<string, { feddes: unknown; root: unknown; description: string }>> {
    const r = await fetch(`${RESEARCH_BASE}/research/dndc/crop-presets`);
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    return r.json();
  },
};

// ---- M2: PTF / soil library ----------------------------------------
export interface PTFRequest {
  sand_pct: number;
  silt_pct: number;
  clay_pct: number;
  bulk_density_g_cm3?: number;
  theta_33?: number;
  theta_1500?: number;
  organic_matter_pct?: number;
  organic_carbon_pct?: number;
  topsoil?: boolean;
  method?: "rosetta3_auto" | "carsel_parrish" | "wosten"
         | "rosetta3_h1" | "rosetta3_h2" | "rosetta3_h3" | "rosetta3_h4";
}

export interface PTFResult {
  theta_r: number; theta_s: number; alpha: number; n: number; Ks: number;
  L: number; method: string; covariance: number[][] | null;
}

export const ptf = {
  async predict(req: PTFRequest): Promise<PTFResult> {
    const r = await fetch(`${RESEARCH_BASE}/research/ptf/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(req),
    });
    if (!r.ok) throw new Error(`HTTP ${r.status}: ${await r.text()}`);
    return r.json();
  },
  async usdaClasses(): Promise<Record<string, { sand_pct: number; silt_pct: number; clay_pct: number }>> {
    const r = await fetch(`${RESEARCH_BASE}/research/ptf/usda-classes`);
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    return r.json();
  },
};

// ---- M3: Batch sweep REST endpoints ---------------------------------
export interface BatchParamSpec {
  name: string;
  target: string;
  bounds: [number, number];
  transform?: "linear" | "log" | "logit";
}

export interface BatchObsSpec {
  name: string;
  kind: "theta" | "h" | "c" | "flux" | "cumulative_flux" | "concentration_flux";
  location: Record<string, number | number[]>;
  time_day: number;
}

export interface BatchStartRequest {
  scenario_dir: string;
  params: BatchParamSpec[];
  obs: BatchObsSpec[];
  n: number;
  sampler: "lhs" | "grid" | "uniform";
  workers: number;
  seed?: number;
}

export interface BatchStatus {
  state: "pending" | "running" | "done" | "failed";
  n_total: number;
  n_done: number;
  out_path: string;
  error: string | null;
  n_failed?: number;
}

export const batch = {
  async start(req: BatchStartRequest): Promise<{ job_id: string }> {
    const r = await fetch(`${RESEARCH_BASE}/research/batch/start`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(req),
    });
    if (!r.ok) throw new Error(`HTTP ${r.status}: ${await r.text()}`);
    return r.json();
  },
  async status(jobId: string): Promise<BatchStatus> {
    const r = await fetch(`${RESEARCH_BASE}/research/batch/${jobId}/status`);
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    return r.json();
  },
  resultUrl(jobId: string): string {
    return `${RESEARCH_BASE}/research/batch/${jobId}/result`;
  },
};
