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

// M2: PTF REST wrappers — calls the FastAPI backend at /research/ptf/*

export interface PTFRequest {                                           // M2:
  sand_pct: number;                                                    // M2:
  silt_pct: number;                                                    // M2:
  clay_pct: number;                                                    // M2:
  bulk_density_g_cm3?: number;                                         // M2:
  theta_33?: number;                                                   // M2:
  theta_1500?: number;                                                 // M2:
  organic_matter_pct?: number;                                         // M2:
  organic_carbon_pct?: number;                                         // M2:
  topsoil?: boolean;                                                   // M2:
  method?: "rosetta3_auto" | "carsel_parrish" | "wosten"              // M2:
         | "rosetta3_h1" | "rosetta3_h2" | "rosetta3_h3" | "rosetta3_h4";  // M2:
}                                                                      // M2:

export interface PTFResult {                                           // M2:
  theta_r: number; theta_s: number; alpha: number; n: number; Ks: number;  // M2:
  L: number; method: string; covariance: number[][] | null;           // M2:
}                                                                      // M2:

const PTF_BASE = (import.meta.env.VITE_DNDC_BASE as string)           // M2:
                 ?? "http://127.0.0.1:8765";                           // M2:

export const ptf = {                                                   // M2:
  async predict(req: PTFRequest): Promise<PTFResult> {                 // M2:
    const r = await fetch(`${PTF_BASE}/research/ptf/predict`, {        // M2:
      method: "POST",                                                  // M2:
      headers: { "Content-Type": "application/json" },                // M2:
      body: JSON.stringify(req),                                       // M2:
    });                                                                // M2:
    if (!r.ok) throw new Error(`HTTP ${r.status}: ${await r.text()}`);  // M2:
    return r.json();                                                   // M2:
  },                                                                   // M2:
  async usdaClasses(): Promise<Record<string, { sand_pct: number; silt_pct: number; clay_pct: number }>> {  // M2:
    const r = await fetch(`${PTF_BASE}/research/ptf/usda-classes`);   // M2:
    if (!r.ok) throw new Error(`HTTP ${r.status}`);                   // M2:
    return r.json();                                                   // M2:
  },                                                                   // M2:
};                                                                     // M2:
