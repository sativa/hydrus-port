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

export type OutputFile = {
  name: string;
  path: string;
  size: number;
};

export type LogLine = { stream: "stdout" | "stderr"; text: string };

export const api = {
  listScenarios: () => invoke<Scenario[]>("list_scenarios"),
  detectPython: () => invoke<PythonInfo>("detect_python"),

  start: (args: {
    kind: string;
    input_dir: string;
    output_dir?: string;
    extra_args?: string[];
  }) => invoke<JobMeta>("start_simulation", { args }),

  stop: (id: string) => invoke<void>("stop_simulation", { id }),
  listJobs: () => invoke<JobMeta[]>("list_jobs"),
  getJob: (id: string) => invoke<JobMeta | null>("get_job", { id }),

  listOutputFiles: (dir: string) =>
    invoke<OutputFile[]>("list_output_files", { dir }),
  readText: (path: string, max_bytes?: number) =>
    invoke<string>("read_output_text", { path, maxBytes: max_bytes }),
  parseTable: (path: string) =>
    invoke<Series>("parse_obs_node", { path }),
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
