// Tauri command handlers invoked from the Vue frontend.

use crate::jobs::{JobHandle, JobMeta, JobRegistry};
use crate::parse::{read_numeric_table, Series};
use crate::scenarios::{self, Scenario};
use serde::Serialize;
use std::path::PathBuf;
use std::process::Stdio;
use tauri::{AppHandle, Emitter, State};
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::Command;

fn now_ms() -> i64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0)
}

#[tauri::command]
pub fn list_scenarios() -> Vec<Scenario> {
    scenarios::list()
}

#[tauri::command]
pub fn debug_log(text: String) -> Result<(), String> {
    use std::io::Write;
    let path = std::env::var("HYDRUS_GUI_DEBUG_LOG")
        .unwrap_or_else(|_| "/tmp/gui_debug.log".to_string());
    let mut f = std::fs::OpenOptions::new()
        .create(true).append(true).open(&path)
        .map_err(|e| e.to_string())?;
    writeln!(f, "{}", text).map_err(|e| e.to_string())?;
    Ok(())
}

// --------------------------------------------------------------------
// Python detection
// --------------------------------------------------------------------

#[derive(Serialize)]
pub struct PythonInfo {
    pub executable: String,
    pub version: String,
    pub repo_root: String,
}

#[tauri::command]
pub async fn detect_python() -> Result<PythonInfo, String> {
    // Try the same exec the CLI scripts use ("python3" fallback to "python").
    for cand in ["python3", "python"] {
        if let Ok(out) = Command::new(cand).arg("--version").output().await {
            if out.status.success() {
                let v = String::from_utf8_lossy(if out.stdout.is_empty() {
                    &out.stderr
                } else {
                    &out.stdout
                })
                .trim()
                .to_string();
                return Ok(PythonInfo {
                    executable: cand.into(),
                    version: v,
                    repo_root: scenarios::repo_root().to_string_lossy().into_owned(),
                });
            }
        }
    }
    Err("No python3/python on PATH".into())
}

// --------------------------------------------------------------------
// Start / stop simulation
// --------------------------------------------------------------------

#[derive(serde::Deserialize)]
pub struct StartArgs {
    pub kind: String,        // "hydrus1d" | "swms2d" | "richards3d"
    pub input_dir: String,
    pub output_dir: Option<String>,
    pub extra_args: Option<Vec<String>>,
}

#[tauri::command]
pub async fn start_simulation(
    app: AppHandle,
    registry: State<'_, JobRegistry>,
    args: StartArgs,
) -> Result<JobMeta, String> {
    let id = uuid::Uuid::new_v4().simple().to_string()[..12].to_string();
    let input_dir = PathBuf::from(&args.input_dir);
    let output_dir = args
        .output_dir
        .map(PathBuf::from)
        .unwrap_or_else(|| input_dir.join("out"));
    std::fs::create_dir_all(&output_dir).map_err(|e| e.to_string())?;

    // All three sims dispatch through the unified `hydrus` CLI
    // (hydrus_port.cli) via `python -u -m hydrus_port.cli {1d|2d|3d}`.
    let subcmd = match args.kind.as_str() {
        "hydrus1d" | "1d" => "1d",
        "swms2d"   | "2d" => "2d",
        "richards3d" | "3d" => "3d",
        other => return Err(format!("Unknown kind: {other}")),
    };

    let py = which_python().ok_or("python not found")?;
    let mut cmd = Command::new(&py);
    cmd.arg("-u") // unbuffered stdout for live streaming
        .arg("-m")
        .arg("hydrus_port.cli")
        .arg(subcmd);
    if subcmd != "3d" {
        cmd.arg(&input_dir).arg("-o").arg(&output_dir);
    }
    if let Some(extra) = &args.extra_args {
        cmd.args(extra);
    }
    cmd.current_dir(scenarios::repo_root());
    cmd.stdout(Stdio::piped()).stderr(Stdio::piped());

    let mut child = cmd.spawn().map_err(|e| format!("spawn failed: {e}"))?;
    let stdout = child.stdout.take().ok_or("no stdout")?;
    let stderr = child.stderr.take().ok_or("no stderr")?;

    let meta = JobMeta {
        id: id.clone(),
        kind: args.kind.clone(),
        scenario: input_dir.to_string_lossy().into_owned(),
        input_dir: input_dir.to_string_lossy().into_owned(),
        output_dir: output_dir.to_string_lossy().into_owned(),
        status: "running".into(),
        exit_code: None,
        started_at_ms: now_ms(),
        finished_at_ms: None,
    };
    registry.insert(JobHandle {
        meta: meta.clone(),
        child: Some(child),
    });

    // Spawn forwarder tasks. Each line emits a `job://{id}/log` event.
    spawn_line_forwarder(app.clone(), id.clone(), "stdout".into(), stdout);
    spawn_line_forwarder(app.clone(), id.clone(), "stderr".into(), stderr);

    // Spawn waiter that updates registry status on exit.
    let app2 = app.clone();
    let id2 = id.clone();
    let reg_arc = registry.inner_arc();
    tokio::spawn(async move {
        let mut child = {
            let mut guard = reg_arc.lock();
            guard.get_mut(&id2).and_then(|h| h.child.take())
        };
        if let Some(ref mut c) = child {
            let status = c.wait().await;
            let (st, code) = match status {
                Ok(s) => (
                    if s.success() { "done".to_string() } else { "failed".to_string() },
                    s.code(),
                ),
                Err(e) => {
                    let _ = app2.emit(
                        &format!("job://{}/log", id2),
                        LogLine {
                            stream: "stderr".into(),
                            text: format!("[wait error] {e}"),
                        },
                    );
                    ("failed".into(), None)
                }
            };
            let mut guard = reg_arc.lock();
            if let Some(h) = guard.get_mut(&id2) {
                h.meta.status = st.clone();
                h.meta.exit_code = code;
                h.meta.finished_at_ms = Some(now_ms());
            }
            let _ = app2.emit(
                &format!("job://{}/status", id2),
                serde_json::json!({"status": st, "exit_code": code}),
            );
        }
    });

    Ok(meta)
}

#[tauri::command]
pub async fn stop_simulation(
    registry: State<'_, JobRegistry>,
    id: String,
) -> Result<(), String> {
    if let Some(mut c) = registry.take_child(&id) {
        let _ = c.kill().await;
        registry.update(&id, |m| {
            m.status = "cancelled".into();
            m.finished_at_ms = Some(now_ms());
        });
    }
    Ok(())
}

#[tauri::command]
pub fn list_jobs(registry: State<'_, JobRegistry>) -> Vec<JobMeta> {
    registry.list()
}

#[tauri::command]
pub fn get_job(registry: State<'_, JobRegistry>, id: String) -> Option<JobMeta> {
    registry.get(&id)
}

// --------------------------------------------------------------------
// Output browsing + parsing
// --------------------------------------------------------------------

#[derive(Serialize)]
pub struct OutputFile {
    pub name: String,
    pub path: String,
    pub size: u64,
}

#[tauri::command]
pub fn list_output_files(dir: String) -> Result<Vec<OutputFile>, String> {
    let mut out = Vec::new();
    for e in walkdir::WalkDir::new(&dir).max_depth(2) {
        let e = e.map_err(|err| err.to_string())?;
        if e.file_type().is_file() {
            let meta = e.metadata().map_err(|err| err.to_string())?;
            out.push(OutputFile {
                name: e.file_name().to_string_lossy().into_owned(),
                path: e.path().to_string_lossy().into_owned(),
                size: meta.len(),
            });
        }
    }
    out.sort_by(|a, b| a.name.cmp(&b.name));
    Ok(out)
}

#[tauri::command]
pub fn read_output_text(path: String, max_bytes: Option<usize>) -> Result<String, String> {
    let bytes = std::fs::read(&path).map_err(|e| e.to_string())?;
    let limit = max_bytes.unwrap_or(usize::MAX).min(bytes.len());
    Ok(String::from_utf8_lossy(&bytes[..limit]).into_owned())
}

#[tauri::command]
pub fn read_output_bytes(path: String) -> Result<Vec<u8>, String> {
    std::fs::read(&path).map_err(|e| e.to_string())
}

#[tauri::command]
pub fn parse_obs_node(path: String) -> Result<Series, String> {
    read_numeric_table(std::path::Path::new(&path))
}

#[tauri::command]
pub fn parse_node_inf(path: String) -> Result<Series, String> {
    read_numeric_table(std::path::Path::new(&path))
}

#[tauri::command]
pub fn list_vtu_series(dir: String) -> Result<Vec<String>, String> {
    let mut out = Vec::new();
    for e in walkdir::WalkDir::new(&dir).max_depth(3) {
        let e = e.map_err(|err| err.to_string())?;
        if e.file_type().is_file() {
            let name = e.file_name().to_string_lossy();
            if name.ends_with(".vtu") || name.ends_with(".pvd") {
                out.push(e.path().to_string_lossy().into_owned());
            }
        }
    }
    out.sort();
    Ok(out)
}

// --------------------------------------------------------------------
// helpers
// --------------------------------------------------------------------

#[derive(Clone, Serialize)]
struct LogLine {
    stream: String,
    text: String,
}

fn spawn_line_forwarder<R: tokio::io::AsyncRead + Unpin + Send + 'static>(
    app: AppHandle,
    id: String,
    stream: String,
    reader: R,
) {
    let topic = format!("job://{}/log", id);
    tokio::spawn(async move {
        let mut lines = BufReader::new(reader).lines();
        while let Ok(Some(line)) = lines.next_line().await {
            let _ = app.emit(
                &topic,
                LogLine {
                    stream: stream.clone(),
                    text: line,
                },
            );
        }
    });
}

fn which_python() -> Option<String> {
    use std::process::Command as StdCommand;
    for cand in ["python3", "python"] {
        if let Ok(out) = StdCommand::new(cand).arg("--version").output() {
            if out.status.success() {
                return Some(cand.into());
            }
        }
    }
    None
}

