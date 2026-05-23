// Locate the repo root (containing pyproject.toml + tests/fixtures)
// and enumerate example scenarios. The GUI assumes the desktop crate
// lives at <repo>/desktop/, so the parent of CARGO_MANIFEST_DIR is the
// repo root.

use serde::Serialize;
use std::path::{Path, PathBuf};

pub fn repo_root() -> PathBuf {
    // CARGO_MANIFEST_DIR is set at build time. For runtime fallback,
    // we also accept HYDRUS_PORT_ROOT env var.
    if let Ok(r) = std::env::var("HYDRUS_PORT_ROOT") {
        return PathBuf::from(r);
    }
    let manifest = env!("CARGO_MANIFEST_DIR"); // .../desktop/src-tauri
    Path::new(manifest)
        .parent() // .../desktop
        .and_then(|p| p.parent()) // repo root
        .map(|p| p.to_path_buf())
        .unwrap_or_else(|| PathBuf::from("."))
}

#[derive(Serialize)]
pub struct Scenario {
    pub name: String,
    pub kind: String,    // "hydrus1d" | "swms2d" | "richards3d"
    pub path: String,
    pub description: String,
}

pub fn list() -> Vec<Scenario> {
    let mut out = Vec::new();
    let root = repo_root();
    let fixtures = root.join("tests").join("fixtures");
    if let Ok(rd) = std::fs::read_dir(&fixtures) {
        for entry in rd.flatten() {
            let p = entry.path();
            if !p.is_dir() {
                continue;
            }
            let name = entry.file_name().to_string_lossy().to_string();
            let (kind, input_path) = classify(&name, &p);
            out.push(Scenario {
                name: name.clone(),
                kind,
                // path points at the actual directory the CLI should take
                // as its --input-dir (which may be `<scenario>/inputs`).
                path: input_path.to_string_lossy().into_owned(),
                description: describe(&name),
            });
        }
    }
    // Always include the 3D synthetic column scenario
    out.push(Scenario {
        name: "richards3d_box_column".into(),
        kind: "richards3d".into(),
        path: root.join("tests").join("validate_richards3d.py")
            .to_string_lossy().into_owned(),
        description: "Synthetic 3D infiltration on a tensor-product box".into(),
    });
    out.sort_by(|a, b| a.name.cmp(&b.name));
    out
}

// Returns (kind, input_dir_to_pass_to_CLI).
fn classify(name: &str, p: &Path) -> (String, PathBuf) {
    let n = name.to_lowercase();
    if n.contains("3d") {
        // 3D fixtures: we still record the inputs path for completeness
        let inputs = p.join("inputs");
        return ("richards3d".into(),
                if inputs.is_dir() { inputs } else { p.to_path_buf() });
    }
    // Both HYDRUS-1D and SWMS_2D use Selector.in, so we can't classify
    // by that alone. Distinguish via the unique markers:
    //   SWMS_2D  → GRID.IN  (FE mesh; HYDRUS-1D has no mesh file)
    //   HYDRUS-1D → Profile.dat
    // Files may live at the scenario root or under `inputs/`.
    let candidates = [p.to_path_buf(), p.join("inputs")];
    for c in &candidates {
        if !c.is_dir() { continue; }
        if has_any_file_case_insensitive(c, &["grid.in"]) {
            return ("swms2d".into(), c.clone());
        }
        if has_any_file_case_insensitive(c, &["profile.dat"]) {
            return ("hydrus1d".into(), c.clone());
        }
    }
    ("unknown".into(), p.to_path_buf())
}

fn has_any_file_case_insensitive(dir: &Path, needles: &[&str]) -> bool {
    if let Ok(rd) = std::fs::read_dir(dir) {
        for e in rd.flatten() {
            let n = e.file_name().to_string_lossy().to_lowercase();
            if needles.iter().any(|m| n == *m) {
                return true;
            }
        }
    }
    false
}

fn describe(name: &str) -> String {
    match name {
        "EX1" | "EX.1" => "SWMS_2D Example 1 — column drainage".into(),
        "EX2" | "EX.2" => "SWMS_2D Example 2 — dry-spell redistribution".into(),
        "EX3" | "EX.3" => "SWMS_2D Example 3 — furrow irrigation".into(),
        "EX4" | "EX.4" => "SWMS_2D Example 4 — solute transport".into(),
        _ => String::new(),
    }
}
