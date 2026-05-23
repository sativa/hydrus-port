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
            let kind = classify(&name, &p);
            out.push(Scenario {
                name: name.clone(),
                kind,
                path: p.to_string_lossy().to_string(),
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

fn classify(name: &str, p: &Path) -> String {
    let n = name.to_lowercase();
    if n.contains("3d") {
        return "richards3d".into();
    }
    // hydrus1d fixtures typically have SELECTOR.IN / PROFILE.DAT
    if p.join("SELECTOR.IN").exists() || p.join("PROFILE.DAT").exists() {
        return "hydrus1d".into();
    }
    // swms2d fixtures have SWMS_2D.IN / GRID.IN
    if p.join("SWMS_2D.IN").exists() || p.join("GRID.IN").exists() {
        return "swms2d".into();
    }
    "unknown".into()
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
