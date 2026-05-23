// In-memory registry of running / finished simulation subprocesses.
// Stdout/stderr are forwarded as Tauri events `job://{id}/log` so the
// frontend can render a live console.

use parking_lot::Mutex;
use serde::Serialize;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::process::Child;

#[derive(Default)]
pub struct JobRegistry {
    inner: Arc<Mutex<HashMap<String, JobHandle>>>,
}

pub struct JobHandle {
    pub meta: JobMeta,
    pub child: Option<Child>,
}

#[derive(Clone, Serialize)]
pub struct JobMeta {
    pub id: String,
    pub kind: String,        // "hydrus1d" | "swms2d" | "richards3d"
    pub scenario: String,    // name or path
    pub input_dir: String,
    pub output_dir: String,
    pub status: String,      // "running" | "done" | "failed" | "cancelled"
    pub exit_code: Option<i32>,
    pub started_at_ms: i64,
    pub finished_at_ms: Option<i64>,
}

impl JobRegistry {
    pub fn insert(&self, h: JobHandle) {
        self.inner.lock().insert(h.meta.id.clone(), h);
    }

    pub fn list(&self) -> Vec<JobMeta> {
        self.inner.lock().values().map(|h| h.meta.clone()).collect()
    }

    pub fn get(&self, id: &str) -> Option<JobMeta> {
        self.inner.lock().get(id).map(|h| h.meta.clone())
    }

    pub fn update<F: FnOnce(&mut JobMeta)>(&self, id: &str, f: F) {
        if let Some(h) = self.inner.lock().get_mut(id) {
            f(&mut h.meta);
        }
    }

    pub fn take_child(&self, id: &str) -> Option<Child> {
        self.inner.lock().get_mut(id).and_then(|h| h.child.take())
    }

    pub fn inner_arc(&self) -> Arc<Mutex<HashMap<String, JobHandle>>> {
        Arc::clone(&self.inner)
    }
}
