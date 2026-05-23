// Parse HYDRUS-1D / SWMS_2D ASCII output files into JSON for the
// frontend plots. These formats are fixed-width whitespace tables with
// a small header — we tolerate variations by skipping any line that
// does not start with a number (after optional whitespace).

use serde::Serialize;
use std::collections::HashMap;
use std::fs;
use std::path::Path;

#[derive(Serialize)]
pub struct Series {
    pub headers: Vec<String>,   // column labels
    pub rows: Vec<Vec<f64>>,    // rows of numbers
}

pub fn read_numeric_table(path: &Path) -> Result<Series, String> {
    let text = fs::read_to_string(path).map_err(|e| e.to_string())?;
    let mut headers: Vec<String> = Vec::new();
    let mut rows: Vec<Vec<f64>> = Vec::new();
    let mut last_label_line: Option<Vec<String>> = None;
    for line in text.lines() {
        let s = line.trim();
        if s.is_empty() {
            continue;
        }
        let first = s.split_whitespace().next().unwrap_or("");
        if first.parse::<f64>().is_ok() {
            // numeric row
            let vals: Vec<f64> = s
                .split_whitespace()
                .filter_map(|t| t.parse::<f64>().ok())
                .collect();
            if !vals.is_empty() {
                rows.push(vals);
            }
        } else {
            // header / metadata; remember last alphabetic line as potential header
            let toks: Vec<String> = s
                .split_whitespace()
                .map(|t| t.to_string())
                .collect();
            // Skip lines that are just units like "[T] [L/T] [L]" —
            // they'd otherwise overwrite the real header line that
            // sits just above them in HYDRUS/SWMS .OUT files.
            let looks_like_units = toks.iter().all(|t|
                t.starts_with('[') && t.ends_with(']')
            );
            if toks.len() > 1
                && !toks.iter().any(|t| t.contains('='))
                && !looks_like_units {
                last_label_line = Some(toks);
            }
        }
    }
    if let Some(h) = last_label_line {
        headers = h;
    }
    // Pad headers if shorter than the widest row
    let max_cols = rows.iter().map(|r| r.len()).max().unwrap_or(0);
    while headers.len() < max_cols {
        headers.push(format!("col{}", headers.len() + 1));
    }
    Ok(Series { headers, rows })
}


// --------------------------------------------------------------------
// NOD_INF.OUT — multiple per-node snapshots over time. Each snapshot
// is preceded by a "Time: X.XXXX" header. Snapshots are normalised
// onto a (n_t × n_z) grid keyed by the union of depths seen.
// --------------------------------------------------------------------

#[derive(Serialize)]
pub struct NodInfSeries {
    pub times: Vec<f64>,                          // length n_t
    pub depths: Vec<f64>,                         // length n_z, sorted ascending
    pub vars: HashMap<String, Vec<Vec<f64>>>,     // var -> rows of length n_z, n_t rows
    pub var_names: Vec<String>,
}

pub fn parse_nod_inf(path: &Path) -> Result<NodInfSeries, String> {
    let text = fs::read_to_string(path).map_err(|e| e.to_string())?;

    // Column index map. HYDRUS-1D NOD_INF columns (in order):
    //   Node  Depth  Head  Moisture  K  C  Flux  Sink  Kappa  v/KsTop  Temp
    let col_names = [
        "Node", "Depth", "Head", "Moisture", "K", "C",
        "Flux", "Sink", "Kappa", "vOverKsTop", "Temp",
    ];

    // We collect per-snapshot rows then assemble.
    let mut snapshots: Vec<(f64, Vec<Vec<f64>>)> = Vec::new();
    let mut current_t: Option<f64> = None;
    let mut current_rows: Vec<Vec<f64>> = Vec::new();

    for line in text.lines() {
        let s = line.trim();
        if s.is_empty() {
            continue;
        }
        // Match "Time: X.XXXX"
        if let Some(rest) = s.strip_prefix("Time:") {
            if let Some(t_prev) = current_t.take() {
                snapshots.push((t_prev, std::mem::take(&mut current_rows)));
            }
            let tval: f64 = rest.trim().parse().unwrap_or(f64::NAN);
            current_t = Some(tval);
            continue;
        }
        if current_t.is_none() {
            continue;
        }
        // Numeric row?
        let first = s.split_whitespace().next().unwrap_or("");
        if first.parse::<f64>().is_err() && first.parse::<i64>().is_err() {
            continue;
        }
        let vals: Vec<f64> = s
            .split_whitespace()
            .filter_map(|t| t.parse::<f64>().ok())
            .collect();
        if vals.len() >= 5 {
            current_rows.push(vals);
        }
    }
    if let Some(t_prev) = current_t {
        snapshots.push((t_prev, current_rows));
    }
    if snapshots.is_empty() {
        return Err("no snapshots parsed".into());
    }

    // Assemble: use depths from first snapshot (HYDRUS-1D's mesh is
    // stationary). Sort by depth ascending for clean axes.
    let first = &snapshots[0].1;
    let mut depths: Vec<f64> = first.iter().map(|r| r[1]).collect();
    let mut idx_order: Vec<usize> = (0..depths.len()).collect();
    idx_order.sort_by(|a, b| depths[*a].partial_cmp(&depths[*b]).unwrap_or(std::cmp::Ordering::Equal));
    let depths_sorted: Vec<f64> = idx_order.iter().map(|&i| depths[i]).collect();
    depths = depths_sorted;

    let times: Vec<f64> = snapshots.iter().map(|(t, _)| *t).collect();

    let var_names: Vec<String> = col_names[2..]
        .iter()
        .map(|s| s.to_string())
        .collect();
    let mut vars: HashMap<String, Vec<Vec<f64>>> = HashMap::new();
    for name in &var_names {
        vars.insert(name.clone(), Vec::with_capacity(times.len()));
    }
    for (_t, rows) in &snapshots {
        for (col_idx, name) in var_names.iter().enumerate() {
            let actual_col = col_idx + 2; // skip Node, Depth
            let row_vals: Vec<f64> = idx_order
                .iter()
                .map(|&i| rows.get(i).and_then(|r| r.get(actual_col)).copied().unwrap_or(f64::NAN))
                .collect();
            vars.get_mut(name).unwrap().push(row_vals);
        }
    }

    Ok(NodInfSeries { times, depths, vars, var_names })
}
