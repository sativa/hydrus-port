// Parse HYDRUS-1D / SWMS_2D ASCII output files into JSON for the
// frontend plots. These formats are fixed-width whitespace tables with
// a small header — we tolerate variations by skipping any line that
// does not start with a number (after optional whitespace).

use serde::Serialize;
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
            if toks.len() > 1 && !toks.iter().any(|t| t.contains('=')) {
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
