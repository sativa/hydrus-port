// Tauri 2 entry. Registers all commands and starts the event loop.

mod commands;
mod jobs;
mod scenarios;
mod parse;

use jobs::JobRegistry;

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .manage(JobRegistry::default())
        .invoke_handler(tauri::generate_handler![
            commands::list_scenarios,
            commands::debug_log,
            commands::start_test,
            commands::read_scenario,
            commands::write_scenario,
            commands::start_simulation,
            commands::stop_simulation,
            commands::list_jobs,
            commands::get_job,
            commands::list_output_files,
            commands::read_output_text,
            commands::read_output_bytes,
            commands::parse_obs_node,
            commands::parse_node_inf,
            commands::parse_nod_inf_series,
            commands::parse_swms2d_grid,
            commands::parse_swms2d_field,
            commands::list_vtu_series,
            commands::detect_python,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
