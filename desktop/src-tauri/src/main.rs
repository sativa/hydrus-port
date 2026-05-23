// Prevents a separate console window on Windows release builds.
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

fn main() {
    hydrus_port_desktop_lib::run();
}
