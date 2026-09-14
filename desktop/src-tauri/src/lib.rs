// Prevents additional console window on Windows in release, DO NOT REMOVE!!
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::process::{Child, Command};
use std::sync::Mutex;

/// The Python `ragdesk serve` child process, killed when the app exits.
struct ServerChild(Mutex<Option<Child>>);

fn spawn_server() -> Option<Child> {
    // Global flags first, then the subcommand: `ragdesk --db <path> serve ...`
    let mut base: Vec<String> = Vec::new();
    if let Ok(home) = std::env::var("HOME") {
        // Desktop indexes live in the home directory, not the app's cwd.
        let db = std::env::var("RAGDESK_DB").unwrap_or_else(|_| format!("{home}/.ragdesk/index.db"));
        base.push("--db".to_string());
        base.push(db);
    }
    base.push("serve".to_string());
    base.push("--port".to_string());
    base.push("8765".to_string());
    // Optional override for machines where the preset model is not pulled yet.
    if let Ok(model) = std::env::var("RAGDESK_LLM_MODEL") {
        base.push("--llm-model".to_string());
        base.push(model);
    }
    let mut candidates: Vec<(String, Vec<String>)> = Vec::new();

    if let Ok(bin) = std::env::var("RAGDESK_BIN") {
        candidates.push((bin, base.clone()));
    }
    candidates.push(("ragdesk".to_string(), base.clone()));
    if let Ok(home) = std::env::var("HOME") {
        candidates.push((format!("{home}/.local/bin/ragdesk"), base.clone()));
    }
    // Dev mode: run the checkout via uv (cwd is desktop/ during `tauri dev`).
    let project = std::env::var("RAGDESK_PROJECT").unwrap_or_else(|_| "..".to_string());
    candidates.push((
        "uv".to_string(),
        std::iter::once("run".to_string())
            .chain(std::iter::once("--project".to_string()))
            .chain(std::iter::once(project))
            .chain(std::iter::once("ragdesk".to_string()))
            .chain(base.iter().cloned())
            .collect(),
    ));

    for (program, program_args) in candidates {
        match Command::new(&program).args(&program_args).spawn() {
            Ok(child) => return Some(child),
            Err(_) => continue,
        }
    }
    None
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .setup(|app| {
            use tauri::Manager;
            let child = spawn_server();
            app.manage(ServerChild(Mutex::new(child)));
            Ok(())
        })
        .build(tauri::generate_context!())
        .expect("error while building tauri application")
        .run(|app_handle, event| {
            if let tauri::RunEvent::Exit = event {
                use tauri::Manager;
                if let Some(state) = app_handle.try_state::<ServerChild>() {
                    if let Some(mut child) = state.0.lock().unwrap().take() {
                        let _ = child.kill();
                    }
                }
            }
        });
}
