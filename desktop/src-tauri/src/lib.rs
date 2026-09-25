// Prevents additional console window on Windows in release, DO NOT REMOVE!!
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::fs::OpenOptions;
use std::net::{Ipv4Addr, SocketAddr, TcpStream};
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

const PORT: u16 = 8765;

/// The Python `ragdesk serve` child process, killed when the app exits.
struct ServerChild(Arc<Mutex<Option<Child>>>);

fn port_open() -> bool {
    let addr = SocketAddr::from((Ipv4Addr::LOCALHOST, PORT));
    TcpStream::connect_timeout(&addr, Duration::from_millis(300)).is_ok()
}

/// Append the child's output to ~/.ragdesk/serve.log so crashes leave a trace.
fn server_log() -> Stdio {
    let Ok(home) = std::env::var("HOME") else {
        return Stdio::null();
    };
    let dir = format!("{home}/.ragdesk");
    let _ = std::fs::create_dir_all(&dir);
    OpenOptions::new()
        .create(true)
        .append(true)
        .open(format!("{dir}/serve.log"))
        .map(Stdio::from)
        .unwrap_or_else(|_| Stdio::null())
}

fn spawn_server(resources: Option<PathBuf>) -> Option<Child> {
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
    base.push(PORT.to_string());
    // The server exits on its own if this process dies (e.g. SIGTERM quit),
    // so a stale server never keeps the port from the next launch.
    base.push("--watch-parent".to_string());
    // Optional override for machines where the preset model is not pulled yet.
    if let Ok(model) = std::env::var("RAGDESK_LLM_MODEL") {
        base.push("--llm-model".to_string());
        base.push(model);
    }
    let mut candidates: Vec<(String, Vec<String>)> = Vec::new();

    // Bundled runtime first: the DMG ships its own CPython + ragdesk, so the
    // app works on a machine that never ran `uv tool install`. Module
    // invocation on purpose: a console-script shebang would point at the
    // build machine's paths.
    if let Some(dir) = resources {
        let python = dir.join("python").join("bin").join("python3");
        if python.exists() {
            let args = std::iter::once("-m".to_string())
                .chain(std::iter::once("ragdesk".to_string()))
                .chain(base.iter().cloned())
                .collect();
            candidates.push((python.to_string_lossy().into_owned(), args));
        }
    }
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
        match Command::new(&program)
            .args(&program_args)
            // A signed bundle must never write __pycache__ at runtime: the
            // sidecar precompiles at build time, and this keeps a writable
            // copy from growing or changing after signing.
            .env("PYTHONDONTWRITEBYTECODE", "1")
            .stdout(server_log())
            .stderr(server_log())
            .spawn()
        {
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
        .plugin(tauri_plugin_dialog::init())
        .setup(|app| {
            use tauri::Manager;
            let resources = app.path().resource_dir().ok();
            let shared = Arc::new(Mutex::new(spawn_server(resources.clone())));
            app.manage(ServerChild(shared.clone()));
            // Watchdog: if the Python server dies, bring it back (unless another
            // instance already owns the port), so the UI never talks to a corpse.
            thread::spawn(move || loop {
                thread::sleep(Duration::from_secs(3));
                let exited = {
                    let Ok(mut guard) = shared.lock() else {
                        break;
                    };
                    match guard.as_mut() {
                        None => break,
                        Some(child) => child.try_wait().ok().flatten().is_some(),
                    }
                };
                if exited && !port_open() {
                    let fresh = spawn_server(resources.clone());
                    if let Ok(mut guard) = shared.lock() {
                        *guard = fresh;
                    }
                }
            });
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
