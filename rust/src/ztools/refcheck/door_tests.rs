//! The door to the osaurus CLI, and the HTTP reads that replaced two of its verbs.
//!
//! The fault (ZT-4, 2026-11-12): the 30-minute `models-refcheck` ran `osaurus config
//! export` and `osaurus list`; that CLI probes the server for 0.5 s, and when the
//! server was busy it decided the app was not running and opened a SECOND Osaurus.
//! A stub `osaurus` on the PATH records every call, so "never ran it" is a log that
//! does not exist.

use super::tests::{BROKEN, CRACK, export_with};
use super::tests_live::env_in;
use super::*;
use crate::test_env::TestEnv;
use serial_test::serial;
use std::io::{Read, Write};
use std::path::Path;
use std::sync::{Arc, Mutex};

/// A stub Osaurus: answers `/health`, `/admin/config/export` and `/v1/models`
/// as the real server does, and records every request line it was sent.
struct Server {
    url: String,
    seen: Arc<Mutex<Vec<String>>>,
    /// The request lines that arrived while no session held the GPU lock.
    unlocked: Arc<Mutex<Vec<String>>>,
}

fn http(status: &str, body: &str) -> String {
    format!(
        "HTTP/1.1 {status}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    )
}

fn server(export_text: &str, health: &'static str) -> Server {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    let seen = Arc::new(Mutex::new(Vec::new()));
    let log = Arc::clone(&seen);
    let unlocked = Arc::new(Mutex::new(Vec::new()));
    let unlocked_log = Arc::clone(&unlocked);
    let lock_dir = crate::ztools::eval::gpu_lock::lock_dir();
    // The real export wraps the document, as text, in {"json": "<document>"}.
    let wrapped = serde_json::json!({ "json": export_text }).to_string();
    let models =
        r#"{"object":"list","data":[{"id":"foundation"},{"id":"qwen3.8-27b-jang_6d-crack"}]}"#;
    std::thread::spawn(move || {
        for mut stream in listener.incoming().flatten() {
            let mut buf = [0u8; 4096];
            let n = stream.read(&mut buf).unwrap_or(0);
            let head = String::from_utf8_lossy(&buf[..n]).into_owned();
            let line = head.lines().next().unwrap_or("").to_string();
            let reply = if line.starts_with("GET /health") {
                http(health, "{}")
            } else if line.starts_with("GET /admin/config/export?format=json") {
                http("200 OK", &wrapped)
            } else if line.starts_with("GET /v1/models") {
                http("200 OK", models)
            } else {
                http("404 Not Found", "{}")
            };
            if !lock_dir.join("owner").exists() {
                unlocked_log.lock().unwrap().push(line.clone());
            }
            log.lock().unwrap().push(line);
            let _ = stream.write_all(reply.as_bytes());
        }
    });
    Server {
        url,
        seen,
        unlocked,
    }
}

fn argv_log(dir: &Path) -> Option<String> {
    std::fs::read_to_string(dir.join("argv.log")).ok()
}

fn install_crack(dir: &Path) {
    let model = dir.join("MLXModels/dealignai/Qwen3.8-27B-JANG_6D-CRACK");
    std::fs::create_dir_all(&model).unwrap();
    std::fs::write(model.join("config.json"), "{}").unwrap();
}

#[test]
#[serial]
fn a_normal_refcheck_never_runs_the_osaurus_cli() {
    let _env = TestEnv::new();
    let dir = tempfile::tempdir().unwrap();
    install_crack(dir.path());
    let srv = server(&export_with(CRACK).to_string(), "200 OK");
    let mut env = env_in(dir.path(), "http://127.0.0.1:1");
    env.base_url = srv.url.clone();
    let report = run(&env, true).unwrap();
    assert_eq!(report.exit_code(), 0, "{:?}", report.lines);
    assert_eq!(argv_log(dir.path()), None, "the stub osaurus was run");
    let seen = srv.seen.lock().unwrap().join("\n");
    assert!(
        seen.contains("GET /admin/config/export?format=json"),
        "{seen}"
    );
    assert!(seen.contains("GET /v1/models"), "{seen}");
}

#[test]
#[serial]
fn export_and_list_read_the_server_and_say_so_when_it_is_down() {
    let _env = TestEnv::new();
    let dir = tempfile::tempdir().unwrap();
    let srv = server(&export_with(BROKEN).to_string(), "200 OK");
    let mut env = env_in(dir.path(), "http://127.0.0.1:1");
    env.base_url = srv.url;
    assert_eq!(env.export().unwrap()["agents"][2]["model"], BROKEN);
    assert_eq!(
        env.osaurus_list().unwrap(),
        vec!["foundation", "qwen3.8-27b-jang_6d-crack"]
    );
    env.base_url = "http://127.0.0.1:1".into();
    let err = format!("{:#}", env.export().unwrap_err());
    assert!(err.contains("/admin/config/export"), "{err}");
    assert!(env.osaurus_list().is_err());
    assert_eq!(argv_log(dir.path()), None);
}

#[test]
#[serial]
fn an_export_that_is_not_json_is_an_error_not_a_pass() {
    let _env = TestEnv::new();
    let dir = tempfile::tempdir().unwrap();
    let bad = server("not json", "200 OK");
    let mut env = env_in(dir.path(), "http://127.0.0.1:1");
    env.base_url = bad.url;
    assert!(env.export().unwrap_err().to_string().contains("not JSON"));
}

#[test]
#[serial]
fn plan_and_apply_go_through_the_cli_only_after_health_answers() {
    let _env = TestEnv::new();
    let dir = tempfile::tempdir().unwrap();
    let srv = server(&export_with(BROKEN).to_string(), "200 OK");
    let mut env = env_in(dir.path(), "http://127.0.0.1:1");
    env.base_url = srv.url.clone();
    let doc = serde_json::json!({"version": 1, "agents": [{"name": "Disk Audit", "model": CRACK}]});
    assert_eq!(env.plan(&doc).unwrap().code, 0);
    assert_eq!(env.apply(&doc).unwrap().text.trim(), "agents:");
    let log = argv_log(dir.path()).unwrap();
    let lines: Vec<&str> = log.lines().collect();
    assert_eq!(lines.len(), 2, "{log}");
    assert!(
        lines[0].starts_with("config plan ") && lines[0].contains(".json"),
        "{log}"
    );
    assert!(
        lines[1].starts_with("config apply ") && lines[1].contains(".json"),
        "{log}"
    );
    assert!(!log.contains("--prune") && !log.contains("--yes"), "{log}");
    // The probe came before each CLI call, and the document is the minimal one.
    let seen = srv.seen.lock().unwrap();
    assert_eq!(
        seen.iter().filter(|l| l.starts_with("GET /health")).count(),
        2
    );
    let sent: Value =
        serde_json::from_str(&std::fs::read_to_string(dir.path().join("last-apply.json")).unwrap())
            .unwrap();
    assert_eq!(sent, doc);
}

#[test]
#[serial]
fn the_door_refuses_with_a_reason_when_health_does_not_answer() {
    let _env = TestEnv::new();
    let doc = serde_json::json!({"version": 1, "agents": []});
    // Nothing listening, and a server that answers 503: neither may reach the CLI.
    let down = tempfile::tempdir().unwrap();
    let mut env = env_in(down.path(), "http://127.0.0.1:1");
    for verb in ["plan", "apply"] {
        let r = if verb == "plan" {
            env.plan(&doc)
        } else {
            env.apply(&doc)
        };
        let err = format!("{:#}", r.unwrap_err());
        assert!(err.contains("/health") && err.contains("second"), "{err}");
    }
    let busy = server(&export_with(BROKEN).to_string(), "503 Service Unavailable");
    env.base_url = busy.url;
    let err = format!("{:#}", env.plan(&doc).unwrap_err());
    assert!(err.contains("503"), "{err}");
    assert_eq!(
        argv_log(down.path()),
        None,
        "the CLI was run past a dead /health"
    );
}

#[test]
#[serial]
fn a_missing_osaurus_binary_is_an_error_even_when_the_server_is_healthy() {
    let _env = TestEnv::new();
    let dir = tempfile::tempdir().unwrap();
    let srv = server(&export_with(BROKEN).to_string(), "200 OK");
    let mut env = env_in(dir.path(), "http://127.0.0.1:1");
    env.base_url = srv.url;
    env.osaurus = dir.path().join("no-such-osaurus");
    let doc = serde_json::json!({"version": 1});
    assert!(format!("{:#}", env.plan(&doc).unwrap_err()).contains("cannot run"));
}

#[test]
#[serial]
fn every_request_the_door_makes_is_made_under_the_gpu_lock() {
    let _env = TestEnv::new();
    let dir = tempfile::tempdir().unwrap();
    let srv = server(&export_with(CRACK).to_string(), "200 OK");
    let mut env = env_in(dir.path(), "http://127.0.0.1:1");
    env.base_url = srv.url.clone();
    env.export().unwrap();
    env.osaurus_list().unwrap();
    env.plan(&serde_json::json!({"version": 1})).unwrap();
    assert_eq!(srv.seen.lock().unwrap().len(), 3);
    assert_eq!(*srv.unlocked.lock().unwrap(), Vec::<String>::new());
}
