//! `LiveEnv` against a stub `osaurus` script, a fixture `~/MLXModels` and a
//! stub Hugging Face server: the real argv, the real files, no real machine.

use super::tests::{BROKEN, CRACK, export_with};
use super::*;
use crate::ztools::eval::report::HistoryEntry;
use std::io::{Read, Write};
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};

/// A stub that logs its argv, copies a fixture for `export`, and prints
/// canned text for `list`, `plan` and `apply`.
fn stub_osaurus(dir: &Path, export: &Value) -> PathBuf {
    std::fs::write(dir.join("export.json"), export.to_string()).unwrap();
    let script = format!(
        r#"#!/bin/sh
echo "$@" >> "{d}/argv.log"
case "$2" in
  export) while [ $# -gt 0 ]; do [ "$1" = "-o" ] && cp "{d}/export.json" "$2"; shift; done ;;
  plan|apply) cp "$3" "{d}/last-$2.json"; echo "agents:"; exit "${{STUB_EXIT:-0}}" ;;
esac
[ "$1" = list ] && printf 'foundation\nqwen3.8-27b-jang_6d-crack\n\n'
exit 0
"#,
        d = dir.display()
    );
    let path = dir.join("osaurus");
    std::fs::write(&path, script).unwrap();
    std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755)).unwrap();
    path
}

fn env_in(dir: &Path, hf_base: &str) -> LiveEnv {
    LiveEnv {
        osaurus: stub_osaurus(dir, &export_with(BROKEN)),
        models_root: dir.join("MLXModels"),
        families: dir.join("state/models/families.json"),
        hf_base: hf_base.to_string(),
        best: BTreeMap::new(),
        scores: BTreeMap::new(),
    }
}

fn hf_stub(body: &'static str, status: &'static str) -> String {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    std::thread::spawn(move || {
        for stream in listener.incoming().flatten().take(8) {
            let mut stream = stream;
            let mut buf = [0u8; 2048];
            let _ = stream.read(&mut buf);
            let resp = format!(
                "HTTP/1.1 {status}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                body.len()
            );
            let _ = stream.write_all(resp.as_bytes());
        }
    });
    format!("http://{addr}")
}

#[test]
fn export_list_plan_and_apply_use_the_real_argv_and_never_prune() {
    let dir = tempfile::tempdir().unwrap();
    let env = env_in(dir.path(), "http://127.0.0.1:1");
    assert_eq!(env.export().unwrap()["agents"][2]["model"], BROKEN);
    assert_eq!(
        env.osaurus_list().unwrap(),
        vec!["foundation", "qwen3.8-27b-jang_6d-crack"]
    );
    let doc = serde_json::json!({"version": 1, "agents": [{"name": "Disk Audit", "model": CRACK}]});
    assert_eq!(env.plan(&doc).unwrap().code, 0);
    assert_eq!(env.apply(&doc).unwrap().text.trim(), "agents:");
    let log = std::fs::read_to_string(dir.path().join("argv.log")).unwrap();
    let lines: Vec<&str> = log.lines().collect();
    assert!(
        lines[0].starts_with("config export --format json -o "),
        "{log}"
    );
    assert_eq!(lines[1], "list");
    assert!(
        lines[2].starts_with("config plan ") && lines[2].contains(".json"),
        "{log}"
    );
    assert!(
        lines[3].starts_with("config apply ") && lines[3].contains(".json"),
        "{log}"
    );
    assert!(!log.contains("--prune") && !log.contains("--yes"), "{log}");
    // The document osaurus was handed is the minimal one, byte for byte.
    let sent: Value =
        serde_json::from_str(&std::fs::read_to_string(dir.path().join("last-apply.json")).unwrap())
            .unwrap();
    assert_eq!(sent, doc);
}

#[test]
fn a_missing_osaurus_or_a_failed_export_is_an_error_not_a_pass() {
    let dir = tempfile::tempdir().unwrap();
    let mut env = env_in(dir.path(), "http://127.0.0.1:1");
    env.osaurus = dir.path().join("no-such-osaurus");
    assert!(env.export().is_err());
    assert!(env.osaurus_list().is_err());
    let dir2 = tempfile::tempdir().unwrap();
    let mut env = env_in(dir2.path(), "http://127.0.0.1:1");
    std::fs::write(dir2.path().join("export.json"), "not json").unwrap();
    assert!(env.export().unwrap_err().to_string().contains("not JSON"));
    // A non-zero export exit is reported with its code.
    env.osaurus = {
        let p = dir2.path().join("failing");
        std::fs::write(&p, "#!/bin/sh\nexit 2\n").unwrap();
        std::fs::set_permissions(&p, std::fs::Permissions::from_mode(0o755)).unwrap();
        p
    };
    assert!(env.export().unwrap_err().to_string().contains("exited 2"));
    assert!(env.osaurus_list().is_err());
}

#[test]
fn disk_models_lists_org_name_pairs_that_have_a_config_json() {
    let dir = tempfile::tempdir().unwrap();
    let env = env_in(dir.path(), "http://127.0.0.1:1");
    let root = dir.path().join("MLXModels");
    for (org, name, config) in [
        ("dealignai", "Qwen3.8-27B-JANG_6D-CRACK", true),
        ("mlx-community", "Half-Downloaded", false),
    ] {
        std::fs::create_dir_all(root.join(org).join(name)).unwrap();
        if config {
            std::fs::write(root.join(org).join(name).join("config.json"), "{}").unwrap();
        }
    }
    std::fs::create_dir_all(root.join("blobs")).unwrap();
    std::fs::write(root.join("stray.txt"), "x").unwrap();
    assert_eq!(env.disk_models(), vec![CRACK.to_string()]);
    assert_empty!(
        LiveEnv {
            models_root: dir.path().join("absent"),
            ..env
        }
        .disk_models()
    );
}

#[test]
fn the_base_model_comes_from_the_cache_then_from_hugging_face_and_is_cached() {
    let dir = tempfile::tempdir().unwrap();
    let hf = hf_stub(
        r#"{"tags":["safetensors","base_model:quantized:Qwen/Qwen3.8-27B"]}"#,
        "200 OK",
    );
    let env = env_in(dir.path(), &hf);
    assert_eq!(env.base_model(BROKEN).as_deref(), Some("Qwen/Qwen3.8-27B"));
    let cache = std::fs::read_to_string(&env.families).unwrap();
    assert!(
        cache.contains(BROKEN) && cache.contains("Qwen/Qwen3.8-27B"),
        "{cache}"
    );
    // Second look: served from the cache with HF unreachable.
    let offline = LiveEnv {
        hf_base: "http://127.0.0.1:1".into(),
        ..env
    };
    assert_eq!(
        offline.base_model(BROKEN).as_deref(),
        Some("Qwen/Qwen3.8-27B")
    );
    assert_eq!(offline.base_model("other/Unknown-1B"), None);
}

#[test]
fn hugging_face_failures_and_untagged_models_have_no_base_model() {
    let dir = tempfile::tempdir().unwrap();
    let not_found = hf_stub("{}", "404 Not Found");
    assert_eq!(env_in(dir.path(), &not_found).base_model(BROKEN), None);
    let untagged = hf_stub(r#"{"tags":["safetensors"]}"#, "200 OK");
    assert_eq!(env_in(dir.path(), &untagged).base_model(BROKEN), None);
    assert!(
        !dir.path().join("state").exists(),
        "a miss must not write a cache"
    );
}

fn row(task: &str, score: i64, ts: f64, complete: bool) -> HistoryEntry {
    HistoryEntry {
        date: "2026-10-09".into(),
        timestamp: ts,
        task: task.into(),
        score,
        time: None,
        complete,
        fingerprint: None,
    }
}

#[test]
fn think_scores_are_the_mean_of_the_latest_complete_run_with_think_rows() {
    let history = BTreeMap::from([
        (
            "Qwen3.8-27B-JANG_6D-CRACK".to_string(),
            vec![
                row("taxes_qa", 100, 1.0, true),
                row("taxes_qa", 80, 100.0, true),
                row("file_summary_a", 70, 100.5, true),
                row("weekend_fixed", 0, 100.0, true),
                row("taxes_qa", 0, 200.0, false),
            ],
        ),
        ("no-think".to_string(), vec![row("json", 100, 1.0, true)]),
    ]);
    let scores = super::live::think_scores(&history);
    assert_eq!(scores.len(), 1);
    assert_exact!(scores["qwen3.8-27b-jang_6d-crack"], 75.0);
}

#[test]
fn from_home_points_at_the_operators_own_locations() {
    let env = LiveEnv::from_home(BTreeMap::new());
    assert_eq!(env.osaurus, PathBuf::from("osaurus"));
    assert!(env.models_root.ends_with("MLXModels"));
    assert!(
        env.families
            .ends_with(".local/state/ztools/models/families.json")
    );
    assert_eq!(env.hf_base, "https://huggingface.co");
}
