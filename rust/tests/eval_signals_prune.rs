//! Tests for `ztools eval-signals --prune` (`M7`).
//!
//! Confirms that pruning:
//! 1. Removes records whose fingerprints do not match current active task definitions.
//! 2. Leaves `_capabilities` intact.
//! 3. Preserves records with active task fingerprints.
//! 4. Refuses or dry-runs without modifying files.
//! 5. Successfully prunes on `--prune` and writes clean JSON back to disk.

#[path = "support/mod.rs"]
mod support;
pub use support::*;

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use ztools::test_env::TestEnv;

fn repo_root() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("rust/ has a parent")
        .to_path_buf()
}

#[test]
fn eval_signals_prune_cli_help() {
    let out = Command::new(bin())
        .args(["eval-signals", "--help"])
        .output()
        .unwrap();
    assert!(out.status.success());
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("--prune"));
    assert!(stdout.contains("--dry-run"));
}

fn setup_test_data(tmp_dir: &Path) -> (PathBuf, PathBuf, String, String) {
    let signals_file = tmp_dir.join("test_eval_signals.json");
    let config_file = tmp_dir.join("test_config.toml");

    let conf_dir = repo_root().join("conf");
    let tasks_data_dir = repo_root().join("eval_tasks/data");
    fs::write(
        &config_file,
        format!(
            "eval_conf_dirs = [\"{}\"]\neval_tasks_dirs = [\"{}\"]\n",
            conf_dir.display(),
            tasks_data_dir.display()
        ),
    )
    .unwrap();

    // Load active tasks from the shipped configuration to know real active fingerprints
    let roster_inputs = ztools::eval::tasks::RosterInputs::in_dir(&conf_dir);
    let active_tasks =
        ztools::eval::load_all_eval_tasks(&roster_inputs, Some(&tasks_data_dir)).unwrap();
    assert_ne!(active_tasks.len(), 0);

    let sample_active = &active_tasks[0];
    let active_fp = ztools::eval::task_fingerprint(sample_active);

    // Create a signal store with 1 capability block, 1 active record, 1 stale record, 1 unrecorded record
    let json_content = serde_json::json!({
        "test-model": {
            "_capabilities": {
                "cold_start_seconds": 1.5,
                "decode_tokens_per_sec": 30.0
            },
            sample_active.name.clone(): {
                "fingerprint": active_fp,
                "p95_latency": 12.5,
                "samples": 3,
                "timeout": 900
            },
            "stale_task": {
                "fingerprint": "superseded_hash_12345",
                "p95_latency": 25.0,
                "samples": 10
            },
            "unrecorded_task": {
                "p95_latency": 45.0,
                "samples": 5
            }
        }
    });

    fs::write(
        &signals_file,
        serde_json::to_string_pretty(&json_content).unwrap(),
    )
    .unwrap();

    (
        config_file,
        signals_file,
        sample_active.name.clone(),
        active_fp,
    )
}

#[test]
fn eval_signals_prune_cli_workflow() {
    let (_env, tmp_dir) = TestEnv::new().at("HOME");
    let (config_file, signals_file, sample_active_name, active_fp) = setup_test_data(&tmp_dir);

    // 1. Inspect without flags
    let inspect_out = Command::new(bin())
        .args([
            "--config",
            config_file.to_str().unwrap(),
            "eval-signals",
            "--path",
            signals_file.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(
        inspect_out.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&inspect_out.stderr)
    );
    let inspect_stdout = String::from_utf8_lossy(&inspect_out.stdout);
    assert!(inspect_stdout.contains("Active task records: 1"));
    assert!(inspect_stdout.contains("Superseded/unrecorded task records: 2"));

    // 2. Dry run: prints report, does NOT modify file
    let dry_out = Command::new(bin())
        .args([
            "--config",
            config_file.to_str().unwrap(),
            "eval-signals",
            "--path",
            signals_file.to_str().unwrap(),
            "--dry-run",
        ])
        .output()
        .unwrap();
    assert!(
        dry_out.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&dry_out.stderr)
    );
    let dry_stdout = String::from_utf8_lossy(&dry_out.stdout);
    assert!(dry_stdout.contains("Dry run: would prune 2 task record(s)"));
    assert!(dry_stdout.contains("1 active records kept"));

    // File on disk must still contain stale and unrecorded tasks
    let text_after_dry = fs::read_to_string(&signals_file).unwrap();
    let parsed_dry: serde_json::Value = serde_json::from_str(&text_after_dry).unwrap();
    assert!(parsed_dry["test-model"].get("stale_task").is_some());
    assert!(parsed_dry["test-model"].get("unrecorded_task").is_some());

    // 3. Real prune: removes stale & unrecorded, preserves active and capabilities
    let prune_out = Command::new(bin())
        .args([
            "--config",
            config_file.to_str().unwrap(),
            "eval-signals",
            "--path",
            signals_file.to_str().unwrap(),
            "--prune",
        ])
        .output()
        .unwrap();
    assert!(
        prune_out.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&prune_out.stderr)
    );
    let prune_stdout = String::from_utf8_lossy(&prune_out.stdout);
    assert!(prune_stdout.contains("Pruned 2 task record(s)"));

    // Read back file from disk
    let text_after_prune = fs::read_to_string(&signals_file).unwrap();
    let parsed_pruned: serde_json::Value = serde_json::from_str(&text_after_prune).unwrap();
    let model_obj = parsed_pruned["test-model"].as_object().unwrap();

    assert!(
        model_obj.contains_key("_capabilities"),
        "_capabilities must be preserved"
    );
    assert!(
        model_obj.contains_key(&sample_active_name),
        "active task must be preserved"
    );
    assert!(
        !model_obj.contains_key("stale_task"),
        "stale task must be stripped"
    );
    assert!(
        !model_obj.contains_key("unrecorded_task"),
        "unrecorded task must be stripped"
    );

    // Retained task must carry the active fingerprint
    assert_eq!(
        model_obj[&sample_active_name]["fingerprint"]
            .as_str()
            .unwrap(),
        active_fp
    );
}
