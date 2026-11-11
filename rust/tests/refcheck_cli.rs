//! `ztools models refcheck` end to end: the real binary, a stub `osaurus` on
//! the PATH, a fixture `~/MLXModels` and a seeded families cache.
//!
//! The binary must fail on an export pinning `mlx-community/Qwen3.8-27B-8bit`,
//! pass on a clean one, and with repair on turn the broken one into a one-agent
//! plan to the 6D crack and apply exactly that document.

// Refuse a stale build of this file (ZT.MTIMECLASS).
ztools::built_from_disk!();

#[path = "support/mod.rs"]
mod support;
pub use support::*;

use std::fs;
use std::os::unix::fs::PermissionsExt;
use std::path::Path;

const BROKEN: &str = "mlx-community/Qwen3.8-27B-8bit";
const CRACK: &str = "dealignai/Qwen3.8-27B-JANG_6D-CRACK";
const RAPTOR: &str = "OsaurusAI/Raptor-v0.5-8B-A1B-JANG_6M";

fn export(disk_audit: &str) -> String {
    serde_json::json!({
        "version": 1,
        "models": [RAPTOR, CRACK],
        "model_defaults": {"core_model": CRACK, "compaction_model": null},
        "default_agent": {"model": CRACK},
        "configurator": {"model": null},
        "agents": [{"name": "Disk Audit", "model": disk_audit}],
        "automations": [],
        "providers": []
    })
    .to_string()
}

/// A home holding the stub, the fixture models and the seeded cache.
fn sandbox(name: &str, pinned: &str) -> std::path::PathBuf {
    let home = fresh(name);
    for id in [RAPTOR, CRACK] {
        let dir = home.join("MLXModels").join(id);
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join("config.json"), "{}").unwrap();
    }
    let state = home.join(".local/state/ztools/models");
    fs::create_dir_all(&state).unwrap();
    fs::write(
        state.join("families.json"),
        format!(r#"{{"{BROKEN}":"Qwen/Qwen3.8-27B","{CRACK}":"Qwen/Qwen3.8-27B"}}"#),
    )
    .unwrap();
    let bin = home.join("bin");
    fs::create_dir_all(&bin).unwrap();
    fs::write(home.join("export.json"), export(pinned)).unwrap();
    fs::write(home.join("fixed.json"), export(CRACK)).unwrap();
    let h = home.display();
    let script = format!(
        r#"#!/bin/sh
echo "$@" >> "{h}/argv.log"
case "$1 $2" in
  "config export") while [ $# -gt 0 ]; do [ "$1" = "-o" ] && cp "{h}/export.json" "$2"; shift; done ;;
  "config plan") cp "$3" "{h}/plan.json"; printf 'agents:\n  ~ Disk Audit\n      model: {BROKEN} -> {CRACK}\n' ;;
  "config apply") cp "$3" "{h}/apply.json"; cp "{h}/fixed.json" "{h}/export.json" ;;
  list*) printf 'raptor-v0.5-8b-a1b-jang_6m\nqwen3.8-27b-jang_6d-crack\n' ;;
esac
"#
    );
    let osaurus = bin.join("osaurus");
    fs::write(&osaurus, script).unwrap();
    fs::set_permissions(&osaurus, fs::Permissions::from_mode(0o755)).unwrap();
    write_config(&home, "");
    home
}

fn refcheck(home: &Path, args: &[&str]) -> std::process::Output {
    ztool(home)
        .env(
            "PATH",
            format!("{}:/usr/bin:/bin", home.join("bin").display()),
        )
        .args(["models", "refcheck"])
        .args(args)
        .output()
        .unwrap()
}

fn argv_log(home: &Path) -> String {
    fs::read_to_string(home.join("argv.log")).unwrap_or_default()
}

#[test]
fn a_clean_config_passes_with_exit_zero_and_changes_nothing() {
    let home = sandbox("refcheck-clean", CRACK);
    let out = refcheck(&home, &[]);
    assert_eq!(
        exit_code(&out),
        0,
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(String::from_utf8_lossy(&out.stdout).contains("✓ refcheck"));
    assert!(
        !argv_log(&home).contains("config plan"),
        "a clean run must not plan"
    );
}

#[test]
fn the_uninstalled_8bit_fails_without_repair_naming_the_path() {
    let home = sandbox("refcheck-broken", BROKEN);
    let out = refcheck(&home, &["--no-repair"]);
    assert_refused("pinned 8bit", &out, 1, &["1 missing", "0 repaired"]);
    assert!(
        String::from_utf8_lossy(&out.stdout).contains(&format!(
            "agents[Disk Audit].model = {BROKEN}: not installed"
        )),
        "{}",
        String::from_utf8_lossy(&out.stdout)
    );
    assert!(!argv_log(&home).contains("config plan"));
}

#[test]
fn repair_applies_a_one_agent_document_pointing_at_the_6d_crack() {
    let home = sandbox("refcheck-repair", BROKEN);
    for flags in [&[][..], &["--repair"][..]] {
        fs::write(home.join("export.json"), export(BROKEN)).unwrap();
        let out = refcheck(&home, flags);
        // 3: found and repaired, so the routines failure line shows it once.
        assert_refused("repair", &out, 3, &["1 missing", "1 repaired"]);
        let stdout = String::from_utf8_lossy(&out.stdout).into_owned();
        assert!(stdout.contains(&format!("{BROKEN} -> {CRACK}")), "{stdout}");
        let applied: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(home.join("apply.json")).unwrap()).unwrap();
        assert_eq!(
            applied,
            serde_json::json!({"version": 1, "agents": [{"name": "Disk Audit", "model": CRACK}]})
        );
    }
    let log = argv_log(&home);
    assert!(!log.contains("--prune"), "{log}");
    // The second run, on the repaired config, is clean.
    let again = refcheck(&home, &[]);
    assert_eq!(exit_code(&again), 0);
}

#[test]
fn both_repair_flags_together_are_refused() {
    let home = sandbox("refcheck-flags", CRACK);
    let out = refcheck(&home, &["--repair", "--no-repair"]);
    assert_ne!(exit_code(&out), 0);
}
