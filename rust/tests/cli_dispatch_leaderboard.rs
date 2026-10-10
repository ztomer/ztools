//! CLI dispatch integration tests for `ztools model-eval --leaderboard`.

#[path = "support/mod.rs"]
mod support;
pub use support::*;

use std::fs;

fn stdout_of(out: &std::process::Output) -> String {
    assert!(
        out.status.success(),
        "command failed — stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8_lossy(&out.stdout).into_owned()
}

fn seed_eval_history(home: &std::path::Path) {
    let conf_dir = home.join(".config/ztools");
    fs::create_dir_all(&conf_dir).unwrap();
    let sample = r#"{"m":[{"date":"2026-09-01","timestamp":100.0,"task":"json","score":80,"complete":true},{"date":"2026-10-09","timestamp":200.0,"task":"json","score":100,"complete":true}]}"#;
    fs::write(conf_dir.join("eval_history.json"), sample).unwrap();
}

#[test]
fn model_eval_leaderboard_json_and_min_tasks_cli() {
    let home = fresh("eval-min");
    write_config(&home, "");
    seed_eval_history(&home);
    let out = ztool(&home)
        .args([
            "model-eval",
            "--leaderboard",
            "--json-output",
            "--min-tasks",
            "2",
        ])
        .output()
        .unwrap();
    let stdout = stdout_of(&out);
    let parsed: serde_json::Value = serde_json::from_str(&stdout).unwrap();
    assert!(parsed.is_array());
}

#[test]
fn model_eval_leaderboard_sort_by_and_delta_cli() {
    let home = fresh("eval-sort");
    write_config(&home, "");
    seed_eval_history(&home);
    let out = ztool(&home)
        .args(["model-eval", "--leaderboard", "--sort-by", "json"])
        .output()
        .unwrap();
    let stdout = stdout_of(&out);
    assert!(stdout.contains("| Rank | Model | Mean | Delta | Think | JSON |"));
    assert!(stdout.contains("`m`"));
    assert!(stdout.contains("+20.0%"));

    let json_out = ztool(&home)
        .args([
            "model-eval",
            "--leaderboard",
            "--json-output",
            "--sort-by",
            "json",
        ])
        .output()
        .unwrap();
    let parsed: serde_json::Value = serde_json::from_str(&stdout_of(&json_out)).unwrap();
    assert_eq!(parsed[0]["delta"], 20.0);

    let bad_out = ztool(&home)
        .args(["model-eval", "--leaderboard", "--sort-by", "invalid_slot"])
        .output()
        .unwrap();
    assert!(!bad_out.status.success());
    let stderr = String::from_utf8_lossy(&bad_out.stderr);
    assert!(
        stderr.contains("invalid sort slot 'invalid_slot'"),
        "{stderr}"
    );
}

#[test]
fn model_eval_leaderboard_flags_m13_to_m16_cli() {
    let home = fresh("eval-flags");
    write_config(&home, "");
    seed_eval_history(&home);

    // M13: --category
    let cat_out = ztool(&home)
        .args(["model-eval", "--leaderboard", "--category", "json"])
        .output()
        .unwrap();
    assert!(cat_out.status.success());
    let stdout = stdout_of(&cat_out);
    assert!(stdout.contains("`m`"));

    // M14: --csv-output
    let csv_out = ztool(&home)
        .args(["model-eval", "--leaderboard", "--csv-output"])
        .output()
        .unwrap();
    assert!(csv_out.status.success());
    let stdout_csv = stdout_of(&csv_out);
    assert!(
        stdout_csv.starts_with("model,mean,delta,think,json,summarize,filename,vlm,tasks,date\n")
    );
    assert!(stdout_csv.contains("m,100.0,20.0"));

    // M15: --group-by-family
    let fam_out = ztool(&home)
        .args(["model-eval", "--leaderboard", "--group-by-family"])
        .output()
        .unwrap();
    assert!(fam_out.status.success());
    let stdout_fam = stdout_of(&fam_out);
    assert!(stdout_fam.contains("Grouped by Family"));

    // M16: --fail-on-regression (clean delta +20.0% should pass with threshold 5.0)
    let pass_reg = ztool(&home)
        .args(["model-eval", "--leaderboard", "--fail-on-regression", "5.0"])
        .output()
        .unwrap();
    assert!(pass_reg.status.success());

    // M16: --fail-on-regression (drop from 100% to 50% = -50% delta should fail)
    let conf_dir = home.join(".config/ztools");
    let reg_sample = r#"{"m":[{"date":"2026-09-01","timestamp":100.0,"task":"json","score":100,"complete":true},{"date":"2026-10-09","timestamp":200.0,"task":"json","score":50,"complete":true}]}"#;
    fs::write(conf_dir.join("eval_history.json"), reg_sample).unwrap();
    let fail_reg = ztool(&home)
        .args([
            "model-eval",
            "--leaderboard",
            "--fail-on-regression",
            "10.0",
        ])
        .output()
        .unwrap();
    assert!(!fail_reg.status.success());
    let stderr_reg = String::from_utf8_lossy(&fail_reg.stderr);
    assert!(stderr_reg.contains("regression detected"), "{stderr_reg}");
}
