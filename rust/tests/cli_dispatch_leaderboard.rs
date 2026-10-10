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

/// The shipped `conf/`, pinned in the sandbox config so the binary and this
/// test read the same roster.
fn conf_dir() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("rust/ has a parent")
        .join("conf")
}

/// The fingerprint of the roster's `json` task as it stands now. Since M17 the
/// leaderboard scores only rows that carry it, so the seeded history does too.
fn json_fingerprint() -> String {
    let roster = ztools::eval::tasks::RosterInputs::in_dir(&conf_dir());
    let tasks = ztools::eval::load_all_eval_tasks(&roster, None).unwrap();
    let json = tasks
        .iter()
        .find(|t| t.name == "json")
        .expect("roster has json");
    ztools::eval::task_fingerprint(json)
}

fn write_roster_config(home: &std::path::Path) {
    write_config(
        home,
        &format!("eval_conf_dirs = [\"{}\"]\n", conf_dir().display()),
    );
}

/// Two clean `json` runs (80 then 100), each carrying `fingerprint`.
fn history_json(first: i64, second: i64, fingerprint: Option<&str>) -> String {
    let rows: Vec<serde_json::Value> =
        [(100.0, "2026-09-01", first), (200.0, "2026-10-09", second)]
            .into_iter()
            .map(|(ts, date, score)| {
                serde_json::json!({
                    "date": date, "timestamp": ts, "task": "json", "score": score,
                    "complete": true, "fingerprint": fingerprint,
                })
            })
            .collect();
    serde_json::json!({ "m": rows }).to_string()
}

fn seed_eval_history(home: &std::path::Path) {
    write_roster_config(home);
    let conf_dir = home.join(".config/ztools");
    fs::create_dir_all(&conf_dir).unwrap();
    let fp = json_fingerprint();
    fs::write(
        conf_dir.join("eval_history.json"),
        history_json(80, 100, Some(&fp)),
    )
    .unwrap();
}

#[test]
fn model_eval_leaderboard_json_and_min_tasks_cli() {
    let home = fresh("eval-min");
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
    assert!(stdout_csv.starts_with(
        "model,mean,delta,think,json,summarize,filename,vlm,tasks,date,not_current\n"
    ));
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
    let fp = json_fingerprint();
    fs::write(
        conf_dir.join("eval_history.json"),
        history_json(100, 50, Some(&fp)),
    )
    .unwrap();
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

/// M17 end to end: the binary loads the roster, and a row that does not carry
/// the current `json` fingerprint (every row on disk before 2026-10-08, and
/// every row of a retired task) is shown as not measured on the current task,
/// never as a score.
#[test]
fn model_eval_leaderboard_does_not_score_a_stale_row_cli() {
    let home = fresh("eval-stale");
    write_roster_config(&home);
    let conf_dir = home.join(".config/ztools");
    fs::create_dir_all(&conf_dir).unwrap();
    fs::write(
        conf_dir.join("eval_history.json"),
        history_json(80, 100, Some("0000000000000000")),
    )
    .unwrap();
    let out = ztool(&home)
        .args(["model-eval", "--leaderboard"])
        .output()
        .unwrap();
    let stdout = stdout_of(&out);
    assert!(
        stdout.contains("| `m` | not current |"),
        "a superseded json row must not be scored: {stdout}"
    );
    assert!(
        stdout.contains("not measured on the current task"),
        "{stdout}"
    );
}
