//! `report`'s tests: persistence, the aggregates, and the trend table.
//!
//! Split out of `report.rs` for the house 500-line cap along the seam the
//! module already had. The aggregates are driven through the pure
//! `historical_stats` / `render_trends` twins with an EXPLICIT identity set, so
//! no test depends on what some other test registered in the process-wide
//! registry; the registry itself is pinned in `task_fingerprint_tests.rs`.
//!
//! THE CASE THE WHOLE CHANGE IS WRITTEN AGAINST: the M1 transition. `file_summary`
//! asked a different question from 2026-10-08, so a history row under that name
//! from before then is a row about a different task and must be set aside and
//! counted, never averaged.

use super::*;
use crate::ztools::eval::task_fingerprint::{
    Standing, TaskIdentities, remember_current_tasks, task_fingerprint,
};
use crate::ztools::eval::task_loader::{Check, EvalTask};
use serde_json::json;

pub(in crate::ztools::eval) fn outcome(task: &str, score: u8) -> TaskOutcome {
    TaskOutcome {
        task: task.to_string(),
        score,
        status: if score >= 90 { "ok" } else { "fail" }.to_string(),
        ..Default::default()
    }
}

pub(in crate::ztools::eval) fn run(
    model: &str,
    outcomes: Vec<TaskOutcome>,
    complete: bool,
) -> ModelRun {
    let mut r = ModelRun::new(model, &[], outcomes);
    if let Some(c) = r.completeness.as_mut() {
        c.complete = complete;
        if !complete {
            c.missing = vec!["never-ran".to_string()];
            c.reason = "test".to_string();
        }
    }
    r
}

/// The task every save-path test runs. ONE task per name for the whole file, so
/// the process-wide registry holds the same value whatever order the tests run
/// in — the alternative is a test that passes alone and fails in parallel.
fn registered(task_name: &str) -> EvalTask {
    EvalTask::new(
        task_name,
        "the prompt",
        vec![Check::Contains("x".to_string())],
    )
}

/// Register and return the identity set for `tasks`.
fn identities_of(tasks: &[EvalTask]) -> TaskIdentities {
    remember_current_tasks(tasks);
    let mut current = TaskIdentities::new();
    for task in tasks {
        current.insert(task.name.clone(), task_fingerprint(task));
    }
    current
}

/// A history file holding `entries` per model.
fn write_history(dir: &Path, entries: &serde_json::Value) {
    std::fs::write(dir.join("eval_history.json"), entries.to_string()).unwrap();
}

fn stats_for(dir: &Path, current: &TaskIdentities, model: &str) -> ModelStats {
    historical_stats(Some(dir), current)
        .get(model)
        .unwrap_or_else(|| panic!("{model} has no row"))
        .clone()
}

#[test]
fn test_models_never_enter_the_leaderboard() {
    let dir = tempfile::tempdir().unwrap();
    let task = registered("t");
    let current = identities_of(std::slice::from_ref(&task));
    save_historical_results(
        &run("mock-model", vec![outcome("t", 100)], true),
        Some(dir.path()),
    )
    .unwrap();
    save_historical_results(
        &run("fake-70b", vec![outcome("t", 100)], true),
        Some(dir.path()),
    )
    .unwrap();
    save_historical_results(
        &run("real-model", vec![outcome("t", 80)], true),
        Some(dir.path()),
    )
    .unwrap();
    let stats = historical_stats(Some(dir.path()), &current);
    assert!(!stats.contains_key("mock-model"), "{stats:?}");
    assert!(!stats.contains_key("fake-70b"), "{stats:?}");
    assert!(stats.contains_key("real-model"));
}

/// A row the model never answered holds no score, so the history must not
/// hold one either: its placeholder 0 used to be averaged into the mean.
#[test]
fn unmeasured_rows_never_enter_the_history() {
    let dir = tempfile::tempdir().unwrap();
    let current = identities_of(&[registered("t"), registered("down"), registered("big")]);
    let mut outage = outcome("down", 0);
    outage.error = Some("Connection failed".to_string());
    outage.failure_category = crate::ztools::eval::FAIL_INFRA.to_string();
    let mut too_big = outcome("big", 0);
    too_big.error = Some("prompt does not fit".to_string());
    too_big.failure_category = crate::ztools::eval::FAIL_CONTEXT.to_string();
    save_historical_results(
        &run("real-model", vec![outcome("t", 80), outage, too_big], true),
        Some(dir.path()),
    )
    .unwrap();
    let stats = historical_stats(Some(dir.path()), &current);
    assert_eq!(stats["real-model"].runs, 1, "{stats:?}");
    assert_exact!(stats["real-model"].mean, 80.0);
}

#[test]
fn truncated_entries_are_written_marked_and_excluded_from_averages() {
    // MARKED, not dropped: the individual scores exist on disk...
    let dir = tempfile::tempdir().unwrap();
    let current = identities_of(&[registered("easy")]);
    save_historical_results(
        &run("ornith-test", vec![outcome("easy", 100)], false),
        Some(dir.path()),
    )
    .unwrap();
    let raw: BTreeMap<String, Vec<HistoryEntry>> = serde_json::from_str(
        &std::fs::read_to_string(dir.path().join("eval_history.json")).unwrap(),
    )
    .unwrap();
    assert!(
        !raw["ornith-test"][0].complete,
        "verdict travels with the entry"
    );

    // ...but no aggregate ever averages them, and the discrepancy between
    // runs and entry count is surfaced rather than hidden.
    let stats = historical_stats(Some(dir.path()), &current);
    let row = &stats["ornith-test"];
    assert_eq!(
        (row.runs, row.excluded, row.incomplete),
        (0, 1, 1),
        "nothing countable, and the reason is named: {stats:?}"
    );

    save_historical_results(
        &run("ornith-test", vec![outcome("easy", 60)], true),
        Some(dir.path()),
    )
    .unwrap();
    let stats = historical_stats(Some(dir.path()), &current);
    let row = &stats["ornith-test"];
    assert_eq!((row.runs, row.excluded), (1, 1));
    assert_exact!(row.mean, 60.0, "the unclean 100 must not be averaged");
}

/// An entry with no `complete` field at all is trusted as complete, exactly as
/// Python's `.get(..., True)` did — that default predates fingerprints and has
/// nothing to do with them.
#[test]
fn legacy_records_without_the_complete_field_are_trusted() {
    let dir = tempfile::tempdir().unwrap();
    let current = identities_of(&[registered("t")]);
    write_history(
        dir.path(),
        &json!({
            "old-model": [
                {"date": "2026-01-01", "timestamp": 1_767_225_600.0, "task": "t",
                 "score": 90, "time": 1.0,
                 "fingerprint": task_fingerprint(&registered("t"))}
            ]
        }),
    );
    let stats = historical_stats(Some(dir.path()), &current);
    assert_eq!(stats["old-model"].runs, 1, "absent complete means complete");
    assert_eq!(stats["old-model"].excluded, 0);
}

/// EVERY entry on disk before 2026-10-08 looks like this: no `fingerprint` key
/// at all. It must still LOAD — that is the serde contract — and it must not be
/// averaged, because absence means unknown, never current.
#[test]
fn an_entry_with_no_fingerprint_loads_and_is_set_aside() {
    let dir = tempfile::tempdir().unwrap();
    let current = identities_of(&[registered("t")]);
    write_history(
        dir.path(),
        &json!({
            "old-model": [
                {"date": "2026-09-19", "timestamp": 1.0, "task": "t",
                 "score": 100, "time": 12.0, "complete": true}
            ]
        }),
    );
    // It LOADED: absence is not a parse error.
    let entries = load_history_entries(Some(dir.path()));
    assert_eq!(entries["old-model"][0].fingerprint, None);
    // And it is set aside rather than averaged.
    let row = stats_for(dir.path(), &current, "old-model");
    assert_eq!(
        (row.runs, row.excluded, row.superseded),
        (0, 1, 1),
        "{row:?}"
    );
    assert_exact!(row.mean, 0.0);
    // The trend table says so, rather than leaving a silent zero-run model.
    let lines = render_trends(Some(dir.path()), &current);
    assert!(
        lines
            .iter()
            .any(|l| l.contains("old-model") && l.contains("—")),
        "a model with nothing countable keeps its row, with dashes: {lines:?}"
    );
    assert!(
        lines
            .last()
            .is_some_and(|l| l.starts_with("Set aside:") && l.contains("1 entry")),
        "the set-aside count and its reason are on the table: {lines:?}"
    );
}

#[test]
fn zero_scores_count_toward_the_mean() {
    // `if e.get("score")` falsy-for-zero once made a model that scored 0
    // on half its runs look identical to one that never failed.
    let dir = tempfile::tempdir().unwrap();
    let current = identities_of(&[registered("a"), registered("b")]);
    save_historical_results(
        &run("m", vec![outcome("a", 100), outcome("b", 0)], true),
        Some(dir.path()),
    )
    .unwrap();
    let stats = historical_stats(Some(dir.path()), &current)
        .remove("m")
        .unwrap();
    assert_eq!((stats.runs, stats.min, stats.max), (2, 0, 100));
    assert_exact!(stats.mean, 50.0);
}

#[test]
fn winners_take_the_best_score_per_task_across_runs() {
    let runs = vec![
        run("a", vec![outcome("t1", 90), outcome("t2", 40)], true),
        run("b", vec![outcome("t1", 95), outcome("t2", 40)], true),
    ];
    let winners = compute_task_winners(&runs);
    assert_eq!(winners["t1"].0, "b");
    assert_eq!(winners["t2"].0, "a", "tie keeps the first winner seen");
}

#[test]
fn trends_render_worst_first_with_an_excluded_column() {
    let dir = tempfile::tempdir().unwrap();
    let current = identities_of(&[registered("t")]);
    save_historical_results(
        &run("slow-model", vec![outcome("t", 40)], true),
        Some(dir.path()),
    )
    .unwrap();
    save_historical_results(
        &run("fast-model", vec![outcome("t", 90)], true),
        Some(dir.path()),
    )
    .unwrap();
    let lines = render_trends(Some(dir.path()), &current);
    assert!(lines[0].starts_with("Historical Trends"));
    assert!(lines[1].contains("Superseded"), "{lines:?}");
    assert!(lines[2].contains("fast-model"), "{lines:?}");
    assert!(
        lines[3].contains("slow-model"),
        "sorted worst-first: {lines:?}"
    );
    assert_eq!(lines.len(), 4, "nothing set aside: no footer: {lines:?}");
}

/// A save writes the digest of the task it ran, so the next aggregate can tell
/// which task that was.
#[test]
fn saving_a_run_records_the_fingerprint_of_the_task_it_ran() {
    let dir = tempfile::tempdir().unwrap();
    let task = registered("fingerprinted");
    let current = identities_of(std::slice::from_ref(&task));
    save_historical_results(
        &run("m", vec![outcome("fingerprinted", 80)], true),
        Some(dir.path()),
    )
    .unwrap();
    let entries = load_history_entries(Some(dir.path()));
    assert_eq!(
        entries["m"][0].fingerprint.as_deref(),
        Some(task_fingerprint(&task).as_str())
    );
    let row = stats_for(dir.path(), &current, "m");
    assert_eq!(
        (row.runs, row.excluded, row.superseded),
        (1, 0, 0),
        "{row:?}"
    );
}

/// THE M1 TRANSITION, as the case this change is written against.
///
/// `file_summary` asked for absolute paths and no contents before 2026-10-08,
/// and for repo-relative paths with each file's own contents after. The CURRENT
/// task is built from the shipped roster — the real thing, prompt and all, no
/// GPU involved — and the history holds one row from before the change (an
/// obviously-old digest) and one from after (the current one).
#[test]
fn a_replaced_task_has_its_old_history_set_aside_not_averaged() {
    let tasks = crate::ztools::eval::load_all_eval_tasks(&shipped_conf(), None)
        .expect("the shipped roster loads without a server");
    let current_task = tasks
        .iter()
        .find(|t| t.name == "file_summary")
        .expect("the roster carries file_summary")
        .clone();
    let current =
        TaskIdentities::from([("file_summary".to_string(), task_fingerprint(&current_task))]);

    let dir = tempfile::tempdir().unwrap();
    write_history(
        dir.path(),
        &json!({
            "gemma-4-e4b": [
                // Before 2026-10-08: another task under the same name. A
                // literal stands in for the old digest rather than the old
                // prompt, so the test pins the RULE and not a historical text.
                {"date": "2026-09-19", "timestamp": 1.0, "task": "file_summary",
                 "score": 100, "time": 20.0, "complete": true,
                 "fingerprint": "pre-m1-file-summary"},
                // And one from before fingerprints existed at all.
                {"date": "2026-09-01", "timestamp": 0.5, "task": "file_summary",
                 "score": 90, "time": 18.0, "complete": true},
                // After: the task as it stands now.
                {"date": "2026-10-09", "timestamp": 2.0, "task": "file_summary",
                 "score": 60, "time": 30.0, "complete": true,
                 "fingerprint": task_fingerprint(&current_task)}
            ]
        }),
    );

    let row = stats_for(dir.path(), &current, "gemma-4-e4b");
    assert_eq!(
        row.runs, 1,
        "only the current fingerprint is averaged: {row:?}"
    );
    assert_eq!(row.excluded, 2, "both old rows are set aside: {row:?}");
    assert_eq!(row.superseded, 2, "and both are superseded: {row:?}");
    assert_exact!(row.mean, 60.0, "the 100 and the 90 must not be averaged in");

    let lines = render_trends(Some(dir.path()), &current);
    let row = lines
        .iter()
        .find(|l| l.contains("gemma-4-e4b"))
        .expect("the model's row is on the table");
    let tail = format!("{:>5} {:>5} {:>5} {:>9} {:>9}", 60, 60, 1, 2, 2);
    assert!(
        row.ends_with(&tail),
        "one run averaged, two set aside: {row:?}"
    );
    assert!(
        lines
            .last()
            .is_some_and(|l| l.starts_with("Set aside:") && l.contains("2 entries")),
        "the table says how many were set aside and why: {lines:?}"
    );
}

/// A model whose ENTIRE history predates the current task is still a model
/// with history. Dropping its row would read as a model nobody ever ran.
#[test]
fn a_model_whose_whole_history_is_old_still_gets_a_row() {
    let dir = tempfile::tempdir().unwrap();
    let current = identities_of(&[registered("file_summary")]);
    write_history(
        dir.path(),
        &json!({
            "bonsai-27b": [
                {"date": "2026-09-19", "timestamp": 1.0, "task": "file_summary",
                 "score": 85, "time": 20.0, "complete": true},
                {"date": "2026-09-19", "timestamp": 1.1, "task": "file_summary",
                 "score": 80, "time": 21.0, "complete": true,
                 "fingerprint": "some-other-version"}
            ],
            "fresh-model": [
                {"date": "2026-10-09", "timestamp": 2.0, "task": "file_summary",
                 "score": 70, "time": 25.0, "complete": true,
                 "fingerprint": task_fingerprint(&registered("file_summary"))}
            ]
        }),
    );
    let lines = render_trends(Some(dir.path()), &current);
    assert!(
        lines[2].contains("fresh-model"),
        "the model with countable history comes first: {lines:?}"
    );
    assert!(
        lines[3].contains("bonsai-27b") && lines[3].contains("—"),
        "the all-old model keeps its row with dashes, not zeros: {lines:?}"
    );
    assert!(
        lines
            .last()
            .is_some_and(|l| l.starts_with("Set aside:") && l.contains("2 entries")),
        "its two set-aside entries are counted: {lines:?}"
    );
}

/// The shipped conf, for the case written against the real M1 roster.
fn shipped_conf() -> crate::ztools::eval::tasks::RosterInputs {
    let conf = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("the checkout that holds rust/")
        .join("conf");
    crate::ztools::eval::tasks::RosterInputs {
        inputs: conf.join("eval_inputs.toml"),
        vision: conf.join("eval_vision.toml"),
    }
}

/// A truncated run of a superseded task is BOTH, and is set aside once.
#[test]
fn a_truncated_superseded_entry_is_counted_once_and_both_ways() {
    let dir = tempfile::tempdir().unwrap();
    let task = registered("t");
    let current = TaskIdentities::from([("t".to_string(), task_fingerprint(&task))]);
    write_history(
        dir.path(),
        &json!({
            "m": [
                {"date": "2026-09-19", "timestamp": 1.0, "task": "t",
                 "score": 100, "time": 20.0, "complete": false,
                 "fingerprint": "pre-m1-digest"}
            ]
        }),
    );
    let row = stats_for(dir.path(), &current, "m");
    assert_eq!(
        (row.runs, row.excluded, row.superseded, row.incomplete),
        (0, 1, 1, 1),
        "one entry, excluded once, with both reasons named: {row:?}"
    );
    assert!(!Standing::Superseded.counts());
    // And the footer counts the TOTAL, not the sum of its reasons: adding the
    // two clauses here would announce two set-aside entries out of a file that
    // holds one.
    let footer = render_trends(Some(dir.path()), &current)
        .last()
        .expect("a table with an excluded entry has a footer")
        .clone();
    assert!(
        footer.starts_with("Set aside: 1 entry not averaged"),
        "the total is one: {footer}"
    );
    assert!(
        footer.contains("1 against a superseded task") && footer.contains("1 from a truncated run"),
        "both reasons are named: {footer}"
    );
    assert!(
        footer.ends_with("a truncated run of a replaced task is both"),
        "and the overlap is said out loud: {footer}"
    );
}
