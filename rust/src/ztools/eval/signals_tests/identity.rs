//! What happens to a per-task series when the task under its name is REPLACED.
//!
//! The store keeps a RUNNING AGGREGATE per (model, task) — p95, retries, parse
//! failures — and not a list of samples, so there is no way to mark one sample
//! as belonging to the old prompt and average the rest. The only honest move is
//! to start a new series when the identity changes and COUNT what was set
//! aside, which is what these cases pin.
//!
//! The last case runs against the REAL `conf/eval_signals.json`, because that
//! file is the whole reason the rule exists: every series in it was accumulated
//! against a prompt that no longer exists.

use super::super::*;
use crate::ztools::eval::task_fingerprint::remember_current_tasks;
use crate::ztools::eval::task_loader::EvalTask;
use crate::ztools::eval::tasks::RosterInputs;
use crate::ztools::eval::{load_all_eval_tasks, task_fingerprint};

/// The task whose observations the cases below accumulate. One name per case,
/// so no two cases ever share a series.
fn task(name: &str, prompt: &str) -> EvalTask {
    EvalTask::new(name, prompt, Vec::new())
}

fn series<'a>(signals: &'a SignalStore, model: &str, name: &str) -> &'a serde_json::Value {
    &signals[model][name]
}

/// The same task name, a DIFFERENT question: the M1 transition in miniature.
#[test]
fn a_replaced_task_starts_a_new_series_and_counts_what_it_set_aside() {
    let mut signals = SignalStore::new();
    let before = task("file_summary", "name the files at these absolute paths");
    let after = task("file_summary", "name the files, with their contents below");

    record_task_signal(&mut signals, "m", &before, 30.0, true, true);
    record_task_signal(&mut signals, "m", &before, 40.0, false, true);
    let row = series(&signals, "m", "file_summary");
    assert_eq!(row["samples"], 2);
    assert_eq!(row["total_retries"], 1);

    // Now the task is replaced and one observation arrives under the new one.
    record_task_signal(&mut signals, "m", &after, 12.0, false, false);
    let row = series(&signals, "m", "file_summary");
    assert_eq!(row["samples"], 1, "the series restarts: {row}");
    assert_eq!(
        row["p95_latency"], 12.0,
        "the old p95 is not blended into the new prompt's: {row}"
    );
    assert_eq!(row["total_retries"], 0, "retries restart with the series");
    assert_eq!(row["parse_failures"], 0);
    assert_eq!(
        row["fingerprint"],
        serde_json::json!(task_fingerprint(&after)),
        "the series says which task it is now measuring"
    );
    assert_eq!(
        superseded_samples(&signals, "m", "file_summary"),
        2,
        "the two discarded observations are counted, not vanished"
    );
    assert_eq!(
        series_fingerprint(&signals, "m", "file_summary").as_deref(),
        Some(task_fingerprint(&after).as_str())
    );
}

/// A second replacement counts both losses.
#[test]
fn a_second_replacement_adds_to_the_set_aside_count() {
    let mut signals = SignalStore::new();
    let first = task("twice", "one");
    let second = task("twice", "two");
    let third = task("twice", "three");
    for t in [&first, &second, &third] {
        record_task_signal(&mut signals, "m", t, 10.0, false, false);
    }
    assert_eq!(superseded_samples(&signals, "m", "twice"), 2);
    assert_eq!(series(&signals, "m", "twice")["samples"], 1);
}

/// A series never refuse to learn a timeout. This is the property that stops a
/// wedged server idling: whatever happens to a task's identity, a request is
/// still sized by the largest of the configured, derived and default ceilings.
#[test]
fn a_restarted_series_still_learns_a_timeout() {
    let mut signals = SignalStore::new();
    let slow = task("timeout-restart", "the slow prompt");
    let fast = task("timeout-restart", "the fast prompt");
    record_task_signal(&mut signals, "m", &slow, 1000.0, false, false);
    assert_eq!(series(&signals, "m", "timeout-restart")["timeout"], 1500);

    // The prompt changes and the first observation back is quick. The learned
    // timeout is re-derived from the new series, never left absent.
    record_task_signal(&mut signals, "m", &fast, 2.0, false, false);
    let timeout = series(&signals, "m", "timeout-restart")["timeout"]
        .as_u64()
        .expect("a series always carries a timeout ceiling");
    assert!(
        timeout >= default_eval_timeout(),
        "the documented floor still bounds the request: {timeout}"
    );
}

/// The name-based writer: the registry resolves the digest, and a name nobody
/// registered keeps accumulating exactly as it did before fingerprints existed
/// (there is nothing to compare against, and resetting every observation would
/// be worse than the disease).
#[test]
fn the_name_path_resolves_the_registry_and_unknowns_still_accumulate() {
    let mut signals = SignalStore::new();
    let registered = task("identity-registry-task", "the prompt");
    remember_current_tasks(std::slice::from_ref(&registered));
    record_signal(
        &mut signals,
        "m",
        "identity-registry-task",
        10.0,
        false,
        false,
    );
    record_signal(
        &mut signals,
        "m",
        "identity-registry-task",
        20.0,
        false,
        false,
    );
    assert_eq!(
        series(&signals, "m", "identity-registry-task")["fingerprint"],
        serde_json::json!(task_fingerprint(&registered)),
        "the writer resolved the digest from the registry"
    );
    assert_eq!(
        series(&signals, "m", "identity-registry-task")["samples"],
        2,
        "the same task accumulates"
    );

    record_signal(
        &mut signals,
        "m",
        "no-such-task-registered",
        5.0,
        false,
        false,
    );
    record_signal(
        &mut signals,
        "m",
        "no-such-task-registered",
        7.0,
        false,
        false,
    );
    let row = series(&signals, "m", "no-such-task-registered");
    assert!(row.get("fingerprint").is_none(), "{row}");
    assert_eq!(row["samples"], 2, "two unknowns still accumulate: {row}");
    assert_eq!(
        superseded_samples(&signals, "m", "no-such-task-registered"),
        0
    );
}

/// THE REAL STORE. Every series in `conf/eval_signals.json` was accumulated
/// against prompts that no longer exist — `file_summary` asked for absolute
/// paths and no contents before 2026-10-08 — so the first fingerprinted
/// observation against each of them must SUPERSEDE the old series, not join it.
///
/// Read-only: the file on disk is never written by this case.
#[test]
fn the_shipped_store_supersedes_its_own_series_rather_than_blending_them() {
    let text = std::fs::read_to_string(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .expect("the checkout that holds rust/")
            .join("conf/eval_signals.json"),
    )
    .expect("the shipped signal store");
    // It LOADS: no new key anywhere, and none required.
    let mut signals: SignalStore = serde_json::from_str(&text).expect("the store parses");

    // The current `file_summary` task, from the shipped roster.
    let files = RosterInputs::in_dir(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .expect("the checkout that holds rust/")
            .join("conf")
            .as_path(),
    );
    let tasks = load_all_eval_tasks(&files, None).expect("the shipped roster loads");
    let current = tasks
        .iter()
        .find(|t| t.name == "file_summary")
        .expect("the roster carries file_summary");

    // The first model with a file_summary series that has samples in it.
    let model = signals
        .iter()
        .find(|(_, entry)| {
            entry
                .get("file_summary")
                .and_then(|t| t.get("samples"))
                .and_then(serde_json::Value::as_u64)
                .is_some_and(|n| n > 0)
        })
        .map(|(name, _)| name.clone())
        .expect("the shipped store holds a model with file_summary samples");
    let stale = signals[&model]["file_summary"]["samples"]
        .as_u64()
        .expect("samples is a count");
    assert!(
        signals[&model]["file_summary"].get("fingerprint").is_none(),
        "the shipped store predates fingerprints"
    );

    record_task_signal(&mut signals, &model, current, 25.0, false, false);

    let row = series(&signals, &model, "file_summary");
    assert_eq!(row["samples"], 1, "a fresh series: {row}");
    assert_eq!(row["p95_latency"], 25.0, "not blended with the old: {row}");
    assert_eq!(
        superseded_samples(&signals, &model, "file_summary"),
        stale,
        "the whole old series is counted as set aside"
    );
    assert_eq!(
        row["fingerprint"],
        serde_json::json!(task_fingerprint(current))
    );
}
