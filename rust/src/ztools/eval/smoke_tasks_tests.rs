//! The five built-in smoke tasks, pinned row for row.
//!
//! The smoke suite is what `--suite smoke` (the DEFAULT suite) runs: it is the
//! first thing a model is measured on, before the 27-row full roster is spent
//! on it. Until now the only assertion over it was `cases.len() == 5`, so every
//! name, every check string and every validator could change and the suite
//! stayed "five tasks long" — a model scored against a different benchmark
//! than last month's numbers, silently.
//!
//! The table below is the roster as it stands. Its shape mirrors
//! `tasks_tests.rs`, which pins the full roster the same way.
//!
//! The second half of this file pins what a Debug rendering cannot see: that
//! every check list is still SATISFIABLE (a typo'd needle — `transient_event`
//! singular — makes a task impossible to pass, so the model is reported as
//! broken when the PROMPT is wrong) and still DISCRIMINATING (a check list that
//! a refusal passes is not measuring anything).
//!
//! The third half pins M5: that every prompt this suite sends fits every
//! context window `conf/models/*.toml` documents. The smoke path sends its
//! prompts without consulting `eval/context_fit.rs`, and "these are small" is a
//! claim about numbers nobody wrote down — the same claim that let
//! `file_summary` grow past `foundation`'s window and score a 0 the model never
//! earned.

use super::*;
use crate::test_env::TestEnv;
use crate::ztools::eval::context_fit::MAX_CHARS_PER_TOKEN;
use crate::ztools::eval::context_fit::{context_refusal, prompt_bytes, tokens_at_least};
use crate::ztools::eval::model_resolve::documented_context_window;
use std::path::Path;

/// `(name, checks)` — the smoke roster row for row. Each check is rendered as
/// its `Debug` form, which names the `run_check` arm that grades it, so this
/// table pins the validator AND its arguments: the needle, the case
/// sensitivity, and the count the JSON array must have.
const SMOKE_TABLE: &[(&str, &[&str])] = &[
    (
        "Weekend Planner (JSON Extraction)",
        &[
            "Contains(\"transient_events\")",
            "Contains(\"Summer Rib Fest\")",
            "Contains(\"Magic Show\")",
            "JsonArrayLen(\"transient_events\", 2)",
        ],
    ),
    (
        "Twitter Summarizer (Markdown formatting)",
        &[
            "Contains(\"##\")",
            "ContainsAny([\"- \", \"* \"])",
            "ContainsLower(\"rust\")",
            "NotContainsLower(\"```html\")",
        ],
    ),
    (
        "Image Renamer (Constraint adherence)",
        &[
            "Contains(\".jpg\")",
            "Contains(\"_\")",
            "NotContains(\" \")",
            "NotContainsLower(\"here is\")",
        ],
    ),
    (
        "Twitter Summarizer (Factual Consistency)",
        &[
            "NotContains(\"@elonmusk\")",
            "NotContains(\"@realDonaldTrump\")",
            "ContainsAny([\"john_doe\", \"@john_doe\"])",
            "ContainsAny([\"jane_smith\", \"@jane_smith\"])",
            "ContainsAny([\"2026-08\", \"August\"])",
            "NotContains(\"2025\")",
            "NotContains(\"2024\")",
        ],
    ),
    ("File Summary (Content detail)", &["FileSummary(50)"]),
];

/// One reference answer per task: what a model that does the task produces.
/// Scored through the REAL `run_check`, after the REAL `clean_model_output`,
/// with the JSON extracted exactly as `eval_model` extracts it.
const REFERENCE_ANSWERS: &[&str] = &[
    // transient_events present, both named events kept, and exactly two rows —
    // which is what the JsonArrayLen arm parses out of the answer.
    r#"{"transient_events": [{"name": "Summer Rib Fest", "location": "Vaughan Park", "day": "Friday"}, {"name": "Magic Show", "location": "Vaughan Library", "day": "Friday"}]}"#,
    "## Highlights\n- New Rust version released\n- Elision explained\n",
    "red_sports_car_sunny_beach.jpg",
    "## Timeline\n- @john_doe launched the new API (@john_doe | 2026-08-01)\n- @jane_smith calls it fast (@jane_smith | 2026-08-02)\n",
    r#"[{"path": "lib/parser.py", "desc": "parses web pages and extracts article metadata"}, {"path": "lib/validator.py", "desc": "validates the extracted JSON fields against the schema"}, {"path": "lib/fetcher.py", "desc": "fetches URLs over HTTP and saves the response body"}, {"path": "lib/reporter.py", "desc": "writes the weekly markdown report from the parsed rows"}]"#,
];

fn rendered(task: &EvalTask) -> Vec<String> {
    task.checks.iter().map(|c| format!("{c:?}")).collect()
}

#[test]
fn the_smoke_roster_is_pinned_row_for_row() {
    let tasks = get_built_in_smoke_tasks();
    assert_eq!(tasks.len(), SMOKE_TABLE.len());
    for (task, (name, checks)) in tasks.iter().zip(SMOKE_TABLE) {
        assert_eq!(task.name, *name, "task name");
        let roles: Vec<&str> = task.messages.iter().map(|m| m.role.as_str()).collect();
        assert_eq!(roles, ["user"], "{name}: one user message, no system role");
        assert!(
            !task.parse_json,
            "{name}: the smoke prompts ask for JSON in the TEXT, so nothing may \
             request response_format — that changes the wire shape"
        );
        assert_eq!(
            task.messages.iter().map(|m| m.images.len()).sum::<usize>(),
            0,
            "{name}: the smoke suite is text-only"
        );
        assert_eq!(rendered(task), *checks, "{name}: check list");
    }
}

/// Every task's checks are satisfiable: the reference answer scores 100%. A
/// check list that cannot be satisfied makes `eval_model` report `failed` for
/// every model forever, which reads as "the model is broken" when the PROMPT
/// and the needle disagree.
#[test]
fn every_smoke_task_has_a_reference_answer_that_passes_every_check() {
    let tasks = get_built_in_smoke_tasks();
    assert_eq!(tasks.len(), REFERENCE_ANSWERS.len());
    for (task, answer) in tasks.iter().zip(REFERENCE_ANSWERS) {
        let cleaned = super::super::clean_model_output(answer);
        let parsed = super::super::extract_json(&cleaned);
        let missed: Vec<String> = task
            .checks
            .iter()
            .filter(|c| !super::super::run_check(c, &cleaned, parsed.as_ref()))
            .map(|c| format!("{c:?}"))
            .collect();
        assert!(
            missed.is_empty(),
            "{}: no reference answer passes {missed:?}",
            task.name
        );
    }
}

/// And they DISCRIMINATE: a refusal cannot pass every task. Note the
/// invariant is "cannot reach `passed`", NOT "passes nothing" — the
/// `NotContains*` arms are satisfied by any text at all (an empty string
/// contains no space and no ```` ```html ````), which is why a task made only
/// of them would measure nothing, asserted below.
#[test]
fn a_non_answer_cannot_pass_any_smoke_task() {
    let refusal = "I'm sorry, I can't help with that.";
    for task in get_built_in_smoke_tasks() {
        let cleaned = super::super::clean_model_output(refusal);
        let parsed = super::super::extract_json(&cleaned);
        let hits = task
            .checks
            .iter()
            .filter(|c| super::super::run_check(c, &cleaned, parsed.as_ref()))
            .count();
        assert!(
            hits < task.checks.len(),
            "{}: a refusal passes {hits} of {} checks, so the task reads as `passed`",
            task.name,
            task.checks.len()
        );
        // At least one check must REQUIRE content, or the task is vacuous.
        let requiring = task
            .checks
            .iter()
            .filter(|c| !matches!(c, Check::NotContains(_) | Check::NotContainsLower(_)))
            .count();
        assert_ne!(
            requiring, 0,
            "{}: every check is a NotContains arm, which any answer passes",
            task.name
        );
    }
}

/// M5: every smoke prompt fits every DOCUMENTED context window, the smallest
/// included.
///
/// The smoke path never consults `eval/context_fit.rs`: `model_eval::eval_model`
/// (`--suite smoke`, the default suite) serialises `case.messages` and sends them,
/// because these prompts are small. That was once said of `file_summary` too, and
/// its prompt grew to ~22.6 KB against `foundation`'s 4096-token window — nothing
/// compared the two, the request could not succeed, and the task landed in the
/// table as a 0 reading "this model summarises files badly" when the model was
/// never shown the files (`context_fit.rs`'s header). A belief nobody measures is
/// a belief that decays. This is the measurement, and it runs on every gate
/// rather than inside a sweep nobody schedules.
///
/// The bytes counted are the ones the runner REFUSES on:
/// `context_fit::prompt_bytes` over the task's own messages — the same call
/// `run_eval_inner` makes before the budget and the retries, and the same text
/// `eval_model` puts in the request body. Not a re-implementation: that sum has
/// one definition and both paths call it.
///
/// The windows scanned are the ones `conf/models/*.toml` DOCUMENTS. A file stem
/// is the model-family name `documented_context_window` matches a served id
/// against, so `foundation.toml`'s window is what a request for `foundation` is
/// refused on. A family that documents no window is never refused for anything,
/// so it is not a window this gate can claim to have covered.
#[test]
fn every_smoke_prompt_fits_every_documented_context_window() {
    // Sandboxed, so a peer's sandbox cannot redirect the conf read underneath
    // us: with `ZTOOLS_CONF_DIR` unpointed, a concurrent `TestEnv` could make
    // `documented_context_window` answer `None`, and this test would pass by
    // finding no windows at all. Pointed at the SHIPPED conf, so the windows are
    // the real ones this repo ships.
    let env = TestEnv::new();
    let shipped_conf = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("CARGO_MANIFEST_DIR is <repo>/rust")
        .join("conf");
    env.set_managed("ZTOOLS_CONF_DIR", shipped_conf.as_os_str());

    let mut windows: Vec<(String, u64)> = Vec::new();
    let models_dir = shipped_conf.join("models");
    for entry in std::fs::read_dir(&models_dir).expect("the shipped conf/models dir is readable") {
        let path = entry.expect("a dir entry is readable").path();
        if path.extension().and_then(|e| e.to_str()) != Some("toml") {
            continue;
        }
        // Owned, because the outlives-the-loop list is the scan's own result and
        // `path` dies with the iteration it came from.
        let Some(family) = path.file_stem().and_then(|s| s.to_str()).map(str::to_owned) else {
            continue;
        };
        if let Some(window) = documented_context_window(&family) {
            windows.push((family, window));
        }
    }
    windows.sort_unstable();
    assert!(
        !windows.is_empty(),
        "no conf/models/*.toml documents a context_window, so this gate scanned \
         nothing and would pass whatever the smoke prompts grow into"
    );

    let tasks = get_built_in_smoke_tasks();
    assert!(!tasks.is_empty(), "the smoke roster is empty");
    for task in &tasks {
        // The bytes that WILL be sent: what the runner refuses on, and what
        // `eval_model` serialises into the request body.
        let bytes = prompt_bytes(&task.messages);
        let tokens = tokens_at_least(u64::try_from(bytes).unwrap_or(u64::MAX));
        for (family, window) in &windows {
            let why = context_refusal(family, bytes);
            assert!(
                why.is_none(),
                "{} sends {bytes} bytes, which is at least {tokens} tokens at \
                 {MAX_CHARS_PER_TOKEN} chars/token -- {family} documents a \
                 {window}-token window, so the runner would refuse this task \
                 instead of measuring it: {why:?}",
                task.name
            );
        }
    }

    // And with room to spare, stated as ONE number so a failure names the
    // margin rather than only the boundary: the LARGEST smoke prompt against the
    // SMALLEST documented window.
    let largest = tasks
        .iter()
        .map(|t| prompt_bytes(&t.messages))
        .max()
        .expect("the smoke roster is non-empty");
    let largest_tokens = tokens_at_least(u64::try_from(largest).unwrap_or(u64::MAX));
    let smallest = windows.iter().map(|(_, w)| *w).min().expect("non-empty");
    assert!(
        largest_tokens < smallest,
        "the largest smoke prompt is {largest} bytes ({largest_tokens} tokens at \
         {MAX_CHARS_PER_TOKEN} chars/token) and the smallest documented window is \
         {smallest} tokens: a prompt that grows now has no headroom at all"
    );
    drop(env);
}
