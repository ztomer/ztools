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

use super::*;

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
