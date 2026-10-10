//! The weekend rows measure what the planner SENDS (W5).
//!
//! Until 2026-10-10 the json slot was ranked on a private copy of the planner's
//! schema: a bare array whose system prompt ORDERED defaults (`weather:
//! "indoor"`, `price: $20-30 or Free`, `target_ages: "6-13 years"`) -- the
//! mandated-placeholder class production had removed -- over a fixed-venue list
//! production has never structured with a model. A model was picked for
//! `weekend_model` by how well it obeyed a prompt the planner no longer sends.
//!
//! These tests hold the rows to production by IDENTITY, not by resemblance: the
//! system message is `structure_system(..)` itself, the user message is the
//! production renderer's output, and a drift in either side turns them red.

use super::super::prompts::{
    WEEKEND_EVAL_LOCATION, WEEKEND_NOISE_RESULTS, WEEKEND_SEARCH_RESULTS, WEEKEND_SIGNAL_EVENTS,
    WEEKEND_STRUCTURE_ROWS, weekend_eval_window,
};
use super::*;
use crate::ztools::weekend::phases::structure_system;

fn conf_dir() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .join("conf")
}

fn tasks() -> Vec<EvalTask> {
    roster(&RosterInputs::in_dir(&conf_dir())).expect("roster loads from the shipped conf")
}

/// The roster rows that grade the planner's JSON phase. Named, not derived
/// from a prefix: a derived set that matched nothing would pass by vacuity.
const STRUCTURE_ROWS: &[&str] = &[
    "weekend_transient",
    "weekend_transient_mixed",
    "weekend_transient_schema",
    "json",
];

#[test]
fn every_weekend_json_row_sends_the_production_structure_system_prompt() {
    let tasks = tasks();
    let production = structure_system(2026);
    for name in STRUCTURE_ROWS {
        let task = tasks
            .iter()
            .find(|t| t.name == *name)
            .unwrap_or_else(|| panic!("{name} is not in the roster"));
        let system = task
            .messages
            .iter()
            .find(|m| m.role == "system")
            .unwrap_or_else(|| panic!("{name} sends no system message"));
        assert_eq!(
            system.content, production,
            "{name} sends a system prompt the planner does not send"
        );
    }
}

fn by_name<'a>(tasks: &'a [EvalTask], name: &str) -> &'a EvalTask {
    tasks
        .iter()
        .find(|t| t.name == name)
        .unwrap_or_else(|| panic!("{name} is not in the roster"))
}

fn user_of(task: &EvalTask) -> &str {
    &task
        .messages
        .iter()
        .rfind(|m| m.role == "user")
        .expect("a user message")
        .content
}

fn score_of(task: &EvalTask, answer: &str) -> i64 {
    let parsed: Option<serde_json::Value> = serde_json::from_str(answer).ok();
    super::super::task_loader::check_graded_score(&task.checks[0], answer, parsed.as_ref())
        .expect("a graded check")
}

/// The user messages are the production renderers' output over this module's
/// data: one structure batch, and the monolithic fallback over search results
/// floated and marked the way `fetch_events_corpus` hands them on.
#[test]
fn the_weekend_user_messages_are_rendered_by_the_production_renderers() {
    use crate::ztools::weekend::{monolithic_user, prioritise_in_window, structure_user};
    let tasks = tasks();
    let batch = structure_user(WEEKEND_STRUCTURE_ROWS);
    for name in ["weekend_transient", "weekend_transient_schema", "json"] {
        assert_eq!(user_of(by_name(&tasks, name)), batch, "{name}");
    }
    let (start, end) = weekend_eval_window();
    let corpus = prioritise_in_window(WEEKEND_SEARCH_RESULTS, start, end);
    assert_eq!(
        user_of(by_name(&tasks, "weekend_transient_mixed")),
        monolithic_user(&corpus, WEEKEND_EVAL_LOCATION, start, end)
    );
}

/// The data is in the shapes production builds: a structure batch is at most
/// `STRUCTURE_BATCH` rows that the phase's own row reader accepts, every search
/// result is a `- ` line the extract phase reads, and the window marks exactly
/// the signal -- so the fixture's dates agree with its labels.
#[test]
fn the_weekend_data_is_production_shaped_and_carries_one_set_of_facts() {
    use crate::ztools::weekend::{IN_WINDOW_MARK, STRUCTURE_BATCH, extracted_rows};
    let lines: Vec<&str> = WEEKEND_STRUCTURE_ROWS.lines().collect();
    assert_eq!(lines.len(), STRUCTURE_BATCH, "one full structure batch");
    assert_eq!(extracted_rows(WEEKEND_STRUCTURE_ROWS), lines);
    for row in &lines {
        assert_eq!(row.split(" | ").count(), 6, "NAME..description: {row}");
    }
    assert!(WEEKEND_SEARCH_RESULTS.lines().all(|l| l.starts_with("- ")));
    assert_eq!(
        WEEKEND_SEARCH_RESULTS.lines().count(),
        WEEKEND_SIGNAL_EVENTS.len() + WEEKEND_NOISE_RESULTS.len()
    );
    let (start, end) = weekend_eval_window();
    let corpus = crate::ztools::weekend::prioritise_in_window(WEEKEND_SEARCH_RESULTS, start, end);
    let marked: Vec<&str> = corpus
        .lines()
        .filter_map(|l| l.strip_prefix(IN_WINDOW_MARK))
        .collect();
    assert_eq!(marked.len(), WEEKEND_SIGNAL_EVENTS.len(), "{corpus}");
    for (line, event) in marked.iter().zip(WEEKEND_SIGNAL_EVENTS) {
        assert!(
            line.trim_start().starts_with(&format!("- {event}:")),
            "{line}"
        );
    }
    // The same ten events in both shapes: each row's NAME is in its result's title.
    for (row, event) in lines.iter().zip(WEEKEND_SIGNAL_EVENTS) {
        let name = row.split(" | ").next().unwrap_or_default();
        assert!(event.starts_with(name), "{name} vs {event}");
    }
}

/// What a correct structure answer looks like: every row, keyed, string
/// fields, "" for what the row does not state.
fn faithful_structure_answer() -> String {
    let events: Vec<serde_json::Value> = WEEKEND_STRUCTURE_ROWS
        .lines()
        .map(|row| {
            let f: Vec<&str> = row.split(" | ").collect();
            let known = |s: &str| if s == "unknown" { "" } else { s }.to_string();
            let days: Vec<&str> = f[2].trim_start_matches("April ").split('-').collect();
            let iso = |d: &str| format!("2026-04-{d}");
            serde_json::json!({
                "name": f[0], "location": known(f[1]), "target_ages": f[4],
                "price": known(f[3]), "start_date": iso(days[0]),
                "end_date": iso(days[days.len() - 1]), "duration": "",
                "place_kind": "", "weather": "", "day": "", "description": f[5],
            })
        })
        .collect();
    serde_json::json!({ "transient_events": events }).to_string()
}

/// The keyed shape production asks for and parses is what scores: a faithful
/// structure answer is full marks on all three structure rows, and the mixed
/// row scores precision AND recall against the stated signal and noise.
#[test]
fn the_graders_score_the_keyed_shape_production_parses() {
    let tasks = tasks();
    let answer = faithful_structure_answer();
    let wire = serde_json::json!({"choices": [{"message": {"content": answer}}]});
    let events = crate::ztools::weekend::parse_llm_events(&wire).expect("production parses it");
    assert_eq!(events.len(), WEEKEND_STRUCTURE_ROWS.lines().count());
    for name in ["weekend_transient", "weekend_transient_schema", "json"] {
        assert_eq!(score_of(by_name(&tasks, name), &answer), 100, "{name}");
    }

    let mixed = by_name(&tasks, "weekend_transient_mixed");
    let keyed = |names: &[&str]| {
        let rows: Vec<_> = names
            .iter()
            .map(|n| serde_json::json!({"name": n}))
            .collect();
        serde_json::json!({ "transient_events": rows }).to_string()
    };
    assert_eq!(score_of(mixed, &keyed(WEEKEND_SIGNAL_EVENTS)), 100);
    let half = &WEEKEND_SIGNAL_EVENTS[..5];
    let with_noise: Vec<&str> = half.iter().chain(WEEKEND_NOISE_RESULTS).copied().collect();
    let (recall_only, polluted) = (
        score_of(mixed, &keyed(half)),
        score_of(mixed, &keyed(&with_noise)),
    );
    assert!(
        recall_only < 100,
        "half the signal is not full marks: {recall_only}"
    );
    assert!(
        polluted < recall_only,
        "kept noise costs precision: {polluted} vs {recall_only}"
    );
}

/// An answer the planner's parser DROPS scores nothing on any weekend JSON row,
/// however well its content would grade. A bare array -- the shape the old eval
/// prompt asked for -- or a keyed list whose field is not a string loses every
/// row in `structure_to_json`; ranking a model up for it is ranking it on a
/// result production throws away.
#[test]
fn an_answer_the_planner_cannot_parse_scores_zero_on_every_weekend_json_row() {
    let tasks = tasks();
    let faithful: serde_json::Value =
        serde_json::from_str(&faithful_structure_answer()).expect("json");
    let bare = faithful["transient_events"].to_string();
    let mut typed = faithful;
    typed["transient_events"][0]["price"] = serde_json::json!(null);
    for name in STRUCTURE_ROWS {
        let task = by_name(&tasks, name);
        for (what, answer) in [("bare array", &bare), ("null field", &typed.to_string())] {
            let (score, reason) = match &task.checks[0] {
                Check::Graded(g) => g.score(answer, serde_json::from_str(answer).ok().as_ref()),
                other => panic!("{name}: {other:?}"),
            };
            assert_eq!(
                score, 0,
                "{name} scored a {what} the planner drops: {reason}"
            );
        }
    }
}

/// The eval's per-family prompt quirks rewrite nothing the planner sends, so
/// every family is ranked on production's bytes (the planner applies no
/// quirks of its own).
#[test]
fn no_model_quirk_rewrites_a_weekend_json_row() {
    use super::super::quirks::{MODEL_FAMILIES, apply_model_quirks};
    let tasks = tasks();
    for name in STRUCTURE_ROWS {
        let wire: Vec<serde_json::Value> = by_name(&tasks, name)
            .messages
            .iter()
            .map(|m| serde_json::json!({"role": m.role, "content": m.content}))
            .collect();
        for family in MODEL_FAMILIES.iter().chain(&["gemma4", "default"]) {
            assert_eq!(
                apply_model_quirks(&wire, family),
                wire,
                "{name} under {family}"
            );
        }
    }
}
