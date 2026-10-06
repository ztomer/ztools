//! The roster's file-summary rows, asserted on the RENDERED prompt.
//!
//! Split out of `tasks_tests.rs` and into its own file for one reason: the pins in
//! `prompts/file_summary_pins_tests.rs` render from the TEMPLATES, so every one of
//! them stays green if `tasks.rs` hands the runner the un-rendered path list — the
//! exact prompt this whole change replaced. The gap is closed by building the real
//! roster and looking at what the model would be sent.

use super::*;
use crate::ztools::eval::prompts::file_summary::FILE_SUMMARY_CONTENTS_SLOT;
use crate::ztools::eval::prompts::file_summary::{
    FILE_SUMMARY_EXCERPT_BYTES, FILE_SUMMARY_FILE_LIST,
};

fn roster_tasks() -> Vec<EvalTask> {
    roster(&RosterInputs::in_dir(
        &Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .join("conf"),
    ))
    .expect("roster loads from the shipped conf")
}

fn user_prompt(tasks: &[EvalTask], name: &str) -> String {
    tasks
        .iter()
        .find(|t| t.name == name)
        .unwrap_or_else(|| panic!("{name} is not in the roster"))
        .messages
        .last()
        .expect("every task has a user message")
        .content
        .clone()
}

/// THE FIX AT THE SEAM, not at the template: the prompt the runner would send
/// carries every listed file's own text.
///
/// Calibrated: making `file_summary_tasks` pass the bare templates — which every
/// template-level pin still accepts — fails here.
#[test]
fn the_roster_sends_the_files_own_content_not_just_their_paths() {
    let tasks = roster_tasks();
    for name in ["file_summary", "file_summary_mixed"] {
        let prompt = user_prompt(&tasks, name);
        assert!(
            !prompt.contains(FILE_SUMMARY_CONTENTS_SLOT),
            "{name} shipped the unfilled slot: the model is told to rely on content \
             it was never given"
        );
        for row in FILE_SUMMARY_FILE_LIST.lines() {
            let row = row.trim();
            assert!(
                prompt.contains(&format!("--- BEGIN {row} ")),
                "{name} does not fence {row}'s content"
            );
        }
        assert_eq!(
            prompt.matches("--- BEGIN ").count(),
            FILE_SUMMARY_FILE_LIST.lines().count(),
            "{name}: exactly one excerpt per listed row"
        );
        assert!(
            prompt.contains("DO NOT infer functionality from file names,"),
            "{name} lost the rule the content makes followable"
        );
    }
}

/// The bound holds at the seam too, in BYTES, because that is what the context
/// window is spent in.
#[test]
fn the_rosters_prompts_are_inside_the_context_budget_they_claim() {
    let tasks = roster_tasks();
    for name in ["file_summary", "file_summary_mixed"] {
        let prompt = user_prompt(&tasks, name);
        let ceiling = (FILE_SUMMARY_FILE_LIST.lines().count() * FILE_SUMMARY_EXCERPT_BYTES) + 4096;
        assert!(
            prompt.len() <= ceiling,
            "{name} is {} bytes over its {ceiling}-byte ceiling",
            prompt.len()
        );
        // And the excerpt bounds are visible in what was sent, so a later edit to
        // either constant shows up here rather than only in the template tests.
        assert!(
            prompt.contains("(showing"),
            "{name}: a truncated excerpt is silent"
        );
    }
}

/// The mixed row's validator is handed the SAME text the model is shown — it
/// reads its signal set out of it. Handing it the template instead would score
/// the mixed variant against a prompt the model never saw, which is the drift
/// `the_source_is_the_prompt_the_model_was_shown` states as a rule.
#[test]
fn the_mixed_validator_is_handed_the_prompt_the_model_was_shown() {
    let tasks = roster_tasks();
    let prompt = user_prompt(&tasks, "file_summary_mixed");
    let Check::Graded(Graded::MixedFileSummary { source }) = tasks
        .iter()
        .find(|t| t.name == "file_summary_mixed")
        .expect("row exists")
        .checks
        .first()
        .expect("one check")
    else {
        panic!("file_summary_mixed is graded by validate_mixed_file_summary with a source");
    };
    assert_eq!(
        source.as_str(),
        prompt.as_str(),
        "the validator's source must be the rendered prompt the model saw"
    );
}

/// An unreadable listed file must stop the roster, not produce a prompt with a
/// hole in it. `roster()` is on the path of every eval run, so a silent skip
/// would be a sweep that quietly measures a shorter task.
#[test]
fn a_missing_listed_file_is_an_error_not_a_shorter_prompt() {
    // The listed rows are absolute paths into this repo, so the only way to make
    // one unreadable without touching the working tree is to render a template
    // whose slot is fine but whose FILE LIST is empty -- which is not reachable
    // from here. So the guard is asserted where it lives instead: the renderer
    // refuses a slotless template, and `render_both` checks both before reading.
    let stripped = FILE_SUMMARY_PROMPT.replace(FILE_SUMMARY_CONTENTS_SLOT, "");
    let err = crate::ztools::eval::prompts::file_summary::render_both(
        &stripped,
        FILE_SUMMARY_PROMPT_MIXED,
    )
    .expect_err("a slotless template must not render")
    .to_string();
    assert!(err.contains(FILE_SUMMARY_CONTENTS_SLOT), "{err}");
}
