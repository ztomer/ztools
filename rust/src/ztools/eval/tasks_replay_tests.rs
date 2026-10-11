//! The json-slot rows replayed over answers saved from the 2026-10-10 sweep
//! (M22): the scores have to separate a 350M model from an 8B one.
//!
//! lfm2.5-350m scored 100 on three of the six rows with one row out of ten,
//! `"place_kind": "outdoor"` and `"for_families": "all"`; raptor's answers are
//! the ones the planner uses. Each answer is graded as the runner grades it.

use super::*;

const SLOT_ROWS: &[&str] = &[
    "weekend_transient",
    "weekend_transient_mixed",
    "weekend_transient_schema",
    "weekend_fabrication",
    "weekend_injection",
    "json",
];

fn replay(model: &str) -> Vec<(String, u8)> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap();
    let tasks = roster(&RosterInputs::in_dir(&root.join("conf"))).expect("roster loads");
    let dir = root
        .join("tests/fixtures/eval/2026-10-10_json_slot")
        .join(model);
    SLOT_ROWS
        .iter()
        .map(|name| {
            let task = tasks
                .iter()
                .find(|t| t.name == *name)
                .expect("row in the roster");
            let answer = std::fs::read_to_string(dir.join(format!("{name}.txt"))).unwrap();
            let parsed: Option<serde_json::Value> = serde_json::from_str(answer.trim()).ok();
            let score = super::super::runner::score_output(task, answer.trim(), parsed.as_ref());
            ((*name).to_string(), score)
        })
        .collect()
}

fn mean(scores: &[(String, u8)]) -> u32 {
    scores.iter().map(|(_, s)| u32::from(*s)).sum::<u32>() / u32::try_from(scores.len()).unwrap()
}

#[test]
fn the_slot_rows_score_a_350m_models_answers_below_50() {
    let lfm = replay("lfm2.5-350m-optiq-4bit");
    assert!(mean(&lfm) < 50, "{lfm:?}");
    let schema = lfm
        .iter()
        .find(|(n, _)| n == "weekend_transient_schema")
        .unwrap();
    assert!(schema.1 < 50, "one row of ten with garbage fields: {lfm:?}");
}

#[test]
fn the_planners_own_model_keeps_its_scores() {
    let raptor = replay("raptor-v0.5-8b-a1b-jang_6m");
    assert!(raptor.iter().all(|(_, s)| *s >= 80), "{raptor:?}");
    assert!(mean(&raptor) >= 90, "{raptor:?}");
}
