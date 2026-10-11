//! The roster pinned to the Python `TASKS` table as it stood on its last run
//! (2026-09-13): names in order, message roles, `parse_json`, and which
//! validator grades each — so a port that drops or reshapes a task is caught
//! here, not in a sweep nobody compares to the old numbers.
//!
//! One deliberate departure since: `weekend_fixed`, `weekend_fixed_mixed` and
//! their `detailed_json` alias were RETIRED on 2026-10-10 (W5). They graded a
//! year-round-venue structuring call the planner does not make — its fixed
//! venues are a curated list (`weekend_cache.rs`) — so they ranked the json
//! slot on a prompt nobody sends. What the remaining weekend rows send is
//! pinned to production in `tasks_weekend_tests.rs`.

use super::*;

fn conf_dir() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .join("conf")
}

fn tasks() -> Vec<EvalTask> {
    roster(&RosterInputs::in_dir(&conf_dir())).expect("roster loads from the shipped conf")
}

fn validator_name(check: &Check) -> &'static str {
    let Check::Graded(g) = check else {
        panic!("roster tasks carry exactly one graded check, got {check:?}");
    };
    graded_name(g)
}

/// The validator a check grades by; the planner's parse gate is a gate in
/// front of one, not a validator of its own (`tasks_weekend_tests.rs`).
fn graded_name(g: &Graded) -> &'static str {
    match g {
        Graded::PlannerEvents { inner, .. } => graded_name(inner),
        Graded::DetailedJson { .. } => "validate_detailed_json",
        Graded::MixedSignal { .. } => "validate_mixed_signal",
        Graded::Filename { .. } => "validate_filename",
        Graded::MixedFilename { .. } => "validate_mixed_filename",
        Graded::Summary { .. } => "validate_summary",
        Graded::MixedSummary { .. } => "validate_mixed_summary",
        Graded::FileSummary => "validate_file_summary",
        Graded::MixedFileSummary { .. } => "validate_mixed_file_summary",
        Graded::StrictSchema { .. } => "validate_strict_schema",
        Graded::NoContradiction { .. } => "validate_no_contradiction",
        Graded::NoLeak => "validate_no_leak",
        Graded::FactualAccuracy { .. } => "validate_factual_accuracy",
        Graded::FactualCoverage { .. } => "validate_factual_coverage",
        Graded::ImageDescription => "validate_image_description",
        Graded::Attribution { .. } => "validate_attribution",
        Graded::NoFabrication { .. } => "validate_no_fabrication",
        Graded::ResistsInjection { .. } => "validate_resists_injection",
    }
}

/// `(name, roles, parse_json, validator)` — the Python table, row for row.
const PYTHON_TABLE: &[(&str, &str, bool, &str)] = &[
    (
        "weekend_transient",
        "system,user",
        true,
        "validate_detailed_json",
    ),
    (
        "weekend_transient_mixed",
        "system,user",
        true,
        "validate_mixed_signal",
    ),
    ("filename", "user", false, "validate_filename"),
    ("image_rename", "user", true, "validate_mixed_filename"),
    ("summarize", "user", false, "validate_summary"),
    ("file_summary", "system,user", true, "validate_file_summary"),
    ("rename_mixed", "user", true, "validate_mixed_filename"),
    ("summarize_mixed", "user", false, "validate_mixed_summary"),
    (
        "file_summary_mixed",
        "system,user",
        false,
        "validate_mixed_file_summary",
    ),
    ("filename_mixed", "user", true, "validate_mixed_filename"),
    (
        "image_rename_mixed",
        "user",
        true,
        "validate_mixed_filename",
    ),
    (
        "weekend_transient_schema",
        "system,user",
        false,
        "validate_strict_schema",
    ),
    (
        "summarize_contradiction",
        "user",
        false,
        "validate_no_contradiction",
    ),
    ("filename_leak", "user", false, "validate_no_leak"),
    (
        "summarize_factual_accuracy",
        "user",
        false,
        "validate_factual_accuracy",
    ),
    ("image_real", "user", false, "validate_image_description"),
    (
        "weekend_fabrication",
        "system,user",
        true,
        "validate_no_fabrication",
    ),
    (
        "filename_injection",
        "user",
        false,
        "validate_resists_injection",
    ),
    (
        "summarize_misattribution",
        "user",
        false,
        "validate_attribution",
    ),
    (
        "summarize_factual_coverage",
        "user",
        false,
        "validate_factual_coverage",
    ),
    // Added 2026-09-19, after the Python table: the summarize slot's injection gate.
    (
        "summarize_injection",
        "user",
        false,
        "validate_resists_injection",
    ),
    (
        "weekend_injection",
        "system,user",
        true,
        "validate_resists_injection",
    ),
    ("json", "system,user", true, "validate_detailed_json"),
];

#[test]
fn roster_matches_the_python_table_row_for_row() {
    let tasks = tasks();
    assert_eq!(tasks.len(), PYTHON_TABLE.len());
    for (task, (name, roles, parse_json, validator)) in tasks.iter().zip(PYTHON_TABLE) {
        assert_eq!(task.name, *name);
        let got_roles: Vec<&str> = task.messages.iter().map(|m| m.role.as_str()).collect();
        assert_eq!(got_roles.join(","), *roles, "{name}: roles");
        assert_eq!(task.parse_json, *parse_json, "{name}: parse_json");
        assert_eq!(task.checks.len(), 1, "{name}: one graded check");
        assert_eq!(
            validator_name(&task.checks[0]),
            *validator,
            "{name}: validator"
        );
    }
}

#[test]
fn filename_tasks_are_rendered_with_the_real_input_never_the_template() {
    let tasks = tasks();
    for name in ["filename", "filename_leak"] {
        let t = tasks.iter().find(|t| t.name == name).unwrap();
        let prompt = &t.messages[0].content;
        assert!(!prompt.contains("{text}"), "{name} sends the bare template");
        assert!(
            prompt.starts_with("Give a short 2-4 word summary of: Screenshot showing login e"),
            "{name}: {prompt}"
        );
    }
    let filename = tasks.iter().find(|t| t.name == "filename").unwrap();
    let Check::Graded(Graded::Filename { source }) = &filename.checks[0] else {
        panic!("filename is graded by validate_filename with a source");
    };
    assert_eq!(
        source,
        "Screenshot showing login error: Invalid credentials. Please try again."
    );
}

#[test]
fn the_source_is_the_prompt_the_model_was_shown() {
    // Relevance and attribution are judged against what the model saw.
    let tasks = tasks();
    for name in [
        "summarize",
        "summarize_mixed",
        "summarize_misattribution",
        "weekend_fabrication",
        "weekend_injection",
        "filename_injection",
    ] {
        let t = tasks.iter().find(|t| t.name == name).unwrap();
        let prompt = &t.messages.last().unwrap().content;
        let Check::Graded(g) = &t.checks[0] else {
            unreachable!()
        };
        // The weekend probes sit behind the planner's parse gate (M20).
        let g = match g {
            Graded::PlannerEvents { inner, .. } => inner.as_ref(),
            other => other,
        };
        let source = match g {
            Graded::Summary { source }
            | Graded::MixedSummary { source }
            | Graded::Attribution { source }
            | Graded::NoFabrication { source, .. }
            | Graded::ResistsInjection { source, .. } => source,
            other => panic!("{name}: unexpected {other:?}"),
        };
        assert_eq!(source, prompt, "{name}: source must be the prompt");
    }
}

#[test]
fn image_real_is_the_only_task_that_sends_images_and_sends_all_three() {
    let tasks = tasks();
    for t in &tasks {
        let images: usize = t.messages.iter().map(|m| m.images.len()).sum();
        if t.name == "image_real" {
            assert_eq!(images, 3, "all fixtures in one call");
            assert!(t.messages[0].images[0].starts_with("data:image/png;base64,"));
            // Wire shape: content parts, not an `images` key.
            let wire = serde_json::to_value(&t.messages[0]).unwrap();
            let parts = wire["content"].as_array().expect("content parts");
            assert_eq!(parts[0]["type"], "text");
            assert_eq!(parts[1]["type"], "image_url");
            assert!(wire.get("images").is_none());
        } else {
            assert_eq!(images, 0, "{} must not carry images", t.name);
        }
    }
}

#[test]
fn the_json_alias_is_weekend_transient_under_the_slot_name() {
    let tasks = tasks();
    let by = |n: &str| tasks.iter().find(|t| t.name == n).unwrap();
    assert_eq!(by("json").messages, by("weekend_transient").messages);
    assert_eq!(by("json").checks, by("weekend_transient").checks);
    // Retired with weekend_fixed (W5): no row may answer to these names, or a
    // sweep would write fresh history under a name that means a dead prompt.
    for retired in ["weekend_fixed", "weekend_fixed_mixed", "detailed_json"] {
        assert!(
            tasks.iter().all(|t| t.name != retired),
            "{retired} is back in the roster"
        );
    }
}

#[test]
fn missing_inputs_are_errors_not_guesses() {
    let td = tempfile::tempdir().unwrap();
    let files = RosterInputs::in_dir(td.path());
    let err = roster(&files).unwrap_err().to_string();
    assert!(err.contains("eval inputs at"), "{err}");
    std::fs::write(
        td.path().join("eval_inputs.toml"),
        "[test_inputs]\nsummarize = \"x\"\n",
    )
    .unwrap();
    let err = roster(&files).unwrap_err().to_string();
    assert!(err.contains("no [test_inputs].filename"), "{err}");
}

/// `[test_inputs]` holds exactly the rows the roster renders, so no input reads
/// like data a task is graded on when nothing loads it. It carried
/// `weekend_fixed` / `weekend_transient` answer arrays, a timeline, a file
/// list and image pairs that no code had read since the Python runtime died —
/// the weekend pair in the very bare-array shape W5 retired.
#[test]
fn the_inputs_file_holds_only_the_rows_the_roster_reads() {
    let text = std::fs::read_to_string(conf_dir().join("eval_inputs.toml")).unwrap();
    let val: toml::Value = toml::from_str(&text).unwrap();
    let mut keys: Vec<&str> = val["test_inputs"]
        .as_table()
        .expect("[test_inputs]")
        .keys()
        .map(String::as_str)
        .collect();
    keys.sort_unstable();
    // `filename` IS read: `missing_inputs_are_errors_not_guesses` above is the
    // roster refusing to load without it.
    assert_eq!(keys, ["filename"], "rows the roster never reads");
}
