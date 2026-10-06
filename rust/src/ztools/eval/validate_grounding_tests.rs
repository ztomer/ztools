//! Does the scorer now SEE the difference between reading a file and reading its
//! name?
//!
//! [`legacy_validate_file_summary`] below is the pre-2026-10-05 scorer, copied
//! verbatim from `validate.rs` at HEAD. It is here so the before/after is
//! MEASURED BY CODE rather than by arithmetic: the control assertions run both
//! scorers over the same two answers and pin that the old one could not tell
//! them apart while the new one can. Deleting it once the finding is on the
//! record would leave the claim resting on this file's comments.

use super::super::*;
use super::*;
use crate::ztools::eval::prompts::file_summary::FILE_SUMMARY_FILE_LIST;
use serde_json::json;

/// The listed paths, verbatim, as a model would echo them back.
fn listed() -> Vec<&'static str> {
    FILE_SUMMARY_FILE_LIST.lines().collect()
}

/// What a model emits when it has the paths and NOTHING ELSE: fluent, plausible,
/// name-derived, and carrying the content verbs the old scorer counted.
///
/// Every one of these is longer than `stem + 25` characters, so none of them
/// trips `is_filename_echo`. That is the finding in one list: the echo check has
/// 25 characters of slack, and "parses the manifest and loads metadata" spends it
/// on exactly the elaboration the old prompt asked for.
///
/// Each carries at least one [`CONTENT_VERBS`] entry, so seventeen of seventeen
/// reach the top rung. Five needed a word added to get there, which is itself the
/// point: under the old scorer the ONLY thing separating a guess from a perfect
/// score was whether the sentence happened to contain a word from the list.
fn name_guesses() -> Vec<serde_json::Value> {
    let descs = [
        "documentation covering setup and configuration for the repository",
        "instructions agents load before building or testing this codebase",
        "configuration file with settings and options for the application",
        "configuration file for the weekend planner with place options",
        "configuration file for the twitter client with api options",
        "configuration file for the rename module with model settings",
        "configuration file for the foundation model client settings",
        "configuration file for gemma model settings and options",
        "configuration file for qwen model settings and parameters",
        "documentation of model quirks and best practices",
        "documents how the test suite handles failures and coverage",
        "parses the project manifest and loads package metadata",
        "converts units between systems and validates the ranges",
        "renames image files using a vision model client",
        "database storage layer that saves and loads model records",
        "twitter summarizer module with api client configuration",
        "formats weekend output and handles the colour table",
    ];
    rows(&descs)
}

/// What a competent reader writes after reading the same files' excerpts.
fn content_descriptions() -> Vec<serde_json::Value> {
    let descs = [
        "local CLI tools for Osaurus: summarize a twitter timeline, plan family weekends, rename images",
        "agent instructions for the repo: the 500-line file size limit, the pre-commit hook, the coverage floor",
        "main config: the llm server url, the per-task max tokens budget, the best_models slots and timeouts",
        "weekend planner config: the exclude_places list of venues, and that top-level keys must precede tables",
        "twitter config: the chrome cookies db, the state_file and output_dir, and the max scrolls count",
        "image renamer config: the tesseract command and the mlx models directory",
        "model family config: the 4096 context window and the max tokens cap for the on-device model",
        "gemma family config: field_mapping aliases and per-task prompt templates",
        "qwen family config: field_mapping aliases and per-task prompt templates",
        "measured model quirks: the DuckDuckGo search wall, ddgs and primp, per-model prompt quirks",
        "how this repo tests: golden parity tests against the retired harness, structural gates, coverage rules",
        "path helper that expands a leading tilde against the home directory",
        "narrowing conversions with saturating boundaries: pid, duration to milliseconds, exact u64 to f64",
        "a shim re-exporting the split rename package: mod, helpers and vlm",
        "resolves the store directories and reads the newest dated plan back via weekend_store_dir",
        "module index for the twitter summarizer: browser collector, cookies, capture, endpoints",
        "renders the saved weekend markdown: the two section headings and the two table headers",
    ];
    rows(&descs)
}

fn rows(descs: &[&str]) -> Vec<serde_json::Value> {
    listed()
        .iter()
        .zip(descs)
        .map(|(path, desc)| json!({ "path": path, "desc": desc }))
        .collect()
}

fn answer(items: &[serde_json::Value]) -> String {
    serde_json::to_string(items).unwrap()
}

/// The pre-2026-10-05 scorer, verbatim: `detailed` meant "contains one of
/// [`CONTENT_VERBS`]", and the score was a bucket on `detailed / num_files`.
/// [`CONTENT_VERBS`] and `is_filename_echo` are the live ones — they did not
/// change — so this copies only the counting and the ladder.
fn legacy_validate_file_summary(raw: &str) -> (u8, String) {
    let items: Vec<serde_json::Value> = match serde_json::from_str(raw) {
        Ok(serde_json::Value::Array(items)) => items,
        _ => return (0, "not a list".to_string()),
    };
    let generic_desc = regex::Regex::new(GENERIC_DESC_RE).unwrap();
    let num_files = items.len();
    let mut detailed_count = 0usize;
    let mut echo_count = 0usize;
    for item in items {
        let Some(obj) = item.as_object() else {
            continue;
        };
        let path = obj.get("path").and_then(|v| v.as_str()).unwrap_or("");
        let desc = obj.get("desc").and_then(|v| v.as_str()).unwrap_or("");
        if path.is_empty() || desc.is_empty() {
            continue;
        }
        let desc_lower = desc.to_lowercase();
        let desc_stripped = desc_lower.trim();
        if generic_desc.is_match(desc_stripped) || is_filename_echo(path, desc_stripped) {
            echo_count += 1;
            continue;
        }
        if CONTENT_VERBS.iter().any(|kw| desc_lower.contains(kw)) {
            detailed_count += 1;
        }
    }
    let score = if detailed_count * 10 >= num_files * 8 {
        100
    } else if detailed_count * 2 >= num_files {
        85
    } else if detailed_count >= 2 {
        70
    } else if detailed_count >= 1 {
        50
    } else {
        25
    };
    let _ = echo_count;
    (score, String::new())
}

/// THE FINDING, pinned: on the real seventeen-row list, the old scorer ranked a
/// pure name-based guess ABOVE a description written from the file content — 100
/// against 85.
///
/// Not "indistinguishable": inverted. Every one of the seventeen guesses contained
/// a content verb, so `detailed_count` was 17/17 and the top rung was earned
/// without reading anything, while seven of the seventeen TRUTHFUL sentences
/// ("path helper that expands a leading tilde against the home directory") name
/// no verb from the list at all and cost 15 points each. The metric and the
/// instruction the prompt was built around pointed in opposite directions, and
/// the metric was the one feeding `[best_models]`.
#[test]
fn the_old_scorer_ranked_a_fabrication_above_the_truth() {
    let guesses = answer(&name_guesses());
    let read = answer(&content_descriptions());
    let (guess_score, _) = legacy_validate_file_summary(&guesses);
    let (read_score, _) = legacy_validate_file_summary(&read);
    assert_eq!(guess_score, 100, "a name-based guess scored full marks");
    assert_eq!(
        read_score, 85,
        "the truthful answer lost fifteen points to it"
    );
    assert!(
        guess_score > read_score,
        "the inversion is the finding: guess {guess_score} beat the truth {read_score}"
    );
}

/// THE FIX, pinned: the same two answers through the live scorer. The grounded
/// answer must now win by a margin the old one could not express, and the reason
/// string must say which descriptions it could not support.
#[test]
fn a_name_based_guess_is_now_beaten_by_a_description_read_from_the_content() {
    let guesses = answer(&name_guesses());
    let read = answer(&content_descriptions());

    let (guess_score, guess_reason) = validate_file_summary(&guesses);
    let (read_score, read_reason) = validate_file_summary(&read);

    assert!(
        read_score > guess_score,
        "grounded {read_score} ({read_reason}) must beat name-derived {guess_score} \
         ({guess_reason})"
    );
    assert!(
        guess_reason.contains("not supported by the file's content"),
        "the score must SAY that the guesses were ungrounded, got: {guess_reason}"
    );
    assert_eq!(
        read_reason, "",
        "a fully grounded answer reports nothing: {read_reason}"
    );
}

/// Per-row, so a failure names the row instead of a total. This is the control
/// that stops the total from hiding one row that no name guess could ever clear
/// and one row every guess accidentally cleared.
#[test]
fn every_row_grounds_on_its_content_and_none_on_its_name() {
    let guesses = name_guesses();
    let read = content_descriptions();
    let mut cleared_by_name = Vec::new();
    let mut missed_by_read = Vec::new();
    for (i, path) in listed().iter().enumerate() {
        let guess_desc = guesses[i]["desc"].as_str().unwrap().to_lowercase();
        let read_desc = read[i]["desc"].as_str().unwrap().to_lowercase();
        let facts = facts_for(path).unwrap_or_else(|| panic!("no pinned truth for {path}"));
        if grounded(facts, &guess_desc) {
            cleared_by_name.push(path.to_string());
        }
        if !grounded(facts, &read_desc) {
            missed_by_read.push((path.to_string(), read_desc.clone()));
        }
    }
    assert_empty!(
        &cleared_by_name,
        "these rows are cleared by a name-derived sentence, so they measure nothing"
    );
    assert_empty!(
        &missed_by_read,
        "these rows are NOT cleared by a description written from the content: {:?}",
        missed_by_read
    );
}

/// The brief's minimum bar: a description that is only a restatement of the path
/// must not score like one grounded in content.
#[test]
fn a_filename_echo_never_reaches_a_grounded_description_score() {
    let paths = listed();
    let read = content_descriptions();
    let mut full = Vec::new();
    for (path, item) in paths.iter().zip(&read) {
        full.push(json!({ "path": path, "desc": item["desc"] }));
    }
    let (grounded_score, _) = validate_file_summary(&answer(&full));

    // Every row replaced by its own stem re-spaced: the echo, spelled out.
    let mut echoed = Vec::new();
    for path in &paths {
        let stem = std::path::Path::new(path)
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap()
            .replace('_', " ");
        echoed.push(json!({ "path": path, "desc": stem }));
    }
    let (echo_score, echo_reason) = validate_file_summary(&answer(&echoed));

    assert_eq!(
        echo_score, 25,
        "a pure filename restatement scores the floor, got {echo_score} ({echo_reason})"
    );
    assert!(
        echo_score + 50 <= grounded_score,
        "the echo must be far below a description read from the content: \
         {echo_score} vs {grounded_score}"
    );
}

/// The ground truth is keyed by row and covers the list EXACTLY: a listed file
/// with no pinned truth has nothing to be graded against and would silently fall
/// back to the verb heuristic; a pinned row that left the list is dead weight in
/// the scorer and a lie in the header.
#[test]
fn the_pinned_truth_covers_the_listed_rows_exactly() {
    assert_eq!(
        pinned_rows(),
        listed()
            .iter()
            .map(|p| {
                p.rsplit_once("/Users/ztomer/Projects/ztools/")
                    .unwrap_or_else(|| panic!("{p} is not repo-absolute"))
                    .1
            })
            .collect::<Vec<_>>(),
        "FILE_SUMMARY_GROUND_TRUTH must be the file list, row for row and in order"
    );
    assert_eq!(pinned_rows().len(), 17);
}

/// A basename resolves without its directory (`units.rs`), which is only
/// unambiguous while every basename is unique. A second `mod.rs` would silently
/// resolve to whichever row came first.
#[test]
fn every_basename_is_unique() {
    let mut seen = std::collections::HashSet::new();
    for row in pinned_rows() {
        let base = std::path::Path::new(row)
            .file_name()
            .and_then(|b| b.to_str())
            .expect("every pinned row names a file");
        assert!(
            seen.insert(base.to_string()),
            "two pinned rows share the basename {base}: a bare-basename answer \
             cannot be resolved, so matching it would pick whichever came first"
        );
    }
}

/// ROT GATE. A pinned phrase that no longer occurs in its file would leave the
/// task scoring a description against a truth the repo has moved on from — and
/// nothing else in the tree would notice.
#[test]
fn every_pinned_truth_still_appears_in_its_file() {
    use crate::ztools::eval::validators::text_match::identifying_tokens;
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .to_path_buf();
    for (rel, facts) in FILE_SUMMARY_GROUND_TRUTH {
        let text = std::fs::read_to_string(root.join(rel)).unwrap_or_else(|e| panic!("{rel}: {e}"));
        let haystack = text.to_lowercase();
        for fact in *facts {
            let tokens = identifying_tokens(fact);
            assert_nonempty!(
                &tokens,
                "{rel}: the phrase {fact:?} has no identifying token"
            );
            let hits = tokens.iter().filter(|t| haystack.contains(*t)).count();
            assert!(
                hits * 2 >= tokens.len(),
                "{rel}: only {hits}/{} identifying tokens of {fact:?} still occur in the \
                 file ({tokens:?}) -- the pinned truth has drifted from the file, so the \
                 scorer is grading against something the repo no longer says",
                tokens.len()
            );
        }
    }
}

/// THE RULE IS NOT VACUOUS, and this is the control for it.
///
/// `grounded` asks for half a row's phrases. Every shipped row carries exactly
/// two, so on the real list `covered * 2 >= 2` and `covered >= 1` are the same
/// predicate — and the seventeen-row controls therefore CANNOT tell them apart
/// (calibrated: relaxing the rule to `covered >= 1` leaves all seven green).
/// That is a property of the shipped table, not of the rule, so the rule's
/// generality is asserted on rows of four instead. It is what makes adding a third
/// phrase to a row a real decision rather than a free one.
#[test]
fn a_row_needs_half_its_phrases_not_merely_one() {
    let four: &[&str] = &["alpha beta", "gamma delta", "epsilon zeta", "eta theta"];
    assert!(
        !grounded(four, "alpha beta and nothing else"),
        "one of four"
    );
    assert!(
        grounded(four, "alpha beta plus gamma delta"),
        "two of four is half"
    );
    // A row of one is cleared by that one: the rule must not demand a phrase a
    // single-phrase row does not have.
    assert!(grounded(&["alpha beta"], "alpha beta"));
    assert!(!grounded(&["alpha beta"], "gamma delta"));
    // And no phrase at all is never cleared, rather than vacuously true.
    assert!(!grounded(&[], "anything"));
}

/// What the instrument is and is not, asserted rather than assumed.
///
/// `grounded` measures LEXICAL grounding: does the description name what only
/// this file says. That is what the anti-inference rule is about, and it is what a
/// name-based guess cannot fake. It is not entailment — a description that names
/// the truth and then contradicts it still clears, and this pins that limit
/// instead of leaving it to be discovered. Catching the contradiction needs a
/// per-row FORBIDDEN list, which is a second hand-pinned table per file and a
/// different validator; it is not claimed here.
#[test]
fn grounding_is_lexical_and_therefore_blind_to_a_contradiction() {
    let facts: &[&str] = &["leading tilde", "home directory"];
    assert!(!grounded(facts, "path helper for the project manifest"));
    assert!(grounded(facts, "expands a leading tilde against the home"));
    // The stated limit: naming it and then denying it still counts as grounded.
    assert!(
        grounded(
            facts,
            "expands a leading tilde, which has nothing to do with the home directory"
        ),
        "lexical grounding cannot see a contradiction; this is the documented limit, \
         not a claim that the answer is right"
    );
}

/// The lookup, pinned shape by shape.
///
/// The roster's prompt lists ABSOLUTE paths, but what comes back is whatever the
/// model chose to echo: the listed path, a repo-relative one, a bare basename, or
/// the same with `./` and a trailing colon from a `## path: summary` header. A
/// resolution rule that quietly stopped accepting any of those would leave every
/// listed row falling back to the verb heuristic — and because the fallback still
/// produces a score, nothing would go red. Calibrated: removing the bare-basename
/// arm leaves every other control in this file green, which is why it is asserted
/// here.
///
/// An UNLISTED path and an empty path must both resolve to nothing, for the same
/// reason: `Check::FileSummary` and the smoke fixtures name other repos' files.
#[test]
fn a_row_resolves_from_every_shape_a_model_echoes_and_from_nothing_else() {
    for spelled in [
        "/Users/ztomer/Projects/ztools/rust/src/units.rs",
        "rust/src/units.rs",
        "./rust/src/units.rs",
        "units.rs",
        "UNITS.RS",
        "/Users/ztomer/Projects/ztools/rust/src/units.rs:",
        "  rust/src/units.rs  ",
    ] {
        assert_eq!(
            facts_for(spelled),
            Some(&["saturating conversions", "whole milliseconds"][..]),
            "{spelled:?} must resolve to its row's truth"
        );
    }
    for other in [
        "",
        "   ",
        "rust/src/units.py",
        "some/other/manifest.rs",
        "units",
    ] {
        assert_eq!(facts_for(other), None, "{other:?} is not a listed row");
    }
}
