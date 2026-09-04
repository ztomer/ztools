//! Roster entry parsing, scoring, and substitution -- the `roster` submodule's tests.

use crate::ztools::eval::model_resolve::*;

pub(super) fn entry(model: &str, size: &str) -> RosterEntry {
    RosterEntry {
        model: model.to_string(),
        parameter_size: size.to_string(),
    }
}

#[test]
fn missing_model_404_is_recognised_not_other_404s_or_statuses() {
    let body = r#"{"error": {"message": "Model 'X' is not installed or registered"}}"#;
    assert!(is_missing_model_error(404, body));
    assert!(!is_missing_model_error(404, "404 page not found"));
    assert!(!is_missing_model_error(503, "Server is at capacity"));
    assert!(!is_missing_model_error(404, ""));
}

#[test]
fn parameter_billions_parses_b_m_and_garbage() {
    assert_eq!(parameter_billions(&entry("a", "27B")), 27.0);
    assert_eq!(parameter_billions(&entry("a", "4M")), 0.004);
    assert_eq!(parameter_billions(&entry("a", "")), 0.0);
    assert_eq!(parameter_billions(&entry("a", "junk")), 0.0);
}

#[test]
fn empty_roster_and_installed_model_substitute_nothing() {
    let (model, reason) = substitute_model("gone-70b", &[], &default_fallback_chain());
    assert_eq!(model, "gone-70b");
    assert!(reason.is_none());

    let roster = vec![entry("live-model", "8B")];
    let (model, reason) = substitute_model("live-model", &roster, &[]);
    assert_eq!(model, "live-model");
    assert!(reason.is_none());
}

#[test]
fn same_family_prefers_the_largest_installed_model() {
    let roster = vec![
        entry("gemma-4-e2b-it-4bit", "4B"),
        entry("gemma-4-12b-it-mxfp8", "12B"),
    ];
    let (model, reason) =
        substitute_model("gemma-4-99b-it-mxfp8", &roster, &default_fallback_chain());
    assert_eq!(model, "gemma-4-12b-it-mxfp8");
    let reason = reason.expect("substitution happened");
    assert!(
        reason.contains("largest installed 'gemma' model"),
        "{reason}"
    );
}

#[test]
fn no_family_match_falls_through_to_the_preference_chain_then_biggest() {
    let roster = vec![
        entry("qwen3.6-35b-a3b-mxfp8", "35B"),
        entry("laguna-70b", "70B"),
    ];
    // No laguna-family configured name exists; chain names foundation first,
    // then qwopus/qwen/gemma/nemotron/laguna -> qwen wins over laguna by order.
    let (model, _) = substitute_model("ghost-70b", &roster, &default_fallback_chain());
    assert_eq!(model, "qwen3.6-35b-a3b-mxfp8");

    // Chain exhausted: biggest model on the roster.
    let (model, reason) = substitute_model("ghost-70b", &roster, &["nothing-matches"]);
    assert_eq!(model, "laguna-70b");
    let reason = reason.expect("substitution happened");
    assert!(
        reason.contains("nothing in the preference chain"),
        "{reason}"
    );
}

#[test]
fn size_tiebreak_beats_name_sorting() {
    // "qwen3.10" sorts below "qwen3.8" alphabetically but is bigger.
    let roster = vec![entry("qwen3.8-8b", "8B"), entry("qwen3.10-27b", "27B")];
    let (model, _) = substitute_model("qwen-gone", &roster, &[]);
    assert_eq!(model, "qwen3.10-27b");
}

#[test]
fn roster_entry_from_json_accepts_and_rejects_precisely() {
    use serde_json::json;
    let full = json!({"model": "m-7b", "details": {"parameter_size": "7B"}});
    assert_eq!(RosterEntry::from_json(&full), Some(entry("m-7b", "7B")));

    // details absent entirely: the entry still counts, size just unknown.
    let bare = json!({"model": "m"});
    assert_eq!(RosterEntry::from_json(&bare), Some(entry("m", "")));

    // A non-string parameter_size must not poison the entry.
    let numeric = json!({"model": "m", "details": {"parameter_size": 7}});
    assert_eq!(RosterEntry::from_json(&numeric), Some(entry("m", "")));

    // No usable model name: no entry at all.
    assert_eq!(RosterEntry::from_json(&json!({"details": {}})), None);
    assert_eq!(RosterEntry::from_json(&json!({"model": ""})), None);
    assert_eq!(RosterEntry::from_json(&json!({"model": 42})), None);
}

#[test]
fn parameter_billions_k_suffix_unknown_suffixes_and_parse_failures() {
    assert!(
        (parameter_billions(&entry("a", "640K")) - 0.00064).abs() < 1e-12,
        "640K is 0.00064B"
    );
    assert_eq!(
        parameter_billions(&entry("a", "27X")),
        0.0,
        "unknown suffix"
    );
    assert_eq!(
        parameter_billions(&entry("a", "junk!")),
        0.0,
        "non-numeric body"
    );
    assert_eq!(
        parameter_billions(&entry("a", " 12B ")),
        12.0,
        "surrounding whitespace"
    );
    assert_eq!(
        parameter_billions(&entry("a", "12.5B")),
        12.5,
        "fractional sizes"
    );
}

#[test]
fn substitute_model_family_miss_falls_through_to_the_chain() {
    // 'qwen-gone' resolves to the qwen family but no qwen model is served:
    // the family arm finds nothing, so the chain decides.
    let roster = vec![entry("laguna-70b", "70B")];
    let (model, reason) = substitute_model("qwen-gone-27b", &roster, &["laguna"]);
    assert_eq!(model, "laguna-70b");
    let reason = reason.expect("substitution happened");
    assert!(reason.contains("no 'qwen' model is either"), "{reason}");
}

#[test]
fn substitute_model_prefers_earlier_chain_entries_over_bigger_models() {
    let roster = vec![entry("gemma-2b", "2B"), entry("nemotron-90b", "90B")];
    let (model, reason) = substitute_model("ghost-model", &roster, &["gemma", "nemotron"]);
    assert_eq!(model, "gemma-2b", "chain order beats roster size");
    assert!(reason.unwrap().contains("falling back to 'gemma-2b'"));
}
