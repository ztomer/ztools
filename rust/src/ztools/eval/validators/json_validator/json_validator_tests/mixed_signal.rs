//! `validate_mixed_signal` scoring: signal/noise precision and recall.

use serde_json::json;

use crate::ztools::eval::validators::json_validator::*;

#[test]
fn test_validate_mixed_signal_requested_count_above_total_uses_total() {
    let data = json!({"activities": [{"name": "Alpha"}]});
    let signal = vec!["Alpha".to_string()];
    let empty: Vec<String> = Vec::new();
    // asking for 9 when only 1 signal item exists: target stays at 1
    assert_eq!(
        validate_mixed_signal(&data, "find 9 things", Some(&signal), Some(&empty)),
        (100, "".to_string())
    );
}

#[test]
fn test_validate_mixed_signal_empty_items() {
    assert_eq!(
        validate_mixed_signal(&json!({}), "", None, None),
        (0, "no items found".to_string())
    );
}

#[test]
fn test_validate_mixed_signal_counts_tp_and_fp() {
    let data = json!({"activities": [
        {"name": "Alpha"}, {"name": "Beta"}, {"name": "Gamma"}, {"name": "Delta"}
    ]});
    let signal = vec!["Alpha".to_string(), "Beta".to_string()];
    let noise = vec!["Gamma".to_string()];
    // tp=2, fp=1 -> recall 1.0, precision 2/3 -> round(83.33)=83
    let (score, reason) =
        validate_mixed_signal(&data, "", Some(&signal), Some(&noise));
    assert_eq!(score, 83);
    assert_eq!(reason, "included 1/1 noise items");
}

#[test]
fn test_validate_mixed_signal_requested_count_caps_recall_target() {
    let signal = vec!["A".to_string(), "B".to_string(), "C".to_string()];
    let empty: Vec<String> = Vec::new();
    // prompt asks for 2, all 2 present: recall target is min(asked, total) = 2
    let data = json!({"activities": [{"name": "A"}, {"name": "B"}]});
    assert_eq!(
        validate_mixed_signal(&data, "find 2 things", Some(&signal), Some(&empty)),
        (100, "".to_string())
    );
    // only 1 of the requested 2 present
    let data = json!({"activities": [{"name": "A"}]});
    let (score, reason) =
        validate_mixed_signal(&data, "find 2 things", Some(&signal), Some(&empty));
    assert_eq!(score, 75);
    assert_eq!(reason, "missed 1/2 signal items");
}

#[test]
fn test_validate_mixed_signal_zero_sets_score_full() {
    let data = json!({"activities": [{"name": "Alpha"}]});
    let empty: Vec<String> = Vec::new();
    // tp=fp=0 with no requested signal at all: perfect score by convention
    assert_eq!(
        validate_mixed_signal(&data, "", Some(&empty), Some(&empty)),
        (100, "".to_string())
    );
}

#[test]
fn test_validate_mixed_signal_parses_source_sections() {
    let data = json!({"activities": [
        {"name": "Alpha"}, {"name": "Beta"}, {"name": "Gamma"}
    ]});
    let source = "Signal:\n- Alpha: good one\n- Beta\nNOISE\n- Gamma: known bad\n";
    // parsed signal=[Alpha,Beta] noise=[Gamma]: tp=2 fp=1 -> 83
    let (score, reason) = validate_mixed_signal(&data, source, None, None);
    assert_eq!(score, 83);
    assert_eq!(reason, "included 1/1 noise items");
}

#[test]
fn test_validate_mixed_signal_precision_zero_when_nothing_matches() {
    let data = json!({"activities": [{"name": "Omega"}]});
    let signal = vec!["Alpha".to_string()];
    // tp=0 fp=0 with a non-empty signal set -> precision 0
    let (score, reason) =
        validate_mixed_signal(&data, "nothing relevant", Some(&signal), None);
    assert_eq!(score, 0);
    assert_eq!(reason, "missed 1/1 signal items");
}

#[test]
fn test_validate_mixed_signal_name_extraction_variants() {
    let data = json!({"activities": [
        {"event": "Eve One"},
        {"title": "Tit Two"},
        {"desc": "no name field"},
        {"label": "also nothing"},
        "Str Three",
        99
    ]});
    let signal = vec![
        "Eve One".to_string(),
        "Tit Two".to_string(),
        "Str Three".to_string(),
    ];
    let empty: Vec<String> = Vec::new();
    // event/title fallbacks and string items count; items without a usable
    // name are skipped entirely; non-name matches neither set
    assert_eq!(
        validate_mixed_signal(&data, "", Some(&signal), Some(&empty)),
        (100, "".to_string())
    );
}
