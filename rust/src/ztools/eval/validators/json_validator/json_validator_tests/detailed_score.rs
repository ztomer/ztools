//! `validate_detailed_json` scoring: detail bands, caps, and failure truncation.

use serde_json::{json, Value};

use crate::ztools::eval::validators::json_validator::*;

use super::support::{detailed_items, SRC_FULL, SRC_LOW, SRC_MED, SRC_NONE};

#[test]
fn test_validate_detailed_json_clean_full_score() {
    let data = json!({"fixed_activities": detailed_items()});
    // structure 15 + count-good 15 + quality 40 + unique bonus 25, no source given
    assert_eq!(validate_detailed_json(&data, ""), (95, "".to_string()));
}

#[test]
fn test_validate_detailed_json_detail_bands() {
    // most-have-details band (5/6 >= 0.8): +32 quality, no failure recorded
    let mut items = detailed_items();
    items.truncate(5);
    items.push(json!({"name": "Plain Item"}));
    let data = json!({"fixed_activities": items});
    assert_eq!(
        validate_detailed_json(&data, ""),
        (15 + 10 + 32 + 25, "".to_string())
    );

    // some-but-not-most band (3/6 < 0.8): no quality points + explicit failure
    let mut items = detailed_items();
    items.truncate(6);
    items[3] = json!({"name": "Bare Three"});
    items[4] = json!({"name": "Bare Four"});
    items[5] = json!({"name": "Bare Five"});
    let data = json!({"fixed_activities": items});
    let (score, reason) = validate_detailed_json(&data, "");
    assert_eq!(score, (15 + 10) + 25);
    assert_eq!(reason, "only 3/6 have details");
}

#[test]
fn test_validate_detailed_json_zero_details_failure() {
    let items: Vec<Value> = ["Aurora", "Borealis", "Cascade", "Driftwood", "Ember"]
        .iter()
        .map(|n| json!({"name": n}))
        .collect();
    let data = json!({"fixed_activities": items});
    // structure 15 + count-ok 10 + no quality + unique 25
    let (score, reason) = validate_detailed_json(&data, "");
    assert_eq!(score, 50);
    assert!(
        reason.contains("no items with details"),
        "reason was: {reason}"
    );
}

#[test]
fn test_validate_detailed_json_duplicate_penalty_replaces_unique_bonus() {
    let mut items = detailed_items();
    items.push(items[0].clone()); // 9 rows, 8 unique names
    let data = json!({"fixed_activities": items});
    // duplicate ratio 1/9 = 11.1% > 10% -> penalty floor(0.111*20)=2, no unique bonus
    let (score, reason) = validate_detailed_json(&data, "");
    assert_eq!(score, 15 + 15 + 40 - 2);
    assert!(reason.contains("duplicates (11%)"), "reason was: {reason}");
}

#[test]
fn test_validate_detailed_json_all_four_source_caps() {
    let data = json!({"fixed_activities": detailed_items()});
    // raw score with no source: 15+15+40+25 = 95; source weight adds up to +30
    // ratio >= 0.8: raw 125 capped at MAX_SCORE_HIGH_SOURCE (100)
    let (score, reason) = validate_detailed_json(&data, SRC_FULL);
    assert_eq!(score, 100);
    assert!(reason.is_empty(), "reason was: {reason}");
    // ratio 0.625: raw 110 capped at 85
    assert_eq!(
        validate_detailed_json(&data, SRC_MED),
        (MAX_SCORE_MED_SOURCE, "".to_string())
    );
    // ratio 0.25: raw 102 capped at 70
    assert_eq!(
        validate_detailed_json(&data, SRC_LOW),
        (MAX_SCORE_LOW_SOURCE, "".to_string())
    );
    // ratio 0.0: raw 95 capped at 50
    let (score, reason) = validate_detailed_json(&data, SRC_NONE);
    assert_eq!(score, MAX_SCORE_NO_SOURCE);
    assert_eq!(reason, "not from input (hallucinated)");
}

#[test]
fn test_validate_detailed_json_constant_column_cap() {
    let items: Vec<Value> = [
        ("Unique One", "12 Alpha St"),
        ("Unique Two", "34 Beta Ave"),
        ("Unique Three", "56 Gamma Rd"),
    ]
    .iter()
    .map(|(name, _loc)| json!({"name": name, "price": "$5", "target_ages": "all"}))
    .collect();
    let data = json!({"fixed_activities": items});
    // raw 15+40+25=80, constant columns cap at 55
    let (score, reason) = validate_detailed_json(&data, "");
    assert_eq!(score, 55);
    assert!(
        reason.contains("constant across every row"),
        "reason was: {reason}"
    );
    assert!(reason.contains("price"), "reason was: {reason}");
}

#[test]
fn test_validate_detailed_json_item_count_caps_do_not_lower_good_scores() {
    let all = detailed_items();
    // <5 items: cap is 85, raw 80 stays; too-few count failure is recorded
    let few: Vec<Value> = all.iter().take(4).cloned().collect();
    assert_eq!(
        validate_detailed_json(&json!({"fixed_activities": few}), ""),
        (80, "only 4 items (need 8+)".to_string())
    );
    // 5..7 items: cap is 95, raw 90 stays
    let some: Vec<Value> = all.iter().take(6).cloned().collect();
    assert_eq!(
        validate_detailed_json(&json!({"fixed_activities": some}), ""),
        (90, "".to_string())
    );
}

#[test]
fn test_validate_detailed_json_truncates_failures_to_three() {
    let item = json!({"name": "Same"});
    let data = json!({"fixed_activities": [item.clone(), item.clone(), item]});
    let (score, reason) = validate_detailed_json(&data, "unrelated text here");
    // 15 structure, no quality, dup penalty floor(2/3*20)=13, near-dup and
    // no-source caps clamp far above the actual value of 2
    assert_eq!(score, 2);
    assert_eq!(
        reason,
        "only 3 items (need 8+); no items with details; duplicates (66%)"
    );
    assert!(!reason.contains("hallucinated"), "reason was: {reason}");
}

#[test]
fn test_validate_detailed_json_non_object_items_get_empty_names() {
    // strings keep their text as the name; other scalar types contribute an
    // empty name, which counts toward the duplicate ratio
    let data = json!({"activities": ["Solo", 7]});
    let (score, reason) = validate_detailed_json(&data, "");
    assert_eq!(score, 15 - 10);
    assert_eq!(
        reason,
        "only 2 items (need 8+); no items with details; duplicates (50%)"
    );
}
