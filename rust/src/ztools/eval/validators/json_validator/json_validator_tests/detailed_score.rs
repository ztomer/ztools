//! `validate_detailed_json` scoring: detail bands, caps, and failure truncation.

use serde_json::{json, Value};

use crate::ztools::eval::validators::json_validator::*;

use super::support::{detailed_items, SRC_FULL, SRC_MED, SRC_NONE};

#[test]
fn test_validate_detailed_json_clean_full_score() {
    let data = json!({"fixed_activities": detailed_items()});
    // structure 15 + count-ok 10 + quality 40 + unique bonus 20, no source given
    assert_eq!(validate_detailed_json(&data, ""), (85, String::new()));
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
        (15 + 10 + 32 + 20, String::new())
    );

    // some-but-not-most band (3/6 < 0.8): no quality points + explicit failure
    let mut items = detailed_items();
    items.truncate(6);
    items[3] = json!({"name": "Bare Three"});
    items[4] = json!({"name": "Bare Four"});
    items[5] = json!({"name": "Bare Five"});
    let data = json!({"fixed_activities": items});
    let (score, reason) = validate_detailed_json(&data, "");
    assert_eq!(score, (15 + 10) + 20);
    assert_eq!(reason, "only 3/6 have details");
}

#[test]
fn test_validate_detailed_json_zero_details_failure() {
    let items: Vec<Value> = ["Aurora", "Borealis", "Cascade", "Driftwood", "Ember"]
        .iter()
        .map(|n| json!({"name": n}))
        .collect();
    let data = json!({"fixed_activities": items});
    // structure 15 + count-ok 10 + no quality + unique 20
    let (score, reason) = validate_detailed_json(&data, "");
    assert_eq!(score, 45);
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
    assert_eq!(score, 15 + 10 + 40 - 2);
    assert!(reason.contains("duplicates (11%)"), "reason was: {reason}");
}

#[test]
fn test_validate_detailed_json_all_four_source_caps() {
    let data = json!({"fixed_activities": detailed_items()});
    // raw score with no source: 15+10+40+20 = 85; source weight adds up to +30
    // ratio >= 0.8: raw 115 capped by the item-count cap (8 < 10) at 95
    let (score, reason) = validate_detailed_json(&data, SRC_FULL);
    assert_eq!(score, 95);
    assert!(reason.is_empty(), "reason was: {reason}");
    // ratio 0.625: raw 100 capped by the item-count cap at 95, then at
    // MAX_SCORE_MED_SOURCE by the source cap
    assert_eq!(
        validate_detailed_json(&data, SRC_MED),
        (MAX_SCORE_MED_SOURCE, String::new())
    );
    // ratio 0.375: the shared SRC_LOW grounds 2/8 = 0.25, which falls below
    // the 0.3 low threshold into NO_SOURCE — so the LOW cap needs its own
    // three-item source here.
    assert_eq!(
        validate_detailed_json(&data, "kappa zeta lambda mu nu xi quiet river stones"),
        (MAX_SCORE_LOW_SOURCE, String::new())
    );
    // ratio 0.0: raw 85 capped by the item-count cap at 85, then at
    // MAX_SCORE_NO_SOURCE by the source cap
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
    // <5 items: cap is 85, raw 75 stays; too-few count failure is recorded
    let few: Vec<Value> = all.iter().take(4).cloned().collect();
    assert_eq!(
        validate_detailed_json(&json!({"fixed_activities": few}), ""),
        (75, "only 4 items (need 10+)".to_string())
    );
    // 5..9 items: cap is 95, raw 85 stays
    let some: Vec<Value> = all.iter().take(6).cloned().collect();
    assert_eq!(
        validate_detailed_json(&json!({"fixed_activities": some}), ""),
        (85, String::new())
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
        "only 3 items (need 10+); no items with details; duplicates (66%)"
    );
    assert!(!reason.contains("hallucinated"), "reason was: {reason}");
}

#[test]
fn test_validate_detailed_json_scalar_names_mirror_python_str() {
    // Python names non-dict scalars with str(item), so the number 7 counts
    // as the name "7" for uniqueness: 15 + 0 + 0 + 20 = 35. The old test
    // pinned empty names (score 5), which no Python input can produce.
    let data = json!({"activities": ["Solo", 7]});
    let (score, reason) = validate_detailed_json(&data, "");
    assert_eq!(score, 35);
    assert_eq!(reason, "only 2 items (need 10+); no items with details");
}
