//! `validate_json` scoring: count, validity, and source-ratio bands.

use serde_json::{json, Value};

use crate::ztools::eval::validators::json_validator::*;

use super::support::{detailed_items, SRC_FULL, SRC_LOW, SRC_MED, SRC_NONE};

#[test]
fn test_validate_json_scalar_input_has_no_items() {
    assert_eq!(
        validate_json(&json!(42), ""),
        (0, "no items found".to_string())
    );
}

#[test]
fn test_validate_json_count_bands() {
    let mk = |n: usize| -> Value {
        let items: Vec<Value> = (0..n)
            .map(|i| json!({"name": format!("Item number {}", i)}))
            .collect();
        json!({"activities": items})
    };
    // 6 items: structure 20 + ok-count 15 + validity 30
    assert_eq!(validate_json(&mk(6), ""), (65, "".to_string()));
    // 3 items: too-few failure, structure 20 + validity 30
    let (score, reason) = validate_json(&mk(3), "");
    assert_eq!(score, 50);
    assert_eq!(reason, "only 3 items (need 8+)");
}

#[test]
fn test_validate_json_validity_bands() {
    let mixed = |invalid: usize| -> Value {
        let mut items: Vec<Value> = (0..8 - invalid)
            .map(|i| json!({"name": format!("Item number {}", i)}))
            .collect();
        for _ in 0..invalid {
            items.push(json!(99));
        }
        json!({"activities": items})
    };
    // 7/8 valid = 0.875 >= 0.7 -> half validity weight + failure note
    let (score, reason) = validate_json(&mixed(1), "");
    assert_eq!(score, 20 + 25 + 15);
    assert_eq!(reason, "only 7/8 items are valid");
    // 3/8 valid = 0.375 < 0.7 -> no validity points
    let (score, reason) = validate_json(&mixed(5), "");
    assert_eq!(score, 20 + 25);
    assert_eq!(reason, "only 3/8 items are valid");
}

#[test]
fn test_validate_json_source_ratio_bands() {
    let data = json!({"activities": detailed_items()});
    // ratio 1.0 -> +25; total caps at exactly MAX_SCORE
    assert_eq!(validate_json(&data, SRC_FULL), (100, "".to_string()));
    // 5/8 match = 0.625 -> +12
    assert_eq!(
        validate_json(&data, SRC_MED),
        (20 + 25 + 30 + 12, "".to_string())
    );
    // 2/8 match = 0.25 -> +6
    assert_eq!(
        validate_json(&data, SRC_LOW),
        (20 + 25 + 30 + 6, "".to_string())
    );
    // ratio 0 -> hallucinated
    let (score, reason) = validate_json(&data, SRC_NONE);
    assert_eq!(score, 75);
    assert_eq!(reason, "not from input (hallucinated)");
}
