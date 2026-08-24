//! The three public validators: plain JSON, detailed JSON, and signal/noise.
//!
//! Split out of json_validator.rs for the 500-line production cap. These are the
//! entry points; everything they call lives in the sibling modules.

use serde_json::Value;
use std::collections::HashSet;

use super::names::_names_match;
use super::super::contract::{parse_signal_noise, requested_item_count};
use super::super::defects::{
    constant_column_ratio, generic_location_ratio, near_duplicate_ratio, CONSTANT_COLUMN_LIMIT,
    CONSTANT_COLUMN_MAX_SCORE, GENERIC_LOCATION_LIMIT, GENERIC_LOCATION_MAX_SCORE,
    NEAR_DUPLICATE_LIMIT, NEAR_DUPLICATE_MAX_SCORE,
};

use super::items::{extract_list_from_dict, has_item_details, is_valid_list_item};
use super::source::check_source_extraction;
use super::weights::*;

pub fn validate_json(data: &Value, source_text: &str) -> (i64, String) {
    let items = extract_list_from_dict(data);
    if items.is_empty() {
        return (0, "no items found".to_string());
    }
    let mut score = JSON_STRUCTURE_WEIGHT;
    let mut failures = Vec::new();

    if items.len() >= MIN_ITEMS_GOOD {
        score += JSON_COUNT_GOOD;
    } else if items.len() >= MIN_ITEMS_OK {
        score += JSON_COUNT_OK;
    } else {
        failures.push(format!(
            "only {} items (need {}+)",
            items.len(),
            MIN_ITEMS_GOOD
        ));
    }

    let valid_items = items.iter().filter(|i| is_valid_list_item(i)).count();
    if valid_items == items.len() {
        score += JSON_VALIDITY_WEIGHT;
    } else if valid_items as f64 >= items.len() as f64 * JSON_VALIDITY_THRESHOLD {
        score += JSON_COUNT_OK;
        failures.push(format!(
            "only {}/{} items are valid",
            valid_items,
            items.len()
        ));
    } else {
        failures.push(format!(
            "only {}/{} items are valid",
            valid_items,
            items.len()
        ));
    }

    if !source_text.is_empty() && !items.is_empty() {
        let source_ratio = check_source_extraction(&items, source_text);
        if source_ratio >= 0.8 {
            score += JSON_SOURCE_WEIGHT;
        } else if source_ratio >= 0.5 {
            score += JSON_SOURCE_WEIGHT / 2;
        } else if source_ratio > 0.0 {
            score += JSON_SOURCE_WEIGHT / 4;
        } else {
            failures.push("not from input (hallucinated)".to_string());
        }
    }

    (score.min(MAX_SCORE), failures.join("; "))
}

pub fn validate_detailed_json(data: &Value, source_text: &str) -> (i64, String) {
    let items = extract_list_from_dict(data);
    if items.is_empty() {
        return (0, "no items found".to_string());
    }
    let mut score = DETAILED_STRUCTURE_WEIGHT;
    let mut failures = Vec::new();

    if items.len() >= MIN_ITEMS_GOOD {
        score += DETAILED_COUNT_GOOD;
    } else if items.len() >= MIN_ITEMS_OK {
        score += DETAILED_COUNT_OK;
    } else {
        failures.push(format!(
            "only {} items (need {}+)",
            items.len(),
            MIN_ITEMS_GOOD
        ));
    }

    let valid_with_details = items.iter().filter(|i| has_item_details(i)).count();
    let all_have_details = valid_with_details == items.len();
    let most_have_details = valid_with_details as f64 >= items.len() as f64 * 0.8;

    if all_have_details {
        score += DETAILED_QUALITY_WEIGHT;
    } else if most_have_details {
        score += DETAILED_QUALITY_WEIGHT * 8 / 10;
    } else if valid_with_details == 0 {
        failures.push("no items with details".to_string());
    } else {
        failures.push(format!(
            "only {}/{} have details",
            valid_with_details,
            items.len()
        ));
    }

    let names: Vec<String> = items
        .iter()
        .map(|i| match i {
            Value::Object(m) => m
                .get("name")
                .map(|v| v.to_string().trim_matches('"').to_string())
                .unwrap_or_default(),
            Value::String(s) => s.clone(),
            _ => String::new(),
        })
        .collect();
    let unique_names: HashSet<String> = names.iter().filter(|n| !n.is_empty()).cloned().collect();
    if unique_names.len() < names.len() {
        let duplicate_ratio = (names.len() - unique_names.len()) as f64 / names.len() as f64;
        if duplicate_ratio > 0.1 {
            score -= (duplicate_ratio * 20.0) as i64;
            failures.push(format!(
                "duplicates ({}%)",
                (duplicate_ratio * 100.0) as i64
            ));
        }
    } else {
        score += JSON_QUALITY_WEIGHT;
    }

    let mut source_ratio = 0.0;
    if !source_text.is_empty() && !items.is_empty() {
        source_ratio = check_source_extraction(&items, source_text);
        if source_ratio >= 0.8 {
            score += DETAILED_SOURCE_WEIGHT;
        } else if source_ratio >= 0.5 {
            score += DETAILED_SOURCE_WEIGHT / 2;
        } else if source_ratio > 0.0 {
            score += DETAILED_SOURCE_WEIGHT / 4;
        } else {
            failures.push("not from input (hallucinated)".to_string());
        }
    }

    let generic = generic_location_ratio(&items);
    if generic >= GENERIC_LOCATION_LIMIT {
        failures.push(format!(
            "{}% of locations are generic placeholders",
            (generic * 100.0) as i64
        ));
        score = score.min(GENERIC_LOCATION_MAX_SCORE);
    }

    let (constant_ratio, constant_names) = constant_column_ratio(&items);
    if constant_ratio >= CONSTANT_COLUMN_LIMIT {
        failures.push(format!(
            "constant across every row: {}",
            constant_names.join(", ")
        ));
        score = score.min(CONSTANT_COLUMN_MAX_SCORE);
    }

    if items.len() < MIN_ITEMS_OK {
        score = score.min(MAX_SCORE - DETAILED_COUNT_GOOD);
    } else if items.len() < MIN_ITEMS_GOOD {
        score = score.min(MAX_SCORE - (DETAILED_COUNT_GOOD - DETAILED_COUNT_OK));
    }

    let near_dupes = near_duplicate_ratio(&items);
    if near_dupes >= NEAR_DUPLICATE_LIMIT {
        failures.push(format!(
            "{}% of rows repeat an earlier venue",
            (near_dupes * 100.0) as i64
        ));
        score = score.min(NEAR_DUPLICATE_MAX_SCORE);
    }

    if !source_text.is_empty() {
        let max_allowed = if source_ratio >= SOURCE_THRESHOLD_HIGH {
            MAX_SCORE_HIGH_SOURCE
        } else if source_ratio >= SOURCE_THRESHOLD_MED {
            MAX_SCORE_MED_SOURCE
        } else if source_ratio >= SOURCE_THRESHOLD_LOW {
            MAX_SCORE_LOW_SOURCE
        } else {
            MAX_SCORE_NO_SOURCE
        };
        let truncated_failures = failures
            .into_iter()
            .take(DETAIL_REQUIRED_FIELDS)
            .collect::<Vec<_>>()
            .join("; ");
        return (score.min(max_allowed), truncated_failures);
    }

    let truncated_failures = failures
        .into_iter()
        .take(DETAIL_REQUIRED_FIELDS)
        .collect::<Vec<_>>()
        .join("; ");
    (score.min(MAX_SCORE), truncated_failures)
}

pub fn validate_mixed_signal(
    data: &Value,
    source_text: &str,
    signal_items: Option<&[String]>,
    noise_items: Option<&[String]>,
) -> (i64, String) {
    let items = extract_list_from_dict(data);
    if items.is_empty() {
        return (0, "no items found".to_string());
    }

    let (parsed_signal, parsed_noise) = if signal_items.is_none() || noise_items.is_none() {
        parse_signal_noise(source_text)
    } else {
        (Vec::new(), Vec::new())
    };

    let signal_set = signal_items.unwrap_or(&parsed_signal);
    let noise_set = noise_items.unwrap_or(&parsed_noise);

    let mut tp = 0;
    let mut fp = 0;

    for item in &items {
        let name = match item {
            Value::Object(map) => map
                .get("name")
                .or_else(|| map.get("event"))
                .or_else(|| map.get("title"))
                .map(|v| v.to_string().trim_matches('"').to_string())
                .unwrap_or_default(),
            Value::String(s) => s.clone(),
            other => other.to_string(),
        };
        if name.is_empty() {
            continue;
        }
        if signal_set.iter().any(|s| _names_match(&name, s)) {
            tp += 1;
        } else if noise_set.iter().any(|n| _names_match(&name, n)) {
            fp += 1;
        }
    }

    let total_signal = signal_set.len();
    let total_noise = noise_set.len();
    let asked_for = requested_item_count(source_text);
    let expected_signal = if let Some(asked) = asked_for {
        if asked < total_signal {
            asked
        } else {
            total_signal
        }
    } else {
        total_signal
    };

    let recall = if expected_signal > 0 {
        (tp as f64 / expected_signal as f64).min(1.0)
    } else {
        1.0
    };
    let precision = if tp + fp > 0 {
        (tp as f64 / (tp + fp) as f64).min(1.0)
    } else if tp == 0 && total_signal == 0 {
        1.0
    } else {
        0.0
    };

    let score = (100.0 * (0.5 * recall + 0.5 * precision)).round() as i64;
    let mut failures = Vec::new();
    if fp > 0 {
        failures.push(format!("included {}/{} noise items", fp, total_noise));
    }
    if tp < expected_signal {
        failures.push(format!(
            "missed {}/{} signal items",
            expected_signal - tp,
            expected_signal
        ));
    }

    (score, failures.join("; "))
}
