//! JSON structure, details, validity, and signal/noise validation.
//!
//! Port of `lib/validators/json_validator.py`.

use serde_json::Value;
use std::collections::HashSet;
use std::sync::LazyLock;

use super::contract::{parse_signal_noise, requested_item_count};
use super::defects::{
    constant_column_ratio, generic_location_ratio, near_duplicate_ratio, CONSTANT_COLUMN_LIMIT,
    CONSTANT_COLUMN_MAX_SCORE, GENERIC_LOCATION_LIMIT, GENERIC_LOCATION_MAX_SCORE,
    NEAR_DUPLICATE_LIMIT, NEAR_DUPLICATE_MAX_SCORE,
};

pub const MAX_SCORE: i64 = 100;
pub const MIN_ITEMS_GOOD: usize = 8;
pub const MIN_ITEMS_OK: usize = 5;
pub const JSON_STRUCTURE_WEIGHT: i64 = 20;
pub const JSON_COUNT_GOOD: i64 = 25;
pub const JSON_COUNT_OK: i64 = 15;
pub const JSON_VALIDITY_WEIGHT: i64 = 30;
pub const JSON_VALIDITY_THRESHOLD: f64 = 0.7;
pub const JSON_SOURCE_WEIGHT: i64 = 25;

pub const DETAILED_STRUCTURE_WEIGHT: i64 = 15;
pub const DETAILED_COUNT_GOOD: i64 = 15;
pub const DETAILED_COUNT_OK: i64 = 10;
pub const DETAILED_QUALITY_WEIGHT: i64 = 40;
pub const DETAILED_SOURCE_WEIGHT: i64 = 30;
pub const JSON_QUALITY_WEIGHT: i64 = 25;
pub const DETAIL_REQUIRED_FIELDS: usize = 3;

pub const SOURCE_THRESHOLD_HIGH: f64 = 0.8;
pub const SOURCE_THRESHOLD_MED: f64 = 0.5;
pub const SOURCE_THRESHOLD_LOW: f64 = 0.2;
pub const MAX_SCORE_HIGH_SOURCE: i64 = 100;
pub const MAX_SCORE_MED_SOURCE: i64 = 85;
pub const MAX_SCORE_LOW_SOURCE: i64 = 70;
pub const MAX_SCORE_NO_SOURCE: i64 = 50;

static STOPWORDS: LazyLock<HashSet<&'static str>> = LazyLock::new(|| {
    [
        "the", "and", "for", "with", "this", "that", "from", "are", "was", "has", "have", "but",
        "not", "you", "all", "can", "her", "his", "had", "they", "been", "will", "would", "could",
        "what", "when", "where", "who", "which", "why", "how",
    ]
    .into_iter()
    .collect()
});

pub const DETAIL_FIELDS: &[&str] = &[
    "name",
    "event",
    "title",
    "activity",
    "place",
    "location",
    "venue",
    "address",
    "where",
    "day",
    "date",
    "when",
    "time",
    "duration",
    "target_ages",
    "age_group",
    "ages",
    "audience",
    "who",
    "price",
    "cost",
    "pricing",
    "weather",
    "type",
    "indoor_outdoor",
    "setting",
    "desc",
    "description",
];

pub fn extract_list_from_dict(data: &Value) -> Vec<Value> {
    if let Value::Object(map) = data {
        let preferred_keys = [
            "activities",
            "items",
            "results",
            "data",
            "fixed_activities",
            "transient_events",
            "events",
            "places",
            "venues",
            "recommendations",
        ];
        for key in preferred_keys {
            if let Some(Value::Array(arr)) = map.get(key) {
                return arr.clone();
            }
        }
        for val in map.values() {
            if val.is_object() {
                let found = extract_list_from_dict(val);
                if !found.is_empty() {
                    return found;
                }
            }
        }
        let mut best: Vec<Value> = Vec::new();
        for val in map.values() {
            if let Value::Array(arr) = val {
                if arr.len() > best.len() {
                    best = arr.clone();
                }
            }
        }
        return best;
    }
    if let Value::Array(arr) = data {
        return arr.clone();
    }
    Vec::new()
}

pub fn is_valid_list_item(item: &Value) -> bool {
    if let Value::String(s) = item {
        return !s.trim().is_empty();
    }
    if let Value::Object(map) = item {
        let valid_fields = [
            "name", "activity", "event", "title", "place", "path", "desc",
        ];
        return valid_fields.iter().any(|f| {
            map.get(*f)
                .map(|v| !v.to_string().trim_matches('"').trim().is_empty())
                .unwrap_or(false)
        });
    }
    false
}

pub fn has_item_details(item: &Value) -> bool {
    let map = match item.as_object() {
        Some(m) => m,
        None => return false,
    };
    let name_fields = ["name", "event", "title", "activity", "place", "path"];
    let has_name = name_fields.iter().any(|f| {
        map.get(*f)
            .map(|v| !v.to_string().trim_matches('"').trim().is_empty())
            .unwrap_or(false)
    });
    if !has_name {
        return map.len() >= 2;
    }
    for field in DETAIL_FIELDS {
        if !name_fields.contains(field) {
            if let Some(val) = map.get(*field) {
                if !val.to_string().trim_matches('"').trim().is_empty() {
                    return true;
                }
            }
        }
    }
    map.len() >= 2
}

pub fn _norm_name(name: &str) -> String {
    let cleaned = name
        .chars()
        .filter(|c| c.is_alphanumeric() || *c == ' ')
        .collect::<String>();
    cleaned
        .to_lowercase()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

pub fn _name_tokens(name: &str) -> HashSet<String> {
    let norm = name
        .chars()
        .map(|c| {
            if c.is_alphanumeric() {
                c.to_ascii_lowercase()
            } else {
                ' '
            }
        })
        .collect::<String>();
    norm.split_whitespace()
        .filter(|t| t.len() >= 3 && !STOPWORDS.contains(t))
        .map(|t| t.to_string())
        .collect()
}

pub fn _names_match(a: &str, b: &str) -> bool {
    let na = _norm_name(a);
    let nb = _norm_name(b);
    if na.is_empty() || nb.is_empty() {
        return false;
    }
    if na.contains(&nb) || nb.contains(&na) {
        return true;
    }
    let ta = _name_tokens(a);
    let tb = _name_tokens(b);
    if ta.is_empty() || tb.is_empty() {
        return false;
    }
    let intersection_count = ta.intersection(&tb).count();
    if intersection_count >= 2 {
        return true;
    }
    let longest_a = ta
        .iter()
        .max_by_key(|t| t.len())
        .cloned()
        .unwrap_or_default();
    let longest_b = tb
        .iter()
        .max_by_key(|t| t.len())
        .cloned()
        .unwrap_or_default();
    (longest_a.len() >= 5 && nb.contains(&longest_a))
        || (longest_b.len() >= 5 && na.contains(&longest_b))
}

pub fn check_source_extraction(items: &[Value], source_text: &str) -> f64 {
    if items.is_empty() || source_text.is_empty() {
        return 0.0;
    }
    let source_lower = source_text.to_lowercase();
    let source_terms: HashSet<String> = source_lower
        .split_whitespace()
        .map(|t| {
            t.chars()
                .filter(|c| c.is_alphanumeric())
                .collect::<String>()
        })
        .filter(|t| t.len() >= 3 && !STOPWORDS.contains(t.as_str()))
        .collect();

    if source_terms.is_empty() {
        return 0.0;
    }

    let mut matches = 0;
    for item in items {
        let item_text = match item {
            Value::Object(map) => map
                .values()
                .map(|v| v.to_string().trim_matches('"').to_lowercase())
                .collect::<Vec<_>>()
                .join(" "),
            Value::String(s) => s.to_lowercase(),
            other => other.to_string().to_lowercase(),
        };
        if item_text.is_empty() {
            continue;
        }
        let item_terms: HashSet<String> = item_text
            .split_whitespace()
            .map(|t| {
                t.chars()
                    .filter(|c| c.is_alphanumeric())
                    .collect::<String>()
            })
            .filter(|t| t.len() >= 3 && !STOPWORDS.contains(t.as_str()))
            .collect();

        if item_terms.intersection(&source_terms).count() >= 2 {
            matches += 1;
            continue;
        }
        let primary = match item {
            Value::Object(map) => map
                .get("name")
                .or_else(|| map.get("event"))
                .or_else(|| map.get("title"))
                .map(|v| v.to_string().trim_matches('"').to_string())
                .unwrap_or_default(),
            _ => String::new(),
        };
        let search = if !primary.is_empty() {
            _norm_name(&primary)
        } else {
            item_text
        };
        if search.len() >= 4 && source_lower.contains(&search) {
            matches += 1;
        }
    }
    matches as f64 / items.len() as f64
}

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

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_validate_json_empty() {
        let (score, reason) = validate_json(&json!({}), "");
        assert_eq!(score, 0);
        assert_eq!(reason, "no items found");
    }

    #[test]
    fn test_validate_json_valid_list() {
        let data = json!({
            "activities": [
                {"name": "High Park Zoo", "location": "1873 Bloor St W", "price": "Free"},
                {"name": "Ripley's Aquarium", "location": "288 Bremner Blvd", "price": "$45"},
                {"name": "Ontario Science Centre", "location": "770 Don Mills Rd", "price": "$22"},
                {"name": "Royal Ontario Museum", "location": "100 Queens Park", "price": "$26"},
                {"name": "Riverdale Farm", "location": "201 Winchester St", "price": "Free"},
                {"name": "Toronto Zoo", "location": "2000 Meadowvale Rd", "price": "$30"},
                {"name": "Centerville Theme Park", "location": "Centre Island", "price": "$40"},
                {"name": "Allan Gardens", "location": "19 Horticultural Ave", "price": "Free"}
            ]
        });
        let (score, _) = validate_json(&data, "");
        assert_eq!(score, 75); // structure (20) + count (25) + validity (30) = 75 (no source given)
    }

    #[test]
    fn test_validate_detailed_json_detects_generic_location() {
        let data = json!({
            "fixed_activities": [
                {"name": "Activity 1", "location": "Indoor venue", "target_ages": "all", "price": "Free"},
                {"name": "Activity 2", "location": "Outdoor location", "target_ages": "all", "price": "Free"},
                {"name": "Activity 3", "location": "Indoor space", "target_ages": "all", "price": "Free"},
                {"name": "Activity 4", "location": "Venue", "target_ages": "all", "price": "Free"},
                {"name": "Activity 5", "location": "Place", "target_ages": "all", "price": "Free"}
            ]
        });
        let (score, reason) = validate_detailed_json(&data, "");
        assert!(score <= GENERIC_LOCATION_MAX_SCORE);
        assert!(
            reason.contains("generic placeholders"),
            "reason was: {reason}"
        );
    }

    #[test]
    fn test_validate_detailed_json_detects_near_duplicates_with_acronym() {
        let data = json!({
            "fixed_activities": [
                {"name": "Royal Ontario Museum", "location": "Toronto", "price": "$20"},
                {"name": "The ROM", "location": "Toronto", "price": "$20"},
                {"name": "Ontario Science Centre", "location": "Toronto", "price": "$20"},
                {"name": "The OSC", "location": "Toronto", "price": "$20"},
                {"name": "Toronto Zoo", "location": "Toronto", "price": "$20"}
            ]
        });
        let (score, reason) = validate_detailed_json(&data, "");
        assert!(score <= NEAR_DUPLICATE_MAX_SCORE);
        assert!(
            reason.contains("repeat an earlier venue"),
            "reason was: {reason}"
        );
    }
}
