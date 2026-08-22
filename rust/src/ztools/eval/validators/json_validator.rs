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

    fn detailed_items() -> Vec<Value> {
        [
            ("Kappa Zeta", "12 Alpha St", "$5"),
            ("Lambda Mu", "34 Beta Ave", "$6"),
            ("Nu Xi", "56 Gamma Rd", "$7"),
            ("Omicron Pi", "78 Delta St", "$8"),
            ("Rho Sigma", "90 Epsilon Ave", "$9"),
            ("Tau Upsilon", "24 Hotel St", "$10"),
            ("Phi Chi", "36 India Ave", "$11"),
            ("Psi Omega", "48 Juliet Rd", "$12"),
        ]
        .iter()
        .map(|(name, loc, price)| json!({"name": name, "location": loc, "price": price}))
        .collect()
    }

    // Marker-pair sources over detailed_items(). Every item carries two
    // distinctive name tokens plus neutral location/price text, so listing
    // marker pairs in the source controls exactly how many items ground:
    // FULL grounds 8/8 (some via the primary-name containment fallback),
    // MED grounds 5/8 = 0.625, LOW grounds 2/8 = 0.25, NONE grounds 0.
    const SRC_FULL: &str =
        "kappa zeta lambda mu nu xi omicron pi rho sigma tau upsilon phi chi psi omega guide";
    const SRC_MED: &str = "kappa zeta lambda mu nu xi omicron pi rho sigma guide";
    const SRC_LOW: &str = "kappa zeta lambda mu quiet river stones";
    const SRC_NONE: &str = "quiet river stones only";

    #[test]
    fn test_extract_list_from_dict_prefers_known_keys_in_order() {
        let data = json!({"items": [1], "events": [1, 2]});
        assert_eq!(extract_list_from_dict(&data).len(), 1);
    }

    #[test]
    fn test_extract_list_from_dict_recurses_into_nested_objects_first() {
        let data = json!({"meta": {"info": {"results": [7, 8]}}, "tags": [1, 2, 3, 4]});
        let found = extract_list_from_dict(&data);
        assert_eq!(found, vec![json!(7), json!(8)]);
    }

    #[test]
    fn test_extract_list_from_dict_falls_back_to_largest_array() {
        let data = json!({"nested": {"x": 5}, "notes": ["a"], "tags": ["x", "y", "z"]});
        assert_eq!(
            extract_list_from_dict(&data),
            vec![json!("x"), json!("y"), json!("z")]
        );
    }

    #[test]
    fn test_extract_list_from_dict_handles_non_objects_and_arrays() {
        assert_eq!(extract_list_from_dict(&json!([1, 2])), vec![json!(1), json!(2)]);
        assert!(extract_list_from_dict(&json!("plain")).is_empty());
        assert!(extract_list_from_dict(&json!(42)).is_empty());
        assert!(extract_list_from_dict(&json!(null)).is_empty());
        assert!(extract_list_from_dict(&json!({"a": 1, "b": "two"})).is_empty());
    }

    #[test]
    fn test_is_valid_list_item_matrix() {
        assert!(is_valid_list_item(&json!("Zoo day")));
        assert!(!is_valid_list_item(&json!("   ")));
        assert!(!is_valid_list_item(&json!(5)));
        assert!(!is_valid_list_item(&json!([1])));
        assert!(is_valid_list_item(&json!({"name": "High Park"})));
        assert!(is_valid_list_item(&json!({"path": "docs/readme"})));
        assert!(!is_valid_list_item(&json!({"name": "", "title": "  "})));
        assert!(!is_valid_list_item(&json!({"other": "x", "desc": ""})));
    }

    #[test]
    fn test_has_item_details_matrix() {
        assert!(!has_item_details(&json!("not an object")));
        assert!(!has_item_details(&json!(null)));
        // no name field but >=2 entries counts as detailed
        assert!(has_item_details(&json!({"time": "3pm", "cost": "$2"})));
        assert!(!has_item_details(&json!({"place": "Here"})));
        // no name field at all and fewer than 2 entries: not detailed
        assert!(!has_item_details(&json!({"time": "3pm"})));
        // name plus a populated detail field
        assert!(has_item_details(&json!({"name": "X", "price": "$5"})));
        // name present but its detail-field siblings are empty: falls back to map size
        assert!(has_item_details(&json!({"name": "X", "desc": ""})));
        assert!(!has_item_details(&json!({"name": "X"})));
    }

    #[test]
    fn test_norm_name_and_tokens() {
        assert_eq!(_norm_name(" High-Park  Zoo! "), "highpark zoo");
        let tokens = _name_tokens("The Big-Foot and 12 Zoo-goers!");
        assert_eq!(
            tokens,
            ["big", "foot", "goers", "zoo"].into_iter().map(String::from).collect()
        );
    }

    #[test]
    fn test_names_match_containment_and_token_overlap() {
        // containment arm
        assert!(_names_match("Central Park", "park"));
        // >=2 shared tokens arm
        assert!(_names_match("Maple Grove Park", "Maple Grove Arena"));
        // empty normalized names never match
        assert!(!_names_match("!!!", "Central Park"));
        assert!(!_names_match("Central Park", "***"));
        // one side has no usable tokens after stopword/length filtering
        assert!(!_names_match("The Who", "Clap Your Hands Say Yeah"));
        // no containment and no long-token anchor
        assert!(!_names_match("Art Cafe", "Cafe Bar"));
        assert!(!_names_match("Waterfront Market", "Marketplace Lofts"));
    }

    #[test]
    fn test_names_match_longest_token_arms() {
        // longest token of A found inside B
        assert!(_names_match("Modern Gallery Annex", "Gallery Bistro"));
        // longest token of B found inside A (symmetric arm)
        assert!(_names_match("Gallery Bistro", "Modern Gallery Annex"));
    }

    #[test]
    fn test_check_source_extraction_guards() {
        let items = vec![json!({"name": "Alpha Beta"})];
        assert_eq!(check_source_extraction(&[], "some source"), 0.0);
        assert_eq!(check_source_extraction(&items, ""), 0.0);
        // every source word is a stopword or under 3 chars -> no terms to match
        assert_eq!(check_source_extraction(&items, "of the a an to"), 0.0);
    }

    #[test]
    fn test_check_source_extraction_object_string_and_other_items() {
        // object item matches via term overlap
        let objects = vec![json!({"name": "Alpha Beta Gamma"})];
        assert_eq!(
            check_source_extraction(&objects, "alpha beta gamma delta report"),
            1.0
        );
        // string item matches via terms...
        let strings = vec![json!("alpha beta")];
        assert_eq!(check_source_extraction(&strings, "alpha beta zone"), 1.0);
        // ...and a non-string item falls back to raw-text containment ("other" arm)
        let others = vec![json!(12345)];
        assert_eq!(check_source_extraction(&others, "route 12345 north"), 1.0);
    }

    #[test]
    fn test_check_source_extraction_skips_empty_text_and_counts_partials() {
        let items = vec![json!({"name": ""}), json!({"name": "Alpha Beta"})];
        let ratio = check_source_extraction(&items, "alpha beta gamma");
        assert!((ratio - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_check_source_extraction_primary_name_fallbacks() {
        // term overlap is zero ("who" is a stopword) but the normalized primary is in the source
        let who = vec![json!({"name": "The Who"})];
        assert_eq!(check_source_extraction(&who, "tickets for the who tribute night"), 1.0);
        // falls back through event fields when name is absent; single shared term
        // is below the >=2 threshold so the normalized-name containment decides
        let titled = vec![json!({"event": "Winterfolk"})];
        assert_eq!(check_source_extraction(&titled, "the winterfolk lineup"), 1.0);
        let fenice = vec![json!({"title": "La Fenice"})];
        assert_eq!(
            check_source_extraction(&fenice, "an evening at la fenice opera house"),
            1.0
        );
        // normalized search shorter than 4 chars is not searched at all
        let short = vec![json!({"name": "Oh!"})];
        assert_eq!(check_source_extraction(&short, "oh what a night"), 0.0);
    }

    #[test]
    fn test_validate_json_scalar_input_has_no_items() {
        assert_eq!(validate_json(&json!(42), ""), (0, "no items found".to_string()));
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
        assert_eq!(validate_json(&data, SRC_MED), (20 + 25 + 30 + 12, "".to_string()));
        // 2/8 match = 0.25 -> +6
        assert_eq!(validate_json(&data, SRC_LOW), (20 + 25 + 30 + 6, "".to_string()));
        // ratio 0 -> hallucinated
        let (score, reason) = validate_json(&data, SRC_NONE);
        assert_eq!(score, 75);
        assert_eq!(reason, "not from input (hallucinated)");
    }

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
        assert!(reason.contains("no items with details"), "reason was: {reason}");
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
        .map(|(name, _loc)| {
            json!({"name": name, "price": "$5", "target_ages": "all"})
        })
        .collect();
        let data = json!({"fixed_activities": items});
        // raw 15+40+25=80, constant columns cap at 55
        let (score, reason) = validate_detailed_json(&data, "");
        assert_eq!(score, 55);
        assert!(reason.contains("constant across every row"), "reason was: {reason}");
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
        assert_eq!(
            score,
            15 - 10
        );
        assert_eq!(
            reason,
            "only 2 items (need 8+); no items with details; duplicates (50%)"
        );
    }

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
}
