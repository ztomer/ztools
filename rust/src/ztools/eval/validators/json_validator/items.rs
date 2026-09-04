//! Item shape: finding the list in a response and judging one entry.
//!
//! Split out of json_validator.rs for the 500-line production cap.

use serde_json::Value;

use super::weights::DETAIL_FIELDS;

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
