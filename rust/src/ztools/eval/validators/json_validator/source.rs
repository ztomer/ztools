//! Grounding: what fraction of the returned items actually came from the source.
//!
//! Split out of json_validator.rs for the 500-line production cap.

use serde_json::Value;
use std::collections::HashSet;

use super::names::_norm_name;
use super::weights::STOPWORDS;

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
