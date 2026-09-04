//! Name normalisation and fuzzy matching between an item and a source mention.
//!
//! Split out of json_validator.rs for the 500-line production cap.

use std::collections::HashSet;

use super::weights::STOPWORDS;

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
