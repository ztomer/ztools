//! Report defect detectors: generic locations, constant columns, and near duplicates.
//!
//! Port of `lib/validators/report_defects.py`.

use regex::Regex;
use serde_json::Value;
use std::collections::HashSet;
use std::sync::LazyLock;

pub const GENERIC_LOCATION_LIMIT: f64 = 0.5;
pub const GENERIC_LOCATION_MAX_SCORE: i64 = 45;
pub const CONSTANT_COLUMN_LIMIT: f64 = 0.5;
pub const CONSTANT_COLUMN_MAX_SCORE: i64 = 55;
pub const NEAR_DUPLICATE_LIMIT: f64 = 0.3;
pub const NEAR_DUPLICATE_MAX_SCORE: i64 = 50;

static GENERIC_LOCATION_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)^\s*(?:(?:in|out)door\s+(?:venue|location|activity|space)|(?:various|multiple|several)\s+(?:locations?|venues?|places?)|n/?a|tbd|tba|unknown|unspecified|not\s+specified|none|local\s+(?:area|venue)|nearby|online|virtual|venue|location|place|address|city)\s*\.?\s*$").unwrap()
});

static DUP_NOISE_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)\b(?:the|a|an|of|at|in|on|and|museum|centre|center|park)\b").unwrap()
});

pub fn generic_location_ratio(items: &[Value]) -> f64 {
    let rows: Vec<&serde_json::Map<String, Value>> =
        items.iter().filter_map(|i| i.as_object()).collect();
    if rows.is_empty() {
        return 0.0;
    }
    let generic = rows
        .iter()
        .filter(|r| {
            if let Some(loc) = r.get("location") {
                let loc_str = loc.to_string().trim_matches('"').trim().to_string();
                GENERIC_LOCATION_RE.is_match(&loc_str)
            } else {
                false
            }
        })
        .count();
    generic as f64 / rows.len() as f64
}

pub fn constant_column_ratio(items: &[Value]) -> (f64, Vec<String>) {
    let rows: Vec<&serde_json::Map<String, Value>> =
        items.iter().filter_map(|i| i.as_object()).collect();
    if rows.len() < 3 {
        return (0.0, Vec::new());
    }
    let exempt = ["day", "weather", "name"];
    let first = rows[0];
    let names: Vec<&String> = first
        .keys()
        .filter(|k| !exempt.contains(&k.as_str()))
        .collect();
    if names.is_empty() {
        return (0.0, Vec::new());
    }
    let mut constant: Vec<String> = Vec::new();
    for key in names {
        let first_val = rows[0]
            .get(key)
            .map(|v| v.to_string().trim_matches('"').trim().to_lowercase())
            .unwrap_or_default();
        if !first_val.is_empty() {
            let all_same = rows.iter().all(|r| {
                let v = r
                    .get(key)
                    .map(|val| val.to_string().trim_matches('"').trim().to_lowercase())
                    .unwrap_or_default();
                v == first_val
            });
            if all_same {
                constant.push(key.clone());
            }
        }
    }
    (
        constant.len() as f64
            / first
                .keys()
                .filter(|k| !exempt.contains(&k.as_str()))
                .count() as f64,
        constant,
    )
}

fn words_list(name: &str) -> Vec<String> {
    let clean = name
        .chars()
        .map(|c| {
            if c.is_alphanumeric() {
                c.to_ascii_lowercase()
            } else {
                ' '
            }
        })
        .collect::<String>();
    clean.split_whitespace().map(|s| s.to_string()).collect()
}

fn distinct_name_tokens(name: &str) -> HashSet<String> {
    let words = words_list(name);
    let joined = words.join(" ");
    let stripped = DUP_NOISE_RE.replace_all(&joined, " ");
    stripped
        .split_whitespace()
        .filter(|w| w.len() > 2 || w.chars().all(|c| c.is_ascii_digit()))
        .map(|w| w.to_string())
        .collect()
}

fn acronym_of(name: &str) -> String {
    let noise: HashSet<&str> = ["the", "a", "an", "of", "and", "at"].into_iter().collect();
    let words: Vec<String> = words_list(name)
        .into_iter()
        .filter(|w| !noise.contains(w.as_str()))
        .collect();
    if words.len() > 1 {
        words.iter().filter_map(|w| w.chars().next()).collect()
    } else {
        String::new()
    }
}

pub fn near_duplicate_ratio(items: &[Value]) -> f64 {
    let names: Vec<String> = items
        .iter()
        .map(|item| match item {
            Value::Object(m) => m
                .get("name")
                .map(|v| v.to_string().trim_matches('"').to_string())
                .unwrap_or_default(),
            Value::String(s) => s.clone(),
            _ => String::new(),
        })
        .collect();
    if names.is_empty() {
        return 0.0;
    }
    let mut kept: Vec<(HashSet<String>, String)> = Vec::new();
    let mut dupes = 0;
    for name in &names {
        let tokens = distinct_name_tokens(name);
        if tokens.is_empty() {
            continue;
        }
        let acronym = acronym_of(name);
        let single = if tokens.len() == 1 {
            tokens.iter().next().cloned().unwrap_or_default()
        } else {
            String::new()
        };
        let mut duplicate = false;
        for (seen_tokens, seen_acronym) in &kept {
            if tokens.is_subset(seen_tokens) || seen_tokens.is_subset(&tokens) {
                duplicate = true;
                break;
            }
            let seen_single = if seen_tokens.len() == 1 {
                seen_tokens.iter().next().cloned().unwrap_or_default()
            } else {
                String::new()
            };
            if (!single.is_empty() && single == *seen_acronym)
                || (!seen_single.is_empty() && seen_single == acronym)
            {
                duplicate = true;
                break;
            }
        }
        if duplicate {
            dupes += 1;
        } else {
            kept.push((tokens, acronym));
        }
    }
    dupes as f64 / names.len() as f64
}
