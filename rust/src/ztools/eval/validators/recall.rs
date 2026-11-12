//! Planted-fact scorers for the Hebrew roster tasks (LM.PROBES, design section 4).
//!
//! Both are deterministic and measure one thing: of the facts planted in the prompt,
//! how many came back. [`validate_recall`] reads prose (names and numbers, plus the
//! `(@handle | time)` attribution form); [`validate_json_fields`] reads a JSON answer
//! (`key` carrying `value`). Partial credit is the point: a task that only ever scores
//! 0 or 100 is a gate, and the design admits a task to the ranking only when it
//! separates at least three scores (`discrimination.rs`).

use regex::Regex;
use serde_json::Value;

/// Weight of fact recall in a prose score; the rest is the attribution form.
pub const RECALL_WEIGHT: f64 = 70.0;

/// Score `text`: recall of `items`, then the attribution form of each of `handles`,
/// then a penalty per `forbidden` token present. With no `handles` recall is the
/// whole score.
#[must_use]
pub fn validate_recall(
    text: &str,
    items: &[String],
    handles: &[String],
    forbidden: &[String],
) -> (i64, String) {
    if text.trim().is_empty() {
        return (0, "empty response".to_string());
    }
    let lower = text.to_lowercase();
    let found = |needle: &String| lower.contains(&needle.to_lowercase());
    let missing = items.iter().filter(|i| !found(i)).count();
    let recall = share(items.len() - missing, items.len());
    let mut notes = Vec::new();
    if missing > 0 {
        notes.push(format!(
            "missing {missing} of {} planted facts",
            items.len()
        ));
    }
    let mut score = if handles.is_empty() {
        recall * 100.0
    } else {
        let formed = handles.iter().filter(|h| has_attribution(text, h)).count();
        if formed < handles.len() {
            notes.push(format!(
                "{} of {} bullets lack the (@handle | time) form",
                handles.len() - formed,
                handles.len()
            ));
        }
        share(formed, handles.len()).mul_add(100.0 - RECALL_WEIGHT, recall * RECALL_WEIGHT)
    };
    let leaked = forbidden.iter().filter(|f| found(f)).count();
    if leaked > 0 {
        notes.push(format!("forbidden facts present: {leaked}"));
    }
    score = fraction(leaked).mul_add(-FORBIDDEN_PENALTY, score);
    (rounded(score.max(0.0)), notes.join("; "))
}

/// Points off for each `forbidden` token the answer carries (a fact from the part of
/// the prompt the task said to ignore, or one that was never there).
pub const FORBIDDEN_PENALTY: f64 = 15.0;

fn fraction(n: usize) -> f64 {
    f64::from(u32::try_from(n).unwrap_or(u32::MAX))
}

/// `part` of `whole` as a fraction; an empty whole is fully satisfied.
fn share(part: usize, whole: usize) -> f64 {
    if whole == 0 {
        return 1.0;
    }
    fraction(part) / fraction(whole)
}

fn rounded(score: f64) -> i64 {
    // 0..=100 by construction, so the text round trip cannot overflow.
    format!("{score:.0}").parse().unwrap_or(0)
}

/// `(@handle | time)`: the handle, a pipe and a time with a digit, inside parentheses.
fn has_attribution(text: &str, handle: &str) -> bool {
    Regex::new(&format!(
        r"\(\s*@{}\s*\|\s*[^)]*\d[^)]*\)",
        regex::escape(handle)
    ))
    .is_ok_and(|re| re.is_match(text))
}

/// Score a JSON answer: each `(key, value)` pair counts when some object anywhere in
/// the answer has that key holding something that contains `value`.
#[must_use]
pub fn validate_json_fields(answer: &Value, expected: &[(String, String)]) -> (i64, String) {
    if matches!(answer, Value::String(_) | Value::Null) {
        return (0, "the answer is not JSON".to_string());
    }
    let hits = expected.iter().filter(|(k, v)| holds(answer, k, v)).count();
    let missed = expected.len() - hits;
    let note = if missed == 0 {
        String::new()
    } else {
        format!("{missed} of {} fields missing or wrong", expected.len())
    };
    (rounded(share(hits, expected.len()) * 100.0), note)
}

/// Whether any object under `node` has `key` holding a value that contains `needle`.
fn holds(node: &Value, key: &str, needle: &str) -> bool {
    match node {
        Value::Object(map) => map
            .iter()
            .any(|(k, v)| (k == key && text_of(v).contains(needle)) || holds(v, key, needle)),
        Value::Array(items) => items.iter().any(|v| holds(v, key, needle)),
        _ => false,
    }
}

fn text_of(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        other => other.to_string(),
    }
}

#[cfg(test)]
#[path = "recall_tests.rs"]
mod tests;
