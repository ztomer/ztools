//! A weekend answer's fields, graded by the planner's own parse (M22).
//!
//! The weekend graders read a row's shape. On the 2026-10-10 json-slot sweep
//! lfm2.5-350m scored 100 with `"place_kind": "outdoor"`, `"for_families":
//! "all"` and `"duration": "outdoor"`, and with one row out of the ten it was
//! handed. Each field the planner closes is graded here by the function that
//! closes it (`weekend::fields`, `weekend::setting`, `weekend::suitability`), so
//! the grade and the planner cannot disagree about what a value is. A field
//! left empty or "unknown" is an honest answer and costs nothing; a value the
//! planner would throw away costs that field.

use serde_json::Value;

use super::json_validator::names_match;
use crate::ztools::weekend::{fields, setting, suitability};

/// Whether the planner's parse keeps a field's raw value.
type Keeps = fn(&str) -> bool;

/// The fields the planner closes, each with whether it keeps `raw`.
const CLOSED: &[(&str, Keeps)] = &[
    ("price", |raw| fields::price_value(raw) != "unknown"),
    ("duration", |raw| !fields::duration_value(raw).is_empty()),
    ("for_families", |raw| {
        !suitability::family_answer(raw).is_empty()
    }),
    ("weather", |raw| !setting::weather_label(raw).is_empty()),
    // A kind of place is a noun; a weather label in its slot says nothing
    // about the place and is exactly what the noun was asked for instead of.
    ("place_kind", |raw| setting::weather_label(raw).is_empty()),
];

/// What one row says that the planner would throw away, as `field=value`.
#[must_use]
pub fn field_defects(event: &Value) -> Vec<String> {
    CLOSED
        .iter()
        .filter_map(|(field, keeps)| {
            let raw = event.get(*field).and_then(Value::as_str)?.trim();
            let unknown = raw.is_empty() || raw.eq_ignore_ascii_case("unknown");
            (!unknown && !keeps(raw)).then(|| format!("{field}={raw}"))
        })
        .collect()
}

/// The share of an answer that survives the planner, 0-100, and why not all.
///
/// The share of closed fields the planner keeps, times -- when `rows` names
/// the input rows every one of which should come back (the structure phase:
/// one input row, one output row) -- the share of them that did. An answer
/// with no `transient_events` list is the parse gate's business, not this.
#[must_use]
pub fn validate_planner_fields(data: &Value, rows: &[String]) -> (i64, String) {
    let Some(events) = data.get("transient_events").and_then(Value::as_array) else {
        return (100, String::new());
    };
    let mut reasons = Vec::new();
    let checked = events.len() * CLOSED.len();
    let defects: Vec<String> = events.iter().flat_map(field_defects).collect();
    if !defects.is_empty() {
        reasons.push(format!(
            "{} of {checked} fields the planner discards ({})",
            defects.len(),
            defects
                .iter()
                .take(4)
                .cloned()
                .collect::<Vec<_>>()
                .join(", ")
        ));
    }
    // The share as a fraction kept/total, so it stays exact until the round.
    let (mut kept, mut total) = if checked == 0 {
        (1, 1)
    } else {
        (checked - defects.len(), checked)
    };
    if !rows.is_empty() {
        let names: Vec<&str> = events
            .iter()
            .filter_map(|e| e.get("name").and_then(Value::as_str))
            .collect();
        let back = rows
            .iter()
            .filter(|row| names.iter().any(|n| names_match(n, row)))
            .count();
        if back < rows.len() {
            reasons.push(format!("{back} of {} input rows came back", rows.len()));
        }
        kept *= back;
        total *= rows.len();
    }
    let percent = (kept * 100 + total / 2) / total;
    (i64::try_from(percent).unwrap_or(0), reasons.join("; "))
}

#[cfg(test)]
#[path = "planner_fields_tests.rs"]
mod tests;
