//! Constant-column detection (weakness class C4). Ported from
//! `weekend/enforce.py`.

use std::collections::HashMap;

use super::WeekendEvent;
/// Fields whose value the prompt asks the model to always supply, paired with
/// the aliases the parsed item may carry them under. These are the columns a
/// model fills mechanically when it has nothing real to say.
pub const CONSTANT_COLUMN_FIELDS: &[(&str, &[&str])] = &[
    ("Target Age(s)", &["target_ages", "age_group"]),
    ("Estimated Price", &["price", "cost"]),
    ("Duration", &["duration"]),
];

/// The values that actually shipped, filled from the instructions rather than
/// from any event. Kept as data so a fourth instance is one line, not a new
/// check -- and so the historical evidence for this class stays readable.
pub const PROMPT_CONSTANTS: &[(&str, &[&str])] = &[
    ("Estimated Price", &["$18-35", "18-35"]),
    ("Duration", &["2-3 hours"]),
];

/// A column of one row is trivially constant. Flagging it would be noise, and a
/// check that cries wolf on every single-row table is a check the reader learns
/// to skip.
const MIN_ROWS_FOR_CONSTANT: usize = 2;

fn column_value(ev: &WeekendEvent, aliases: &[&str]) -> String {
    for alias in aliases {
        let value = match *alias {
            "target_ages" | "age_group" => &ev.target_ages,
            "price" | "cost" => &ev.price,
            "duration" => &ev.duration,
            _ => "",
        };
        let trimmed = value.trim();
        if !trimmed.is_empty() {
            return trimmed.to_string();
        }
    }
    String::new()
}

/// Report columns that are identical on every row **and** equal to a value the
/// configuration or the prompt supplied.
///
/// The conjunction is what makes it precise. *Constant* alone fires on a
/// genuinely uniform column (every event in one city). *Equal to a configured
/// value* alone fires on the one row that legitimately matches. Together they
/// say: this column was answered from the question.
///
/// Returns notes and **changes nothing**. A mechanically-filled column is a
/// signal that the extraction is wrong, not a reason to delete the rows.
pub fn flag_constant_columns(
    events: &[WeekendEvent],
    suspects: &HashMap<String, Vec<String>>,
) -> Vec<String> {
    if events.len() < MIN_ROWS_FOR_CONSTANT {
        return Vec::new();
    }

    let mut notes = Vec::new();
    for (label, aliases) in CONSTANT_COLUMN_FIELDS {
        let values: Vec<String> = events.iter().map(|ev| column_value(ev, aliases)).collect();
        let first = &values[0];
        if first.is_empty() {
            continue;
        }
        // `if first.is_empty()` above is load-bearing: it is what keeps an empty
        // column from ever matching. A blank cell is an honest "unknown".
        if values[1..].iter().any(|v| !v.eq_ignore_ascii_case(first)) {
            continue;
        }
        let suspects_for_label = suspects.get(*label).map(|v| v.as_slice()).unwrap_or(&[]);
        let is_suspect = suspects_for_label
            .iter()
            .filter(|s| !s.trim().is_empty())
            .any(|s| first.eq_ignore_ascii_case(s.trim()));
        if !is_suspect {
            continue;
        }
        notes.push(format!(
            "{label}: every one of {} rows reads {first:?}, which is the configured \
             value -- the column was answered from the prompt, not from the events. \
             Rows kept; treat the column as unverified.",
            events.len()
        ));
    }
    notes
}
