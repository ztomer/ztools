//! Post-parse enforcement of constraints the prompt can only ask for politely.
//!
//! Classes C3, C5, C7 and C8 in `docs/REPORT_WEAKNESS_CLASSES.md` share one
//! shape: a rule was stated in a prompt and never checked in code, so whether
//! it held depended on the model's mood. A user constraint that is only ever a
//! suggestion to a model is not a feature.
//!
//! Everything here is a pure function over the parsed event list -- no LLM, no
//! network -- so it is deterministic and cheap to test. Each returns
//! `(kept_events, notes)`; the notes are surfaced to the operator rather than
//! discarded, so a filtered run says what it dropped and why.

use std::collections::{HashMap, HashSet};

use chrono::{Datelike, NaiveDate};

use super::WeekendEvent;

/// Connector words carry no identifying signal, so requiring them would make a
/// config entry miss a re-worded venue ("The Art of the Brick" vs "Art of Brick").
const CONNECTORS: &[&str] = &["of", "the", "and", "at", "in", "a", "an", "on", "for"];

/// Words that mark a specific, time-limited seasonal event as an exception to
/// an excluded venue (C8).
pub const SEASONAL_EVENT_MARKERS: &[&str] = &[
    "festival",
    "fair",
    "lumina",
    "spooktacular",
    "lights",
    "exhibit",
    "market",
    "harvest",
    "show",
    "carnival",
    "spectacular",
    "concert",
    "expo",
    "pumpkin",
    "maple",
];

/// Venue words that settle indoor/outdoor without consulting a forecast. Kept
/// deliberately small: only terms where an "outdoor" label is unambiguously
/// wrong. These are STEMS, matched as substrings, so plurals are covered:
/// "librar" catches both "library" and "libraries".
pub const INDOOR_MARKERS: &[&str] = &[
    "indoor",
    "trampoline park",
    "museum",
    "play centre",
    "play center",
    "playground",
    "librar",
    "cinema",
    "aquarium",
    "arcade",
    "bowling",
];

pub const OUTDOOR_MARKERS: &[&str] = &[
    "high park",
    "conservation",
    "nature walk",
    "botanical garden",
    "provincial park",
    "national park",
    "hiking trail",
];

/// Fold typographic punctuation and whitespace for constraint matching.
///
/// Scraped venue names use typographic punctuation; a hand-written config uses
/// ASCII. "Ripley's" in conf/weekend.toml did NOT match a scraped
/// "Ripley's Aquarium of Canada" because of U+2019 vs U+0027 -- found by a real
/// `wk` run, after the exclusion filter had already been declared working.
pub fn normalize_for_match(text: &str) -> String {
    let folded: String = text
        .chars()
        .map(|c| match c {
            '\u{2019}' | '\u{2018}' | '\u{02BC}' => '\'',
            '\u{201C}' | '\u{201D}' => '"',
            '\u{2013}' | '\u{2014}' | '\u{2212}' => '-',
            '\u{00A0}' => ' ',
            other => other,
        })
        .collect();
    folded
        .to_lowercase()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

/// Identifying tokens of a venue name.
///
/// Three things are deliberately discarded, each because it caused a real miss:
/// - possessive `'s`, which otherwise tokenises to a stray "s" the scraped name
///   will not have ("Canada's Wonderland" vs "Wonderland Canada")
/// - single characters, which carry no signal
/// - connector words (see `CONNECTORS`)
fn significant_tokens(text: &str) -> HashSet<String> {
    let norm = normalize_for_match(text);
    // Remove possessive "'s" at a word boundary, as Python's `re.sub(r"'s\b",
    // "", ...)` does, so "canada's" -> "canada".
    let chars: Vec<char> = norm.chars().collect();
    let mut cleaned = String::with_capacity(norm.len());
    let mut i = 0;
    while i < chars.len() {
        if chars[i] == '\''
            && i + 1 < chars.len()
            && chars[i + 1] == 's'
            && (i + 2 >= chars.len() || !chars[i + 2].is_ascii_alphanumeric())
        {
            i += 2;
        } else {
            cleaned.push(chars[i]);
            i += 1;
        }
    }

    let mut tokens = HashSet::new();
    let mut cur = String::new();
    for c in cleaned.chars() {
        if c.is_ascii_alphanumeric() {
            cur.push(c);
        } else if !cur.is_empty() {
            if cur.len() > 1 && !CONNECTORS.contains(&cur.as_str()) {
                tokens.insert(cur.clone());
            }
            cur.clear();
        }
    }
    if !cur.is_empty() && cur.len() > 1 && !CONNECTORS.contains(&cur.as_str()) {
        tokens.insert(cur);
    }
    tokens
}

/// Tokens an entry REQUIRES of a candidate name.
///
/// A parenthetical is an annotation, not a requirement: the config's
/// "Royal Ontario Museum (ROM)" must still match a venue called simply
/// "Royal Ontario Museum". Dropping it keeps the match conservative -- the
/// remaining tokens are all still required.
fn required_tokens(entry: &str) -> HashSet<String> {
    let mut without_parens = String::with_capacity(entry.len());
    let mut depth = 0usize;
    for c in entry.chars() {
        match c {
            '(' => {
                depth += 1;
                without_parens.push(' ');
            }
            ')' if depth > 0 => {
                depth -= 1;
                without_parens.push(' ');
            }
            _ if depth > 0 => without_parens.push(' '),
            _ => without_parens.push(c),
        }
    }
    let tokens = significant_tokens(&without_parens);
    if !tokens.is_empty() {
        tokens
    } else {
        significant_tokens(entry)
    }
}

/// Does `entry` name the same venue as `haystack`?
///
/// Class NAME-MATCHED-BY-CONTAINMENT. Containment matching assumes the config's
/// wording is a contiguous substring of the scraped wording, which it usually
/// is not: the scraper interleaves and reorders words. `"Sky Zone Toronto"` is
/// not a substring of `"Sky Zone Trampoline Park (Vaughan/Toronto)"`, so an
/// excluded venue shipped in a real run.
///
/// The rule is token-SUBSET: every significant token of the entry must appear
/// in the haystack, in any order, with anything interleaved. That is still
/// conservative -- ALL tokens are required, so "Toronto Zoo" does not match
/// "Toronto Islands" -- but it survives word order, interpolated words and
/// punctuation, which containment does not. Contiguous containment is kept as
/// an additional accept so that nothing which matched before stops matching.
pub fn matches_exclusion(entry: &str, haystack: &str) -> bool {
    let entry_n = normalize_for_match(entry);
    let hay_n = normalize_for_match(haystack);
    if entry_n.is_empty() {
        return false;
    }
    if hay_n.contains(&entry_n) {
        return true;
    }
    let entry_tokens = required_tokens(entry);
    if entry_tokens.is_empty() {
        return false;
    }
    entry_tokens.is_subset(&significant_tokens(&hay_n))
}

/// Check if an item at an excluded venue is a specific seasonal event exception.
///
/// Generic visits to excluded places (e.g. 'Toronto Zoo') are dropped. But
/// specific, time-limited seasonal events/exhibits (e.g. 'Terra Lumina at
/// Toronto Zoo') are allowed as exceptions.
pub fn is_seasonal_event_exception(name: &str, hit_exclusion: &str) -> bool {
    let name_norm = normalize_for_match(name);
    let hit_norm = normalize_for_match(hit_exclusion);

    let has_event_marker = SEASONAL_EVENT_MARKERS.iter().any(|m| name_norm.contains(m));
    let is_more_specific = name_norm.split_whitespace().count() > hit_norm.split_whitespace().count();

    has_event_marker && is_more_specific
}

/// C8. Remove rows matching the user's `exclude_places` unless it is a seasonal event.
pub fn drop_excluded_places(
    events: Vec<WeekendEvent>,
    excluded: &[String],
) -> (Vec<WeekendEvent>, Vec<String>) {
    let mut kept = Vec::new();
    let mut notes = Vec::new();
    for ev in events {
        let haystack = normalize_for_match(&format!("{} {}", ev.name, ev.location));
        let hit = excluded
            .iter()
            .find(|p| matches_exclusion(p, &haystack))
            .cloned();
        if let Some(hit) = hit {
            if is_seasonal_event_exception(&ev.name, &hit) {
                notes.push(format!(
                    "kept seasonal event '{}' at '{}'",
                    ev.name, hit
                ));
                kept.push(ev);
            } else {
                notes.push(format!(
                    "dropped '{}' — matches excluded place '{}'",
                    ev.name, hit
                ));
            }
        } else {
            kept.push(ev);
        }
    }
    (kept, notes)
}

/// C5. Correct weather label inversions (indoor/outdoor).
///
/// Local LLMs occasionally invert weather labels (e.g. High Park or Nature Walk
/// labeled as 'indoor', or Trampoline Park labeled as 'outdoor'). Only
/// clear-cut cases are corrected; ambiguous venues are left alone.
pub fn correct_weather_labels(mut events: Vec<WeekendEvent>) -> (Vec<WeekendEvent>, Vec<String>) {
    let mut notes = Vec::new();
    for ev in events.iter_mut() {
        let weather = ev.weather.to_lowercase();
        let text = normalize_for_match(&format!("{} {}", ev.name, ev.location));
        if weather == "outdoor" {
            if let Some(marker) = INDOOR_MARKERS.iter().find(|m| text.contains(*m)) {
                ev.weather = "indoor".to_string();
                notes.push(format!(
                    "corrected '{}' from 'outdoor' to 'indoor' (name contains {marker:?})",
                    ev.name
                ));
            }
        } else if weather == "indoor" {
            if let Some(marker) = OUTDOOR_MARKERS.iter().find(|m| text.contains(*m)) {
                ev.weather = "outdoor".to_string();
                notes.push(format!(
                    "corrected '{}' from 'indoor' to 'outdoor' (name contains {marker:?})",
                    ev.name
                ));
            }
        }
    }
    (events, notes)
}

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

/// Does this row's date range overlap `[start, end]`?
///
/// Returns Some(true)/Some(false), or None when the row carries no parseable
/// date. A long-running exhibition (e.g. late June to mid August) is IN the
/// plan if it spans the weekend, even though neither of its endpoints falls
/// inside it. There is one decision, in one place.
pub fn window_overlap(ev: &WeekendEvent, start: NaiveDate, end: NaiveDate) -> Option<bool> {
    let year = start.year();
    let first = super::dates::parse_any_date(&ev.start_date, year);
    let last = super::dates::parse_any_date(&ev.end_date, year);
    let (mut first, mut last) = match (first, last) {
        (Some(f), Some(l)) => (f, l),
        (Some(f), None) => (f, f),
        (None, Some(l)) => (l, l),
        (None, None) => return None,
    };
    if last < first {
        std::mem::swap(&mut first, &mut last);
    }
    Some(!(last < start || first > end))
}

/// C3. A dated event outside the plan's weekend is dropped.
///
/// Only rows that actually carry a parseable date are judged. A row with no
/// date is NOT dropped here -- undated rows are class C7's problem, and
/// silently discarding them would hide that rather than fix it.
pub fn drop_events_outside_window(
    events: Vec<WeekendEvent>,
    start: NaiveDate,
    end: NaiveDate,
) -> (Vec<WeekendEvent>, Vec<String>) {
    let mut kept = Vec::new();
    let mut notes = Vec::new();
    for ev in events {
        let overlaps = window_overlap(&ev, start, end);
        let Some(overlaps) = overlaps else {
            kept.push(ev);
            continue;
        };
        if !overlaps {
            let year = start.year();
            let starts = super::dates::parse_any_date(&ev.start_date, year);
            let ends = super::dates::parse_any_date(&ev.end_date, year);
            let first = starts.or(ends);
            let last = ends.or(starts);
            if let (Some(first), Some(last)) = (first, last) {
                notes.push(format!(
                    "dropped '{}' — runs {}..{}, outside {}..{}",
                    ev.name, first, last, start, end
                ));
            }
        } else {
            kept.push(ev);
        }
    }
    (kept, notes)
}

/// Make `day` agree with the row's own dates, or blank it.
///
/// Where the row's dates overlap the plan window, `day` is derived from them.
/// Where they do not overlap at all the row is left for
/// `drop_events_outside_window`. Where there are no dates, `day` is left
/// alone: it cannot be verified, and inventing one would be class C4 again.
pub fn reconcile_day_with_dates(
    mut events: Vec<WeekendEvent>,
    start: NaiveDate,
    end: NaiveDate,
) -> (Vec<WeekendEvent>, Vec<String>) {
    let mut notes = Vec::new();
    for ev in events.iter_mut() {
        let year = start.year();
        let mut first = match super::dates::parse_any_date(&ev.start_date, year) {
            Some(d) => d,
            None => continue,
        };
        let mut last = super::dates::parse_any_date(&ev.end_date, year).unwrap_or(first);
        if last < first {
            std::mem::swap(&mut first, &mut last);
        }

        let lo = first.max(start);
        let hi = last.min(end);
        if lo > hi {
            continue; // entirely outside the window; not this function's job
        }

        let mut covered = Vec::new();
        let mut cursor = lo;
        while cursor <= hi {
            covered.push(cursor.format("%A").to_string());
            cursor = cursor.succ_opt().unwrap();
        }

        let stated = ev.day.trim().to_string();
        if !stated.is_empty() && covered.contains(&stated) {
            continue;
        }
        let corrected = if covered.len() == 1 {
            covered[0].clone()
        } else {
            String::new()
        };
        ev.day = corrected.clone();
        if !stated.is_empty() {
            notes.push(format!(
                "'{}': day {stated:?} is not within {}..{} — {}",
                ev.name,
                first,
                last,
                if corrected.is_empty() {
                    "cleared".to_string()
                } else {
                    format!("corrected to {corrected:?}")
                }
            ));
        }
    }
    (events, notes)
}

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

#[cfg(test)]
#[path = "../weekend_enforce_tests.rs"]
mod tests;
