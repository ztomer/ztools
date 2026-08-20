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

use std::collections::HashSet;

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

#[cfg(test)]
#[path = "../weekend_enforce_tests.rs"]
mod tests;
