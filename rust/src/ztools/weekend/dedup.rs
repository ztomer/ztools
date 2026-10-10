//! The same event twice is one row.
//!
//! On 2026-10-10 the Thanksgiving plan listed the robotics workshop twice and
//! the Markham Farmers' Market twice. Two corpus results named each event, the
//! extractor answered each result, and the refine prompt's "merge any
//! near-duplicates" did not merge them -- on the replay it carried both
//! "Robotics Workshop For Kids" and "Robotics Workshop For Kids at Vaughan
//! Mills" through untouched. Whether two rows are the same event is
//! VERIFIABLE from the rows, so it is decided here, deterministically, and
//! never left to the model.
//!
//! Two rows are one event when all three hold:
//! - **name**: the same words once dates and the venue's words are set aside
//!   ("... at Vaughan Mills", "— Oct 9 - Oct 12 in Toronto", when the venue
//!   and city are in either row's location);
//! - **dates**: their ranges overlap, or one row has no date to compare;
//! - **location**: one's place words are a subset of the other's, or one is
//!   unknown -- "Fall Harvest Market" at Sugar Beach and one in Milton are two.
//!
//! A name that ADDS words to the other ("Fall Harvest & Street Eats Market")
//! is the same event only on evidence both rows state: both dated and
//! overlapping, both placed and agreeing, and the shorter name at least two
//! words. Three page titles for one Sugar Beach market survived as three rows
//! on 2026-10-10 because only venue words could be added.
//!
//! The richer row is kept (more known fields; the first on a tie) and any
//! field it does not know is filled from its twin, so a merge never loses a
//! value. Every merge is named.

use std::collections::HashSet;

use chrono::NaiveDate;

use super::WeekendEvent;
use super::enforce::significant_tokens;

/// A field's value counts as known when it says something.
fn known(value: &str) -> bool {
    let v = value.trim();
    !v.is_empty() && !v.eq_ignore_ascii_case("unknown")
}

/// How many of the row's descriptive fields are known.
fn richness(ev: &WeekendEvent) -> usize {
    [
        &ev.location,
        &ev.price,
        &ev.target_ages,
        &ev.start_date,
        &ev.end_date,
        &ev.description,
        &ev.weather,
        &ev.duration,
    ]
    .iter()
    .filter(|v| known(v))
    .count()
}

fn span(ev: &WeekendEvent, year: i32) -> Option<(NaiveDate, NaiveDate)> {
    let first = super::dates::parse_any_date(&ev.start_date, year);
    let last = super::dates::parse_any_date(&ev.end_date, year);
    match (first, last) {
        (Some(f), Some(l)) => Some((f.min(l), f.max(l))),
        (Some(d), None) | (None, Some(d)) => Some((d, d)),
        (None, None) => None,
    }
}

fn dates_agree(a: &WeekendEvent, b: &WeekendEvent, year: i32) -> bool {
    match (span(a, year), span(b, year)) {
        (Some((a1, a2)), Some((b1, b2))) => a1 <= b2 && b1 <= a2,
        _ => true,
    }
}

fn places_agree(a: &WeekendEvent, b: &WeekendEvent) -> bool {
    if !known(&a.location) || !known(&b.location) {
        return true;
    }
    let (x, y) = (
        significant_tokens(&a.location),
        significant_tokens(&b.location),
    );
    x.is_subset(&y) || y.is_subset(&x)
}

/// Words that date a title rather than name the event.
const DATE_WORDS: &[&str] = &[
    "jan",
    "january",
    "feb",
    "february",
    "mar",
    "march",
    "apr",
    "april",
    "may",
    "jun",
    "june",
    "jul",
    "july",
    "aug",
    "august",
    "sep",
    "sept",
    "september",
    "oct",
    "october",
    "nov",
    "november",
    "dec",
    "december",
    "mon",
    "monday",
    "tue",
    "tuesday",
    "wed",
    "wednesday",
    "thu",
    "thursday",
    "fri",
    "friday",
    "sat",
    "saturday",
    "sun",
    "sunday",
    "weekend",
];

fn dates_a_title(word: &str) -> bool {
    let digits = word.trim_end_matches(|c: char| c.is_ascii_alphabetic());
    (!digits.is_empty() && digits.chars().all(|c| c.is_ascii_digit())) || DATE_WORDS.contains(&word)
}

/// How two names relate once dates and venue words are set aside.
#[derive(Debug, PartialEq, Eq)]
enum Names {
    Different,
    Same,
    /// One adds words of its own to all of the other's.
    Extends,
}

fn names_agree(a: &WeekendEvent, b: &WeekendEvent) -> Names {
    let venue: HashSet<String> = significant_tokens(&a.location)
        .union(&significant_tokens(&b.location))
        .cloned()
        .collect();
    let core = |ev: &WeekendEvent| -> HashSet<String> {
        significant_tokens(&ev.name)
            .into_iter()
            .filter(|w| !venue.contains(w) && !dates_a_title(w))
            .collect()
    };
    let (x, y) = (core(a), core(b));
    let (short, long) = if x.len() <= y.len() {
        (&x, &y)
    } else {
        (&y, &x)
    };
    if short.is_empty() || !short.is_subset(long) {
        Names::Different
    } else if short.len() == long.len() {
        Names::Same
    } else if short.len() >= 2 {
        Names::Extends
    } else {
        Names::Different
    }
}

/// Both rows state a date span and a venue -- the evidence an extended name
/// needs.
fn both_stated(a: &WeekendEvent, b: &WeekendEvent, year: i32) -> bool {
    span(a, year).is_some() && span(b, year).is_some() && known(&a.location) && known(&b.location)
}

/// Are these two rows the same event? See the module doc for the tests.
#[must_use]
pub fn same_event(a: &WeekendEvent, b: &WeekendEvent, year: i32) -> bool {
    let agree = dates_agree(a, b, year) && places_agree(a, b);
    match names_agree(a, b) {
        Names::Same => agree,
        Names::Extends => agree && both_stated(a, b, year),
        Names::Different => false,
    }
}

/// Fill every field `into` does not know from `from`.
fn fill_from(into: &mut WeekendEvent, from: &WeekendEvent) {
    for (field, other) in [
        (&mut into.location, &from.location),
        (&mut into.price, &from.price),
        (&mut into.target_ages, &from.target_ages),
        (&mut into.start_date, &from.start_date),
        (&mut into.end_date, &from.end_date),
        (&mut into.dates, &from.dates),
        (&mut into.description, &from.description),
        (&mut into.weather, &from.weather),
        (&mut into.duration, &from.duration),
    ] {
        if !known(field) && known(other) {
            field.clone_from(other);
        }
    }
}

/// Merge rows that are the same event, keeping the richer of each pair and
/// filling its gaps from the other. Order is the order of first appearance.
/// Returns the rows and one note per merge.
#[must_use]
pub fn merge_duplicate_events(
    events: Vec<WeekendEvent>,
    year: i32,
) -> (Vec<WeekendEvent>, Vec<String>) {
    let mut kept: Vec<WeekendEvent> = Vec::new();
    let mut notes = Vec::new();
    for ev in events {
        let Some(twin) = kept.iter_mut().find(|k| same_event(k, &ev, year)) else {
            kept.push(ev);
            continue;
        };
        let (mut keep, other) = if richness(&ev) > richness(twin) {
            (ev, twin.clone())
        } else {
            (twin.clone(), ev)
        };
        fill_from(&mut keep, &other);
        notes.push(format!(
            "merged '{}' into '{}' — the same event twice",
            other.name, keep.name
        ));
        *twin = keep;
    }
    (kept, notes)
}

#[cfg(test)]
#[path = "dedup_tests.rs"]
mod tests;
