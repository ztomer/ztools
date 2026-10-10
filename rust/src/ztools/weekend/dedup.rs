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
//! - **name**: the same words, or one name is the other plus only words that
//!   name the venue ("... at Vaughan Mills", when "Vaughan Mills" is in either
//!   row's location) and connectors;
//! - **dates**: their ranges overlap, or one row has no date to compare;
//! - **location**: one's place words are a subset of the other's, or one is
//!   unknown -- "Fall Harvest Market" at Sugar Beach and one in Milton are two.
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

fn names_agree(a: &WeekendEvent, b: &WeekendEvent) -> bool {
    let (x, y) = (significant_tokens(&a.name), significant_tokens(&b.name));
    if x.is_empty() || y.is_empty() {
        return false;
    }
    let (short, long) = if x.len() <= y.len() {
        (&x, &y)
    } else {
        (&y, &x)
    };
    if !short.is_subset(long) {
        return false;
    }
    let venue: HashSet<String> = significant_tokens(&a.location)
        .union(&significant_tokens(&b.location))
        .cloned()
        .collect();
    long.difference(short).all(|w| venue.contains(w))
}

/// Are these two rows the same event? See the module doc for the three tests.
#[must_use]
pub fn same_event(a: &WeekendEvent, b: &WeekendEvent, year: i32) -> bool {
    names_agree(a, b) && dates_agree(a, b, year) && places_agree(a, b)
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
