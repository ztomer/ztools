//! Whether a row is fit to recommend to THIS family at all.
//!
//! Three gates, each a class a real plan shipped (2026-10-09, Thanksgiving):
//!
//! - **Age (A1).** "Baby and Me (birth to 12 months)" and "French Meetup for
//!   Adults" were recommended to children aged 14, 11 and 6. Age only counted
//!   when the model filled `target_ages`, it only ever ADDED score, and the
//!   model left the field empty for nearly every row — while the name said
//!   exactly who the event was for. A row's age range is now read from the
//!   declared field AND the name AND the description, and a row that fits
//!   none of the children is dropped, not merely ranked lower.
//! - **Listing-page titles (A2).** A row's venue was "Vaughan Events This
//!   Weekend & Things to Do - Oct 2026": the title of the aggregator page the
//!   events were scraped from, copied into the venue column of every row.
//!   A listing page is neither a venue nor an event.
//! - **Duplicates of fixed venues (A3).** "Kortright Centre for Conservation"
//!   appeared in BOTH halves of the plan: once as a year-round venue and once
//!   as a "transient event" whose name was just the venue's.

use std::sync::LazyLock;

use regex::Regex;

use super::WeekendEvent;

pub use super::listing::{is_listing_page_title, names_no_event, reject_listing_page_titles};

/// The top of an open range ("18+", "adults").
pub const OPEN_END: u32 = 120;

/// Inclusive age range `(lo, hi)` in whole years.
pub type AgeRange = (u32, u32);

/// Words that say an event welcomes children generally. When one is present a
/// restrictive WORD ("adults", "baby") is read as one audience among several
/// ("kids and adults", "family baby-sign class"), never as the whole audience.
const INCLUSIVE: &[&str] = &[
    "all ages", "family", "families", "kids", "children", "child", "everyone",
];

/// Audience words and the ages they mean, matched at a word START ("teens"
/// and "teenagers", never "canteen"). "Young adult" is read before "adults";
/// a matched word is blanked so it is not read twice. Singular "adult" and
/// "senior" are deliberately absent: "Admission $10/adult" is a price, and
/// "Senior Kindergarten" is five-year-olds.
const AUDIENCE_WORDS: &[(&str, AgeRange)] = &[
    (r"\byoung adults?\b", (12, 18)),
    (r"\bnewborn", (0, 1)),
    (r"\binfant", (0, 1)),
    (r"\bbab(?:y|ies)\b", (0, 1)),
    (r"\btoddler", (1, 3)),
    (r"\bpre-?school", (3, 5)),
    (r"\bearlyon\b", (0, 6)),
    (r"\btweens?\b", (9, 12)),
    (r"\bteen", (13, 19)),
    (
        r"\badults\b|\badults? only\b|\bfor adults?\b",
        (18, OPEN_END),
    ),
    (r"\bseniors\b", (60, OPEN_END)),
];

static AUDIENCE: LazyLock<Vec<(Regex, AgeRange)>> = LazyLock::new(|| {
    AUDIENCE_WORDS
        .iter()
        .map(|(pattern, ages)| (re(pattern), *ages))
        .collect()
});

fn re(pattern: &str) -> Regex {
    // The patterns are literals in this file; a bad one fails every test.
    Regex::new(pattern).unwrap_or_else(|e| panic!("bad suitability pattern {pattern}: {e}"))
}

/// "12 months", "6-18 months", "birth to 12 months".
static MONTHS: LazyLock<Regex> =
    LazyLock::new(|| re(r"(?:(\d{1,2}|birth)\s*(?:-|–|to)\s*)?(\d{1,2})\s*(?:months?|mos)\b"));
/// "ages 6-12", "6 to 12 years", "birth to 6 years old".
static SPAN: LazyLock<Regex> = LazyLock::new(|| {
    re(
        r"(?:\bages?\s*(\d{1,2}|birth)\s*(?:-|–|to)\s*(\d{1,2}))|(?:(\d{1,2}|birth)\s*(?:-|–|to)\s*(\d{1,2})\s*(?:years?|yrs?|y/o|year-olds?))",
    )
});
/// "18+", "ages 5+", "ages 5 and up".
static OPEN: LazyLock<Regex> =
    LazyLock::new(|| re(r"(?:\b(\d{1,2})\+)|(?:\bages?\s*(\d{1,2})\s*(?:and up|and older|\+))"));
/// "kids under 5", "ages under 5".
static UNDER: LazyLock<Regex> =
    LazyLock::new(|| re(r"\b(?:kids|children|ages?)\s+under\s*(\d{1,2})"));
/// Any number, for a field that is ONLY an age field.
static NUMBER: LazyLock<Regex> = LazyLock::new(|| re(r"\d{1,3}"));

fn num(m: Option<regex::Match<'_>>) -> Option<u32> {
    let text = m?.as_str();
    if text == "birth" {
        return Some(0);
    }
    text.parse().ok()
}

fn widen(acc: Option<AgeRange>, next: AgeRange) -> AgeRange {
    acc.map_or(next, |(lo, hi)| (lo.min(next.0), hi.max(next.1)))
}

/// The age range stated in free text (a name or a description), if any.
///
/// Only numbers in an AGE CONTEXT count here — "Saturday 10am, $5" is not
/// ages 5 to 10 — and every match widens the range, so "ages 3-5 and 6-12"
/// reads as 3 to 12 rather than as its first clause.
#[must_use]
pub fn stated_range(text: &str) -> Option<AgeRange> {
    let lower = text.to_lowercase();
    let mut range = None;
    // An age in months is an infant's: "birth to 12 months" is ages 0 to 1.
    for c in MONTHS.captures_iter(&lower) {
        if let Some(hi) = num(c.get(2)) {
            let lo = num(c.get(1)).unwrap_or(0);
            range = Some(widen(range, (lo.min(hi) / 12, lo.max(hi) / 12)));
        }
    }
    let in_months = range.is_some();
    for c in SPAN.captures_iter(&lower) {
        let lo = num(c.get(1)).or_else(|| num(c.get(3)));
        let hi = num(c.get(2)).or_else(|| num(c.get(4)));
        // A span already read in months is not also a span in years.
        if let (Some(lo), Some(hi), false) = (lo, hi, in_months) {
            range = Some(widen(range, (lo.min(hi), lo.max(hi))));
        }
    }
    for c in OPEN.captures_iter(&lower) {
        if let Some(lo) = num(c.get(1)).or_else(|| num(c.get(2))) {
            range = Some(widen(range, (lo, OPEN_END)));
        }
    }
    for c in UNDER.captures_iter(&lower) {
        if let Some(n) = num(c.get(1)) {
            range = Some(widen(range, (0, n.saturating_sub(1))));
        }
    }
    if range.is_none() && !INCLUSIVE.iter().any(|w| lower.contains(w)) {
        let mut rest = lower;
        for (word, ages) in AUDIENCE.iter() {
            if word.is_match(&rest) {
                range = Some(widen(range, *ages));
                rest = word.replace_all(&rest, " ").into_owned();
            }
        }
    }
    range
}

/// The range a `target_ages` FIELD declares. Bare numbers count here, since
/// the field holds nothing else ("6-12", "7-14 yrs", "6,10").
#[must_use]
pub fn declared_range(field: &str) -> Option<AgeRange> {
    if let Some(range) = stated_range(field) {
        return Some(range);
    }
    let nums: Vec<u32> = NUMBER
        .find_iter(field)
        .filter_map(|m| m.as_str().parse().ok())
        .filter(|n| *n <= OPEN_END)
        .collect();
    let (lo, hi) = (*nums.iter().min()?, *nums.iter().max()?);
    if field.contains('+') {
        Some((lo, OPEN_END))
    } else {
        Some((lo, hi))
    }
}

/// Every range the row states about itself, from each of its three sources.
#[must_use]
pub fn row_ranges(ev: &WeekendEvent) -> Vec<AgeRange> {
    [
        declared_range(&ev.target_ages),
        stated_range(&ev.name),
        stated_range(&ev.description),
    ]
    .into_iter()
    .flatten()
    .collect()
}

/// How many of `ages` fit EVERY range the row states, or `None` when the row
/// states no range at all (it cannot be judged, which is not the same as fit).
#[must_use]
pub fn children_who_fit(ev: &WeekendEvent, ages: &[u32]) -> Option<usize> {
    let ranges = row_ranges(ev);
    if ranges.is_empty() {
        return None;
    }
    Some(
        ages.iter()
            .filter(|a| ranges.iter().all(|(lo, hi)| (lo..=hi).contains(a)))
            .count(),
    )
}

/// A1: drop rows that fit none of the children. Unjudgeable rows stay.
#[must_use]
pub fn drop_unsuitable_for_ages(
    events: Vec<WeekendEvent>,
    ages: &[u32],
) -> (Vec<WeekendEvent>, Vec<String>) {
    let mut notes = Vec::new();
    let kept = events
        .into_iter()
        .filter(|ev| {
            if ages.is_empty() || children_who_fit(ev, ages) != Some(0) {
                return true;
            }
            let ranges: Vec<String> = row_ranges(ev)
                .iter()
                .map(|(lo, hi)| {
                    if *hi >= OPEN_END {
                        format!("{lo}+")
                    } else {
                        format!("{lo}-{hi}")
                    }
                })
                .collect();
            notes.push(format!(
                "dropped '{}' — for ages {}, fits none of the children ({})",
                ev.name,
                ranges.join(" and "),
                super::family::ages_label(ages)
            ));
            false
        })
        .collect();
    (kept, notes)
}

/// A3: drop a transient row that is only a fixed venue under another heading.
///
/// Such a row's every identifying word is a word of the venue's name. A real
/// event AT the venue ("Kortright Maple Syrup Festival") names something the
/// venue does not, and stays.
#[must_use]
pub fn drop_duplicates_of_fixed(
    transient: Vec<WeekendEvent>,
    fixed: &[WeekendEvent],
) -> (Vec<WeekendEvent>, Vec<String>) {
    let words = |s: &str| -> Vec<String> {
        super::normalize_for_match(s)
            .split(|c: char| !c.is_alphanumeric())
            .filter(|w| w.len() > 2 && !super::CONNECTORS.contains(w))
            .map(String::from)
            .collect()
    };
    let mut notes = Vec::new();
    let kept = transient
        .into_iter()
        .filter(|ev| {
            let name = words(&ev.name);
            let twin = fixed.iter().find(|f| {
                let venue = words(&f.name);
                !name.is_empty() && name.iter().all(|w| venue.contains(w))
            });
            twin.is_none_or(|f| {
                notes.push(format!(
                    "dropped transient '{}' — it is the fixed venue '{}'",
                    ev.name, f.name
                ));
                false
            })
        })
        .collect();
    (kept, notes)
}

#[cfg(test)]
#[path = "suitability_tests.rs"]
mod tests;
