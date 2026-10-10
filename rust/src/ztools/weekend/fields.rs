//! A field holds a value of its own kind, copied from the source, or is
//! unknown (W2).
//!
//! The model fills a field it cannot answer with whatever is near: on
//! 2026-10-10 Pumpkinfest's price and duration both came back "Unlimited"
//! (from "Unlimited midway rides"), Pumpkins After Dark's price "$10,000+"
//! (from "10,000+ glowing pumpkins"), and a description landed in a duration.
//! Whether a value is a price at all, and whether a dollar amount is one the
//! sources state, are verifiable -- so they are checked here, never asked.

use std::sync::LazyLock;

use regex::Regex;

use super::WeekendEvent;
use super::enforce::normalize_for_match;

/// An amount of money, or a word that stands for one.
static PRICE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r"(?i)\$\s?\d|\d\s?(?:cad|dollars?)\b|\bfree\b|by donation|pay what you can|\bpwyc\b|no cost|included with",
    )
    .expect("static regex")
});

/// A clock time or a length of time.
static DURATION: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r"(?i)\d{1,2}(?::\d{2})?\s?(?:am|pm)\b|\d{1,2}:\d{2}|\d\s?(?:hours?|hrs?|minutes?|mins?|days?)\b|\b(?:all|full|half)[ -]day\b",
    )
    .expect("static regex")
});

/// A dollar amount as written: "$5", "$10,000", "$12.50".
static DOLLARS: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\$\s?(\d[\d,]*(?:\.\d+)?)").expect("static regex"));

/// The price as a row stores it: the model's value when it reads as a price,
/// else "unknown" (the renderer's missing-value sentinel).
#[must_use]
pub fn price_value(raw: &str) -> String {
    let raw = raw.trim();
    if PRICE.is_match(raw) {
        raw.to_string()
    } else {
        "unknown".to_string()
    }
}

/// The duration as a row stores it: the model's value when it reads as a
/// time or a length, else empty.
#[must_use]
pub fn duration_value(raw: &str) -> String {
    let raw = raw.trim();
    if DURATION.is_match(raw) {
        raw.to_string()
    } else {
        String::new()
    }
}

/// Clear a price whose dollar amounts the corpus never states as dollars.
/// With no corpus there is nothing to judge against, and nothing changes.
#[must_use]
pub fn clear_unsourced_prices(
    events: Vec<WeekendEvent>,
    corpus: &str,
) -> (Vec<WeekendEvent>, Vec<String>) {
    let mut notes = Vec::new();
    if corpus.trim().is_empty() {
        return (events, notes);
    }
    let stated: Vec<String> = DOLLARS
        .captures_iter(&normalize_for_match(corpus))
        .map(|c| c[1].to_string())
        .collect();
    let events = events
        .into_iter()
        .map(|mut ev| {
            let amounts: Vec<String> = DOLLARS
                .captures_iter(&ev.price)
                .map(|c| c[1].to_string())
                .collect();
            if !amounts.is_empty() && !amounts.iter().any(|a| stated.contains(a)) {
                notes.push(format!(
                    "cleared price '{}' of '{}' — no such amount in the fetched corpus",
                    ev.price, ev.name
                ));
                "unknown".clone_into(&mut ev.price);
            }
            ev
        })
        .collect();
    (events, notes)
}

#[cfg(test)]
#[path = "fields_tests.rs"]
mod tests;
