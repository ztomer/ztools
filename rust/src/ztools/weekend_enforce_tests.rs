//! Tests for the post-parse constraint enforcement ported from
//! `references/weekend/enforce.py` (weakness classes C3/C5/C8 in
//! `docs/REPORT_WEAKNESS_CLASSES.md`).
//!
//! These are pure functions over the parsed event list -- no LLM, no network --
//! so they are deterministic and cheap to test. Each drops/corrects and reports
//! a note rather than failing silently.

use crate::ztools::weekend::{drop_excluded_places, matches_exclusion, WeekendEvent};

fn event(name: &str, location: &str) -> WeekendEvent {
    WeekendEvent {
        name: name.into(),
        location: location.into(),
        price: "$10".into(),
        target_ages: "6-12".into(),
        day: "Sat".into(),
        dates: "Aug 8".into(),
        description: "desc".into(),
        is_transient: true,
        score: 4.0,
    }
}

/// The CLASS: the config's wording is not a contiguous substring of the scraped
/// wording. Each row below is a variant containment silently missed by the
/// naive token matcher.
#[test]
fn matcher_survives_reordering_interpolation_and_punctuation() {
    let should_match = [
        ("Sky Zone Toronto", "Sky Zone Trampoline Park (Vaughan/Toronto)"),
        ("Canada's Wonderland", "Wonderland Canada thrill rides"),
        ("Ripley's", "Ripley\u{2019}s Aquarium of Canada"),
        ("Museum of Illusions", "Illusions Museum Toronto"),
        ("Royal Ontario Museum (ROM)", "Royal Ontario Museum"),
    ];
    for (entry, scraped) in should_match {
        assert!(
            matches_exclusion(entry, scraped),
            "{entry:?} should match {scraped:?}"
        );
    }
}

/// All tokens are required, so a shared word is not enough to drop a row.
#[test]
fn matcher_stays_conservative_and_does_not_over_drop() {
    let should_not_match = [
        ("Toronto Zoo", "Toronto Islands ferry"),
        ("Toronto Islands", "Toronto Zoo"),
        ("CN Tower", "Tower of London exhibit"),
        ("Little Canada", "Canada Day at the Zoo"),
    ];
    for (entry, scraped) in should_not_match {
        assert!(
            !matches_exclusion(entry, scraped),
            "{entry:?} must NOT match {scraped:?}"
        );
    }
}

/// Regression: a REAL wk run on 2026-08-02 shipped "Ripley\u{2019}s Aquarium of
/// Canada" while conf/weekend.toml said "Ripley's" (U+0027) -- the typographic
/// apostrophe defeated the match.
#[test]
fn typographic_apostrophe_does_not_defeat_the_exclusion() {
    let items = vec![
        event("Ripley\u{2019}s Aquarium of Canada", "Toronto, Ontario"),
        event("Union Summer", "Union Station Plaza"),
    ];
    let (kept, notes) = drop_excluded_places(items, &["Ripley's".to_string()]);
    assert_eq!(notes.len(), 1, "{notes:?}");
    assert!(notes[0].contains("Ripley"), "{notes:?}");
    assert_eq!(kept.len(), 1, "{kept:?}");
    assert_eq!(kept[0].name, "Union Summer");
}

/// A specific, time-limited seasonal event at an excluded venue is an
/// exception; a generic visit is not.
#[test]
fn seasonal_event_exception_is_kept_but_generic_visit_is_dropped() {
    let exclusions = vec!["Toronto Zoo".to_string()];
    let seasonal = event("Terra Lumina Light Festival at Toronto Zoo", "Toronto Zoo");
    let generic = event("A Day at Your Toronto Zoo", "Toronto Zoo");
    let (kept, notes) =
        drop_excluded_places(vec![seasonal.clone(), generic.clone()], &exclusions);
    assert_eq!(kept.len(), 1, "{notes:?}");
    assert_eq!(kept[0].name, "Terra Lumina Light Festival at Toronto Zoo");
    assert!(
        notes.iter().any(|n| n.contains("kept seasonal event")),
        "{notes:?}"
    );
}

/// C8's bar is ZERO excluded venues in the output, not "some exclusion fired".
/// One venue dropping does not license "no excluded venue shipped".
#[test]
fn zero_excluded_venues_in_the_output_not_merely_one_drop() {
    let excluded = [
        "Canada's Wonderland",
        "Ontario Science Centre",
        "Toronto Zoo",
        "The Art of the Brick",
        "Reptilia",
        "ROM",
        "Ripley's",
        "Little Canada",
        "LEGOLAND",
        "CN Tower",
        "Museum of Illusions",
        "Royal Ontario Museum (ROM)",
        "Toronto Islands",
        "Sky Zone Toronto",
        "LEGOLAND Discovery Centre Toronto",
    ];
    let scraped = vec![
        event("Sky Zone Trampoline Park", "Vaughan/Toronto"),
        event("LEGOLAND Discovery Centre Toronto", "Vaughan Mills"),
        event("Ripley\u{2019}s Aquarium of Canada", "Toronto, Ontario"),
        event("Canada Day at Your Toronto Zoo", "Toronto Zoo"),
        event("Harbour Kite Festival", "Pier 4"),
    ];
    let (kept, notes) = drop_excluded_places(scraped, &excluded.map(String::from));
    assert_eq!(kept.len(), 1, "{notes:?}");
    assert_eq!(kept[0].name, "Harbour Kite Festival");
    assert_eq!(notes.len(), 4, "{notes:?}");
}