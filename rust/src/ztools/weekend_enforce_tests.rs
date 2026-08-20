//! Tests for the post-parse constraint enforcement ported from
//! `references/weekend/enforce.py` (weakness classes C3/C5/C8 in
//! `docs/REPORT_WEAKNESS_CLASSES.md`).
//!
//! These are pure functions over the parsed event list -- no LLM, no network --
//! so they are deterministic and cheap to test. Each drops/corrects and reports
//! a note rather than failing silently.

use crate::ztools::weekend::{
    correct_weather_labels, drop_excluded_places, flag_constant_columns, matches_exclusion,
    WeekendEvent,
};
use std::collections::HashMap;

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
        start_date: "".into(),
        end_date: "".into(),
        weather: "".into(),
        duration: "".into(),
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

/// C5: a local LLM labelled a trampoline park "outdoor". Clear-cut cases are
/// corrected; the note names the marker that fired.
#[test]
fn an_impossible_weather_label_is_corrected() {
    let mut tramp = event("Sky Zone Trampoline Park", "Toronto");
    tramp.weather = "outdoor".into();
    let (fixed, notes) = correct_weather_labels(vec![tramp]);
    assert_eq!(fixed[0].weather, "indoor", "{notes:?}");
    assert!(!notes.is_empty() && notes[0].contains("trampoline park"), "{notes:?}");
}

/// The reverse inversion: High Park / nature walk labeled "indoor".
#[test]
fn an_outdoor_venue_labeled_indoor_is_corrected() {
    let mut park = event("High Park Nature Walk", "Toronto");
    park.weather = "indoor".into();
    let (fixed, notes) = correct_weather_labels(vec![park]);
    assert_eq!(fixed[0].weather, "outdoor", "{notes:?}");
    assert!(!notes.is_empty() && notes[0].contains("high park"), "{notes:?}");
}

/// Ambiguous venues are left alone -- a generic cafe is not corrected either way.
#[test]
fn an_ambiguous_venue_is_left_alone() {
    let mut cafe = event("Maple Cafe", "Vaughan");
    cafe.weather = "outdoor".into();
    let (fixed, notes) = correct_weather_labels(vec![cafe]);
    assert_eq!(fixed[0].weather, "outdoor");
    assert!(notes.is_empty(), "{notes:?}");
}

// ---------------------------------------------------------------------------
// C4 constant-column detection (ported from test_constant_columns.py)
// ---------------------------------------------------------------------------

fn suspects(ages: &str) -> HashMap<String, Vec<String>> {
    HashMap::from([
        (
            "Estimated Price".to_string(),
            vec!["$18-35".to_string(), "18-35".to_string()],
        ),
        ("Duration".to_string(), vec!["2-3 hours".to_string()]),
        ("Target Age(s)".to_string(), vec![ages.to_string()]),
    ])
}

fn rows_with(n: usize, field: &str, value: &str) -> Vec<WeekendEvent> {
    (0..n)
        .map(|i| {
            let mut ev = event(&format!("event {i}"), "Vaughan");
            match field {
                "target_ages" => ev.target_ages = value.into(),
                "price" => ev.price = value.into(),
                "duration" => ev.duration = value.into(),
                _ => {}
            }
            ev
        })
        .collect()
}

/// 5.2's actual failure: every row carrying the configured family range.
#[test]
fn the_shipped_target_age_constant_is_flagged() {
    let notes = flag_constant_columns(&rows_with(5, "target_ages", "6-13"), &suspects("6-13"));
    assert_eq!(notes.len(), 1, "{notes:?}");
    assert!(notes[0].contains("Target Age(s)"), "{notes:?}");
    assert!(notes[0].contains("6-13"), "{notes:?}");
    // The note must say the rows were kept, or a reader will assume a drop.
    assert!(notes[0].to_lowercase().contains("kept"), "{notes:?}");
}

/// The other two shipped instances: Duration and Estimated Price.
#[test]
fn the_shipped_duration_and_price_constants_are_flagged() {
    let mut rows = rows_with(4, "duration", "2-3 hours");
    for ev in rows.iter_mut() {
        ev.price = "$18-35".into();
    }
    let notes = flag_constant_columns(&rows, &suspects("6-13"));
    let joined = notes.join(" ");
    assert!(joined.contains("Duration"), "{notes:?}");
    assert!(joined.contains("Estimated Price"), "{notes:?}");
}

/// A constant column that is NOT a configured value is ordinary, not C4.
#[test]
fn a_constant_that_is_not_a_configured_value_is_not_flagged() {
    assert!(
        flag_constant_columns(&rows_with(4, "target_ages", "all ages"), &suspects("6-13"))
            .is_empty()
    );
}

/// A column that varies is not flagged -- a shared word is not the defect.
#[test]
fn a_column_that_varies_is_not_flagged() {
    let mut rows = rows_with(3, "target_ages", "6-13");
    rows[1].target_ages = "all ages".into();
    rows[2].target_ages = "8+".into();
    assert!(flag_constant_columns(&rows, &suspects("6-13")).is_empty());
}

/// An empty cell is an honest 'unknown'. The guard is load-bearing only when a
/// suspect list itself contains the empty string.
#[test]
fn empty_cells_are_not_a_constant_column() {
    assert!(
        flag_constant_columns(&rows_with(4, "target_ages", ""), &suspects("6-13")).is_empty()
    );
    let mut pathological = suspects("6-13");
    pathological.insert("Target Age(s)".to_string(), vec!["6-13".to_string(), "".to_string()]);
    assert!(flag_constant_columns(&rows_with(4, "target_ages", ""), &pathological).is_empty());
}

/// A single row is trivially constant; flagging it would be noise.
#[test]
fn one_row_is_never_a_constant_column() {
    assert!(
        flag_constant_columns(&rows_with(1, "target_ages", "6-13"), &suspects("6-13")).is_empty()
    );
    assert!(flag_constant_columns(&[], &suspects("6-13")).is_empty());
}

/// Matching is case and space insensitive: "6-13 " and "2-3 Hours" are the same
/// answer wearing different clothes.
#[test]
fn matching_is_case_and_space_insensitive() {
    let mut rows = rows_with(3, "target_ages", " 6-13 ");
    for ev in rows.iter_mut() {
        ev.duration = "2-3 Hours".into();
    }
    let joined = flag_constant_columns(&rows, &suspects("6-13")).join(" ");
    assert!(joined.contains("Target Age(s)"), "{joined}");
    assert!(joined.contains("Duration"), "{joined}");
}