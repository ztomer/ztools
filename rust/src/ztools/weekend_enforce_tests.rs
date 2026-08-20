//! Tests for the post-parse constraint enforcement ported from
//! `references/weekend/enforce.py` (weakness classes C3/C5/C8 in
//! `docs/REPORT_WEAKNESS_CLASSES.md`).
//!
//! These are pure functions over the parsed event list -- no LLM, no network --
//! so they are deterministic and cheap to test. Each drops/corrects and reports
//! a note rather than failing silently.

use crate::ztools::weekend::{
    correct_weather_labels, drop_events_outside_window, drop_excluded_places,
    drop_unsourced_rows, find_dates_in, flag_constant_columns, in_window_count,
    matches_exclusion, parse_any_date, reconcile_day_with_dates, row_is_sourced,
    window_overlap, WeekendEvent,
};
use chrono::NaiveDate;
use std::collections::HashMap;

fn d(y: i32, m: u32, day: u32) -> NaiveDate {
    NaiveDate::from_ymd_opt(y, m, day).unwrap()
}

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

// ---------------------------------------------------------------------------
// C3 window enforcement (ported from test_report_class_fixes.py / test_g3_edge_cases.py)
// ---------------------------------------------------------------------------

fn dated(name: &str, start_date: &str, end_date: &str) -> WeekendEvent {
    let mut ev = event(name, "Vaughan");
    ev.start_date = start_date.into();
    ev.end_date = end_date.into();
    ev
}

/// "Aug 15", "09 Aug 2026", "Sun 09 Aug" and ISO all parse; a four-digit year
/// in the text wins over the fallback year.
#[test]
fn dates_parse_across_the_supported_shapes() {
    assert_eq!(find_dates_in("Aug 15", 2026), vec![d(2026, 8, 15)]);
    assert_eq!(find_dates_in("August 15, 2026", 2026), vec![d(2026, 8, 15)]);
    assert_eq!(find_dates_in("Sun 09 Aug", 2026), vec![d(2026, 8, 9)]);
    assert_eq!(find_dates_in("2026-08-08", 2026), vec![d(2026, 8, 8)]);
    // A past explicit year is NOT promoted into the fallback year.
    assert_eq!(
        parse_any_date("festival ran 2019-07-01", 2026),
        Some(d(2019, 7, 1))
    );
    // Durations are not dates.
    assert!(find_dates_in("2-3 hours", 2026).is_empty());
    // The shared parse is what window_overlap uses.
    assert_eq!(
        window_overlap(&dated("e", "Aug 15", "Aug 16"), d(2026, 8, 7), d(2026, 8, 9)),
        Some(false)
    );
}

/// A row spanning the weekend is IN the plan; a row fully outside is OUT;
/// an undated row is not judged here.
#[test]
fn window_overlap_judges_spans_not_endpoints() {
    let start = d(2026, 8, 7);
    let end = d(2026, 8, 9);
    assert_eq!(
        window_overlap(&dated("Monet", "2026-06-29", "2026-08-16"), start, end),
        Some(true)
    );
    assert_eq!(
        window_overlap(&dated("Canada Day", "2026-07-01", "2026-07-01"), start, end),
        Some(false)
    );
    assert_eq!(window_overlap(&event("no dates", "X"), start, end), None);
    // End date with no start.
    assert_eq!(
        window_overlap(&dated("e", "", "2026-08-08"), start, end),
        Some(true)
    );
    assert_eq!(
        window_overlap(&dated("e", "", "2026-07-01"), start, end),
        Some(false)
    );
    // A reversed range is swapped, not silently judged empty.
    assert_eq!(
        window_overlap(&dated("e", "2026-08-16", "2026-06-29"), start, end),
        Some(true)
    );
}

#[test]
fn dated_events_outside_the_window_are_dropped() {
    let start = d(2026, 8, 7);
    let end = d(2026, 8, 9);
    let (kept, notes) = drop_events_outside_window(
        vec![
            dated("Monet", "2026-06-29", "2026-08-16"),
            dated("Canada Day", "2026-07-01", "2026-07-01"),
        ],
        start,
        end,
    );
    assert_eq!(kept.len(), 1, "{notes:?}");
    assert_eq!(kept[0].name, "Monet");
    assert_eq!(notes.len(), 1, "{notes:?}");
    assert!(notes[0].contains("2026-07-01"), "{notes:?}");
}

/// day agrees with the row's own dates, or is blanked -- never guessed.
#[test]
fn reconcile_day_derives_or_clears() {
    let start = d(2026, 8, 7);
    let end = d(2026, 8, 9);

    // Single date in window -> day corrected to that weekday.
    let mut single = dated("Solo", "2026-08-08", "");
    single.day = "Monday".into();
    let (fixed, notes) = reconcile_day_with_dates(vec![single], start, end);
    assert_eq!(fixed[0].day, "Saturday", "{notes:?}");
    assert!(!notes.is_empty() && notes[0].contains("corrected"), "{notes:?}");

    // Multi-day range with a stated day outside it -> cleared, not guessed.
    let mut multi = dated("Long", "2026-08-07", "2026-08-09");
    multi.day = "Monday".into();
    let (fixed, notes) = reconcile_day_with_dates(vec![multi], start, end);
    assert_eq!(fixed[0].day, "", "{notes:?}");
    assert!(!notes.is_empty() && notes[0].contains("cleared"), "{notes:?}");

    // A row entirely outside the window is not this function's job.
    let mut old = dated("Old", "2026-07-01", "2026-07-01");
    old.day = "Wednesday".into();
    let (fixed, notes) = reconcile_day_with_dates(vec![old], start, end);
    assert_eq!(fixed[0].day, "Wednesday");
    assert!(notes.is_empty(), "{notes:?}");

    // A reversed range is swapped before the window overlap is judged.
    let mut backwards = dated("Backwards", "2026-08-09", "2026-08-08");
    backwards.day = "Tuesday".into();
    let (fixed, notes) = reconcile_day_with_dates(vec![backwards], start, end);
    assert_eq!(notes.len(), 1, "{notes:?}");
    assert!(
        fixed[0].day == "Saturday" || fixed[0].day == "Sunday" || fixed[0].day.is_empty(),
        "{}",
        fixed[0].day
    );
}

// ---------------------------------------------------------------------------
// Provenance gate (drop_unsourced_rows / row_is_sourced)
// ---------------------------------------------------------------------------

/// A name that traces to the fetched corpus survives; an invented one is dropped.
#[test]
fn a_row_that_traces_to_the_corpus_survives_but_invention_is_dropped() {
    let corpus = "Harbour Kite Festival at Pier 4 Toronto this weekend";
    let real = event("Harbour Kite Festival", "Pier 4");
    let invented = event("Quantum Levitation Workshop", "Mississauga");
    let (kept, notes) = drop_unsourced_rows(vec![real, invented], corpus);
    assert_eq!(kept.len(), 1, "{notes:?}");
    assert_eq!(kept[0].name, "Harbour Kite Festival");
    assert_eq!(notes.len(), 1, "{notes:?}");
    assert!(notes[0].contains("Quantum Levitation"), "{notes:?}");
}

/// Coverage is a fraction, not all-or-nothing: 2 of 3 significant words is
/// enough (0.66 >= 0.6), one of three is not (0.33 < 0.6).
#[test]
fn provenance_coverage_is_a_fraction_not_all_or_nothing() {
    let corpus = "apple banana cherry daily events in vaughan";
    // "apple banana" both in corpus -> 2/2 kept.
    assert!(row_is_sourced("Apple Banana Festival", corpus));
    // "apple plum" -> plum missing, 1/2 = 0.5 < 0.6 -> dropped.
    assert!(!row_is_sourced("Apple Plum Festival", corpus));
}

/// An unnamed row has nothing to check; empty corpus judges nothing.
#[test]
fn unnamed_rows_and_empty_corpus_are_never_dropped() {
    assert!(row_is_sourced("", "anything"));
    assert!(row_is_sourced("   ", "anything"));

    let (kept, notes) = drop_unsourced_rows(vec![event("X", "Y")], "");
    assert_eq!(kept.len(), 1);
    assert!(notes.is_empty());
}

// ---------------------------------------------------------------------------
// Supply prioritisation (supply.py)
// ---------------------------------------------------------------------------

fn friday() -> chrono::NaiveDate {
    chrono::NaiveDate::parse_from_str("2026-08-07", "%Y-%m-%d").unwrap()
}

fn sunday() -> chrono::NaiveDate {
    chrono::NaiveDate::parse_from_str("2026-08-09", "%Y-%m-%d").unwrap()
}

/// In-window candidates float to the top marked, order preserved within each
/// group; out-of-window lines stay below in their own order.
#[test]
fn in_window_lines_float_to_the_top_marked_but_nothing_is_removed() {
    let corpus = "First entry\nAug 15 festival all month\nSecond entry\nAug 8 happening this weekend";
    let out = crate::ztools::weekend::prioritise_in_window(corpus, friday(), sunday());
    let lines: Vec<&str> = out.lines().collect();
    // Both in-window lines (Aug 8) float up, marked; everything else follows in order.
    assert_eq!(lines.len(), 4, "{out:?}");
    assert!(lines[0].starts_with("[THIS WEEKEND]"), "{out:?}");
    assert!(lines[0].ends_with("happening this weekend"), "{out:?}");
    assert_eq!(lines[1], "First entry");
    assert_eq!(lines[2], "Aug 15 festival all month");
    assert_eq!(lines[3], "Second entry");
    assert!(out.contains("Aug 15 festival all month"), "nothing may be removed");
}

/// A corpus with no dated candidates is unchanged verbatim -- inventing a
/// marker would tell the model something untrue.
#[test]
fn corpus_with_no_dated_candidates_is_returned_unchanged() {
    let corpus = "Evergreen venue listing\nAnother evergreen listing";
    let out = crate::ztools::weekend::prioritise_in_window(corpus, friday(), sunday());
    assert_eq!(out, corpus);
}

/// The count reports how many candidates land in the window -- the number that
/// distinguishes a supply problem from a model problem.
#[test]
fn in_window_count_counts_only_lines_that_mention_the_window() {
    let corpus = "Aug 8 festival\nEvergreen\nAug 9 show\nAug 15 out of window";
    assert_eq!(in_window_count(corpus, friday(), sunday()), 2);
    assert_eq!(in_window_count("no dates here at all", friday(), sunday()), 0);
}