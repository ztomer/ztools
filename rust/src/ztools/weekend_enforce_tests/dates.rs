//! Date parsing, window overlap, day reconciliation, and provenance coverage.

use super::support::{d, event};
use crate::ztools::weekend::{
    drop_events_outside_window, drop_unsourced_rows, find_dates_in, parse_any_date,
    reconcile_day_with_dates, row_is_sourced, window_overlap, WeekendEvent,
};

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
        window_overlap(
            &dated("e", "Aug 15", "Aug 16"),
            d(2026, 8, 7),
            d(2026, 8, 9)
        ),
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
    assert!(
        !notes.is_empty() && notes[0].contains("corrected"),
        "{notes:?}"
    );

    // Multi-day range with a stated day outside it -> cleared, not guessed.
    let mut multi = dated("Long", "2026-08-07", "2026-08-09");
    multi.day = "Monday".into();
    let (fixed, notes) = reconcile_day_with_dates(vec![multi], start, end);
    assert_eq!(fixed[0].day, "", "{notes:?}");
    assert!(
        !notes.is_empty() && notes[0].contains("cleared"),
        "{notes:?}"
    );

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
