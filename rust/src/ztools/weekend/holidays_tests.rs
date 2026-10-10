//! The Ontario rule table, pinned date by date for two years.
//!
//! The expected dates are written out by hand from the published 2026 and
//! 2027 calendars rather than computed, so a rule that drifts (a "second
//! Monday" read as the first, Victoria Day on May 25 itself) goes red here
//! instead of quietly shifting a long weekend by a week.

use super::*;

fn d(y: i32, m: u32, day: u32) -> NaiveDate {
    NaiveDate::from_ymd_opt(y, m, day).unwrap()
}

fn table(year: i32) -> Vec<(&'static str, NaiveDate)> {
    Province::Ontario
        .holidays(year)
        .into_iter()
        .map(|h| (h.name, h.date))
        .collect()
}

#[test]
fn ontario_2026_holidays_fall_on_their_published_dates() {
    assert_eq!(
        table(2026),
        vec![
            ("New Year's Day", d(2026, 1, 1)),
            ("Family Day", d(2026, 2, 16)),
            ("Good Friday", d(2026, 4, 3)),
            ("Victoria Day", d(2026, 5, 18)),
            ("Canada Day", d(2026, 7, 1)),
            ("Civic Holiday", d(2026, 8, 3)),
            ("Labour Day", d(2026, 9, 7)),
            ("Thanksgiving", d(2026, 10, 12)),
            ("Christmas Day", d(2026, 12, 25)),
            // Boxing Day 2026 is a Saturday: observed the following Monday.
            ("Boxing Day", d(2026, 12, 28)),
        ]
    );
}

#[test]
fn ontario_2027_holidays_fall_on_their_published_dates() {
    assert_eq!(
        table(2027),
        vec![
            ("New Year's Day", d(2027, 1, 1)),
            ("Family Day", d(2027, 2, 15)),
            ("Good Friday", d(2027, 3, 26)),
            // May 24 2027 is itself a Monday: the last Monday before May 25.
            ("Victoria Day", d(2027, 5, 24)),
            ("Canada Day", d(2027, 7, 1)),
            ("Civic Holiday", d(2027, 8, 2)),
            ("Labour Day", d(2027, 9, 6)),
            ("Thanksgiving", d(2027, 10, 11)),
            // Christmas on a Saturday, Boxing Day on a Sunday: Mon and Tue.
            ("Christmas Day", d(2027, 12, 27)),
            ("Boxing Day", d(2027, 12, 28)),
        ]
    );
}

#[test]
fn easter_matches_known_years() {
    assert_eq!(easter_sunday(2024), d(2024, 3, 31));
    assert_eq!(easter_sunday(2025), d(2025, 4, 20));
    assert_eq!(easter_sunday(2026), d(2026, 4, 5));
    assert_eq!(easter_sunday(2027), d(2027, 3, 28));
    assert_eq!(easter_sunday(2038), d(2038, 4, 25));
}

#[test]
fn a_sunday_canada_day_is_observed_monday() {
    // July 1 2029 is a Sunday.
    assert!(Province::Ontario.holidays(2029).contains(&Holiday {
        name: "Canada Day",
        date: d(2029, 7, 2)
    }));
}

#[test]
fn holiday_on_finds_the_day_and_only_the_day() {
    assert_eq!(
        Province::Ontario
            .holiday_on(d(2026, 10, 12))
            .map(|h| h.name),
        Some("Thanksgiving")
    );
    assert_eq!(Province::Ontario.holiday_on(d(2026, 10, 19)), None);
}

#[test]
fn province_parses_code_and_name_and_refuses_the_rest() {
    assert_eq!(Province::parse(" ON "), Ok(Province::Ontario));
    assert_eq!(Province::parse("Ontario"), Ok(Province::Ontario));
    let err = Province::parse("QC").unwrap_err();
    assert!(
        err.contains("\"qc\"") && err.contains("supported: ON"),
        "{err}"
    );
}

#[test]
fn load_province_reads_the_location_table_and_states_every_failure() {
    let dir = tempfile::tempdir().unwrap();
    let write = |name: &str, body: &str| {
        let p = dir.path().join(name);
        std::fs::write(&p, body).unwrap();
        p.to_string_lossy().into_owned()
    };
    let good = write(
        "good.toml",
        "[location]\ncity = \"Vaughan\"\nprovince = \"ON\"\n",
    );
    let none = write("none.toml", "[location]\ncity = \"Vaughan\"\n");
    let other = write("other.toml", "[location]\nprovince = \"BC\"\n");
    let no_table = write("no_table.toml", "[llm]\nphase_retries = 1\n");
    let missing = dir
        .path()
        .join("absent.toml")
        .to_string_lossy()
        .into_owned();

    // The first file WITH a [location] table decides; files without one are skipped.
    assert_eq!(
        load_province(&[missing.clone(), no_table.clone(), good]),
        Ok(Province::Ontario)
    );
    assert!(
        load_province(&[none])
            .unwrap_err()
            .contains("without `province`")
    );
    assert!(load_province(&[other]).unwrap_err().contains("\"bc\""));
    assert!(
        load_province(&[missing, no_table])
            .unwrap_err()
            .contains("no weekend config with a [location] table")
    );
}

/// The shipped config names a province the table covers. Without this, a typo
/// in `conf/weekend.toml` would only surface as a failed scheduled run.
#[test]
fn the_shipped_weekend_config_names_a_supported_province() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .join("conf/weekend.toml");
    assert_eq!(
        load_province(&[path.to_string_lossy().into_owned()]),
        Ok(Province::Ontario)
    );
}

/// The window the planner AND the status page use. Thanksgiving 2026 is the
/// weekend the old Friday-plus-two window missed.
#[test]
fn the_plan_window_reaches_a_holiday_monday_and_only_a_holiday_monday() {
    use crate::ztools::weekend::plan_window;
    let on = Province::Ontario;
    // Thanksgiving 2026: from the Thursday run, during it, and ON the Monday.
    for today in [
        d(2026, 10, 8),
        d(2026, 10, 10),
        d(2026, 10, 11),
        d(2026, 10, 12),
    ] {
        assert_eq!(
            plan_window(today, on),
            (d(2026, 10, 9), d(2026, 10, 12)),
            "{today}"
        );
    }
    // The Tuesday after is the next, ordinary weekend.
    assert_eq!(
        plan_window(d(2026, 10, 13), on),
        (d(2026, 10, 16), d(2026, 10, 18))
    );
    // An ordinary Monday steps forward, not back.
    assert_eq!(
        plan_window(d(2026, 10, 19), on),
        (d(2026, 10, 23), d(2026, 10, 25))
    );
    // Every Monday holiday in both years extends its weekend.
    for (thursday, monday) in [
        (d(2026, 2, 12), d(2026, 2, 16)),
        (d(2026, 5, 14), d(2026, 5, 18)),
        (d(2026, 7, 30), d(2026, 8, 3)),
        (d(2026, 9, 3), d(2026, 9, 7)),
        (d(2026, 12, 24), d(2026, 12, 28)),
        (d(2027, 2, 11), d(2027, 2, 15)),
        (d(2027, 5, 20), d(2027, 5, 24)),
        (d(2027, 7, 29), d(2027, 8, 2)),
        (d(2027, 9, 2), d(2027, 9, 6)),
        (d(2027, 10, 7), d(2027, 10, 11)),
        (d(2027, 12, 23), d(2027, 12, 27)),
    ] {
        let friday = thursday + Duration::days(1);
        assert_eq!(plan_window(thursday, on), (friday, monday), "{thursday}");
    }
    // Good Friday is already the window's first day; Easter Monday is not a
    // holiday in Ontario, so the window stays Friday to Sunday.
    assert_eq!(
        plan_window(d(2026, 4, 2), on),
        (d(2026, 4, 3), d(2026, 4, 5))
    );
}
