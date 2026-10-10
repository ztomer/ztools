//! What the planner asks the engines (`weekend/queries.rs`).

use super::*;

fn d(y: i32, m: u32, day: u32) -> chrono::NaiveDate {
    chrono::NaiveDate::from_ymd_opt(y, m, day).unwrap()
}

#[test]
fn test_build_search_queries_derives_month_from_target_friday() {
    let queries = build_search_queries(d(2026, 9, 4), d(2026, 9, 6), None);
    assert_nonempty!(&queries);
    assert!(
        queries
            .iter()
            .any(|q| q.contains("September") && q.contains("2026"))
    );
    assert!(
        queries
            .iter()
            .any(|q| q.contains("harvest festival farm pumpkin"))
    );
    assert!(!queries.iter().any(|q| q.contains("August")));

    let jan_queries = build_search_queries(d(2027, 1, 1), d(2027, 1, 3), None);
    assert!(
        jan_queries
            .iter()
            .any(|q| q.contains("January") && q.contains("2027"))
    );
    assert!(
        jan_queries
            .iter()
            .any(|q| q.contains("winter festival holiday lights"))
    );
}

/// The weekend being planned is asked about BY ITS DATES, and first: on
/// Thanksgiving 2026 every query was month-scoped, and the pages titled for
/// that weekend ("Things To Do In Vaughan This Weekend | October 9-11, 2026")
/// were never asked for.
#[test]
fn the_window_is_asked_about_by_its_dates_and_first() {
    let queries = build_search_queries(d(2026, 10, 9), d(2026, 10, 12), None);
    for (city, q) in ["Vaughan", "Markham", "Richmond Hill", "Toronto"]
        .iter()
        .zip(&queries)
    {
        assert_eq!(*q, format!("{city} events this weekend October 9-12 2026"));
    }
}

/// A window that crosses a month names both months.
#[test]
fn a_window_across_a_month_names_both_months() {
    let queries = build_search_queries(d(2026, 10, 30), d(2026, 11, 1), None);
    assert_eq!(
        queries[0],
        "Vaughan events this weekend October 30 - November 1 2026"
    );
}

/// The holiday that made it a long weekend is searched for by name, and only
/// when the window holds one.
#[test]
fn a_holiday_in_the_window_is_searched_for_by_name() {
    let with = build_search_queries(d(2026, 10, 9), d(2026, 10, 12), Some("Thanksgiving"));
    let named: Vec<&String> = with
        .iter()
        .filter(|q| q.starts_with("Thanksgiving weekend"))
        .collect();
    assert_eq!(
        named,
        [
            "Thanksgiving weekend family events near Toronto 2026",
            "Thanksgiving weekend harvest festival farm pumpkin near Toronto 2026",
        ],
        "{with:?}"
    );
    let without = build_search_queries(d(2026, 10, 2), d(2026, 10, 4), None);
    assert!(
        !without
            .iter()
            .any(|q| q.contains("weekend family events near")),
        "{without:?}"
    );
}

/// "GTA" is Grand Theft Auto to every engine: all four "GTA ..." queries on
/// 2026-10-10 came back about the game, on Bing and on Brave.
#[test]
fn no_query_names_the_region_by_its_acronym() {
    let queries = build_search_queries(d(2026, 10, 9), d(2026, 10, 12), Some("Thanksgiving"));
    assert!(
        !queries
            .iter()
            .any(|q| q.split_whitespace().any(|w| w.eq_ignore_ascii_case("gta"))),
        "{queries:?}"
    );
}
