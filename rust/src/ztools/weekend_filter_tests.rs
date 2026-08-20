//! Tests for the weekend planner's search-result filters and its display
//! formatting. Split from `weekend_tests.rs` for the house 400-line cap.
//!
//! These are the rules that decide what a live web search is allowed to put in
//! front of the family: a directory page, a foreign city, or a realtor listing
//! must never reach the plan, and a search title has to survive cleaning with
//! its meaning intact.

use crate::ztools::weekend::{
    flag_constant_columns, format_weather_display, render_weekend_plan_gorgeous, WeekendEvent,
};
use crate::ztools::weekend_cache::{
    clean_venue_or_event_title, has_region_evidence, is_directory_or_list_page, load_exclusions,
};

fn event(name: &str, location: &str, description: &str) -> WeekendEvent {
    WeekendEvent {
        name: name.into(),
        location: location.into(),
        price: "$10".into(),
        target_ages: "6-12".into(),
        day: "Sat".into(),
        dates: "Aug 8".into(),
        description: description.into(),
        is_transient: true,
        score: 4.0,
    }
}

/// A title in a script the plan cannot render is not a GTA venue — it is a
/// scraped result from somewhere else entirely.
#[test]
fn a_non_latin_title_is_treated_as_a_directory_page() {
    assert!(is_directory_or_list_page("Достопримечательности Торонто"));
    assert!(!is_directory_or_list_page("Kortright Centre"));
}

/// A bare city name is a place, not something to do in it.
#[test]
fn a_bare_city_name_is_not_an_activity() {
    for bare in [
        "Toronto",
        "vaughan",
        "Ontario",
        "Canada",
        "Toronto, Ontario",
    ] {
        assert!(
            is_directory_or_list_page(bare),
            "{bare} should be filtered out"
        );
    }
}

/// Real-estate and hotel listings match the same searches family events do.
#[test]
fn address_and_listing_shapes_are_filtered_out() {
    assert!(is_directory_or_list_page(
        "123 Maple Road Suite 400, Vaughan"
    ));
    assert!(is_directory_or_list_page("3 bedroom rental in Vaughan"));
}

/// Listicles ("10 Best Things To Do…") are directories of other pages, not
/// events with a time and a place.
#[test]
fn a_numbered_listicle_is_filtered_out() {
    assert!(is_directory_or_list_page("10 Best Things To Do in Vaughan"));
    assert!(is_directory_or_list_page("25 top attractions"));
    // A number that is not the start of a listicle must survive.
    assert!(!is_directory_or_list_page("5 Pin Bowling at Woodbridge"));
}

/// A result naming a foreign city is rejected outright, however many GTA words
/// it also contains — the wrong continent is not a partial match.
#[test]
fn a_foreign_city_beats_any_local_token() {
    assert!(!has_region_evidence("New York street festival"));
    assert!(!has_region_evidence("Toronto Blue Jays play in Chicago"));
    assert!(has_region_evidence("Family day at Vaughan Mills"));
}

/// Aggregator suffixes get stripped so the venue keeps its own name rather
/// than being titled after the site that listed it.
#[test]
fn aggregator_suffixes_are_stripped_from_titles() {
    for site in [
        "Yelp",
        "Narcity",
        "Realtor.ca",
        "HotelsByDay",
        "Medium",
        "Tripadvisor",
        "TikTok",
        "YouTube",
    ] {
        let raw = format!("**Kortright Centre Vaughan** - {site}");
        assert_eq!(
            clean_venue_or_event_title(&raw).as_deref(),
            Some("Kortright Centre Vaughan"),
            "suffix {site} was not stripped"
        );
    }
}

/// A title that cleans down to nothing usable is dropped rather than shown.
#[test]
fn a_title_that_cleans_down_to_nothing_is_dropped() {
    // Stripping the aggregator suffix leaves a fragment with no region
    // evidence, so there is nothing left to trust.
    assert_eq!(clean_venue_or_event_title("Some Cafe - Yelp Toronto"), None);
    // No region evidence at all.
    assert_eq!(clean_venue_or_event_title("Some Random Cafe"), None);
    // A directory page.
    assert_eq!(clean_venue_or_event_title("Toronto"), None);
}

/// With no exclusion file present the built-in defaults still apply. An empty
/// list would silently disable the filter, letting the venues the operator
/// deliberately excluded back into the plan.
#[test]
fn exclusions_fall_back_to_the_built_in_defaults() {
    let config = crate::config::ZtoolsConfig {
        weekend_exclusions_paths: vec!["/definitely/not/a/file.toml".to_string()],
        ..crate::config::ZtoolsConfig::default()
    };
    let excl = load_exclusions(&config);
    assert!(!excl.is_empty());
    assert!(excl.iter().any(|e| e.contains("Wonderland")), "{excl:?}");
}

/// An event with no location is titled by its name alone — "Rib Fest ()" would
/// read as missing data rather than as an event with nowhere stated.
#[test]
fn an_event_without_a_location_renders_as_its_name_alone() {
    let out = render_weekend_plan_gorgeous(
        "Aug 7-9",
        "Sunny",
        &[event("Nameless Venue", "", "fixed thing")],
        &[event("Nameless Event", "", "transient thing")],
    );
    assert!(out.contains("Nameless Venue"), "{out}");
    assert!(out.contains("Nameless Event"), "{out}");
    assert!(!out.contains("()"), "an empty location was rendered: {out}");
}

/// Flagging a constant column needs at least two rows to compare; one row is
/// not evidence that a column never varies.
#[test]
fn a_single_row_flags_no_constant_columns() {
    assert!(flag_constant_columns(&[]).is_empty());
    assert!(flag_constant_columns(&[event("Solo", "Vaughan", "d")]).is_empty());
}

#[test]
fn identical_columns_across_rows_are_flagged() {
    let flags = flag_constant_columns(&[event("A", "Vaughan", "d"), event("B", "Markham", "d")]);
    assert!(flags.contains(&"price".to_string()), "{flags:?}");
    assert!(flags.contains(&"target_ages".to_string()), "{flags:?}");
}

/// A forecast line the parser cannot decompose is passed through verbatim
/// rather than dropped — a half-understood forecast still beats no forecast.
#[test]
fn an_unparseable_forecast_line_is_passed_through() {
    let out = format_weather_display("Daily Forecast:\n2026-08-07: hot and humid");
    assert_eq!(out, "2026-08-07: hot and humid");

    // A line whose date is not a date is passed through too.
    let out = format_weather_display("Saturday: 28.2°C, Clear (0.0mm)");
    assert_eq!(out, "Saturday: 28.2°C, Clear (0.0mm)");
}

/// With nothing parseable at all the display falls back to a stated default
/// rather than rendering an empty weather field.
#[test]
fn an_empty_forecast_falls_back_to_a_stated_default() {
    let out = format_weather_display("Daily Forecast:\n\n");
    assert!(out.contains("Fri"), "{out}");
    assert!(out.contains("Sun"), "{out}");
}
