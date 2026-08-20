//! Unit tests for Rust Weekend Planner module.

use super::*;

#[test]
fn test_matches_exclusion() {
    assert!(matches_exclusion(
        "Sky Zone Toronto",
        "Sky Zone Trampoline Park, Toronto"
    ));
    assert!(matches_exclusion(
        "Toronto Zoo",
        "Canada Day at Your Toronto Zoo"
    ));
    assert!(!matches_exclusion("Toronto Zoo", "Toronto Islands Park"));
}

#[test]
fn test_matches_exclusion_possessives() {
    assert!(matches_exclusion("Ripley's Aquarium", "Ripley’s Aquarium of Canada"));
}

#[test]
fn test_filter_exclusions() {
    let events = vec![
        WeekendEvent {
            name: "Canada Day at Your Toronto Zoo".into(),
            location: "Toronto Zoo".into(),
            price: "$20".into(),
            target_ages: "6-13".into(),
            day: "Friday".into(),
            dates: "2026-07-01".into(),
            description: "Zoo event".into(),
            is_transient: true,
            score: 0.0,
            start_date: "".into(),
            end_date: "".into(),
            weather: "".into(),
            duration: "".into(),
},
        WeekendEvent {
            name: "Local Library Story Time".into(),
            location: "Vaughan Library".into(),
            price: "Free".into(),
            target_ages: "3-8".into(),
            day: "Saturday".into(),
            dates: "2026-08-08".into(),
            description: "Library story time".into(),
            is_transient: true,
            score: 0.0,
            start_date: "".into(),
            end_date: "".into(),
            weather: "".into(),
            duration: "".into(),
},
    ];
    let exclusions = vec!["Toronto Zoo".into()];
    let (filtered, notes) = drop_excluded_places(events, &exclusions);
    assert_eq!(filtered.len(), 1, "{notes:?}");
    assert_eq!(filtered[0].name, "Local Library Story Time");
}

#[test]
fn test_flag_constant_columns() {
    let events = vec![
        WeekendEvent {
            name: "Event 1".into(),
            location: "Vaughan".into(),
            price: "$18-35 per child or free".into(),
            target_ages: "6-13".into(),
            day: "Friday".into(),
            dates: "2026-08-07".into(),
            description: "Test description 1".into(),
            is_transient: true,
            score: 0.0,
            start_date: "".into(),
            end_date: "".into(),
            weather: "".into(),
            duration: "".into(),
},
        WeekendEvent {
            name: "Event 2".into(),
            location: "Toronto".into(),
            price: "$18-35 per child or free".into(),
            target_ages: "6-13".into(),
            day: "Saturday".into(),
            dates: "2026-08-08".into(),
            description: "Test description 2".into(),
            is_transient: true,
            score: 0.0,
            start_date: "".into(),
            end_date: "".into(),
            weather: "".into(),
            duration: "".into(),
},
    ];

    // The column is constant AND carries the configured family range, so it is
    // flagged as answered from the prompt.
    let mut suspects = std::collections::HashMap::new();
    suspects.insert("Target Age(s)".to_string(), vec!["6-13".to_string()]);
    let constants = flag_constant_columns(&events, &suspects);
    assert_eq!(constants.len(), 1, "{constants:?}");
    assert!(constants[0].contains("Target Age(s)"), "{constants:?}");
    assert!(constants[0].to_lowercase().contains("kept"), "{constants:?}");

    // Same constant, but no suspect value configured: NOT the C4 defect.
    let none: std::collections::HashMap<String, Vec<String>> = std::collections::HashMap::new();
    assert!(flag_constant_columns(&events, &none).is_empty());
}

#[test]
fn test_is_directory_or_list_page() {
    assert!(is_directory_or_list_page(
        "Best Kids Parks in Vaughan Ontario | TikTok"
    ));
    assert!(is_directory_or_list_page(
        "25+ Best Mother's Day Events in Toronto & GTA"
    ));
    assert!(is_directory_or_list_page(
        "10 Easy Adventures That Get Kids Off Screens"
    ));
    assert!(is_directory_or_list_page(
        "30 North Park Road 207 - Thornhill City Centre - MLS® Listing"
    ));
    assert!(!is_directory_or_list_page(
        "Kortright Centre for Conservation"
    ));
    assert!(!is_directory_or_list_page("Air Riderz Trampoline Park"));
}

#[test]
fn test_clean_venue_or_event_title() {
    assert_eq!(
        clean_venue_or_event_title("Air Riderz Trampoline Park in Toronto: Bounce into Adventure"),
        Some("Air Riderz Trampoline Park in Toronto: Bounce into Adventure".to_string())
    );
    assert_eq!(
        clean_venue_or_event_title(
            "Candyland Indoor Play Centre (Vaughan, Canada) - Đa... - Tripadvisor"
        ),
        Some("Candyland Indoor Play Centre (Vaughan, Canada)".to_string())
    );
    assert_eq!(
        clean_venue_or_event_title("Playdium Vaughan – YouTube"),
        Some("Playdium Vaughan".to_string())
    );
    assert_eq!(
        clean_venue_or_event_title("Best Kids Parks in Vaughan Ontario | TikTok"),
        None
    );
    assert_eq!(clean_venue_or_event_title("ab"), None);
}

#[test]
fn test_load_exclusions() {
    let excl = load_exclusions(&crate::config::ZtoolsConfig::default());
    assert!(!excl.is_empty());
    assert!(excl
        .iter()
        .any(|e| e.contains("Zoo") || e.contains("Wonderland")));
}

#[test]
fn test_load_cached_activities() {
    let (transient, fixed) = load_cached_activities(&crate::config::ZtoolsConfig::default());
    assert!(!transient.is_empty() || !fixed.is_empty());
}

/// With no exclusion file configured the curated list still loads, and it is
/// the *unfiltered* list. This is the branch CI takes; it used to depend on
/// whether the developer happened to have `~/.config/weekend.toml`.
#[test]
fn cached_activities_load_with_no_exclusion_file_configured() {
    let config = crate::config::ZtoolsConfig {
        weekend_exclusions_paths: vec![],
        ..crate::config::ZtoolsConfig::default()
    };
    let (_, fixed) = load_cached_activities(&config);
    assert!(!fixed.is_empty());
}

/// A configured exclusion file is read, and what it names is removed.
#[test]
fn a_configured_exclusion_file_filters_the_curated_list() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("weekend.toml");
    std::fs::write(&path, "exclude_places = [\n  \"Kortright\",\n]\n").unwrap();
    let config = crate::config::ZtoolsConfig {
        weekend_exclusions_paths: vec![path.to_string_lossy().into_owned()],
        ..crate::config::ZtoolsConfig::default()
    };
    let excl = load_exclusions(&config);
    assert_eq!(excl, vec!["Kortright".to_string()]);
    let (_, fixed) = load_cached_activities(&config);
    assert!(
        !fixed.iter().any(|e| e.name.contains("Kortright")),
        "an excluded venue survived: {:?}",
        fixed.iter().map(|e| &e.name).collect::<Vec<_>>()
    );
}

#[test]
fn test_format_weekend_plan_empty_and_populated() {
    let plan1 = format_weekend_plan(&[], &[], "Vaughan", "6-13", "Aug 07-09", "Sat 24C (clear)");
    assert!(plan1.contains("# Weekend Plan: Aug 07-09 (Vaughan)"));
    assert!(plan1.contains("Score"));
    // The forecast is the caller's, printed verbatim -- not refetched for a
    // hardcoded weekend in August.
    assert!(plan1.contains("Sat 24C (clear)"), "{plan1}");
    assert!(plan1.contains("*No fixed activities listed.*"), "{plan1}");

    let transient = vec![WeekendEvent {
        name: "Harvest Fair".into(),
        location: "Vaughan".into(),
        price: "$10".into(),
        target_ages: "All Ages".into(),
        day: "Saturday".into(),
        dates: "2026-08-08".into(),
        description: "Annual harvest fair".into(),
        is_transient: true,
        score: 0.0,
        start_date: "".into(),
        end_date: "".into(),
        weather: "".into(),
        duration: "".into(),
}];
    let plan2 = format_weekend_plan(
        &transient,
        &[],
        "Vaughan",
        "6-13",
        "Aug 07-09",
        "Sat 24C (clear)",
    );
    assert!(plan2.contains("Harvest Fair"));
}

#[test]
fn test_format_weather_display() {
    let raw = "Daily Forecast:\n2026-08-07: 28.2°C, Clear (0.0mm)\n2026-08-08: 32.0°C, Precipitation (1.2mm)\n2026-08-09: 29.7°C, Clear (0.1mm)";
    let formatted = format_weather_display(raw);
    assert!(formatted.contains("Fri 28.2°C (clear)"));
    assert!(formatted.contains("Sat 32.0°C (precipitation)"));
    assert!(formatted.contains("Sun 29.7°C (clear)"));
}

#[test]
fn test_apply_scores_sorts_by_score() {
    let mut events = vec![
        crate::ztools::weekend::WeekendEvent {
            name: "A".into(),
            location: "Toronto".into(),
            price: "$10".into(),
            target_ages: "6-12".into(),
            day: "Fri".into(),
            dates: "Aug 7".into(),
            description: "fun".into(),
            is_transient: true,
            score: 0.0,
            start_date: "".into(),
            end_date: "".into(),
            weather: "".into(),
            duration: "".into(),
},
        crate::ztools::weekend::WeekendEvent {
            name: "B".into(),
            location: "Vaughan".into(),
            price: "Free".into(),
            target_ages: "all".into(),
            day: "Sat".into(),
            dates: "Aug 8".into(),
            description: "outdoor festival".into(),
            is_transient: true,
            score: 0.0,
            start_date: "".into(),
            end_date: "".into(),
            weather: "".into(),
            duration: "".into(),
},
    ];
    // Event B has more populated fields (outdoor description) and matching ages.
    crate::ztools::weekend::apply_scores(&mut events, "sunny clear warm", "6-12");
    // Scores should be computed and sorted descending.
    assert!(events[0].score >= events[1].score);
    assert!(events.iter().all(|e| e.score > 0.0));
}

#[test]
fn test_apply_scores_empty_ages() {
    let mut events = vec![crate::ztools::weekend::WeekendEvent {
        name: "X".into(),
        location: "Y".into(),
        price: "".into(),
        target_ages: "".into(),
        day: "".into(),
        dates: "".into(),
        description: "".into(),
        is_transient: true,
        score: 0.0,
        start_date: "".into(),
        end_date: "".into(),
        weather: "".into(),
        duration: "".into(),
}];
    // Empty age range: no age bonus, but populated fields still score.
    crate::ztools::weekend::apply_scores(&mut events, "rain", "");
    assert!(events[0].score > 0.0);
}

#[test]
/// The fan-out runs and yields nothing when neither the search nor the model
/// can be reached. Both endpoints point at a closed port so the outcome does
/// not depend on DuckDuckGo being up -- this test used to hit the live site
/// thirteen times, and whether it answered moved the coverage number.
fn test_fetch_duckduckgo_events() {
    let config = crate::config::ZtoolsConfig {
        duckduckgo_url: "http://127.0.0.1:1/".into(),
        osaurus_url: "http://127.0.0.1:1".into(),
        llm_timeout_secs: 1,
        ..crate::config::ZtoolsConfig::default()
    };
    let ctx = crate::ztools::weekend::PlanContext {
        location: "Vaughan".into(),
        ages: "6-12".into(),
        date_range: "Aug 7 to Aug 9".into(),
        year: 2026,
        exclusions: "none".into(),
    };
    let (events, corpus) = fetch_duckduckgo_events(
        "Vaughan",
        chrono::NaiveDate::parse_from_str("2026-08-07", "%Y-%m-%d").unwrap(),
        chrono::NaiveDate::parse_from_str("2026-08-09", "%Y-%m-%d").unwrap(),
        "sunny",
        &ctx,
        &config,
    );
    assert!(
        events.is_empty(),
        "unreachable search and model must yield nothing, not invented events: {events:?}"
    );
    assert!(corpus.is_empty());
}

fn sample_event(name: &str, location: &str, score: f32) -> crate::ztools::weekend::WeekendEvent {
    crate::ztools::weekend::WeekendEvent {
        name: name.into(),
        location: location.into(),
        price: "$10".into(),
        target_ages: "6-12".into(),
        day: "Friday".into(),
        dates: "Aug 7".into(),
        description: format!("{} in {}", name, location),
        is_transient: true,
        score,
        start_date: "".into(),
        end_date: "".into(),
        weather: "".into(),
        duration: "".into(),
    }
}

mod weekend_parse_tests;

mod weekend_filter_tests;

mod weekend_phases_tests;
