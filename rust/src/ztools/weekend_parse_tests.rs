//! Additional weekend tests. Split from `weekend_tests.rs` for the file cap.
use super::*;

#[test]
fn test_parse_weather_json_sunny() {
    let json: serde_json::Value = serde_json::json!({
        "daily": {
            "time": ["2026-08-07", "2026-08-08"],
            "temperature_2m_max": [28.2, 32.0],
            "precipitation_sum": [0.0, 0.0]
        }
    });
    let forecast = crate::ztools::weekend::parse_weather_json(&json).unwrap();
    assert!(forecast.contains("Daily Forecast"));
    assert!(forecast.contains("2026-08-07: 28.2°C"));
    assert!(forecast.contains("Clear"));
}

#[test]
fn test_parse_weather_json_precipitation() {
    let json: serde_json::Value = serde_json::json!({
        "daily": {
            "time": ["2026-08-07"],
            "temperature_2m_max": [22.0],
            "precipitation_sum": [5.0]
        }
    });
    let forecast = crate::ztools::weekend::parse_weather_json(&json).unwrap();
    assert!(forecast.contains("Precipitation"));
    assert!(forecast.contains("5.0mm"));
}

#[test]
fn test_parse_weather_json_empty_returns_none() {
    let json: serde_json::Value = serde_json::json!({
        "daily": {
            "time": [],
            "temperature_2m_max": [],
            "precipitation_sum": []
        }
    });
    assert!(crate::ztools::weekend::parse_weather_json(&json).is_none());
}

#[test]
fn test_parse_weather_json_missing_fields_returns_none() {
    assert!(crate::ztools::weekend::parse_weather_json(&serde_json::json!({})).is_none());
    assert!(
        crate::ztools::weekend::parse_weather_json(&serde_json::json!({"daily": {}})).is_none()
    );
}

#[test]
fn test_parse_llm_events_valid_response() {
    let resp: serde_json::Value = serde_json::json!({
        "choices": [{"message": {"content": "{\"transient_events\":[{\"name\":\"Rib Fest\",\"location\":\"Vaughan\",\"target_ages\":\"all\",\"price\":\"free\",\"start_date\":\"2026-08-07\",\"day\":\"Friday\",\"description\":\"Summer festival\"}]}"}}]
    });
    let events = crate::ztools::weekend::parse_llm_events(&resp).unwrap();
    assert_eq!(events.len(), 1);
    assert_eq!(events[0].name, "Rib Fest");
    assert_eq!(events[0].location, "Vaughan");
    assert!(events[0].is_transient);
}

#[test]
fn test_parse_llm_events_with_code_fence() {
    let resp: serde_json::Value = serde_json::json!({
        "choices": [{"message": {"content": "```json\n{\"transient_events\":[{\"name\":\"Magic Show\"}]}\n```"}}]
    });
    let events = crate::ztools::weekend::parse_llm_events(&resp).unwrap();
    assert_eq!(events.len(), 1);
    assert_eq!(events[0].name, "Magic Show");
}

#[test]
fn test_parse_llm_events_empty_defaults() {
    let resp: serde_json::Value = serde_json::json!({
        "choices": [{"message": {"content": "{\"transient_events\":[{\"name\":\"X\"}]}"}}]
    });
    let events = crate::ztools::weekend::parse_llm_events(&resp).unwrap();
    assert_eq!(events.len(), 1);
    assert_eq!(events[0].price, "unknown");
    assert_eq!(events[0].target_ages, "unknown");
    assert_eq!(events[0].day, "This Weekend");
    assert!(
        events[0].description.is_empty(),
        "a missing description is a missing value, never a copy of the name: {:?}",
        events[0]
    );
}

#[test]
fn test_parse_llm_events_invalid_returns_none() {
    assert!(crate::ztools::weekend::parse_llm_events(&serde_json::json!({})).is_none());
    assert!(
        crate::ztools::weekend::parse_llm_events(&serde_json::json!({"choices": []})).is_none()
    );
    assert!(
        crate::ztools::weekend::parse_llm_events(
            &serde_json::json!({"choices": [{"message": {"content": "not json"}}]})
        )
        .is_none()
    );
}

#[test]
fn test_render_weekend_plan_gorgeous_fixed_activities() {
    let fixed = vec![
        sample_event("Museum", "Toronto", 4.5),
        sample_event("Zoo", "Vaughan", 3.8),
    ];
    let output =
        crate::ztools::weekend::render_weekend_plan_gorgeous("Aug 7-9", "Sunny 28°C", &fixed, &[]);
    assert!(output.contains("Weekend Plan: Aug 7-9"));
    assert!(output.contains("Sunny 28°C"));
    assert!(output.contains("Fixed / Year-Round Activities"));
    assert!(output.contains("Museum"));
    assert!(output.contains("Zoo"));
    assert!(output.contains("* 4.5/5"));
}

#[test]
fn test_render_weekend_plan_gorgeous_transient_events() {
    let transient = vec![sample_event("Rib Fest", "Vaughan", 4.0)];
    let output = crate::ztools::weekend::render_weekend_plan_gorgeous(
        "Aug 7-9",
        "Clear 25°C",
        &[],
        &transient,
    );
    assert!(output.contains("Transient / Limited-Time Events"));
    assert!(output.contains("Rib Fest"));
}

#[test]
fn test_render_weekend_plan_gorgeous_empty() {
    let output = crate::ztools::weekend::render_weekend_plan_gorgeous("Aug 7-9", "Sunny", &[], &[]);
    assert!(output.contains("Weekend Plan"));
    assert!(!output.contains("Fixed / Year-Round"));
    assert!(!output.contains("Transient / Limited-Time"));
}

#[test]
fn test_seasonal_keywords_all_months() {
    use crate::ztools::weekend::seasonal_keywords;
    assert_nonempty!(seasonal_keywords("June"));
    assert_nonempty!(seasonal_keywords("July"));
    assert_nonempty!(seasonal_keywords("August"));
    assert_nonempty!(seasonal_keywords("September"));
    assert_nonempty!(seasonal_keywords("December"));
    assert_nonempty!(seasonal_keywords("March"));
}

#[test]
fn test_matches_exclusion_empty() {
    // An empty exclusion never matches.
    assert!(!crate::ztools::weekend::matches_exclusion("anything", ""));
}

/// A model that emits gemma-style keys (`activity`/`place`/`ages`) instead of
/// the canonical schema must still yield a named, located event. Port of
/// `test_dict_with_alt_keys` in `test_weekend_llm_1.py`: the Python pipeline
/// normalizes alternate keys before structuring, the Rust one used to drop
/// them silently and emit an empty name/location.
#[test]
fn test_parse_llm_events_normalizes_alternate_keys() {
    let resp: serde_json::Value = serde_json::json!({
        "choices": [{"message": {"content": "{\"transient_events\":[{\"activity\":\"My Activity\",\"place\":\"My Place\",\"ages\":\"5-10\",\"cost\":\"Free\",\"indoor_outdoor\":\"outdoor\",\"event_date\":\"Saturday\",\"time\":\"2 hours\"}]}"}}]
    });
    let events = crate::ztools::weekend::parse_llm_events(&resp).unwrap();
    assert_eq!(events.len(), 1);
    assert_eq!(events[0].name, "My Activity");
    assert_eq!(events[0].location, "My Place");
    assert_eq!(events[0].target_ages, "5-10");
    assert_eq!(events[0].price, "Free");
    assert_eq!(events[0].weather, "outdoor");
    assert_eq!(events[0].day, "Saturday");
    assert_eq!(events[0].duration, "2 hours");
}

/// A present canonical key wins over an alternate one, even when the
/// canonical value is empty: presence, not emptiness, decides. Port of
/// `test_field_mapping_already_has_standard`.
#[test]
fn test_parse_llm_events_canonical_key_beats_alternate_key() {
    let resp: serde_json::Value = serde_json::json!({
        "choices": [{"message": {"content": "{\"transient_events\":[{\"name\":\"Bar\",\"activity\":\"Foo\"}]}"}}]
    });
    let events = crate::ztools::weekend::parse_llm_events(&resp).unwrap();
    assert_eq!(events.len(), 1);
    assert_eq!(events[0].name, "Bar");
}
