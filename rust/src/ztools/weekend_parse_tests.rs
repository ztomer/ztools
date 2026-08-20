//! Additional weekend tests. Split from weekend_tests.rs for the file cap.
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
}

#[test]
fn test_parse_llm_events_invalid_returns_none() {
    assert!(crate::ztools::weekend::parse_llm_events(&serde_json::json!({})).is_none());
    assert!(
        crate::ztools::weekend::parse_llm_events(&serde_json::json!({"choices": []})).is_none()
    );
    assert!(crate::ztools::weekend::parse_llm_events(
        &serde_json::json!({"choices": [{"message": {"content": "not json"}}]})
    )
    .is_none());
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
    crate::ztools::weekend::apply_scores(&mut events, "sunny clear warm", "6-12");
    assert!(events[0].score >= events[1].score);
    assert!(events.iter().all(|e| e.score > 0.0));
}

#[test]
fn test_compute_score_indoor_gets_bonus() {
    // Indoor activity gets a +1.0 bonus (before the /2 division).
    let ev = crate::ztools::weekend::WeekendEvent {
        name: "Museum".into(),
        location: "Toronto".into(),
        price: "Free".into(),
        target_ages: "all".into(),
        day: "Sat".into(),
        dates: "Aug 8".into(),
        description: "indoor exhibits".into(),
        is_transient: true,
        score: 0.0,
        start_date: "".into(),
        end_date: "".into(),
        weather: "".into(),
        duration: "".into(),
    };
    let score = crate::ztools::weekend::compute_score(&ev, "rain precipitation", "all");
    // 5 fields = 3.0 base, +1.0 indoor, +0.5 location > 5 chars = 4.5 / 2 = 2.25.
    assert!((score - 2.25).abs() < 0.01, "expected ~2.25, got {score}");
}

#[test]
fn test_compute_score_age_overlap_bonus() {
    // Overlapping age ranges give a +3.0 bonus (before /2).
    let ev = crate::ztools::weekend::WeekendEvent {
        name: "Camp".into(),
        location: "Vaughan Park".into(),
        price: "$20".into(),
        target_ages: "6-12".into(),
        day: "Mon".into(),
        dates: "Aug 7".into(),
        description: "fun".into(),
        is_transient: true,
        score: 0.0,
        start_date: "".into(),
        end_date: "".into(),
        weather: "".into(),
        duration: "".into(),
    };
    let score = crate::ztools::weekend::compute_score(&ev, "cloudy", "6-12");
    // 5 fields = 3.0, age overlap (6-12 ∩ 6-12 = 7 ≥ 2) = +3.0,
    // price not free = +0.5, location > 5 = +0.5. Total 7.0 / 2 = 3.5.
    assert!((score - 3.5).abs() < 0.01, "expected ~3.5, got {score}");
}

#[test]
fn test_compute_score_caps_at_5() {
    // A perfect event should not exceed 5.0.
    let ev = crate::ztools::weekend::WeekendEvent {
        name: "Perfect".into(),
        location: "Vaughan Woods".into(),
        price: "$50".into(),
        target_ages: "6-12".into(),
        day: "Fri".into(),
        dates: "Aug 7".into(),
        description: "outdoor sunny fun festival".into(),
        is_transient: true,
        score: 0.0,
        start_date: "".into(),
        end_date: "".into(),
        weather: "".into(),
        duration: "".into(),
    };
    let score = crate::ztools::weekend::compute_score(&ev, "sunny clear warm", "6-12");
    assert!(score <= 5.0, "score should cap at 5.0, got {score}");
    assert!(score >= 4.5, "perfect event should score high, got {score}");
}

#[test]
fn test_seasonal_keywords_all_months() {
    use crate::ztools::weekend::_seasonal_keywords;
    assert!(_seasonal_keywords("June").is_some());
    assert!(_seasonal_keywords("July").is_some());
    assert!(_seasonal_keywords("August").is_some());
    assert!(_seasonal_keywords("September").is_some());
    assert!(_seasonal_keywords("December").is_some());
    assert!(_seasonal_keywords("March").is_some());
}

#[test]
fn test_matches_exclusion_empty() {
    // An empty exclusion never matches.
    assert!(!crate::ztools::weekend::matches_exclusion("anything", ""));
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
    crate::ztools::weekend::apply_scores(&mut events, "rain", "");
    assert!(events[0].score > 0.0);
}
