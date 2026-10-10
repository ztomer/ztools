//! The fit score measures fit: age share and weather, never filled fields or
//! price. Each expectation is hand-computed from the two terms in `score.rs`.

use super::*;

fn ev(target_ages: &str, name: &str, description: &str, price: &str) -> WeekendEvent {
    WeekendEvent {
        name: name.into(),
        location: "Vaughan".into(),
        price: price.into(),
        target_ages: target_ages.into(),
        day: "Sat".into(),
        dates: "Oct 10".into(),
        description: description.into(),
        is_transient: true,
        score: 0.0,
        start_date: String::new(),
        end_date: String::new(),
        weather: String::new(),
        duration: String::new(),
    }
}

const FAMILY: &[u32] = &[14, 11, 6];

fn close(actual: f32, expected: f32) {
    assert!(
        (actual - expected).abs() < 0.001,
        "expected {expected}, got {actual}"
    );
}

#[test]
fn a_price_is_not_a_fit_and_free_is_never_worse() {
    let paid = compute_score(&ev("6-14", "Art Camp", "indoor", "$25"), "", FAMILY);
    let free = compute_score(&ev("6-14", "Art Camp", "indoor", "Free"), "", FAMILY);
    assert!(free >= paid, "free {free} must not score below paid {paid}");
    close(free, paid);
}

#[test]
fn filling_in_fields_earns_nothing() {
    // The old score paid 3 points for five filled fields and half a point for
    // a long venue name. The same fit with every optional field blank must
    // score the same.
    let full = ev("6-14", "Art Camp", "indoor", "$25");
    let mut bare = full.clone();
    bare.location.clear();
    bare.price.clear();
    close(
        compute_score(&full, "", FAMILY),
        compute_score(&bare, "", FAMILY),
    );
}

#[test]
fn age_points_are_the_share_of_children_the_row_admits() {
    // 6-14 admits all three: 3.0. 10-14 admits two of three: 2.0. 13-19: one.
    close(age_points(&ev("6-14", "x", "", ""), FAMILY), 3.0);
    close(age_points(&ev("10-14", "x", "", ""), FAMILY), 2.0);
    close(
        age_points(&ev("", "Toastmasters for Teens", "", ""), FAMILY),
        1.0,
    );
    close(age_points(&ev("", "Baby and Me", "", ""), FAMILY), 0.0);
    // A row that says nothing about ages is unjudged: half marks.
    close(
        age_points(&ev("unknown", "Sing and Sign", "", ""), FAMILY),
        1.5,
    );
    // No family on record: nothing to judge against, half marks.
    close(age_points(&ev("6-14", "x", "", ""), &[]), 1.5);
}

#[test]
fn an_unlabelled_row_ranks_below_one_known_to_fit_everyone() {
    let mut rows = vec![
        ev("", "Sing and Sign", "", ""),
        ev("", "Family Robotics Workshop (ages 6-14)", "", ""),
    ];
    apply_scores(&mut rows, "", FAMILY);
    assert_eq!(rows[0].name, "Family Robotics Workshop (ages 6-14)");
    close(rows[0].score, 3.0);
    close(rows[1].score, 1.5);
}

#[test]
fn weather_points_follow_the_forecast() {
    close(weather_points(&ev("", "x", "indoor play", ""), "rain"), 1.0);
    close(
        weather_points(&ev("", "x", "outdoor fair", ""), "sunny"),
        2.0,
    );
    close(
        weather_points(&ev("", "x", "outdoor fair", ""), "cloudy rain"),
        0.0,
    );
    close(
        weather_points(&ev("", "x", "overcast walk", ""), "cloudy"),
        2.0,
    );
    close(
        weather_points(&ev("", "x", "sunny picnic", ""), "clear skies"),
        2.0,
    );
    close(
        weather_points(&ev("", "x", "sunny picnic", ""), "rain showers"),
        1.0,
    );
    close(weather_points(&ev("", "x", "fun", ""), "sunny"), 0.0);
}

#[test]
fn a_perfect_row_scores_exactly_the_ceiling() {
    let perfect = ev("6-14", "Fair", "outdoor festival", "Free");
    close(compute_score(&perfect, "sunny clear", FAMILY), MAX_SCORE);
}
