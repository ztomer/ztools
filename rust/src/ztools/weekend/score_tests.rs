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
    // Neither carries a weather label: both take the neutral 1.0 on top.
    close(rows[0].score, 4.0);
    close(rows[1].score, 2.5);
}

fn labelled(weather: &str, description: &str) -> WeekendEvent {
    let mut row = ev("", "x", description, "");
    weather.clone_into(&mut row.weather);
    row
}

/// The weather term reads the row's weather LABEL against the forecast, as the
/// Python predecessor did (`item["weather"]`) and this module's doc says. The
/// Rust port read the DESCRIPTION instead, so a row the structure phase
/// labelled "outdoor" scored nothing unless its blurb happened to contain the
/// word, and a blurb saying "warm drinks" earned the clear-sky bonus.
#[test]
fn weather_points_read_the_label_against_the_forecast() {
    let clear = "Fri 19.2°C (clear), Sat 15.6°C (clear)";
    let wet = "Fri 12.0°C (precipitation), Sat 11.0°C (precipitation)";
    close(weather_points(&labelled("outdoor", ""), clear), 2.0);
    close(weather_points(&labelled("outdoor", ""), wet), 0.0);
    close(weather_points(&labelled("indoor", ""), clear), 1.0);
    close(weather_points(&labelled("indoor", ""), wet), 1.0);
    close(weather_points(&labelled("both", ""), clear), 2.0);
    close(weather_points(&labelled("both", ""), wet), 1.0);
    close(weather_points(&labelled("Outdoor ", ""), clear), 2.0);
}

/// The description is not the label: whatever words it holds, they move
/// nothing. The old code paid 2.0 for "sunny picnic" under a clear sky and 0.0
/// for an "outdoor"-labelled fair whose blurb never said "outdoor".
#[test]
fn the_description_never_decides_the_weather_term() {
    let clear = "Sat 15.6°C (clear)";
    close(
        weather_points(&labelled("", "outdoor fair, sunny and warm"), clear),
        1.0,
    );
    close(
        weather_points(&labelled("outdoor", "harvest market"), clear),
        2.0,
    );
    close(
        weather_points(&labelled("indoor", "outdoor fair"), "rain"),
        1.0,
    );
}

/// UNKNOWN IS NOT KNOWN (honest placeholders). A row the model could not
/// place -- the structure prompt now says "" for that -- and a forecast that
/// is missing or mixed earn the neutral middle, like an unjudged age range.
/// They used to earn 0.0: an unlabelled row ranked as though it were an
/// outdoor event in the rain.
#[test]
fn an_unknown_label_or_forecast_scores_neutral() {
    let clear = "Sat 15.6°C (clear)";
    for label in ["", "unknown", "n/a"] {
        close(weather_points(&labelled(label, ""), clear), 1.0);
    }
    close(
        weather_points(&labelled("outdoor", ""), "⚠ Forecast unavailable"),
        1.0,
    );
    close(weather_points(&labelled("outdoor", ""), ""), 1.0);
    // Clear Friday, wet Saturday: which day the row is on is not read here,
    // so the forecast cannot be called either way.
    close(
        weather_points(
            &labelled("outdoor", ""),
            "Fri 19.2°C (clear), Sat 12.0°C (precipitation)",
        ),
        1.0,
    );
}

#[test]
fn a_perfect_row_scores_exactly_the_ceiling() {
    let mut perfect = ev("6-14", "Fair", "", "Free");
    perfect.weather = "outdoor".into();
    close(compute_score(&perfect, "sunny clear", FAMILY), MAX_SCORE);
}
