//! The fit score each row is ranked by: how well it suits THIS family on
//! THIS weekend's weather, out of 5.
//!
//! WHAT IT USED TO MEASURE. Three of the old score's five terms were about the
//! row, not the family: 3 points for how many of five fields the model filled
//! in, half a point for a venue name longer than five characters, and half a
//! point for any price that was not "free" — so a paid event outranked the
//! identical free one, and a row was rewarded for being verbose. Age counted
//! only when `target_ages` was filled, which it almost never was, so every
//! unlabelled row in the Thanksgiving 2026 plan scored an identical 1.7.
//!
//! WHAT IT MEASURES NOW. Two things, both fit:
//! - **Age, 0-3:** the share of the children every age range the row states
//!   admits (see `suitability.rs`, which reads the field, the name and the
//!   description). A row that states no range cannot be judged and gets half
//!   marks — honest about not knowing, and ranked below a row known to fit.
//! - **Weather, 0-2:** an indoor row is weather-proof; an outdoor one fits a
//!   clear forecast and not a wet one.
//!
//! Price is deliberately absent: a price is not a fit, and free is never worse.

use super::WeekendEvent;

/// Points for age fit when every child fits.
pub const AGE_POINTS: f32 = 3.0;
/// The score ceiling.
pub const MAX_SCORE: f32 = 5.0;

/// Score every row against `ages` and the forecast, then sort best first.
pub fn apply_scores(events: &mut [WeekendEvent], weather_str: &str, ages: &[u32]) {
    for ev in events.iter_mut() {
        ev.score = compute_score(ev, weather_str, ages);
    }
    events.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
}

/// A small count as `f32`, exactly (the counts here are children and are tiny).
fn small(n: usize) -> f32 {
    f32::from(u16::try_from(n).unwrap_or(u16::MAX))
}

/// Age fit, 0 to [`AGE_POINTS`].
#[must_use]
pub fn age_points(ev: &WeekendEvent, ages: &[u32]) -> f32 {
    if ages.is_empty() {
        return AGE_POINTS / 2.0;
    }
    super::suitability::children_who_fit(ev, ages).map_or(AGE_POINTS / 2.0, |fit| {
        AGE_POINTS * small(fit) / small(ages.len())
    })
}

/// Weather fit, 0 to 2.
#[must_use]
pub fn weather_points(ev: &WeekendEvent, weather_str: &str) -> f32 {
    let desc_lower = ev.description.to_lowercase();
    let is_outdoor = desc_lower.contains("outdoor");
    let is_indoor = desc_lower.contains("indoor");
    let is_sunny =
        desc_lower.contains("sunny") || desc_lower.contains("clear") || desc_lower.contains("warm");
    let is_cloudy = desc_lower.contains("cloudy")
        || desc_lower.contains("rain")
        || desc_lower.contains("overcast");

    let w_lower = weather_str.to_lowercase();
    let forecast_sunny =
        w_lower.contains("sunny") || w_lower.contains("clear") || w_lower.contains("warm");
    let forecast_cloudy =
        w_lower.contains("cloudy") || w_lower.contains("rain") || w_lower.contains("precipitation");

    if is_indoor {
        1.0
    } else if is_outdoor && forecast_sunny {
        2.0
    } else if is_outdoor && forecast_cloudy {
        0.0
    } else if (is_cloudy && forecast_cloudy) || (is_sunny && forecast_sunny) {
        2.0
    } else if is_sunny || is_cloudy {
        1.0
    } else {
        0.0
    }
}

/// The row's fit, 0 to [`MAX_SCORE`].
#[must_use]
pub fn compute_score(ev: &WeekendEvent, weather_str: &str, ages: &[u32]) -> f32 {
    (age_points(ev, ages) + weather_points(ev, weather_str)).min(MAX_SCORE)
}

#[cfg(test)]
#[path = "score_tests.rs"]
mod tests;
