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
//! - **Weather, 0-2:** read from the row's weather LABEL (indoor / outdoor /
//!   both, set by the structure phase under `WEATHER_RULE`) against the
//!   forecast. An outdoor row fits a clear forecast (2) and not a wet one (0);
//!   "both" fits a clear one (2) and survives a wet one (1); an indoor row is
//!   weather-proof (1). A row with no label, or no forecast, cannot be judged
//!   and gets the neutral 1 -- the same honesty as an unjudged age range.
//!   Each row is judged by the forecast for ITS day(s) -- its dates, else the
//!   weekdays its `day` names, else every day -- taking the best: one wet
//!   Sunday used to make the whole weekend "mixed" and every row neutral. The port used to read the DESCRIPTION here, contrary
//!   to this doc and to the Python original (`item["weather"]`): "sunny
//!   picnic" earned the clear-sky bonus and an "outdoor"-labelled fair whose
//!   blurb never said "outdoor" earned nothing.
//!
//! Price is deliberately absent: a price is not a fit, and free is never worse.

use chrono::{Datelike, NaiveDate, Weekday};

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

/// Weather fit when it cannot be judged: no label, or no clear verdict from
/// the forecast.
pub const WEATHER_NEUTRAL: f32 = 1.0;

/// What the plan's forecast says, as far as the score can use it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Forecast {
    Clear,
    Wet,
    /// Missing (`WEATHER_UNAVAILABLE` carries none of the words), or clear on
    /// some days and wet on others -- which day a row is on is not read here.
    Unknown,
}

fn forecast_of(weather_str: &str) -> Forecast {
    let w = weather_str.to_lowercase();
    let clear = ["sunny", "clear", "warm"].iter().any(|k| w.contains(k));
    let wet = [
        "cloudy",
        "rain",
        "precipitation",
        "overcast",
        "snow",
        "storm",
    ]
    .iter()
    .any(|k| w.contains(k));
    match (clear, wet) {
        (true, false) => Forecast::Clear,
        (false, true) => Forecast::Wet,
        _ => Forecast::Unknown,
    }
}

/// Weather fit for one label under one day's (or the whole) forecast.
fn fit(label: &str, forecast: Forecast) -> f32 {
    match (label, forecast) {
        ("outdoor" | "both", Forecast::Clear) => 2.0,
        ("outdoor", Forecast::Wet) => 0.0,
        _ => WEATHER_NEUTRAL,
    }
}

/// The forecast day by day, from the fetcher's `YYYY-MM-DD: …` lines. Empty
/// when the string carries none (a missing forecast, or a bare summary).
fn daily(weather_str: &str) -> Vec<(NaiveDate, Forecast)> {
    weather_str
        .lines()
        .filter_map(|line| {
            let (date, rest) = line.trim().split_once(':')?;
            let date = NaiveDate::parse_from_str(date.trim(), "%Y-%m-%d").ok()?;
            Some((date, forecast_of(rest)))
        })
        .collect()
}

fn weekday(token: &str) -> Option<Weekday> {
    let t = token.to_lowercase();
    let prefix = t.get(..3)?;
    ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]
        .iter()
        .position(|d| *d == prefix)
        .and_then(|i| Weekday::try_from(u8::try_from(i).ok()?).ok())
}

/// The weekdays `day` names: "Saturday", "Sat & Sun", or a range "Fri-Sun".
fn weekdays_named(day: &str) -> Vec<Weekday> {
    let named: Vec<Weekday> = day
        .split(|c: char| !c.is_alphabetic())
        .filter_map(weekday)
        .collect();
    if let [from, to] = named[..]
        && day.contains('-')
    {
        let mut run = vec![from];
        let mut d = from;
        while d != to {
            d = d.succ();
            run.push(d);
        }
        return run;
    }
    named
}

/// The forecast days a row is on: its dates, else the weekdays it names,
/// else every day.
fn row_days(ev: &WeekendEvent, days: &[NaiveDate]) -> Vec<NaiveDate> {
    let year = days.first().map_or(2026, Datelike::year);
    let start = super::dates::parse_any_date(&ev.start_date, year);
    let end = super::dates::parse_any_date(&ev.end_date, year).or(start);
    if let (Some(start), Some(end)) = (start, end) {
        let on: Vec<NaiveDate> = days
            .iter()
            .copied()
            .filter(|d| (start..=end).contains(d))
            .collect();
        if !on.is_empty() {
            return on;
        }
    }
    let named = weekdays_named(&ev.day);
    let on: Vec<NaiveDate> = days
        .iter()
        .copied()
        .filter(|d| named.contains(&d.weekday()))
        .collect();
    if on.is_empty() { days.to_vec() } else { on }
}

/// Weather fit, 0 to 2: the row's LABEL against the forecast for its day(s).
///
/// The best of its days counts. Anything but "indoor", "outdoor" or "both" is
/// an unknown label and scores [`WEATHER_NEUTRAL`], as does no forecast.
#[must_use]
pub fn weather_points(ev: &WeekendEvent, weather_str: &str) -> f32 {
    let label = ev.weather.trim().to_lowercase();
    let days = daily(weather_str);
    if days.is_empty() {
        return fit(&label, forecast_of(weather_str));
    }
    let dates: Vec<NaiveDate> = days.iter().map(|(d, _)| *d).collect();
    row_days(ev, &dates)
        .iter()
        .filter_map(|d| days.iter().find(|(day, _)| day == d))
        .map(|(_, forecast)| fit(&label, *forecast))
        .fold(0.0, f32::max)
}

/// The row's fit, 0 to [`MAX_SCORE`].
#[must_use]
pub fn compute_score(ev: &WeekendEvent, weather_str: &str, ages: &[u32]) -> f32 {
    (age_points(ev, ages) + weather_points(ev, weather_str)).min(MAX_SCORE)
}

#[cfg(test)]
#[path = "score_tests.rs"]
mod tests;
