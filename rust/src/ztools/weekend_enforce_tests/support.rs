//! Shared helpers for weekend_enforce's test modules.

use crate::ztools::weekend::WeekendEvent;
use chrono::NaiveDate;

pub(super) fn d(y: i32, m: u32, day: u32) -> NaiveDate {
    NaiveDate::from_ymd_opt(y, m, day).unwrap()
}

pub(super) fn event(name: &str, location: &str) -> WeekendEvent {
    WeekendEvent {
        name: name.into(),
        location: location.into(),
        price: "$10".into(),
        target_ages: "6-12".into(),
        day: "Sat".into(),
        dates: "Aug 8".into(),
        description: "desc".into(),
        is_transient: true,
        score: 4.0,
        start_date: "".into(),
        end_date: "".into(),
        weather: "".into(),
        duration: "".into(),
    }
}
