//! Shared helpers for `weekend_enforce`'s test modules.

use crate::ztools::weekend::WeekendEvent;
use jiff::civil::Date;

pub(super) fn d(y: i32, m: u32, day: u32) -> Date {
    crate::ztools::weekend::dates::ymd(y, m, day).unwrap()
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
        start_date: String::new(),
        end_date: String::new(),
        weather: String::new(),
        duration: String::new(),
        for_families: String::new(),
    }
}
