//! What the planner asks the search engines.
//!
//! Every query used to be scoped to the MONTH ("Vaughan family events October
//! 2026"), so what came back was month-wide directories and pages about the
//! month, and nothing ever asked about the weekend being planned -- not its
//! dates, and not the holiday that made it a long weekend. On Thanksgiving
//! 2026 the engines that did answer had pages titled "Things To Do In Vaughan
//! This Weekend | October 9-11, 2026" and "Thanksgiving Weekend Toronto 2026";
//! the planner never asked for them.
//!
//! So the WINDOW is asked about first, by its dates, then the holiday (when
//! the window holds one), then the month-wide queries as before. Order
//! matters downstream: listing pages are followed in result order, so the
//! weekend's own listings are the ones the bounded follow budget reads.
//!
//! The region is spelled "near Toronto", never "GTA": every "GTA ..." query on
//! 2026-10-10 came back about Grand Theft Auto, on Bing and on Brave.

use chrono::NaiveDate;

use super::seasonal_keywords;

/// The municipalities every plan searches, nearest first.
const MUNICIPALITIES: [&str; 4] = ["Vaughan", "Markham", "Richmond Hill", "Toronto"];

/// The window as a listing page writes it: "October 9-12", or "October 30 -
/// November 1" when it crosses a month.
fn window_phrase(d1: NaiveDate, d2: NaiveDate) -> String {
    if d1.format("%B").to_string() == d2.format("%B").to_string() {
        format!(
            "{} {}-{}",
            d1.format("%B"),
            d1.format("%-d"),
            d2.format("%-d")
        )
    } else {
        format!("{} - {}", d1.format("%B %-d"), d2.format("%B %-d"))
    }
}

/// Search queries for the weekend `d1..=d2`, naming `holiday` when the window
/// holds one (the province's holiday table decides; see `holidays.rs`).
#[must_use]
pub fn build_search_queries(d1: NaiveDate, d2: NaiveDate, holiday: Option<&str>) -> Vec<String> {
    let month_name = d1.format("%B").to_string();
    let year = d1.format("%Y").to_string();
    let window = window_phrase(d1, d2);
    let seasonal = seasonal_keywords(&month_name);

    let mut queries: Vec<String> = MUNICIPALITIES
        .iter()
        .map(|city| format!("{city} events this weekend {window} {year}"))
        .collect();

    if let Some(holiday) = holiday {
        queries.push(format!(
            "{holiday} weekend family events near Toronto {year}"
        ));
        queries.push(format!("{holiday} weekend {seasonal} near Toronto {year}"));
    }

    for city in MUNICIPALITIES {
        queries.push(format!("kids activities {city} {month_name} {year}"));
        queries.push(format!(
            "{city} community centre kids programs {month_name}"
        ));
        queries.push(format!("{city} family events {month_name} {year}"));
    }

    queries.push(format!("family events near Toronto {month_name} {year}"));
    queries.push(format!(
        "zoo special events near Toronto {month_name} {year}"
    ));
    queries.push(format!(
        "museum family programs near Toronto {month_name} {year}"
    ));
    queries.push(format!("{seasonal} near Toronto {month_name} {year}"));
    queries
}
