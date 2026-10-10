//! Statutory holidays that turn a weekend into a long weekend.
//!
//! The plan window was always Friday plus two days, so the plan for
//! Thanksgiving 2026 stopped on Sunday Oct 11 and never looked at Monday
//! Oct 12 — the one day of that weekend a family is most likely to spend out.
//!
//! A RULE TABLE, NOT A FEED. Every holiday here is a fixed date or a
//! "Nth weekday of a month" rule, so the table needs no network, no API key
//! and no yearly update, and its dates are pinned for 2026 and 2027 in the
//! tests. The province is DATA (`conf/weekend.toml [location] province`); a
//! province without a table is a stated failure, never a silent "no holidays",
//! because a missing table and a holiday-free year look identical in a plan.

use chrono::{Datelike, Duration, NaiveDate, Weekday};

/// A province whose holiday rules are in this table.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Province {
    Ontario,
}

/// One day off, by name.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Holiday {
    pub name: &'static str,
    pub date: NaiveDate,
}

impl Province {
    /// Parse the config spelling: the two-letter postal code or the name.
    ///
    /// # Errors
    ///
    /// When the province has no rule table, naming the value.
    pub fn parse(raw: &str) -> Result<Self, String> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "on" | "ontario" => Ok(Self::Ontario),
            other => Err(format!(
                "no statutory-holiday table for province {other:?}; supported: ON (Ontario)"
            )),
        }
    }

    /// Every holiday this province observes in `year`, in date order, with
    /// fixed-date holidays moved to the day they are observed.
    #[must_use]
    pub fn holidays(self, year: i32) -> Vec<Holiday> {
        match self {
            Self::Ontario => ontario(year),
        }
    }

    /// The holiday on `date`, if there is one.
    #[must_use]
    pub fn holiday_on(self, date: NaiveDate) -> Option<Holiday> {
        self.holidays(date.year())
            .into_iter()
            .find(|h| h.date == date)
    }
}

/// The `n`th `weekday` of `month` (1-based `n`).
fn nth_weekday(year: i32, month: u32, weekday: Weekday, n: u32) -> NaiveDate {
    // Every month of every year in chrono's range has a 1st and at least four
    // of each weekday, and the callers ask for n <= 3.
    NaiveDate::from_weekday_of_month_opt(year, month, weekday, u8::try_from(n).unwrap_or(1))
        .unwrap_or_default()
}

/// Easter Sunday (Gregorian), by the anonymous Meeus/Jones/Butcher algorithm.
#[must_use]
pub fn easter_sunday(year: i32) -> NaiveDate {
    // The algorithm's single letters, spelled out: a..m in the published form.
    let golden = year % 19;
    let century = year / 100;
    let in_century = year % 100;
    let leap_centuries = century / 4;
    let century_rem = century % 4;
    let lunar_corr = (century + 8) / 25;
    let solar_corr = (century - lunar_corr + 1) / 3;
    let epact = (19 * golden + century - leap_centuries - solar_corr + 15) % 30;
    let in_century_q = in_century / 4;
    let in_century_r = in_century % 4;
    let weekday_off = (32 + 2 * century_rem + 2 * in_century_q - epact - in_century_r) % 7;
    let shift = (golden + 11 * epact + 22 * weekday_off) / 451;
    let total = epact + weekday_off - 7 * shift + 114;
    NaiveDate::from_ymd_opt(
        year,
        u32::try_from(total / 31).unwrap_or(4),
        u32::try_from(total % 31 + 1).unwrap_or(1),
    )
    .unwrap_or_default()
}

/// Victoria Day: the last Monday strictly before May 25.
fn victoria_day(year: i32) -> NaiveDate {
    let may_24 = NaiveDate::from_ymd_opt(year, 5, 24).unwrap_or_default();
    let back = i64::from(may_24.weekday().num_days_from_monday());
    may_24 - Duration::days(back)
}

/// A fixed-date holiday falling on a weekend is observed on the next weekday
/// no other holiday already took (so Christmas on a Saturday and Boxing Day on
/// a Sunday are observed Monday and Tuesday).
fn observe(date: NaiveDate, taken: &[NaiveDate]) -> NaiveDate {
    let mut day = date;
    while matches!(day.weekday(), Weekday::Sat | Weekday::Sun) || taken.contains(&day) {
        day += Duration::days(1);
    }
    day
}

fn ontario(year: i32) -> Vec<Holiday> {
    let fixed = |m, d| NaiveDate::from_ymd_opt(year, m, d).unwrap_or_default();
    let christmas = observe(fixed(12, 25), &[]);
    let mut days = vec![
        Holiday {
            name: "New Year's Day",
            date: observe(fixed(1, 1), &[]),
        },
        Holiday {
            name: "Family Day",
            date: nth_weekday(year, 2, Weekday::Mon, 3),
        },
        Holiday {
            name: "Good Friday",
            date: easter_sunday(year) - Duration::days(2),
        },
        Holiday {
            name: "Victoria Day",
            date: victoria_day(year),
        },
        Holiday {
            name: "Canada Day",
            date: observe(fixed(7, 1), &[]),
        },
        // Not statutory under the ESA, but the province-wide August long
        // weekend nearly every employer and municipality observes.
        Holiday {
            name: "Civic Holiday",
            date: nth_weekday(year, 8, Weekday::Mon, 1),
        },
        Holiday {
            name: "Labour Day",
            date: nth_weekday(year, 9, Weekday::Mon, 1),
        },
        Holiday {
            name: "Thanksgiving",
            date: nth_weekday(year, 10, Weekday::Mon, 2),
        },
        Holiday {
            name: "Christmas Day",
            date: christmas,
        },
        Holiday {
            name: "Boxing Day",
            date: observe(fixed(12, 26), &[christmas]),
        },
    ];
    days.sort_by_key(|h| h.date);
    days
}

/// Read `[location] province` from the first weekend config that has a
/// `[location]` table.
///
/// # Errors
///
/// When no candidate file has a `[location]` table, when it has no
/// `province`, or when the province has no rule table — each named, because
/// every one of them would otherwise plan a long weekend as a short one.
pub fn load_province(paths: &[String]) -> Result<Province, String> {
    for raw in paths {
        let path = crate::manifest::expand_tilde(raw);
        let Ok(content) = std::fs::read_to_string(&path) else {
            continue;
        };
        let Ok(val) = toml::from_str::<toml::Value>(&content) else {
            continue;
        };
        let Some(location) = val.get("location") else {
            continue;
        };
        let province = location
            .get("province")
            .and_then(toml::Value::as_str)
            .ok_or_else(|| {
                format!(
                    "{} has a [location] table without `province`; the plan window \
                     needs it to know which Mondays are holidays",
                    path.display()
                )
            })?;
        return Province::parse(province);
    }
    Err(format!(
        "no weekend config with a [location] table among {paths:?}; cannot tell which \
         Mondays are holidays"
    ))
}

#[cfg(test)]
#[path = "holidays_tests.rs"]
mod tests;
