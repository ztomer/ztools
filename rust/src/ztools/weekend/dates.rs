//! Calendar-date scanning shared between the weekend enforcer and any
//! in-window prioritiser. Ported from `lib/dates.py`.

use chrono::NaiveDate;

/// Full month names, in order. Three-letter prefixes are the matching stems.
const MONTHS: [&str; 12] = [
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
];

/// 1-12 for a month name or its three-letter stem, mirroring `lib/dates.py`.
fn month_number(run: &str) -> Option<u32> {
    let stem = run.get(..3)?;
    MONTHS
        .iter()
        .position(|m| m.starts_with(stem))
        .map(|i| i as u32 + 1)
}

fn push_date(found: &mut Vec<NaiveDate>, year: i32, month: u32, day: u32) {
    if let Some(value) = NaiveDate::from_ymd_opt(year, month, day) {
        if !found.contains(&value) {
            found.push(value);
        }
    }
}

/// Pull explicit calendar dates out of a cell. Durations are not dates.
///
/// `year` is the fallback for formats that omit it; an explicit four-digit year
/// in the text always wins, so a snippet carrying a past year is not silently
/// promoted into this year's plan window.
///
/// Ported from `lib/dates.py` so the enforcer and any future in-window
/// prioritiser cannot drift apart (they already did once: the enforcer read
/// three-letter stems while the prioritiser matched only full month names).
pub fn find_dates_in(value: &str, year: i32) -> Vec<NaiveDate> {
    let mut found = Vec::new();
    if value.is_empty() {
        return found;
    }

    // ISO dates YYYY-MM-DD.
    let chars: Vec<char> = value.chars().collect();
    for i in 0..chars.len().saturating_sub(9) {
        if chars[i + 4] != '-' || chars[i + 7] != '-' {
            continue;
        }
        let all_digits = (0..10).all(|k| k == 4 || k == 7 || chars[i + k].is_ascii_digit());
        if !all_digits {
            continue;
        }
        let at = |k: usize| chars[i + k].to_digit(10).unwrap() as i32;
        push_date(
            &mut found,
            at(0) * 1000 + at(1) * 100 + at(2) * 10 + at(3),
            (at(5) * 10 + at(6)) as u32,
            (at(8) * 10 + at(9)) as u32,
        );
    }

    // Named-month forms: "Aug 15", "August 15, 2026", "Aug. 15 2026",
    // "15 Aug", "09 Aug 2026", "Sun 09 Aug". Tokenise into alphanumeric runs
    // (month names are pure letters, days are 1-2 digits, explicit years 4).
    let lower = value.to_lowercase();
    let mut runs: Vec<(String, bool)> = Vec::new();
    let mut cur = String::new();
    for c in lower.chars() {
        if c.is_ascii_alphanumeric() {
            cur.push(c);
        } else if !cur.is_empty() {
            runs.push((cur.clone(), cur.chars().all(|c| c.is_ascii_digit())));
            cur.clear();
        }
    }
    if !cur.is_empty() {
        runs.push((cur.clone(), cur.chars().all(|c| c.is_ascii_digit())));
    }

    let is_month_word =
        |run: &str| run.chars().all(|c| c.is_ascii_alphabetic()) && month_number(run).is_some();
    let is_short_day = |run: &(String, bool)| run.1 && run.0.len() <= 2;
    for k in 0..runs.len() {
        let (text, is_num) = &runs[k];
        // "Aug 15" / "August 15, 2026"
        if is_month_word(text) {
            if let Some(next) = runs.get(k + 1) {
                if is_short_day(next) {
                    let day = next.0.parse::<u32>().unwrap();
                    let mut yr = year;
                    if let Some(yr_run) = runs.get(k + 2) {
                        if yr_run.1 && yr_run.0.len() == 4 {
                            yr = yr_run.0.parse::<i32>().unwrap();
                        }
                    }
                    push_date(&mut found, yr, month_number(text).unwrap(), day);
                }
            }
        }
        // "15 Aug" / "09 Aug 2026" / "Sun 09 Aug"
        if *is_num && text.len() <= 2 {
            if let Some(month_run) = runs.get(k + 1) {
                if !month_run.1 && is_month_word(&month_run.0) {
                    let day = text.parse::<u32>().unwrap();
                    let mut yr = year;
                    if let Some(yr_run) = runs.get(k + 2) {
                        if yr_run.1 && yr_run.0.len() == 4 {
                            yr = yr_run.0.parse::<i32>().unwrap();
                        }
                    }
                    push_date(&mut found, yr, month_number(&month_run.0).unwrap(), day);
                }
            }
        }
    }

    found
}

/// First explicit date in `value`, or None. Shared with the checker so the
/// enforcer and the candidate prioritiser cannot drift apart.
pub fn parse_any_date(value: &str, year: i32) -> Option<NaiveDate> {
    find_dates_in(value, year).into_iter().next()
}
