//! Calendar-date scanning shared between the weekend enforcer and any
//! in-window prioritiser. Ported from `lib/dates.py`.

use chrono::{Datelike, NaiveDate};

/// The days a plan should cover, seen from `today`: Friday to Sunday, or to
/// Monday when that Monday is a holiday in `province`.
///
/// During a weekend the answer is *this* one, not the next: a plan for the
/// days you are living through is current, not stale. Monday = 0 ... Friday
/// = 4, Saturday = 5, Sunday = 6, so Saturday and Sunday step BACK to their
/// own Friday — and so does a holiday Monday, which is still that weekend.
///
/// ONE definition, used by both the planner and the status page. They used
/// to disagree — the planner walked forward to the next Friday, so a Saturday
/// refresh planned the FOLLOWING weekend while the status page kept saying
/// this one was "not planned" — and the two only agreed Monday to Friday.
///
/// The Monday is the point of the province: the window was always Friday
/// plus two days, so the Thanksgiving 2026 plan stopped on Sunday Oct 11 and
/// never looked at Monday Oct 12 (see `holidays.rs`).
#[must_use]
pub fn plan_window(
    today: NaiveDate,
    province: super::holidays::Province,
) -> (NaiveDate, NaiveDate) {
    let weekday = i64::from(today.weekday().num_days_from_monday());
    let friday = if weekday == 0 && province.holiday_on(today).is_some() {
        today - chrono::Duration::days(3)
    } else {
        today - chrono::Duration::days(weekday - 4)
    };
    let monday = friday + chrono::Duration::days(3);
    let end = if province.holiday_on(monday).is_some() {
        monday
    } else {
        friday + chrono::Duration::days(2)
    };
    (friday, end)
}

/// Full month names, in order.
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

/// 1-12 for a month name or an abbreviation of one ("oct", "sept").
///
/// The run must be a PREFIX of the month name, at least three letters long.
/// It used to be enough for the run's first three letters to match, so
/// "Markham 3" read as March 3, "Junior 5" as June 5 and "Decor 2" as
/// December 2 — and Markham is one of the four cities every query names.
fn month_number(run: &str) -> Option<u32> {
    if run.len() < 3 || !run.chars().all(|c| c.is_ascii_alphabetic()) {
        return None;
    }
    MONTHS
        .iter()
        .position(|m| m.starts_with(run))
        .and_then(|i| u32::try_from(i + 1).ok())
}

/// A day of the month: one or two digits, optionally an English ordinal
/// ("9", "09", "10th", "1st", "22nd", "3rd").
fn day_of(run: &str) -> Option<u32> {
    let digits = ["st", "nd", "rd", "th"]
        .iter()
        .find_map(|suffix| run.strip_suffix(suffix))
        .unwrap_or(run);
    if digits.is_empty() || digits.len() > 2 || !digits.chars().all(|c| c.is_ascii_digit()) {
        return None;
    }
    digits.parse().ok().filter(|d| (1..=31).contains(d))
}

/// A four-digit year.
fn year_of(run: &str) -> Option<i32> {
    if run.len() == 4 && run.chars().all(|c| c.is_ascii_digit()) {
        run.parse().ok()
    } else {
        None
    }
}

/// One alphanumeric run of the text, with the separator text before it.
struct Token {
    text: String,
    gap: String,
}

/// Lower-cased alphanumeric runs: month names are pure letters, days are 1-2
/// digits (with an optional ordinal suffix), explicit years 4 digits.
fn tokenize(value: &str) -> Vec<Token> {
    let mut tokens = Vec::new();
    let mut gap = String::new();
    let mut cur = String::new();
    for c in value.to_lowercase().chars() {
        if c.is_ascii_alphanumeric() {
            cur.push(c);
        } else {
            if !cur.is_empty() {
                tokens.push(Token {
                    text: std::mem::take(&mut cur),
                    gap: std::mem::take(&mut gap),
                });
            }
            gap.push(c);
        }
    }
    if !cur.is_empty() {
        tokens.push(Token { text: cur, gap });
    }
    tokens
}

/// Where the END of a range starts, if token `j` opens one: a dash between
/// the two ends ("Oct 9-12", "October 1–4") or a connecting word ("Sept 19 to
/// Oct 31", "through", "until").
fn range_end_at(tokens: &[Token], j: usize) -> Option<usize> {
    let tok = tokens.get(j)?;
    if matches!(tok.gap.trim(), "-" | "\u{2013}" | "\u{2014}") {
        return Some(j);
    }
    matches!(
        tok.text.as_str(),
        "to" | "through" | "thru" | "until" | "till"
    )
    .then_some(j + 1)
}

/// The day of the month at `j`, unless it is really the hour of a clock time
/// ("- 11:00 am", "- 11 am"): event pages put a time after a date as often as
/// they put a second date there.
fn range_end_day(tokens: &[Token], j: usize) -> Option<u32> {
    let day = day_of(&tokens.get(j)?.text)?;
    let is_time = tokens.get(j + 1).is_some_and(|next| {
        next.gap.starts_with(':') || matches!(next.text.as_str(), "am" | "pm" | "a" | "p")
    });
    (!is_time).then_some(day)
}

const fn ymd(year: i32, month: u32, day: u32) -> Option<NaiveDate> {
    NaiveDate::from_ymd_opt(year, month, day)
}

/// A "Month Day" date at `k`, and the range it opens if one follows:
/// "Oct 9-12, 2026", "October 9 to 12", "Sept 19 – Oct 31",
/// "September 9, 2026 to October 28, 2026".
fn month_first_span(tokens: &[Token], k: usize, year: i32) -> Option<(NaiveDate, NaiveDate)> {
    let month = month_number(&tokens.get(k)?.text)?;
    let day = day_of(&tokens.get(k + 1)?.text)?;
    let own_year = tokens.get(k + 2).and_then(|t| year_of(&t.text));
    let after = if own_year.is_some() { k + 3 } else { k + 2 };
    let start_year = own_year.unwrap_or(year);

    if let Some(e) = range_end_at(tokens, after) {
        // "… to Oct 31[, 2026]": the far end names its own month.
        if let Some(end_month) = tokens.get(e).and_then(|t| month_number(&t.text))
            && let Some(end_day) = range_end_day(tokens, e + 1)
        {
            let rollover = i32::from(end_month < month);
            let end_year = tokens
                .get(e + 2)
                .and_then(|t| year_of(&t.text))
                .unwrap_or(start_year + rollover);
            let start_year = own_year.unwrap_or(end_year - rollover);
            let (start, end) = (
                ymd(start_year, month, day)?,
                ymd(end_year, end_month, end_day)?,
            );
            return Some((start, end.max(start)));
        }
        // "Oct 9-12[, 2026]": a bare day ends it -- but only when no year sat
        // between, because "Oct 5, 2026 - 12 events" is a byline date
        // followed by a sentence, not a range ending on the 12th.
        if own_year.is_none()
            && let Some(end_day) = range_end_day(tokens, e)
            && end_day > day
        {
            let y = tokens
                .get(e + 1)
                .and_then(|t| year_of(&t.text))
                .unwrap_or(year);
            return Some((ymd(y, month, day)?, ymd(y, month, end_day)?));
        }
    }
    let date = ymd(start_year, month, day)?;
    Some((date, date))
}

/// A "Day Month" date at `k`: "15 Aug", "09 Aug 2026", "Sun 09 Aug", "10th October".
///
/// Not when the day already belongs to the month BEFORE it: in "Sept 19 - Oct
/// 31" the 19 is September's, and reading "19 - Oct" as October 19 put a date
/// in the window that the text never named.
fn day_first_date(tokens: &[Token], k: usize, year: i32) -> Option<NaiveDate> {
    let owned = k
        .checked_sub(1)
        .and_then(|p| tokens.get(p))
        .is_some_and(|prev| month_number(&prev.text).is_some());
    if owned {
        return None;
    }
    let day = day_of(&tokens.get(k)?.text)?;
    let month = month_number(&tokens.get(k + 1)?.text)?;
    let y = tokens
        .get(k + 2)
        .and_then(|t| year_of(&t.text))
        .unwrap_or(year);
    ymd(y, month, day)
}

/// ISO dates YYYY-MM-DD, each a one-day span.
fn iso_spans(value: &str, spans: &mut Vec<(NaiveDate, NaiveDate)>) {
    let chars: Vec<char> = value.chars().collect();
    for i in 0..chars.len().saturating_sub(9) {
        if chars[i + 4] != '-' || chars[i + 7] != '-' {
            continue;
        }
        let all_digits = (0..10).all(|k| k == 4 || k == 7 || chars[i + k].is_ascii_digit());
        if !all_digits {
            continue;
        }
        // Every digit was checked above, so `to_digit` is `Some` and the
        // four-digit year fits `i32` by construction.
        let at = |k: usize| chars[i + k].to_digit(10).unwrap_or(0);
        let year = i32::try_from(at(0) * 1000 + at(1) * 100 + at(2) * 10 + at(3)).unwrap_or(0);
        if let Some(date) = ymd(year, at(5) * 10 + at(6), at(8) * 10 + at(9)) {
            spans.push((date, date));
        }
    }
}

/// Every explicit date or date RANGE in `value`, as inclusive `(first, last)`
/// spans; a single date is a one-day span.
///
/// Ranges are spans rather than their two endpoints because the question
/// asked of them is "does this overlap the plan window?" — the enforcer's
/// question (`window_overlap`), which the in-window prioritiser has to share.
/// "Pumpkin patch open daily Sept 19 – Oct 31" is ON over Thanksgiving
/// although neither endpoint is.
///
/// `year` is the fallback for formats that omit it; an explicit four-digit year
/// in the text always wins, so a snippet carrying a past year is not silently
/// promoted into this year's plan window.
#[must_use]
pub fn find_date_spans_in(value: &str, year: i32) -> Vec<(NaiveDate, NaiveDate)> {
    let mut spans = Vec::new();
    if value.is_empty() {
        return spans;
    }
    iso_spans(value, &mut spans);
    let tokens = tokenize(value);
    for k in 0..tokens.len() {
        if let Some(span) = month_first_span(&tokens, k, year) {
            spans.push(span);
        }
        if let Some(date) = day_first_date(&tokens, k, year) {
            spans.push((date, date));
        }
    }
    let mut unique = Vec::new();
    for span in spans {
        if !unique.contains(&span) {
            unique.push(span);
        }
    }
    unique
}

/// Pull explicit calendar dates out of a cell -- every single date, and both
/// ends of every range. Durations are not dates.
///
/// Ported from `lib/dates.py` so the enforcer and the in-window prioritiser
/// cannot drift apart (they already did once: the enforcer read three-letter
/// stems while the prioritiser matched only full month names).
#[must_use]
pub fn find_dates_in(value: &str, year: i32) -> Vec<NaiveDate> {
    let mut found = Vec::new();
    for (first, last) in find_date_spans_in(value, year) {
        for date in <[NaiveDate; 2]>::from((first, last)) {
            if !found.contains(&date) {
                found.push(date);
            }
        }
    }
    found
}

/// First explicit date in `value`, or None. Shared with the checker so the
/// enforcer and the candidate prioritiser cannot drift apart.
#[must_use]
pub fn parse_any_date(value: &str, year: i32) -> Option<NaiveDate> {
    find_dates_in(value, year).into_iter().next()
}
