//! The date shapes real event listings use, judged against the Thanksgiving
//! 2026 window (Fri Oct 9 .. Mon Oct 12).
//!
//! Every quoted line below is verbatim search-result text from the live
//! corpus of 2026-10-10, the run that reported "0/88 mention a date this
//! weekend". The class: the in-window scanner read only "Month D" and "D
//! Month" with bare digits, so an ordinal day, a range that SPANS the window
//! and a city whose name begins like a month were each read wrong -- and
//! `mentions_window` asked whether a single date fell inside the window
//! while the enforcer (`window_overlap`) asks whether a RANGE overlaps it.

use super::support::d;
use crate::ztools::weekend::{find_dates_in, in_window_count, mentions_window, parse_any_date};

fn window() -> (chrono::NaiveDate, chrono::NaiveDate) {
    (d(2026, 10, 9), d(2026, 10, 12))
}

fn in_window(text: &str) -> bool {
    let (start, end) = window();
    mentions_window(text, start, end)
}

/// An ordinal day is a day: "9th October", "Oct. 10th".
#[test]
fn an_ordinal_day_is_read_as_a_date() {
    assert!(in_window(
        "Events & Things To Do In Toronto on 9th October 2026"
    ));
    assert!(in_window("Pumpkin Fest returns Oct. 10th and 11th"));
    assert_eq!(find_dates_in("October 22nd", 2026), vec![d(2026, 10, 22)]);
}

/// A range that spans the window is IN it, though neither end is -- the same
/// verdict `window_overlap` gives a row carrying those two dates.
#[test]
fn a_range_spanning_the_window_mentions_it() {
    assert!(in_window(
        "September 9, 2026 to October 28, 2026, Wednesdays 10:00 AM to 10:45 AM, \
         Saturdays 10:00 AM to 10:45 AM"
    ));
    assert!(in_window("Pumpkin patch open daily Sept 19 - Oct 31"));
    assert!(in_window("Fall Harvest Festival, October 3-11, 2026"));
    assert!(in_window("Corn maze open Sept. 26 through Oct. 12"));
    assert!(in_window("Apple picking Oct 1 \u{2013} 12 at the farm"));
}

/// A range wholly outside the window stays outside it, and its far end is
/// still a date the enforcer can read.
#[test]
fn a_range_outside_the_window_does_not_mention_it() {
    assert!(!in_window(
        "Running from September 18, 2026 to October 4, 2026, Vaughan Culture Days"
    ));
    assert!(!in_window(
        "Toronto International Festival of Authors (TIFA) Dates: October 28-November 1, 2026"
    ));
    assert_eq!(
        find_dates_in("Markham Fair, October 1-4, 2026", 2026),
        vec![d(2026, 10, 1), d(2026, 10, 4)]
    );
    // A December-to-January range rolls the year over.
    assert_eq!(
        find_dates_in("Winter lights Dec 20 - Jan 5, 2027", 2026),
        vec![d(2026, 12, 20), d(2027, 1, 5)]
    );
}

/// A byline date followed by a TIME is not a range ending on that hour:
/// "Oct 5, 2026 - 12:37" must not read as Oct 5..12, and "Sunday, October 4 -
/// 11:00 am" must not read as Oct 4..11.
#[test]
fn a_date_followed_by_a_time_is_not_a_range() {
    assert!(!in_window(
        "Oct 5, 2026 - 12:37 Provincial funding will improve water systems"
    ));
    assert!(!in_window(
        "Gallery show Sunday, October 4 - 11:00 am to 5:00 pm"
    ));
    assert!(!in_window("Open house October 4 - 11 am to 5 pm"));
    // A search engine's byline ("Oct 5, 2026 - ") followed by a sentence that
    // happens to open with a number: the year between them ends the date.
    assert!(!in_window(
        "Oct 5, 2026 - 12 new fall events added this week"
    ));
    assert!(in_window(
        "RHGA Member Gallery Show and Sale Saturday, October 10, 2026 - 11:00 am to 5:00 pm"
    ));
}

/// A word that merely BEGINS like a month is not one. "Markham" read as March,
/// and Markham is one of the four cities every query names.
#[test]
fn a_word_that_begins_like_a_month_is_not_a_month() {
    assert_eq!(
        find_dates_in("182nd Markham Fair 10 years running", 2026),
        vec![]
    );
    assert_eq!(find_dates_in("Junior 5 and up", 2026), vec![]);
    assert_eq!(find_dates_in("Decor 2 for 1", 2026), vec![]);
    // Real abbreviations still read.
    assert_eq!(find_dates_in("Sept 19", 2026), vec![d(2026, 9, 19)]);
    assert_eq!(find_dates_in("Oct. 10", 2026), vec![d(2026, 10, 10)]);
}

/// The range's FIRST day is still what `parse_any_date` hands the enforcer,
/// so a row's start date reads the same as it did before ranges were spans.
#[test]
fn the_first_date_of_a_range_is_its_start() {
    assert_eq!(parse_any_date("Oct 9 - Oct 12", 2026), Some(d(2026, 10, 9)));
    assert_eq!(
        parse_any_date("Sept 19 to Oct 31", 2026),
        Some(d(2026, 9, 19))
    );
}

/// The operator-facing count over a corpus of these shapes: the number the
/// 2026-10-10 run printed as 0.
#[test]
fn the_count_sees_every_shape_that_lands_in_the_window() {
    let (start, end) = window();
    let corpus = "\
- Events & Things To Do In Toronto on 9th October 2026
- Pumpkin patch open daily Sept 19 - Oct 31
- Running from September 18, 2026 to October 4, 2026, Vaughan Culture Days
- 182nd Markham Fair 10 years running
- Fall Harvest Market \u{2014} Oct 9 - Oct 12 in Toronto";
    assert_eq!(in_window_count(corpus, start, end), 3);
}
