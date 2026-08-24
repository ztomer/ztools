//! "Float in-window lines to the top, drop nothing" transient-corpus behavior.

use crate::ztools::weekend::in_window_count;
fn friday() -> chrono::NaiveDate {
    chrono::NaiveDate::parse_from_str("2026-08-07", "%Y-%m-%d").unwrap()
}

fn sunday() -> chrono::NaiveDate {
    chrono::NaiveDate::parse_from_str("2026-08-09", "%Y-%m-%d").unwrap()
}

/// In-window candidates float to the top marked, order preserved within each
/// group; out-of-window lines stay below in their own order.
#[test]
fn in_window_lines_float_to_the_top_marked_but_nothing_is_removed() {
    let corpus =
        "First entry\nAug 15 festival all month\nSecond entry\nAug 8 happening this weekend";
    let out = crate::ztools::weekend::prioritise_in_window(corpus, friday(), sunday());
    let lines: Vec<&str> = out.lines().collect();
    // Both in-window lines (Aug 8) float up, marked; everything else follows in order.
    assert_eq!(lines.len(), 4, "{out:?}");
    assert!(lines[0].starts_with("[THIS WEEKEND]"), "{out:?}");
    assert!(lines[0].ends_with("happening this weekend"), "{out:?}");
    assert_eq!(lines[1], "First entry");
    assert_eq!(lines[2], "Aug 15 festival all month");
    assert_eq!(lines[3], "Second entry");
    assert!(
        out.contains("Aug 15 festival all month"),
        "nothing may be removed"
    );
}

/// A corpus with no dated candidates is unchanged verbatim -- inventing a
/// marker would tell the model something untrue.
#[test]
fn corpus_with_no_dated_candidates_is_returned_unchanged() {
    let corpus = "Evergreen venue listing\nAnother evergreen listing";
    let out = crate::ztools::weekend::prioritise_in_window(corpus, friday(), sunday());
    assert_eq!(out, corpus);
}

/// The count reports how many candidates land in the window -- the number that
/// distinguishes a supply problem from a model problem.
#[test]
fn in_window_count_counts_only_lines_that_mention_the_window() {
    let corpus = "Aug 8 festival\nEvergreen\nAug 9 show\nAug 15 out of window";
    assert_eq!(in_window_count(corpus, friday(), sunday()), 2);
    assert_eq!(
        in_window_count("no dates here at all", friday(), sunday()),
        0
    );
}
