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

/// A followed listing page's entry spans lines, and the date is on a line of
/// its own: the shape Visit Vaughan's events page produced on 2026-10-10,
/// verbatim. Floating that one dated line put "Oct 10Sat+2 dates" at the top
/// of the corpus and left "Woodbridge Fall Fair" in another extract batch, so
/// the extractor saw a date naming no event and an event with no date. The
/// page's lines must keep their order and their neighbours; the dated one is
/// marked where it stands, and a self-contained search result still floats.
#[test]
fn a_followed_pages_entry_is_marked_in_place_and_never_split() {
    let page = "[Events in Vaughan | Visit Vaughan, Ontario]";
    let corpus = [
        "- Evergreen result: a venue open all year".to_string(),
        format!("- {page} add Screemers to my stay"),
        format!("- {page} Oct 10 - 11Sat - Sun+16 dates"),
        format!("- {page} Screemers"),
        format!("- {page} add Woodbridge Fall Fair to my stay"),
        format!("- {page} Oct 10Sat+2 dates"),
        format!("- {page} Woodbridge Fall Fair"),
        "- Fall Harvest Market — Oct 9 - Oct 12 in Toronto | Very Toronto: harvest foods"
            .to_string(),
    ]
    .join("\n");
    let (fri, mon) = (
        chrono::NaiveDate::from_ymd_opt(2026, 10, 9).unwrap(),
        chrono::NaiveDate::from_ymd_opt(2026, 10, 12).unwrap(),
    );
    let out = crate::ztools::weekend::prioritise_in_window(&corpus, fri, mon);
    let lines: Vec<&str> = out.lines().collect();
    let mark = crate::ztools::weekend::IN_WINDOW_MARK;

    // The search result floats, marked.
    assert!(
        lines[0].starts_with(mark) && lines[0].contains("Fall Harvest Market"),
        "{out}"
    );
    // The fair's date sits between its two name lines, marked in place.
    let date = lines
        .iter()
        .position(|l| l.ends_with("Oct 10Sat+2 dates"))
        .expect("the date line survives");
    assert!(
        lines[date].starts_with(mark),
        "the dated line is marked: {out}"
    );
    assert!(
        lines[date - 1].ends_with("add Woodbridge Fall Fair to my stay"),
        "{out}"
    );
    assert!(lines[date + 1].ends_with("] Woodbridge Fall Fair"), "{out}");

    // Nothing removed, nothing duplicated: the output is the input's lines.
    let mut unmarked: Vec<String> = lines
        .iter()
        .map(|l| l.strip_prefix(mark).map_or(*l, str::trim_start).to_string())
        .collect();
    let mut input: Vec<String> = corpus.lines().map(String::from).collect();
    unmarked.sort();
    input.sort();
    assert_eq!(unmarked, input);
}
