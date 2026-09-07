//! Tests for which eval tasks a `--task` filter selects.
//!
//! In a sibling file for the house 500-line cap. The rule exists as its own
//! function because it was previously inlined in an entry point that needs a
//! live model to reach.

use super::*;
#[test]
fn a_full_name_matches_itself() {
    assert!(task_matches_filter("weekend.taxes", "weekend.taxes"));
}

#[test]
fn a_trailing_segment_matches_without_the_namespace() {
    assert!(task_matches_filter("weekend.taxes", "taxes"));
    assert!(task_matches_filter("twitter.summarize", "summarize"));
}

#[test]
fn a_comma_list_matches_any_entry_and_tolerates_spaces() {
    assert!(task_matches_filter("weekend.taxes", "summarize, taxes"));
    assert!(task_matches_filter(
        "twitter.summarize",
        "summarize , taxes"
    ));
    assert!(!task_matches_filter("eval.other", "summarize, taxes"));
}

/// The case the caller turns into a refusal. If this ever returned true
/// for everything, `--task nonsense` would silently run the whole suite.
#[test]
fn a_filter_matching_nothing_matches_nothing() {
    assert!(!task_matches_filter("weekend.taxes", "nonsense"));
    assert!(
        !task_matches_filter("weekend.taxes", "TAXES"),
        "match is case-sensitive"
    );
}

/// An empty entry must not become a wildcard -- every name ends with "".
#[test]
fn an_empty_entry_is_ignored_rather_than_matching_everything() {
    assert!(!task_matches_filter("weekend.taxes", ""));
    assert!(!task_matches_filter("weekend.taxes", ",,"));
    assert!(!task_matches_filter("weekend.taxes", "   "));
    assert!(
        task_matches_filter("weekend.taxes", "taxes,,"),
        "a real entry beside empty ones still selects"
    );
}

/// A prefix is NOT a match: `--task week` must not pull in
/// `weekend.taxes`, or a narrow filter silently widens.
#[test]
fn a_leading_fragment_does_not_match() {
    assert!(!task_matches_filter("weekend.taxes", "week"));
    assert!(!task_matches_filter("weekend.taxes", "weekend"));
}
