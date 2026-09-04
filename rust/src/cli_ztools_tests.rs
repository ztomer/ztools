//! Tests for the parts of `cli_ztools.rs` that are logic rather than
//! orchestration: which tweets a run starts from, and which eval tasks a
//! `--task` filter selects.
//!
//! In a sibling file for the house 500-line cap. Both groups exist because the
//! rules they cover were previously inlined in entry points that need a live
//! model server and a headless browser -- so the rules could not be exercised
//! at all, and one of them (the filter) was quietly wrong.

use super::*;

fn tweet(name: &str) -> serde_json::Value {
    serde_json::json!({
        "screen_name": name,
        "text": "hello",
        "created_at": "Thu Aug 20 12:00:00 +0000 2026",
        "favorite_count": 1,
        "retweet_count": 0,
        "reply_to": null,
    })
}

fn write(dir: &std::path::Path, name: &str, body: &str) -> PathBuf {
    let path = dir.join(name);
    std::fs::write(&path, body).unwrap();
    path
}

#[test]
fn a_well_formed_array_parses() {
    let text = serde_json::to_string(&vec![tweet("a"), tweet("b")]).unwrap();
    let got = tweets_from_json(&text);
    assert_eq!(got.len(), 2);
    assert_eq!(got[0].screen_name, "a");
}

/// Unparseable input yields NOTHING, never a partial list. Half a timeline
/// read as the whole one is a summary that is confidently wrong about what
/// was said, and the caller's fallback sources exist precisely for this.
#[test]
fn malformed_input_yields_nothing_rather_than_a_partial_list() {
    assert!(tweets_from_json("not json").is_empty());
    assert!(tweets_from_json("").is_empty());
    assert!(
        tweets_from_json(r#"[{"screen_name": "a"}]"#).is_empty(),
        "a tweet missing required fields fails the WHOLE parse"
    );
    assert!(
        tweets_from_json(r#"{"tweets": []}"#).is_empty(),
        "valid JSON of the wrong shape is still not a tweet list"
    );
}

#[test]
fn an_absent_or_unreadable_file_yields_nothing() {
    let tmp = tempfile::tempdir().unwrap();
    assert!(tweets_from_file(&tmp.path().join("nope.json")).is_empty());
    // A directory is not a readable file.
    assert!(tweets_from_file(tmp.path()).is_empty());
}

#[test]
fn a_readable_file_yields_its_tweets() {
    let tmp = tempfile::tempdir().unwrap();
    let path = write(
        tmp.path(),
        "tweets.json",
        &serde_json::to_string(&vec![tweet("from-file")]).unwrap(),
    );
    let got = tweets_from_file(&path);
    assert_eq!(got.len(), 1);
    assert_eq!(got[0].screen_name, "from-file");
}

#[test]
fn the_cache_scan_returns_the_first_populated_file_and_names_it() {
    let tmp = tempfile::tempdir().unwrap();
    let good = write(
        tmp.path(),
        "good.json",
        &serde_json::to_string(&vec![tweet("cached")]).unwrap(),
    );
    let (from, got) =
        tweets_from_cache(std::slice::from_ref(&good)).expect("a populated cache is used");
    assert_eq!(from, good, "the caller reports WHICH file it read");
    assert_eq!(got.len(), 1);
}

/// An empty cache file is not an answer. Stopping at one would report
/// "no cached tweets" while a populated candidate sat unread behind it.
#[test]
fn an_empty_cache_file_does_not_shadow_a_populated_one() {
    let tmp = tempfile::tempdir().unwrap();
    let empty = write(tmp.path(), "empty.json", "[]");
    let broken = write(tmp.path(), "broken.json", "not json");
    let missing = tmp.path().join("missing.json");
    let good = write(
        tmp.path(),
        "good.json",
        &serde_json::to_string(&vec![tweet("second")]).unwrap(),
    );

    let (from, got) = tweets_from_cache(&[missing, empty, broken, good.clone()]).unwrap();
    assert_eq!(from, good);
    assert_eq!(got[0].screen_name, "second");
}

#[test]
fn no_usable_candidate_is_none_so_the_caller_can_say_so() {
    let tmp = tempfile::tempdir().unwrap();
    let empty = write(tmp.path(), "empty.json", "[]");
    assert!(tweets_from_cache(&[]).is_none());
    assert!(tweets_from_cache(&[tmp.path().join("nope.json"), empty]).is_none());
}

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
