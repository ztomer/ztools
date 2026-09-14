//! Tests for where `twitter-summarize` finds its tweets.
//!
//! Split from `cli_ztools_tests.rs` when the command itself moved out of
//! `cli_ztools.rs`: these exercise private functions, so they have to live
//! beside the module that owns them.

use std::path::PathBuf;

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
fn test_save_tweets_json_round_trips_through_the_loader() {
    // The A/B contract: what `--fetch-only` writes must be readable back by
    // `--use-cache` (and carry the IDs the gate compares).
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("tweets.json");
    let tweets = vec![crate::ztools::twitter::Tweet {
        id: "111".to_string(),
        screen_name: "u1".into(),
        text: "hello".into(),
        created_at: "now".into(),
        favorite_count: 1,
        retweet_count: 0,
        reply_to: None,
    }];
    save_tweets_json(&tweets, &path).unwrap();
    let back = tweets_from_file(&path);
    assert_eq!(back.len(), 1);
    assert_eq!(back[0].id, "111");
    assert_eq!(back[0].text, "hello");
}

#[test]
fn test_debug_cache_path_is_the_shared_python_path() {
    // Both collectors write `~/.twitter_summary_debug_cache.json`: the Rust
    // `--fetch-only` and Python's `save_debug_cache` must name the same file
    // or the A/B compares a run against itself.
    let path = debug_cache_path().unwrap();
    assert_eq!(
        path.file_name().unwrap(),
        ".twitter_summary_debug_cache.json"
    );
}
