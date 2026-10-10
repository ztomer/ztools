//! The end-to-end `run_summary` path, against a loopback stub.
//!
//! Every test here used to run `ZtoolsConfig::default()`, which named three
//! paths on the developer's disk -- a cache file, the Playwright collector's
//! checkout and that checkout's `conf/twitter.toml` -- and which read the
//! `[fallback]` table from THIS CHECKOUT, so the suite only passed on a
//! developer machine with the repo in the expected place. Two tests also wrote
//! to fixed directories under the system temp dir, which collide between two
//! `cargo test` runs on the same Mac -- and several agent sessions share it.

use std::path::PathBuf;

use super::super::*;
use super::support::{CHAT_BODY, sandboxed_config, stub_server};
use crate::test_env::TestEnv;
use serial_test::serial;

fn tweet(user: &str, text: &str, at: &str, fav: u64, rt: u64) -> Tweet {
    Tweet {
        id: String::new(),
        screen_name: user.to_string(),
        text: text.to_string(),
        created_at: at.to_string(),
        favorite_count: fav,
        retweet_count: rt,
        reply_to: None,
    }
}

#[test]
#[serial]
fn test_call_osaurus_invalid_host() {
    let env = TestEnv::new();
    let cfg = sandboxed_config(&env.root().join("cfg"));
    let res = call_osaurus("http://127.0.0.1:59999", "model", "prompt", 5, &cfg);
    assert!(
        res.is_err(),
        "an unreachable endpoint is an error, not a document"
    );
    drop(env);
}

#[test]
#[serial]
fn test_call_osaurus_and_run_summary_success() {
    let env = TestEnv::new();
    let base_url = stub_server(CHAT_BODY);
    let output = tempfile::tempdir().unwrap();
    let cfg = sandboxed_config(&env.root().join("cfg"));

    let res = run_summary(
        &[tweet(
            "routine_user",
            "Testing twitter summarizer with mock server",
            "14:00",
            3,
            1,
        )],
        output.path(),
        Some(&base_url),
        Some("mock-model"),
        &cfg,
    );
    let path = res.expect("the stub answered, so the summary is written");
    assert!(path.exists());

    let content = std::fs::read_to_string(&path).unwrap();
    assert!(content.contains("Twitter Timeline Summary"), "{content}");
    assert!(content.contains("mock-model"), "{content}");
    // The model's body opens with its own heading, so the writer must not
    // stack an empty `## Summary` preamble on top of it.
    assert!(
        !content.contains("## Summary"),
        "a body with its own heading must not get an empty `## Summary` preamble: {content}"
    );
    assert!(content.contains("## Section"), "{content}");
    drop(env);
}

#[test]
#[serial]
fn test_run_summary_cache_reading() {
    let env = TestEnv::new();
    let base_url = stub_server(
        r###"{"choices": [{"message": {"content": "## Section\n- Cached tweet summary (@cached_user | 10:00)"}}]}"###,
    );
    // A fixture cache, inside the sandbox, not the operator's.
    let cfg = sandboxed_config(&env.root().join("cfg"));
    let cache_file = PathBuf::from(&cfg.twitter_cache_path);
    std::fs::create_dir_all(cache_file.parent().unwrap()).unwrap();
    std::fs::write(
        &cache_file,
        r#"[{"screen_name":"cached_user","text":"Cached tweet content for test","created_at":"10:00","favorite_count":2,"retweet_count":0,"reply_to":null}]"#,
    )
    .unwrap();
    let output = tempfile::tempdir().unwrap();

    let path = run_summary(&[], output.path(), Some(&base_url), None, &cfg)
        .expect("the cached timeline is summarised");
    let doc = std::fs::read_to_string(&path).unwrap();
    assert!(
        doc.contains("Cached tweet summary"),
        "the cached timeline was not summarised: {doc}"
    );
    assert!(doc.contains("1 fetched"), "cache was not read: {doc}");
    drop(env);
}

#[test]
#[serial]
/// No tweets, no cache and no collector: the summarizer must fail rather than
/// write an empty document that reads as "a quiet week on the timeline".
///
/// This test used to DELETE the developer's real
/// `~/.cache/twitter/debug_tweets.json` without restoring it, then shell out to
/// their actual Playwright scraper in the home checkout -- and assert
/// `res.is_err() || res.is_ok()`, which is true of every possible outcome.
fn test_run_summary_empty_fallback() {
    let env = TestEnv::new();
    let mut cfg = sandboxed_config(&env.root().join("cfg"));
    let empty = tempfile::tempdir().unwrap();
    cfg.twitter_cache_path = empty
        .path()
        .join("no-such-cache.json")
        .to_string_lossy()
        .into_owned();
    cfg.twitter_collector_dir = empty.path().to_string_lossy().into_owned();
    let output = tempfile::tempdir().unwrap();

    let res = run_summary(
        &[],
        output.path(),
        Some("http://127.0.0.1:59999"),
        None,
        &cfg,
    );
    assert!(
        res.is_err(),
        "an unreachable LLM with nothing to summarise must fail, not produce a document"
    );
    assert_eq!(
        std::fs::read_dir(output.path()).unwrap().count(),
        0,
        "a failed run must not leave a document behind that reads as a quiet timeline"
    );
    drop(env);
}

#[test]
#[serial]
fn test_run_summary_merges_thinking_into_analysis_section() {
    let env = TestEnv::new();
    let base_url = stub_server(
        r###"{"choices": [{"message": {"content": "## Executive Summary\nKey point<thinking>why it matters</thinking>"}}]}"###,
    );
    let output = tempfile::tempdir().unwrap();
    let cfg = sandboxed_config(&env.root().join("cfg"));

    let res = run_summary(
        &[tweet(
            "routine_user",
            "Testing thinking merge with mock server",
            "14:00",
            1,
            0,
        )],
        output.path(),
        Some(&base_url),
        Some("mock-model"),
        &cfg,
    );
    let content = std::fs::read_to_string(res.expect("the stub answered")).unwrap();
    assert!(content.contains("## Analysis"), "{content}");
    assert!(content.contains("why it matters"), "{content}");
    drop(env);
}

#[test]
#[serial]
fn test_run_summary_fails_when_model_returns_unusable_summary() {
    let env = TestEnv::new();
    let base_url =
        stub_server(r#"{"choices": [{"message": {"content": "junk with no structure"}}]}"#);
    let output = tempfile::tempdir().unwrap();
    let cfg = sandboxed_config(&env.root().join("cfg"));

    let res = run_summary(
        &[tweet(
            "routine_user",
            "Testing critical-output failure",
            "14:00",
            1,
            0,
        )],
        output.path(),
        Some(&base_url),
        Some("mock-model"),
        &cfg,
    );
    let why = format!(
        "{:#}",
        res.expect_err("a critical-quality summary must fail, not be saved")
    );
    // The refused answer is kept for diagnosis, and the reason says where.
    let kept: Vec<_> = std::fs::read_dir(output.path().join("rejected"))
        .expect("a rejected/ directory")
        .flatten()
        .map(|e| e.path())
        .collect();
    // One per model the chain tried (the fixture chain has two).
    assert_eq!(kept.len(), 2, "{kept:?}");
    for path in &kept {
        let body = std::fs::read_to_string(path).unwrap();
        assert!(body.contains("junk with no structure"), "{body}");
        assert!(why.contains(&path.display().to_string()), "{why}");
    }
    drop(env);
}
