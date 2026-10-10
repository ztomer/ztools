//! The pure string helpers: dedup, prompt building, section framing, the
//! quality gate, and the thinking-merge.
//!
//! None of these touch the filesystem, so none of them need the sandbox — but
//! `build_prompt` takes its instructions from a `ZtoolsConfig`, and that
//! config's defaults name `~`, so they take the guard anyway and the audit gate
//! says so out loud.

use super::super::*;
use super::support::sandboxed_config;
use crate::test_env::TestEnv;
use serial_test::serial;

fn tweet(id: &str, user: &str, text: &str, at: &str, fav: u64, rt: u64) -> Tweet {
    Tweet {
        id: id.to_string(),
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
fn test_deduplicate_tweets() {
    let env = TestEnv::new();
    let tweets = vec![
        tweet(
            "1",
            "user1",
            "Breaking news: Rust 2.0 announced today!",
            "12:00",
            10,
            2,
        ),
        tweet(
            "2",
            "user2",
            "RT @user1: Breaking news: Rust 2.0 announced today!",
            "12:01",
            0,
            0,
        ),
    ];
    let deduped = deduplicate_tweets(&tweets);
    assert_eq!(deduped.len(), 1);
    assert_eq!(deduped[0].screen_name, "user1");
    drop(env);
}

#[test]
#[serial]
fn test_build_prompt() {
    let env = TestEnv::new();
    let cfg = sandboxed_config(&env.root().join("cfg"));
    let (prompt, n) = build_prompt(
        &[tweet("1", "user1", "Hello Rust!", "12:00", 5, 1)],
        10_000,
        &cfg.twitter_summarize_prompt,
    );
    assert_eq!(n, 1);
    assert!(prompt.contains("Hello Rust!"));
    assert!(prompt.contains("5 favs, 1 RTs"));
    drop(env);
}

#[test]
#[serial]
fn test_build_prompt_empty_and_budget() {
    let env = TestEnv::new();
    let cfg = sandboxed_config(&env.root().join("cfg"));
    let (prompt_empty, n_empty) = build_prompt(&[], 10_000, &cfg.twitter_summarize_prompt);
    assert_eq!(n_empty, 0);
    assert!(prompt_empty.contains("<timeline>"));

    let tweets = vec![
        tweet("1", "user1", "First tweet content", "12:00", 0, 0),
        tweet("2", "user2", "Second tweet content", "12:05", 0, 0),
    ];
    let (prompt_small, n_small) = build_prompt(&tweets, 100, &cfg.twitter_summarize_prompt);
    assert!(n_small <= 2);
    assert!(prompt_small.contains("user"));
    drop(env);
}

#[test]
fn summary_section_never_prepends_a_heading_a_body_already_has() {
    let with_heading = summary_section_for("## Executive Summary\n\nSomething happened.");
    assert!(
        !with_heading.starts_with("## Summary"),
        "the model's own heading must not end up under an empty `## Summary` preamble: {with_heading:?}"
    );
    assert_eq!(with_heading, "## Executive Summary\n\nSomething happened.");

    assert_eq!(summary_section_for(""), "");
    assert_eq!(summary_section_for("   \n  "), "");
    assert_eq!(summary_section_for("# Single heading"), "# Single heading");

    let bare = summary_section_for("Nothing but prose.");
    assert_eq!(bare, "## Summary\n\nNothing but prose.");
}

// The quality gate's own fixtures, one per degenerate shape, are in
// `quality_tests.rs` beside the module that owns it.

#[test]
fn test_merge_thinking_with_summary_includes_analysis() {
    // Port of test_merge_thinking_with_summary in test_twitter.py.
    let thinking = "Analysis: This is important";
    let summary = "## Summary\nMain content";
    let result = merge_thinking_with_summary(thinking, summary);
    assert!(result.contains("## Analysis"), "{result}");
    assert!(result.contains(thinking), "{result}");
    assert!(result.contains(summary), "{result}");
}

#[test]
fn test_merge_with_empty_thinking_returns_summary_unchanged() {
    assert_eq!(
        merge_thinking_with_summary("", "## Summary\nMain"),
        "## Summary\nMain"
    );
}

#[test]
fn test_extract_thinking_splits_block_from_body() {
    // Verified against the Python original: `cleaned` keeps the raw
    // `<thinking>` block (`remove_thinking_blocks` strips `<think>` tags, not
    // `<thinking>` ones) — the thinking is extracted for the Analysis section,
    // not removed from the body.
    let (thinking, cleaned) =
        extract_thinking("## Summary\nMain<thinking type=\"x\">inner reasoning</thinking>\nTail");
    assert_eq!(thinking, "inner reasoning");
    assert!(cleaned.contains("Main"), "{cleaned}");
    assert!(cleaned.contains("<thinking"), "{cleaned}");
}

#[test]
fn test_extract_thinking_without_block_returns_text_untouched() {
    let (thinking, cleaned) = extract_thinking("## Summary\nMain content");
    assert_eq!(thinking, "");
    assert_eq!(cleaned, "## Summary\nMain content");
}

#[test]
fn test_handle_model_output_passes_good_content_through() {
    // Port of test_success_first_model in test_twit_summarize.py.
    let result = handle_model_output(
        "## Topic\n- fact 1 (@a | 1)\n- fact 2 (@b | 2)\n- fact 3 (@c | 3)",
        &given(&[
            ("a", "1"),
            ("b", "2"),
            ("c", "3"),
            ("d", "4"),
            ("e", "5"),
            ("f", "6"),
            ("g", "7"),
        ]),
    );
    let (text, processed) = result.expect("good content must survive the gate");
    assert!(text.contains("Topic"), "{text}");
    assert_eq!(processed, 7);
}

#[test]
fn test_handle_model_output_merges_thinking_when_present() {
    // Port of test_success_with_thinking: thinking routes through the merge,
    // then the merged text faces the same quality gate.
    let result = handle_model_output(
        "<thinking>reasoning here</thinking>## Topic\n- a (@a | 1)\n- b (@b | 2)\n- c (@c | 3)",
        &given(&[("a", "1"), ("b", "2"), ("c", "3")]),
    );
    let (text, _) = result.expect("merged thinking must survive the gate");
    assert!(text.contains("## Analysis"), "{text}");
    assert!(text.contains("reasoning here"), "{text}");
}

#[test]
fn test_handle_model_output_drops_critical_thinking_output() {
    // Port of test_thinking_critical_skips (single attempt): thinking present
    // but the body is unstructured, so the attempt yields nothing.
    assert!(handle_model_output("<thinking>x</thinking>bad", &given(&[("a", "1")])).is_err());
}

#[test]
fn test_handle_model_output_drops_structureless_content() {
    // Port of test_target_model_with_known_critical_skips (single attempt).
    assert!(handle_model_output("no structure here at all", &given(&[("a", "1")])).is_err());
}

/// A rejected answer carries the gate's reason, so the chain can record WHY
/// the model was passed over rather than "no usable summary".
#[test]
fn test_handle_model_output_names_the_rejection() {
    let looped = "## Topic\n- same fact (@a | 1)\n- same fact (@a | 1)\n- same fact (@a | 1)\n- same fact (@a | 1)";
    let why = handle_model_output(looped, &given(&[("a", "1"); 10])).unwrap_err();
    assert!(why.contains("repeat an earlier bullet"), "{why}");
}

/// The tweets the prompt showed, as `(handle, created_at)`.
fn given(pairs: &[(&str, &str)]) -> Vec<(String, String)> {
    pairs
        .iter()
        .map(|(h, t)| ((*h).to_string(), (*t).to_string()))
        .collect()
}

/// The gate is wired to the tweets themselves: a well-formed answer whose
/// citations name tweets the model was never given is not saved.
#[test]
fn test_handle_model_output_refuses_citations_of_tweets_it_was_not_given() {
    let invented = "## Topic\n- fact 1 (@a | 1)\n- fact 2 (@zz | 9)\n- fact 3 (@c | 3o)";
    let why =
        handle_model_output(invented, &given(&[("a", "1"), ("b", "2"), ("c", "3")])).unwrap_err();
    assert!(why.contains("2 of 3 citations"), "{why}");
}

/// The sandbox config really is self-contained.
///
/// The assertion that makes it a config rather than a hope: every path field
/// is under the test's own root, and the `[fallback]` table
/// `run_summary` refuses to start without is actually in the fixture -- which
/// is the thing these tests were silently getting from this checkout's
/// `conf/twitter.toml`.
#[test]
fn the_sandbox_config_points_every_path_at_its_own_root_and_carries_the_fallback_table() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().to_str().unwrap();
    let cfg = sandboxed_config(dir.path());
    for (name, value) in [
        ("twitter_cache_path", &cfg.twitter_cache_path),
        ("twitter_collector_dir", &cfg.twitter_collector_dir),
        ("search_record_path", &cfg.search_record_path),
    ] {
        assert!(
            value.starts_with(root),
            "{name} escapes the test root: {value}"
        );
    }
    for value in cfg
        .twitter_config_paths
        .iter()
        .chain(cfg.weekend_exclusions_paths.iter())
        .chain(cfg.weekend_region_paths.iter())
    {
        assert!(
            value.starts_with(root),
            "a config path escapes the test root: {value}"
        );
    }
    let policy = crate::ztools::twitter::chain::load_fallback_policy(&cfg.twitter_config_paths)
        .expect("the sandbox config carries the fallback table run_summary needs");
    assert_eq!(policy.models, vec!["fixture-model".to_string()]);
}
