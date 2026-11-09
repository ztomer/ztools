//! The prompt/citation contract (ZT.TWITTERCHAIN, 2026-11-09): what a model is
//! shown is what it may cite, the prompt names no account the timeline lacks,
//! and each model is shown what fits its own window.

use std::os::unix::fs::PermissionsExt;
use std::path::Path;

use super::super::*;
use super::support::{sandboxed_config, stub_server};
use crate::test_env::TestEnv;
use serial_test::serial;

fn tweet(user: &str, text: &str, at: &str) -> Tweet {
    Tweet {
        id: String::new(),
        screen_name: user.to_string(),
        text: text.to_string(),
        created_at: at.to_string(),
        favorite_count: 4,
        retweet_count: 1,
        reply_to: None,
    }
}

/// Every `@handle` the prompt names is one the timeline holds. The shared
/// instructions used to carry `@TechCrunch | 08:00` as their example, and
/// raptor cited it on every bullet of timelines with no such account.
#[test]
fn the_prompt_names_no_handle_the_timeline_lacks() {
    let instructions = crate::config::default_twitter_summarize_prompt();
    let tweets = [
        tweet(
            "alice",
            "The bridge reopened today",
            "Mon Nov 09 01:00:00 +0000 2026",
        ),
        tweet(
            "bob_2",
            "Rates were held steady",
            "Mon Nov 09 02:00:00 +0000 2026",
        ),
    ];
    let (prompt, n) = build_prompt(&tweets, 24_000, &instructions);
    assert_eq!(n, 2);
    let handle = regex::Regex::new(r"@([A-Za-z0-9_]+)").unwrap();
    let foreign: Vec<&str> = handle
        .captures_iter(&prompt)
        .map(|c| c.get(1).unwrap().as_str())
        .filter(|h| !["alice", "bob_2", "handle"].contains(h))
        .collect();
    assert!(
        foreign.is_empty(),
        "the prompt names {foreign:?}:\n{prompt}"
    );
}

/// The citable set is the tweets the prompt SHOWED. A line over the budget is
/// skipped and a later one still fits, so they are not a prefix of the input;
/// a citation of the shown tweet used to be refused as naming no given tweet.
#[test]
#[serial]
fn a_citation_of_a_shown_tweet_is_accepted_when_the_budget_skipped_an_earlier_one() {
    let env = TestEnv::new();
    let base_url = stub_server(
        r###"{"choices": [{"message": {"content": "## News\n- Rates were held steady (@bob_2 | 15:00)"}}]}"###,
    );
    let mut cfg = sandboxed_config(&env.root().join("cfg"));
    cfg.twitter_prompt_max_chars = 200;
    let output = tempfile::tempdir().unwrap();
    let long = "a very long tweet about the harbour ".repeat(20);
    let res = run_summary(
        &[
            tweet("alice", &long, "14:00"),
            tweet("bob_2", "Rates were held steady", "15:00"),
        ],
        output.path(),
        Some(&base_url),
        Some("mock-model"),
        &cfg,
    );
    let path = res.expect("the cited tweet was shown, so the answer stands");
    let doc = std::fs::read_to_string(path).unwrap();
    assert!(doc.contains("1 processed"), "{doc}");
    drop(env);
}

/// A stand-in `fm` that keeps the prompt it was given and answers with one
/// distinct, correctly cited bullet per source line it saw.
fn fake_fm(dir: &Path) -> std::path::PathBuf {
    let path = dir.join("fm");
    let body = format!(
        "#!/bin/sh\ncat > '{seen}'\nprintf '## Executive Summary\\nA day.\\n\\n## News\\n'\n\
         sed -nE 's/^\\[(@[a-z0-9_]+ \\| [^|]+) \\| [^]]*\\]: (.*)$/- \\2 (\\1)/p' '{seen}'\n",
        seen = dir.join("seen.txt").display()
    );
    std::fs::write(&path, body).unwrap();
    std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755)).unwrap();
    path
}

/// foundation's 4096-token window holds prompt AND answer. It was sent the
/// 24 000-character budget raptor gets, which left it no room to answer.
#[test]
#[serial]
fn foundation_is_shown_only_what_fits_its_window_with_room_to_answer() {
    let env = TestEnv::new();
    let models = env.set_path("ZTOOLS_CONF_DIR", "conf_dir").join("models");
    std::fs::create_dir_all(&models).unwrap();
    std::fs::write(models.join("foundation.toml"), "context_window = 4096\n").unwrap();
    let base_url = stub_server(r#"{"data": []}"#);
    let cfg = sandboxed_config(&env.root().join("cfg"));
    let bin = tempfile::tempdir().unwrap();
    let fm = fake_fm(bin.path());
    std::fs::write(
        &cfg.twitter_config_paths[0],
        format!(
            "[fallback]\nmodels = [\"foundation\"]\npreferred = []\non_device = \"{}\"\n",
            fm.display()
        ),
    )
    .unwrap();
    let words = [
        "amber", "basalt", "cedar", "delta", "ember", "fjord", "garnet",
    ];
    let tweets: Vec<Tweet> = (0..60)
        .map(|i| {
            let text = format!(
                "story {i} {} {} {} and a long tail of ordinary words to fill the line out {}",
                words[i % 7],
                words[(i / 7) % 7],
                i * 31,
                "padding ".repeat(20)
            );
            tweet(
                &format!("u{i}"),
                &text,
                &format!("Mon Nov 09 {:02}:{:02}:00 +0000 2026", i / 60, i % 60),
            )
        })
        .collect();
    let output = tempfile::tempdir().unwrap();
    let res = run_summary(
        &tweets,
        output.path(),
        Some(&base_url),
        Some("foundation"),
        &cfg,
    );
    let seen = std::fs::read_to_string(bin.path().join("seen.txt")).unwrap();
    let tokens =
        u64::try_from(seen.len()).unwrap() / crate::ztools::eval::transport::CHARS_PER_TOKEN;
    assert!(
        tokens + budget::DEFAULT_OUTPUT_TOKENS <= 4096,
        "a {}-character prompt leaves foundation no room to answer",
        seen.len()
    );
    let path = res.expect("every bullet cites a shown tweet");
    assert!(
        std::fs::read_to_string(path)
            .unwrap()
            .contains("foundation")
    );
    drop(env);
}
