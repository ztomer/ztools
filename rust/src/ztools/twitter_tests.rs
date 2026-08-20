//! Unit tests for Rust Twitter summarizer module.

use super::*;

#[test]
fn test_deduplicate_tweets() {
    let tweets = vec![
        Tweet {
            screen_name: "user1".into(),
            text: "Breaking news: Rust 2.0 announced today!".into(),
            created_at: "12:00".into(),
            favorite_count: 10,
            retweet_count: 2,
            reply_to: None,
        },
        Tweet {
            screen_name: "user2".into(),
            text: "RT @user1: Breaking news: Rust 2.0 announced today!".into(),
            created_at: "12:01".into(),
            favorite_count: 0,
            retweet_count: 0,
            reply_to: None,
        },
    ];
    let deduped = deduplicate_tweets(&tweets);
    assert_eq!(deduped.len(), 1);
    assert_eq!(deduped[0].screen_name, "user1");
}

#[test]
fn test_build_prompt() {
    let tweets = vec![Tweet {
        screen_name: "user1".into(),
        text: "Hello Rust!".into(),
        created_at: "12:00".into(),
        favorite_count: 5,
        retweet_count: 1,
        reply_to: None,
    }];
    let (prompt, n) = build_prompt(&tweets, 10000);
    assert_eq!(n, 1);
    assert!(prompt.contains("Hello Rust!"));
    assert!(prompt.contains("5 favs, 1 RTs"));
}

#[test]
fn test_build_prompt_empty_and_budget() {
    let (prompt_empty, n_empty) = build_prompt(&[], 10000);
    assert_eq!(n_empty, 0);
    assert!(prompt_empty.contains("<timeline>"));

    let tweets = vec![
        Tweet {
            screen_name: "user1".into(),
            text: "First tweet content".into(),
            created_at: "12:00".into(),
            favorite_count: 0,
            retweet_count: 0,
            reply_to: None,
        },
        Tweet {
            screen_name: "user2".into(),
            text: "Second tweet content".into(),
            created_at: "12:05".into(),
            favorite_count: 0,
            retweet_count: 0,
            reply_to: None,
        },
    ];
    let (prompt_small, n_small) = build_prompt(&tweets, 100);
    assert!(n_small <= 2);
    assert!(prompt_small.contains("user"));
}

#[test]
fn test_check_summary_quality() {
    let (warn_empty, crit_empty) = check_summary_quality("");
    assert!(crit_empty);
    assert!(warn_empty[0].contains("empty"));

    let good_summary = "## Topic Section\n- Fact one about event\n- Fact two about event\n- Fact three about event\nDetailed description of timeline story.";
    let (warn_good, crit_good) = check_summary_quality(good_summary);
    assert!(!crit_good);
    assert!(warn_good.is_empty());

    let (warn_no_head, crit_no_head) = check_summary_quality("Just raw text without headers");
    assert!(crit_no_head);
    assert!(warn_no_head.iter().any(|w| w.contains("headers")));

    let (warn_short, _) = check_summary_quality("## Short Header\n- Bullet item");
    assert!(warn_short.iter().any(|w| w.contains("Very short")));
}

#[test]
fn test_call_osaurus_invalid_host() {
    let res = call_osaurus(
        "http://127.0.0.1:59999",
        "model",
        "prompt",
        &crate::config::ZtoolsConfig::default(),
    );
    assert!(res.is_err());
}

#[test]
fn test_call_osaurus_and_run_summary_success() {
    use std::io::{Read, Write};
    use std::net::TcpListener;
    use std::thread;

    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let base_url = format!("http://{}", addr);

    thread::spawn(move || {
        for mut stream in listener.incoming().flatten() {
            let mut buf = [0u8; 1024];
            let _ = stream.read(&mut buf);

            let request_str = String::from_utf8_lossy(&buf);
            let body = if request_str.contains("/v1/embeddings") {
                r###"{"data": [{"embedding": [0.1, 0.2]}]}"###
            } else {
                r###"{"choices": [{"message": {"content": "## Section\n- Item 1\n- Item 2"}}]}"###
            };

            let resp = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
                body.len(),
                body
            );
            let _ = stream.write_all(resp.as_bytes());
        }
    });

    let tweets = vec![Tweet {
        screen_name: "routine_user".into(),
        text: "Testing twitter summarizer with mock server".into(),
        created_at: "14:00".into(),
        favorite_count: 3,
        retweet_count: 1,
        reply_to: Some("target_user".into()),
    }];

    let temp_dir = std::env::temp_dir().join("ztools_test_twitter_success");
    let res = run_summary(
        &tweets,
        Some(&temp_dir),
        Some(&base_url),
        Some("mock-model"),
        &crate::config::ZtoolsConfig::default(),
    );
    assert!(res.is_ok());
    let path = res.unwrap();
    assert!(path.exists());

    let content = std::fs::read_to_string(&path).unwrap();
    assert!(content.contains("Twitter Timeline Summary"));
    assert!(content.contains("mock-model"));

    let _ = std::fs::remove_dir_all(&temp_dir);
}

#[test]
fn test_run_summary_cache_reading() {
    use std::io::{Read, Write};
    use std::net::TcpListener;
    use std::thread;

    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let base_url = format!("http://{}", addr);

    thread::spawn(move || {
        for mut stream in listener.incoming().flatten() {
            let mut buf = [0u8; 1024];
            let _ = stream.read(&mut buf);

            // Extremely simple check: if it's embeddings, return mock embedding, else chat.
            let request_str = String::from_utf8_lossy(&buf);
            let body = if request_str.contains("/v1/embeddings") {
                r###"{"data": [{"embedding": [0.1, 0.2]}]}"###
            } else {
                r###"{"choices": [{"message": {"content": "## Section\n- Cached tweet summary"}}]}"###
            };

            let resp = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
                body.len(),
                body
            );
            let _ = stream.write_all(resp.as_bytes());
        }
    });

    // A fixture cache, not the operator's. This test used to WRITE
    // ~/.cache/twitter/debug_tweets.json on the developer's own machine.
    let fixture = tempfile::tempdir().unwrap();
    let cache_file = fixture.path().join("debug_tweets.json");
    let mock_json = r#"[{"screen_name":"cached_user","text":"Cached tweet content for test","created_at":"10:00","favorite_count":2,"retweet_count":0,"reply_to":null}]"#;
    std::fs::write(&cache_file, mock_json).unwrap();

    let temp_dir = std::env::temp_dir().join("ztools_test_twitter_cache");
    let res = run_summary(
        &[],
        Some(&temp_dir),
        Some(&base_url),
        None,
        &crate::config::ZtoolsConfig {
            twitter_cache_path: cache_file.to_string_lossy().into_owned(),
            ..crate::config::ZtoolsConfig::default()
        },
    );
    let path = res.expect("summary should be written");
    let doc = std::fs::read_to_string(&path).unwrap();
    assert!(
        doc.contains("Cached tweet summary"),
        "the cached timeline was not summarised: {doc}"
    );
    assert!(doc.contains("1 fetched"), "cache was not read: {doc}");

    let _ = std::fs::remove_dir_all(&temp_dir);
}

#[test]
/// No tweets, no cache and no collector: the summarizer must fail rather than
/// write an empty document that reads as "a quiet week on the timeline".
///
/// This test used to DELETE the developer's real
/// `~/.cache/twitter/debug_tweets.json` without restoring it, then shell out to
/// their actual Playwright scraper in `~/Projects/ztools` -- and assert
/// `res.is_err() || res.is_ok()`, which is true of every possible outcome.
fn test_run_summary_empty_fallback() {
    let empty = tempfile::tempdir().unwrap();
    let temp_dir = std::env::temp_dir().join("ztools_test_twitter_fallback");
    let res = run_summary(
        &[],
        Some(&temp_dir),
        Some("http://127.0.0.1:59999"),
        None,
        &crate::config::ZtoolsConfig {
            twitter_cache_path: empty
                .path()
                .join("no-such-cache.json")
                .to_string_lossy()
                .into_owned(),
            twitter_collector_dir: empty.path().to_string_lossy().into_owned(),
            ..crate::config::ZtoolsConfig::default()
        },
    );
    assert!(
        res.is_err(),
        "an unreachable LLM with nothing to summarise must fail, not produce a document"
    );
    let _ = std::fs::remove_dir_all(&temp_dir);
}
