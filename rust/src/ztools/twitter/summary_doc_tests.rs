//! Byte-level golden for the saved TWITTER SUMMARY DOCUMENT — the markdown
//! `run_summary` writes to `~/Documents/twitter_summaries` and the dashboard
//! renders.
//!
//! The suite pins CSV output byte for byte (`eval/report_csv.rs`); this
//! document, the other artefact a person reads, was checked only with
//! `contains` — so a provenance banner that stopped saying which model
//! answered, a header line that lost a field, or a `## Summary` preamble
//! stacked on a body that already had a heading would all have passed. Both
//! provenance shapes are pinned: the normal run's one quiet line and the
//! degraded run's block that cannot be mistaken for it.
//!
//! PROVENANCE OF THE EXPECTED BYTES: read, not recorded. Each fixture was
//! generated once and every line checked against the writer and the input that
//! produced it. The clock is the one thing a golden cannot freeze, so it is
//! SUBSTITUTED by a pattern that matches only the writer's exact shape — a
//! writer that changes the format stops matching and fails on the raw bytes
//! rather than passing quietly.
//!
//! A GOLDEN IS NOT A CONSTANT, IT IS A DECISION. `ZTOOLS_UPDATE_GOLDENS=1`
//! rewrites a fixture and still fails, so blessing a change is always a
//! deliberate second run.

use std::path::{Path, PathBuf};

use crate::ztools::twitter::Tweet;

/// `CARGO_MANIFEST_DIR` is `<repo>/rust`.
fn fixture_path(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("the crate lives in <repo>/rust")
        .join("tests/fixtures/user_documents")
        .join(name)
}

/// Compare a written document with its committed golden byte for byte.
fn assert_golden(actual: &str, name: &str) {
    let path = fixture_path(name);
    let updating = std::env::var_os("ZTOOLS_UPDATE_GOLDENS").is_some();
    let expected = match std::fs::read_to_string(&path) {
        Ok(text) => text,
        Err(e) => {
            if updating {
                std::fs::create_dir_all(path.parent().expect("a fixture has a parent"))
                    .expect("the fixture directory this repository owns must be creatable");
                std::fs::write(&path, actual).expect("creating the golden");
                panic!("golden {name} did not exist and was created; re-run to verify it");
            }
            panic!(
                "golden {name} is missing at {}: {e}. It must be committed — a golden this \
                 repository does not own is a test that asserts nothing.",
                path.display()
            );
        }
    };
    if actual == expected {
        return;
    }
    if updating {
        std::fs::write(&path, actual).expect("rewriting the golden");
        panic!("golden {name} was rewritten; re-run to verify it");
    }
    let first = actual
        .lines()
        .zip(expected.lines())
        .position(|(a, b)| a != b)
        .unwrap_or_else(|| actual.lines().count().min(expected.lines().count()));
    panic!(
        "{name} no longer matches the saved summary.\n  first difference at line {}\n  \
         written:  {:?}\n  golden:   {:?}\n  lines: written {}, golden {}\nA change here changes \
         what a human reads. Re-read the document before regenerating: \
         ZTOOLS_UPDATE_GOLDENS=1 rewrites the fixture and still fails.",
        first + 1,
        actual.lines().nth(first),
        expected.lines().nth(first),
        actual.lines().count(),
        expected.lines().count(),
    );
}

/// The one line of the document the golden cannot hold still: the moment the
/// summary was taken. Only `**Period:**`'s value is substituted, and only when
/// it matches the writer's exact `%Y-%m-%d %H:%M %:z` shape (local clock, real
/// offset) — so a change to that format fails the comparison instead of
/// silently skipping the line.
const PERIOD: &str = r"(?m)^\*\*Period:\*\* \d{4}-\d{2}-\d{2} \d{2}:\d{2} [+-]\d{2}:\d{2}$";
const PERIOD_STAMP: &str = "**Period:** <LOCAL CLOCK> <LOCAL OFFSET>";

fn without_the_clock(doc: &str) -> String {
    let re = regex::Regex::new(PERIOD).expect("the clock pattern is a valid regex");
    let stamps = re.find_iter(doc).count();
    assert_eq!(
        stamps, 1,
        "expected exactly one `**Period:**` stamp to substitute, found {stamps} in:\n{doc}"
    );
    re.replace_all(doc, PERIOD_STAMP).into_owned()
}

/// A loopback osaurus. `roster` is what `/v1/models` serves — serving the
/// intended model is what keeps a run on the primary tier, so the two fixtures
/// differ by one roster entry and nothing else.
fn stub_server(roster: &'static str, content: &'static str) -> String {
    use std::io::{Read, Write};

    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    std::thread::spawn(move || {
        while let Ok((mut stream, _)) = listener.accept() {
            let mut buf = vec![0u8; 64 * 1024];
            let n = stream.read(&mut buf).unwrap_or(0);
            let req = String::from_utf8_lossy(&buf[..n]).to_string();
            let body = if req.starts_with("GET /v1/models") {
                format!(r#"{{"data":[{roster}]}}"#)
            } else if req.contains("/v1/embeddings") {
                // One embedding per tweet, or `cluster_tweets` discards the
                // answer and the clustering path stops being exercised.
                r#"{"data":[{"embedding":[0.1,0.2]},{"embedding":[0.1,0.2]}]}"#.to_string()
            } else {
                format!(
                    r#"{{"choices":[{{"message":{{"content":{}}}}}]}}"#,
                    serde_json::to_string(content).unwrap()
                )
            };
            let resp = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",
                body.len()
            );
            let _ = stream.write_all(resp.as_bytes());
            let _ = stream.flush();
        }
    });
    format!("http://{addr}")
}

/// A structured answer that passes `check_summary_quality`: its own `##`
/// headings, one bullet per input tweet (never more), each citing its tweet in
/// the `(@handle | timestamp)` form — the first on a wrapped continuation line —
/// and enough prose to clear the length warning.
const SUMMARY_BODY: &str = "\
## Executive Summary

- Two threads dominated the timeline, the louder a rail shutdown on the Lakeshore
  line between Union and Bathurst (@transit_watch | Thu Aug 20 12:00:00 +0000 2026).
- A funding round for small groups closes Friday and has not been picked up
  (@civic_notes | Thu Aug 20 12:04:00 +0000 2026).

## What The Funding Thread Adds

The funding round closes Friday and one account flagged the deadline specifically.";

fn tweets() -> Vec<Tweet> {
    vec![
        Tweet {
            id: "1900000000000000001".to_string(),
            screen_name: "transit_watch".to_string(),
            text: "Lakeshore line is down between Union and Bathurst, shuttle buses running."
                .to_string(),
            created_at: "Thu Aug 20 12:00:00 +0000 2026".to_string(),
            favorite_count: 42,
            retweet_count: 19,
            reply_to: None,
        },
        Tweet {
            id: "1900000000000000002".to_string(),
            screen_name: "civic_notes".to_string(),
            text: "Funding round closes Friday, applications open to small groups.".to_string(),
            created_at: "Thu Aug 20 12:04:00 +0000 2026".to_string(),
            favorite_count: 8,
            retweet_count: 2,
            reply_to: None,
        },
    ]
}

/// A config whose every path points inside a directory this test owns: the
/// shipped `conf/twitter.toml` must not decide what the document says, and the
/// operator's real cache must not be read.
fn config(base_url: &str, tmp: &std::path::Path) -> crate::config::ZtoolsConfig {
    let fallback = tmp.join("twitter.toml");
    // The degraded document's stand-in is one the policy NAMES. It used to be
    // reached through `select_best_model`'s `models.first()` with an empty
    // preference list -- the arbitrary pick that no longer exists.
    std::fs::write(
        &fallback,
        "[fallback]\nmodels = []\npreferred = [\"small-fallback\"]\n",
    )
    .expect("writing the test's own fallback policy");
    crate::config::ZtoolsConfig {
        twitter_model: "golden-model".to_string(),
        twitter_config_paths: vec![fallback.to_string_lossy().into_owned()],
        twitter_cache_path: tmp
            .join("no-such-cache.json")
            .to_string_lossy()
            .into_owned(),
        twitter_prompt_max_chars: 4000,
        osaurus_url: base_url.to_string(),
        llm_timeout_secs: 5,
        ..crate::config::ZtoolsConfig::default()
    }
}

/// Run the whole summary path against a stub and return (path, document).
fn run(roster: &'static str, content: &'static str) -> (std::path::PathBuf, String) {
    let tmp = tempfile::tempdir().expect("a temp dir for the summary's output");
    let out = tmp.path().join("summaries");
    let base_url = stub_server(roster, content);
    let cfg = config(&base_url, tmp.path());
    let path = crate::ztools::twitter::run_summary(
        &tweets(),
        &out,
        Some(&cfg.osaurus_url),
        Some(&cfg.twitter_model),
        &cfg,
    )
    .expect("the stub answered, so a summary must be written");
    let doc = std::fs::read_to_string(&path).expect("the summary document");
    (path, doc)
}

#[test]
fn the_primary_summary_document_matches_its_golden_byte_for_byte() {
    let (path, doc) = run(r#"{"id":"golden-model"}"#, SUMMARY_BODY);
    // The filename shape, pinned without freezing the clock: minute resolution,
    // `_summary` suffix. Two runs in the same minute overwrite each other, which
    // is a property of the name and not something a golden should hide.
    let name = path
        .file_name()
        .expect("a file name")
        .to_string_lossy()
        .into_owned();
    let stem = name
        .strip_suffix("_summary.md")
        .unwrap_or_else(|| panic!("summary file is named {name:?}"));
    assert!(
        regex::Regex::new(r"^\d{4}-\d{2}-\d{2}_\d{4}$")
            .expect("a valid regex")
            .is_match(stem),
        "summary filename is not <YYYY-MM-DD_HHMM>_summary.md: {name:?}"
    );
    assert_golden(&without_the_clock(&doc), "twitter_summary.md");
}

/// The same run with a roster that does NOT serve the intended model. The chain
/// resolves to what the server does serve, and the document must say so in
/// terms no reader can mistake for a normal run (house rule: a degraded state is
/// visually distinct and states WHY).
#[test]
fn the_degraded_summary_document_matches_its_golden_byte_for_byte() {
    let (_path, doc) = run(r#"{"id":"small-fallback-8b"}"#, SUMMARY_BODY);
    assert_golden(&without_the_clock(&doc), "twitter_summary_degraded.md");
}

/// The calibration for the normal-run golden: a different model ANSWER changes
/// the document, so a comparison blind to the provenance banner would go green
/// with the banner deleted.
#[test]
fn changing_the_answering_model_moves_the_document_bytes() {
    let (_path, primary) = run(r#"{"id":"golden-model"}"#, SUMMARY_BODY);
    let (_other, other) = run(r#"{"id":"small-fallback-8b"}"#, SUMMARY_BODY);
    assert_ne!(
        without_the_clock(&primary),
        without_the_clock(&other),
        "a summary from a different model must not be byte-identical to one from the \
         primary: that is the class-C9 defect this golden exists to hold"
    );
    assert!(!primary.contains("DEGRADED OUTPUT"), "{primary}");
    assert!(other.contains("DEGRADED OUTPUT"), "{other}");
}

/// The audit that keeps these goldens honest: the fixtures must be the writer's
/// output for THIS file's input, and every field in them must be one the writer
/// can actually produce. A fixture carrying a value nothing supplied is the
/// differential-blindness trap wearing a golden's clothes.
#[test]
fn the_goldens_were_read_not_merely_recorded() {
    let normal = std::fs::read_to_string(fixture_path("twitter_summary.md")).unwrap();
    let degraded = std::fs::read_to_string(fixture_path("twitter_summary_degraded.md")).unwrap();

    for doc in [&normal, &degraded] {
        // Counts come from the two tweets in `tweets()`, and both were processed:
        // a document whose counts disagree with the input is not a document.
        assert!(
            doc.contains("**Tweets:** 2 fetched, 2 processed"),
            "counts do not match the two tweets this file supplies:\n{doc}"
        );
        assert!(
            doc.contains("**Period:** <LOCAL CLOCK> <LOCAL OFFSET>"),
            "the clock stamp was not substituted, so this fixture froze a moment in time:\n{doc}"
        );
        // The model's own heading survives, and no `## Summary` preamble is
        // stacked on top of it (the writer mints headings only for a body that
        // has none).
        assert!(doc.contains("## Executive Summary"), "{doc}");
        assert!(!doc.contains("## Summary"), "{doc}");
        assert!(doc.contains("## What The Funding Thread Adds"), "{doc}");
        assert!(
            doc.ends_with(
                "The funding round closes Friday and one account flagged the deadline \
                          specifically.\n"
            ),
            "the document must end with the model body, one trailing newline and nothing \
             else:\n{doc}"
        );
    }

    assert!(
        normal.contains("**Model:** golden-model (osaurus, primary)"),
        "a normal run's provenance is one quiet line:\n{normal}"
    );
    assert!(
        !normal.contains("DEGRADED OUTPUT"),
        "a primary run must not claim to be degraded:\n{normal}"
    );
    assert!(degraded.contains("> ⚠ **DEGRADED OUTPUT**"), "{degraded}");
    assert!(
        degraded.contains("> **Backend:** small-fallback-8b (osaurus, fallback)"),
        "{degraded}"
    );
    assert!(
        degraded.contains("answered by small-fallback-8b instead of the intended golden-model"),
        "a degraded run must name both models, or the reader cannot tell what happened:\n{degraded}"
    );
}

/// The `**Period:**` stamp is the local clock labelled with the local offset.
///
/// It used to print a LOCAL time followed by a literal "UTC", which was wrong
/// by the offset on every run (four hours, here). The golden cannot see this:
/// its stamp is substituted. So this reads the live line and checks the label
/// against the offset the clock really has.
#[test]
fn the_period_stamp_names_the_real_offset_not_a_literal_utc() {
    let (_p, doc) = run(r#"{"id":"golden-model"}"#, SUMMARY_BODY);
    let line = doc
        .lines()
        .find(|l| l.starts_with("**Period:**"))
        .expect("a Period line");
    let offset = chrono::Local::now().format("%:z").to_string();
    assert!(
        line.ends_with(&format!(" {offset}")),
        "the Period stamp must carry the local offset {offset}: {line:?}"
    );
    assert!(!line.contains("UTC"), "{line:?}");
}

/// The document's PARAGRAPH structure, asserted on documents rendered live.
///
/// The provenance block must start its own paragraph. Without the blank line
/// `**Tweets:** ...` and `**Model:** ...` are two lines of one paragraph, so
/// every `CommonMark` renderer folds them into a single run-on line and the model
/// line reads as part of the count. `>` may interrupt a paragraph, so the
/// DEGRADED banner survived the missing blank line by luck; the quiet one did
/// not, which is why both shapes are checked.
///
/// It renders rather than reading the fixtures: a structural assertion over
/// committed bytes cannot see a writer that stopped producing them, so it would
/// sit green through exactly the break it was written to catch.
#[test]
fn the_provenance_block_is_its_own_paragraph() {
    // The line the assertions anchor on, named once so they cannot drift apart.
    const COUNT: &str = "**Tweets:** 2 fetched, 2 processed";
    let (_p, quiet) = run(r#"{"id":"golden-model"}"#, SUMMARY_BODY);
    let (_d, degraded) = run(r#"{"id":"small-fallback-8b"}"#, SUMMARY_BODY);

    for doc in [&quiet, &degraded] {
        assert!(
            doc.contains(&format!("{COUNT}\n\n")),
            "the tweets line must be followed by a blank line, or the provenance block is \
             lazy continuation of its paragraph rather than a paragraph of its own:\n{doc}"
        );
    }
    assert!(
        !quiet.contains(&format!("{COUNT}\n**Model:**")),
        "the quiet provenance line is glued to the count:\n{quiet}"
    );
    assert!(
        !degraded.contains(&format!("{COUNT}\n> ⚠")),
        "the degraded block quote is glued to the count:\n{degraded}"
    );

    // Inside that block quote the closing advice is its own paragraph: the
    // banner already put a `>` blank line above its Backend line, and the advice
    // — the one line that says how much to trust the summary — was left as a
    // continuation of `**Why:**` instead of standing apart from it.
    let advice = "> Treat the content below";
    let stands_apart = degraded
        .lines()
        .collect::<Vec<_>>()
        .windows(2)
        .any(|w| w[0] == ">" && w[1].starts_with(advice));
    assert!(
        stands_apart,
        "the closing advice needs a `>` blank line above it, exactly as the banner has \
         above its Backend line:\n{degraded}"
    );
    // And it is still inside the quote, not escaped out of it.
    assert!(
        degraded.lines().any(|l| l.starts_with(advice)),
        "{degraded}"
    );
}
