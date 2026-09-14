//! Weekend corpus parity: the deterministic half of the Rust corpus pipeline
//! (`clean_search_results`, `as_candidate_lines`, `looks_like_aggregator`)
//! must still produce what the retired Python pipeline produced.
//!
//! Until 2026-09-13 this was a two-process gate: this file PRINTED
//! `PARITY <task>|0|<json>` lines and a pytest half recomputed the payloads
//! with `weekend.data._clean_search_results` / `weekend.followup` and diffed
//! them. The Python pipeline is gone; on its last run its payloads over the
//! fixtures in `tests/fixtures/weekend_parity/` were frozen into
//! `expected_python_payloads.json`, and this test asserts the Rust side
//! reproduces them byte for byte.
//!
//! Scope is unchanged: live search rankings and page extraction are
//! corpus-QUALITY concerns and are not byte-compared. The fixtures use only
//! region inputs both filters agreed on; the deliberate divergence (a foreign
//! city beats a local token) is pinned by `a_foreign_city_beats_any_local_token`.

use serde_json::Value;
use ztools::weekend::{
    as_candidate_lines, clean_search_results, looks_like_aggregator, SearchResult, MAX_BODY_LENGTH,
};
use ztools::weekend_cache::load_region_lists;

fn fixtures_dir() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .join("tests/fixtures/weekend_parity")
}

fn fixture(name: &str) -> String {
    std::fs::read_to_string(fixtures_dir().join(name))
        .unwrap_or_else(|e| panic!("fixture {name}: {e}"))
}

/// Region lists from the same checked-in `conf/weekend.toml` the Python
/// pipeline read when the golden was frozen — the lists are inputs.
fn shipped_region() -> ztools::weekend_cache::RegionLists {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .join("conf/weekend.toml");
    load_region_lists(&[path.to_string_lossy().into_owned()])
}

fn corpus_results() -> Vec<SearchResult> {
    let corpus_value: Value =
        serde_json::from_str(&fixture("corpus_results.json")).expect("corpus fixture json");
    corpus_value
        .as_array()
        .expect("corpus fixture array")
        .iter()
        .map(|r| SearchResult {
            title: r["title"].as_str().unwrap_or("").to_string(),
            href: r["href"].as_str().unwrap_or("").to_string(),
            body: r["body"].as_str().unwrap_or("").to_string(),
        })
        .collect()
}

fn expected() -> Value {
    serde_json::from_str(&fixture("expected_python_payloads.json")).expect("expected payloads json")
}

#[test]
fn corpus_cleaning_reproduces_the_frozen_python_corpus() {
    let corpus = clean_search_results(
        &corpus_results(),
        "Event",
        MAX_BODY_LENGTH,
        &shipped_region(),
    );
    assert_eq!(Value::String(corpus), expected()["weekend_corpus"]);
}

#[test]
fn candidate_lines_reproduce_the_frozen_python_lines() {
    let page: Value = serde_json::from_str(&fixture("aggregator_page.json")).expect("page json");
    let candidates = as_candidate_lines(
        page["text"].as_str().unwrap_or(""),
        page["title"].as_str().unwrap_or(""),
    );
    assert_eq!(Value::String(candidates), expected()["weekend_candidates"]);
}

#[test]
fn aggregator_flags_reproduce_the_frozen_python_flags() {
    let titles: Vec<String> =
        serde_json::from_str(&fixture("aggregator_flags_titles.json")).expect("titles json");
    let flags: Vec<bool> = titles.iter().map(|t| looks_like_aggregator(t)).collect();
    assert_eq!(serde_json::json!(flags), expected()["weekend_aggregator"]);
}

#[test]
fn the_corpus_fixture_exercises_every_kernel_rule() {
    // The fixture is only evidence if it reaches each rule.
    let corpus = clean_search_results(
        &corpus_results(),
        "Event",
        MAX_BODY_LENGTH,
        &shipped_region(),
    );
    // Title-only dedupe: two same-title different-body rows collapse to one.
    assert_eq!(corpus.matches("- Vaughan Fall Fair: ").count(), 1);
    // Trailing-punctuation title dedupes into the plain-title row.
    assert!(!corpus.contains("- Vaughan Fall Fair!!!"));
    // Truncation to MAX_BODY_LENGTH: the long row's body is exactly that long.
    let storytime = corpus
        .lines()
        .find(|l| l.contains("Vaughan Public Library Storytime"))
        .expect("storytime row present");
    let body = storytime.split_once(": ").map_or("", |(_, b)| b);
    assert_eq!(body.chars().count(), MAX_BODY_LENGTH);
}

#[test]
fn the_golden_can_fail() {
    // Calibration: a changed title must change the corpus.
    let mut results = corpus_results();
    results[0].title = "Vaughan Fall Fairzed".to_string();
    let mutated = clean_search_results(&results, "Event", MAX_BODY_LENGTH, &shipped_region());
    assert_ne!(Value::String(mutated), expected()["weekend_corpus"]);
}
