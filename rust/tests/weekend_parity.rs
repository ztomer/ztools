//! The CI drift gate between the Rust and Python weekend CORPUS stacks.
//!
//! For every fixture in `tests/fixtures/weekend_parity/`, prints one
//! `PARITY <task>|0|<json-encoded payload>` line computed by the RUST side of
//! the weekend corpus pipeline (the deterministic halves only: corpus cleaning,
//! candidate lines, aggregator classification). The pytest side
//! (`references/tests/test_rust_weekend_parity.py`) computes the same payloads
//! with the PYTHON pipeline (`_clean_search_results`, `as_candidate_lines`,
//! `looks_like_aggregator`) and asserts byte-for-byte agreement.
//!
//! The payload is JSON-encoded so multi-line corpora survive one line of
//! stdout. If this test's output format changes, update the parser — a silent
//! format change would read as "nothing to compare" and green-light exactly
//! the drift this gate exists to catch.

use serde_json::Value;
use ztools::weekend::{
    as_candidate_lines, clean_search_results, looks_like_aggregator, SearchResult, MAX_BODY_LENGTH,
};
use ztools::weekend_cache::load_region_lists;

fn fixture(name: &str) -> String {
    let manifest = env!("CARGO_MANIFEST_DIR");
    std::fs::read_to_string(format!(
        "{manifest}/../tests/fixtures/weekend_parity/{name}"
    ))
    .unwrap_or_else(|e| panic!("fixture {name}: {e}"))
}

/// Region lists from the same checked-in `conf/weekend.toml` the Python
/// pipeline reads — the parity comparison is only meaningful over identical
/// inputs, and the lists are inputs.
fn shipped_region() -> ztools::weekend_cache::RegionLists {
    let manifest = env!("CARGO_MANIFEST_DIR");
    let path = std::path::Path::new(manifest)
        .parent()
        .unwrap()
        .join("conf/weekend.toml");
    load_region_lists(&[path.to_string_lossy().into_owned()])
}

#[test]
fn print_rust_verdicts_for_python_comparison() {
    // — weekend_corpus: clean_search_results over the raw scrape fixture —
    let corpus_value: Value =
        serde_json::from_str(&fixture("corpus_results.json")).expect("corpus fixture json");
    let results: Vec<SearchResult> = corpus_value
        .as_array()
        .expect("corpus fixture array")
        .iter()
        .map(|r| SearchResult {
            title: r["title"].as_str().unwrap_or("").to_string(),
            href: r["href"].as_str().unwrap_or("").to_string(),
            body: r["body"].as_str().unwrap_or("").to_string(),
        })
        .collect();
    let corpus = clean_search_results(&results, "Event", MAX_BODY_LENGTH, &shipped_region());
    println!(
        "PARITY weekend_corpus|0|{}",
        serde_json::to_string(&corpus).expect("corpus json")
    );

    // — weekend_candidates: as_candidate_lines over one aggregator page —
    let page: Value = serde_json::from_str(&fixture("aggregator_page.json")).expect("page json");
    let text = page["text"].as_str().unwrap_or("");
    let title = page["title"].as_str().unwrap_or("");
    let candidates = as_candidate_lines(text, title);
    println!(
        "PARITY weekend_candidates|0|{}",
        serde_json::to_string(&candidates).expect("candidates json")
    );

    // — weekend_aggregator: looks_like_aggregator over the title set —
    let titles: Vec<String> = serde_json::from_str(&fixture("aggregator_flags_titles.json"))
        .expect("aggregator titles json");
    let flags: Vec<bool> = titles.iter().map(|t| looks_like_aggregator(t)).collect();
    println!(
        "PARITY weekend_aggregator|0|{}",
        serde_json::to_string(&flags).expect("aggregator flags json")
    );
}
