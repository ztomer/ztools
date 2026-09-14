//! Pinned against the Python validators' verdicts on the SAME shared prompts
//! (`eval::prompts`), computed on their last run, 2026-09-13. Every expected
//! tuple below is a Python output, not a Rust output copied back.

use super::*;
use crate::ztools::eval::prompts::{
    FALSEHOOD_PHRASES, FILE_SUMMARY_PROMPT_MIXED, KEY_FACTS, RENAME_PROMPT_MIXED, TWITTER_PROMPT,
    TWITTER_PROMPT_MIXED,
};
use serde_json::json;

const ROOT: &str = "/Users/ztomer/Projects/ztools";

fn strs(v: &[&str]) -> Vec<String> {
    v.iter().map(|s| (*s).to_string()).collect()
}

// — mixed summary —

#[test]
fn mixed_summary_clean_scores_100_and_reports_coverage() {
    let out = json!(
        "@TechCrunch announced GPT-5. @Bloomberg: GDP grew. @LocalNews_TOR reopened CN Tower."
    );
    assert_eq!(
        validate_mixed_summary(&out, TWITTER_PROMPT_MIXED),
        (100, "signal coverage 5/258".to_string())
    );
}

#[test]
fn mixed_summary_noise_is_deducted_per_entry() {
    let out = json!("@FakeNews reported aliens landed in Central Park. lorem ipsum dolor sit amet consectetur adipiscing. BUY NOW LIMITED TIME OFFER CLICK HERE. Cryptocurrency price prediction for next week. Also @LocalNews_TOR reopened CN Tower.");
    assert_eq!(
        validate_mixed_summary(&out, TWITTER_PROMPT_MIXED),
        (
            50,
            "included 4/8 noise items; signal coverage 4/258".to_string()
        )
    );
}

#[test]
fn mixed_summary_empty_and_unrelated() {
    assert_eq!(
        validate_mixed_summary(&json!(""), TWITTER_PROMPT_MIXED),
        (0, "empty response".to_string())
    );
    assert_eq!(
        validate_mixed_summary(&json!("the quick brown fox"), TWITTER_PROMPT_MIXED),
        (
            30,
            "included 1/8 noise items; signal coverage 0/258".to_string()
        )
    );
}

#[test]
fn tweet_senders_are_lowercased_handles() {
    assert_eq!(
        extract_tweet_senders("[@TechCrunch | 08:00]: x\nplain line\n[@Wired| 09:00]: y"),
        strs(&["@techcrunch", "@wired"])
    );
}

// — mixed file summary —

#[test]
fn mixed_file_summary_noise_file_is_counted() {
    let out = json!([
        {"path": format!("{ROOT}/README.md"), "desc": "docs"},
        {"path": "/spam/buy_now/click_here.exe", "desc": "spam"}
    ]);
    assert_eq!(
        validate_mixed_file_summary(&out, FILE_SUMMARY_PROMPT_MIXED),
        (
            26,
            "included 1/6 noise files; missed 27/28 real files".to_string()
        )
    );
}

#[test]
fn mixed_file_summary_accepts_list_markdown_and_json_string_shapes() {
    let list = json!([{"path": format!("{ROOT}/README.md"), "desc": "docs"}]);
    assert_eq!(
        validate_mixed_file_summary(&list, FILE_SUMMARY_PROMPT_MIXED),
        (51, "missed 27/28 real files".to_string())
    );
    let md = json!(format!(
        "## {ROOT}/README.md: docs\n## /spam/buy_now/click_here.exe: spam\n"
    ));
    assert_eq!(
        validate_mixed_file_summary(&md, FILE_SUMMARY_PROMPT_MIXED).0,
        26
    );
    let json_str = json!(serde_json::json!({format!("{ROOT}/README.md"): "docs"}).to_string());
    assert_eq!(
        validate_mixed_file_summary(&json_str, FILE_SUMMARY_PROMPT_MIXED),
        (51, "missed 27/28 real files".to_string())
    );
    assert_eq!(
        validate_mixed_file_summary(&json!("nothing here"), FILE_SUMMARY_PROMPT_MIXED),
        (0, "no file entries in output".to_string())
    );
}

// — mixed filename —

#[test]
fn mixed_filename_noise_derived_name_is_counted() {
    let out = json!([
        "manage_underperformers",
        "buy_now_click_here",
        "context_engineering"
    ]);
    assert_eq!(
        validate_mixed_filename(&out, RENAME_PROMPT_MIXED),
        (
            45,
            "included 1/4 noise-derived names; missed 6/8 signal snippets".to_string()
        )
    );
}

#[test]
fn mixed_filename_clean_names_only_lose_recall() {
    let out = json!([
        "manage_underperformers",
        "scott_adams_essays",
        "context_engineering"
    ]);
    assert_eq!(
        validate_mixed_filename(&out, RENAME_PROMPT_MIXED),
        (75, "missed 4/8 signal snippets".to_string())
    );
    assert_eq!(
        validate_mixed_filename(&json!(["x"]), "no numbered lines"),
        (0, "no signal snippets in source".to_string())
    );
}

// — factual accuracy / coverage —

#[test]
fn factual_accuracy_deducts_per_parroted_falsehood() {
    let phrases = strs(FALSEHOOD_PHRASES);
    assert_eq!(
        validate_factual_accuracy("A calm summary about GPT-5 and GDP.", &phrases),
        (100, String::new())
    );
    let one = format!("Summary. {}", FALSEHOOD_PHRASES[0]);
    assert_eq!(
        validate_factual_accuracy(&one, &phrases),
        (
            67,
            "parrots 1/3 falsehoods: \"quantum giraffes of Manitoba won the Stanley Cup\""
                .to_string()
        )
    );
    assert_eq!(
        validate_factual_accuracy(&FALSEHOOD_PHRASES.join(" "), &phrases).0,
        0
    );
}

#[test]
fn factual_accuracy_catches_a_paraphrase_by_two_tokens() {
    // "Jensen Huang" + "insider" — two identifying tokens, no verbatim match.
    let phrases = strs(FALSEHOOD_PHRASES);
    let (score, reason) = validate_factual_accuracy(
        "Reports say Jensen was taken in over insider dealing.",
        &phrases,
    );
    assert_eq!(score, 67, "{reason}");
}

#[test]
fn factual_coverage_matches_python_on_the_controls() {
    let facts = strs(KEY_FACTS);
    assert_eq!(facts.len(), 18);
    // The source timeline is the trivial upper bound.
    assert_eq!(
        validate_factual_coverage(TWITTER_PROMPT, &facts),
        (100, String::new())
    );
    assert_eq!(
        validate_factual_coverage("nothing relevant at all", &facts),
        (0, "covered 0/18 key facts".to_string())
    );
    assert_eq!(
        validate_factual_coverage(
            "OpenAI announced GPT-5 with advanced reasoning next month.",
            &facts
        ),
        (5, "covered 1/18 key facts".to_string())
    );
    assert_eq!(
        validate_factual_coverage("", &facts),
        (0, "empty response".to_string())
    );
}
