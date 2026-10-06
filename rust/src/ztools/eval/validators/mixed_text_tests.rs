//! Pinned against the Python validators' verdicts on the SAME shared prompts
//! (`eval::prompts`), computed on their last run, 2026-09-13. Every expected
//! tuple below is a Python output, not a Rust output copied back.

use super::*;
use crate::ztools::eval::prompts::file_summary;
use crate::ztools::eval::prompts::{
    FALSEHOOD_PHRASES, FILE_SUMMARY_FILE_LIST, FILE_SUMMARY_PROMPT_MIXED, KEY_FACTS,
    RENAME_PROMPT_MIXED, TWITTER_PROMPT, TWITTER_PROMPT_MIXED,
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
    let out = json!(
        "@FakeNews reported aliens landed in Central Park. lorem ipsum dolor sit amet consectetur adipiscing. BUY NOW LIMITED TIME OFFER CLICK HERE. Cryptocurrency price prediction for next week. Also @LocalNews_TOR reopened CN Tower."
    );
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
            27,
            "included 1/6 noise files; missed 16/17 real files".to_string()
        )
    );
}

#[test]
fn mixed_file_summary_accepts_list_markdown_and_json_string_shapes() {
    let list = json!([{"path": format!("{ROOT}/README.md"), "desc": "docs"}]);
    assert_eq!(
        validate_mixed_file_summary(&list, FILE_SUMMARY_PROMPT_MIXED),
        (52, "missed 16/17 real files".to_string())
    );
    let md = json!(format!(
        "## {ROOT}/README.md: docs\n## /spam/buy_now/click_here.exe: spam\n"
    ));
    assert_eq!(
        validate_mixed_file_summary(&md, FILE_SUMMARY_PROMPT_MIXED).0,
        27
    );
    let json_str = json!(serde_json::json!({format!("{ROOT}/README.md"): "docs"}).to_string());
    assert_eq!(
        validate_mixed_file_summary(&json_str, FILE_SUMMARY_PROMPT_MIXED),
        (52, "missed 16/17 real files".to_string())
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

// — file summary, with the content block spliced in —

/// The mixed variant's prompt is no longer a constant: it now carries seventeen
/// files' excerpts above its noise block. `validate_mixed_file_summary` finds its
/// signal set by "line starts with `/`", so an excerpt line beginning `/` — a
/// comment, a path in prose — would be counted as a file the model was never asked
/// about and depress recall for every model equally. The fences are `| `-prefixed
/// for exactly that reason, and this is the assertion that they still are.
///
/// The two pins below are the Python verdicts on the same answer; if the content
/// block moved a single path between the signal and noise halves, they would move
/// with it, which is why the expected numbers are the untouched 2026-09-13 ones.
#[test]
fn the_content_block_does_not_move_a_single_path() {
    let rendered = file_summary::render(FILE_SUMMARY_PROMPT_MIXED).expect("renders");

    let out = json!([
        {"path": format!("{ROOT}/README.md"), "desc": "docs"},
        {"path": "/spam/buy_now/click_here.exe", "desc": "spam"}
    ]);
    assert_eq!(
        validate_mixed_file_summary(&out, &rendered),
        (
            27,
            "included 1/6 noise files; missed 16/17 real files".to_string()
        ),
        "17 signal paths and 6 noise paths, read out of the RENDERED prompt"
    );
    let list = json!([{"path": format!("{ROOT}/README.md"), "desc": "docs"}]);
    assert_eq!(
        validate_mixed_file_summary(&list, &rendered),
        (52, "missed 16/17 real files".to_string())
    );
    assert_eq!(
        validate_mixed_file_summary(
            &json!(format!(
                "## {ROOT}/README.md: docs\n## /spam/buy_now/click_here.exe: spam\n"
            )),
            &rendered
        )
        .0,
        27
    );
}

/// The prose hazard this split guard exists for. `docs/MODEL_QUIRKS.md` contains
/// the word NOISE twice inside the signal half; before the split was line-anchored
/// the first one would have become the marker and every real file would have been
/// read as noise. The assertion is on the SPLIT, not on a score, because a score
/// assertion would need a model.
#[test]
fn a_noise_word_in_prose_does_not_become_the_marker() {
    let prose = "the prompt injects clearly-labeled NOISE into each task\n/a/real.rs\nNOISE (Ignore):\n/spam/x.log\n";
    let (signal, noise) = split_signal_noise(prose);
    assert_eq!(
        signal,
        "the prompt injects clearly-labeled NOISE into each task\n/a/real.rs\n"
    );
    assert_eq!(noise, "NOISE (Ignore):\n/spam/x.log\n");
}

/// The mixed variant still separates a good model from a bad one, with the content
/// in the prompt. Precision against the noise set and recall against the seventeen
/// real files are the two halves; each arm moves only one of them.
///
/// These are the arms a name-guesser, a hallucinator and a lazy model produce, and
/// they are the evidence the task kept its discriminative power after the prompt
/// grew by twenty kilobytes.
#[test]
fn the_mixed_variant_still_separates_guessing_inventing_and_dropping() {
    let rendered = file_summary::render(FILE_SUMMARY_PROMPT_MIXED).expect("renders");
    let signal: Vec<String> = FILE_SUMMARY_FILE_LIST
        .lines()
        .map(|l| l.trim().to_lowercase())
        .collect();

    // A good model: every real file, no noise file.
    let all_of_them = json!(
        signal
            .iter()
            .map(|p| json!({ "path": p, "desc": "reads the file and describes it" }))
            .collect::<Vec<_>>()
    );
    // A hallucinator: everything above PLUS every noise file.
    let mut with_noise = all_of_them.as_array().unwrap().clone();
    for noise in [
        "/fake/path/nonexistent_file.txt",
        "/totally/made/up/directory/garbage.log",
        "/random/asdfghjkl/qwertyuiop.zxc",
        "/spam/buy_now/click_here.exe",
        "/irrelevant/crypto_price_predictions.md",
        "/hallucinated/alien_landing_report.pdf",
    ] {
        with_noise.push(json!({ "path": noise, "desc": "invented" }));
    }
    // A lazy model: four of the seventeen.
    let few = json!(all_of_them.as_array().unwrap()[..4].to_vec());

    let (good, good_reason) = validate_mixed_file_summary(&all_of_them, &rendered);
    let (hallucinating, hall_reason) = validate_mixed_file_summary(&json!(with_noise), &rendered);
    let (lazy, lazy_reason) = validate_mixed_file_summary(&few, &rendered);

    assert_eq!(
        (good, good_reason.as_str()),
        (100, ""),
        "a clean answer scores 100"
    );
    assert_eq!(
        (hallucinating, hall_reason.as_str()),
        (86, "included 6/6 noise files"),
        "inventing the noise set costs precision: recall is perfect and the \
         score still falls"
    );
    assert_eq!(
        (lazy, lazy_reason.as_str()),
        (61, "missed 13/17 real files"),
        "dropping signal files costs recall: precision is perfect and the score \
         still falls"
    );
    assert!(
        hallucinating < good && lazy < good,
        "{good}/{good_reason} must beat hallucinating {hallucinating}/{hall_reason} \
         and lazy {lazy}/{lazy_reason}"
    );
    assert!(
        lazy < hallucinating,
        "answering 4 of 17 real files ({lazy}) must score below answering all 17 \
         plus the noise ({hallucinating})"
    );
}
