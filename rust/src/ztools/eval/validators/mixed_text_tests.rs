//! Pinned against the Python validators' verdicts on the SAME shared prompts
//! (`eval::prompts`), computed on their last run, 2026-09-13. Every expected
//! tuple below is a Python output, not a Rust output copied back.
//!
//! THE 2026-10-08 RE-PIN, AND WHAT IT DELIBERATELY DID NOT MOVE. The
//! file-summary rows became repo-relative and the renderer started reading them
//! through a checkout seam (`file_summary_content::live_root`), which forced
//! `extract_file_paths` to stop recognising a row by "line starts with `/`" and
//! start recognising it by SHAPE. Every tuple below was re-derived and re-run
//! for that change, and every one of them came out IDENTICAL — which is a
//! property worth writing down, because it is not obvious and it is what makes
//! the loosened predicate safe:
//!
//!   * the row COUNT is unchanged (17 signal, 6 noise — pinned by
//!     `the_rendered_prompt_yields_seventeen_signal_rows_and_six_noise_rows`),
//!     because the content block's `| ` prefix keeps every excerpt line out of
//!     the row shape, and
//!   * `prefix_overlap` matches a row against an answer by containment in EITHER
//!     direction, so an answer spelled `/Users/…/README.md` still resolves
//!     against the row `README.md` exactly as it resolved against the absolute
//!     row before.
//!
//! What the change PREVENTS is also pinned, because it is the reason the
//! predicate had to move at all: had the predicate stayed `starts_with('/')`, the
//! relative rows would have read as ZERO signal paths, `recall` would have hit
//! its `signal_paths.is_empty()` arm of 1.0, and every model would have scored
//! full marks for describing nothing. Measured: the one-real-plus-one-noise arm
//! below scores 75 under that defect and 27 under the shape predicate — a 48
//! point inflation, silent, on every model in the table.

use super::*;
use crate::ztools::eval::prompts::file_summary;
use crate::ztools::eval::prompts::{
    FALSEHOOD_PHRASES, FILE_SUMMARY_FILE_LIST, FILE_SUMMARY_PROMPT, FILE_SUMMARY_PROMPT_MIXED,
    KEY_FACTS, RENAME_PROMPT_MIXED, TWITTER_PROMPT, TWITTER_PROMPT_MIXED,
};
use serde_json::json;

/// A model's answer spelling: an absolute path into the repo it was shown.
///
/// The rows in the prompt are repo-relative now, so this is a spelling the model
/// produces and the prompt no longer contains — which is the point.
/// `prefix_overlap` resolves it to its row, and these tests pin that it must.
// path-ok: a model echo's shape, not this repo's path.
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

/// THE COUNT, read out of the RENDERED prompt rather than out of the constant.
///
/// The signal set is read by SHAPE (see `extract_file_paths`), so the two numbers
/// that make the mixed variant mean anything are properties of the rendered text:
/// seventeen rows above the noise marker, six below. Both directions are pinned
/// because both fail silently. A predicate that read sixteen signal rows would
/// quietly inflate every model's recall by a row it was never asked about; one
/// that read eighteen would depress it by the same. A row read out of the NOISE
/// half — an excerpt line that escaped its `| ` prefix — costs precision instead,
/// and a signal row read as noise does both.
///
/// The plain prompt is pinned too, because it is the same seventeen rows and it
/// must not gain a single one: its content block is the same block, and a line in
/// it that looked like a row would move a path between the halves of a task the
/// plain variant does not even have.
#[test]
fn the_rendered_prompt_yields_seventeen_signal_rows_and_six_noise_rows() {
    let rendered_mixed = file_summary::render(FILE_SUMMARY_PROMPT_MIXED).expect("renders");
    let (signal_part, noise_part) = split_signal_noise(&rendered_mixed);
    assert_eq!(
        extract_file_paths(signal_part).len(),
        17,
        "the mixed prompt's signal half must list exactly the seventeen rows, read by shape: \
         {:?}",
        extract_file_paths(signal_part)
    );
    assert_eq!(
        extract_file_paths(noise_part).len(),
        6,
        "the noise half must be exactly the six decoys"
    );
    // Every row the prompt lists is one of them: a row that the shape predicate
    // reads as something else is a row the scorer cannot score.
    let rows: Vec<String> = extract_file_paths(signal_part);
    for row in FILE_SUMMARY_FILE_LIST.lines() {
        assert!(
            rows.iter().any(|r| r == row),
            "{row} is listed but not read as a signal row"
        );
    }
    let rendered_plain = file_summary::render(FILE_SUMMARY_PROMPT).expect("renders");
    assert_eq!(
        extract_file_paths(&rendered_plain).len(),
        17,
        "the plain prompt carries the same seventeen rows and no noise block at all"
    );
}

/// The mixed variant's prompt is no longer a constant: it now carries seventeen
/// files' excerpts above its noise block, and those rows are REPO-RELATIVE and
/// read from the checkout `file_summary::live_root` resolves. An excerpt line
/// beginning `/` — a comment, a TOML value, a path in prose — read as a row would
/// be counted as a file the model was never asked about and depress recall for
/// every model equally; the fences and every excerpt line are `| `-prefixed for
/// exactly that reason, and `the_rendered_prompt_yields_seventeen_signal_rows_and_
/// six_noise_rows` is the assertion that they still are.
///
/// The three pins below are the Python verdicts on the same answers, and the
/// 2026-10-08 re-pin left them IDENTICAL: the row count did not move (the count
/// test above), and `prefix_overlap` resolves an absolutely-spelled answer
/// against a relative row by containment in either direction. If either of those
/// stops being true they move with it, which is why they are pinned here rather
/// than trusted.
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
///
/// RE-PINNED 2026-10-08 (see the module header): the three tuples are UNCHANGED,
/// because the seventeen rows are still read as seventeen rows and the answer's
/// absolute spelling still resolves against them. The ordering pins below are
/// therefore re-derived rather than rebuilt — and they are the reason this test
/// exists at all, so they are worth stating in the failure direction: under the
/// defect the re-pin was defending against, `recall` collapses to 1.0, the lazy
/// arm's 61 becomes 100, and it TIES the good answer. A lazy model that drops
/// thirteen of seventeen files must lose to one that answers all of them, and the
/// only thing that keeps it losing is that the signal set is read by shape.
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
