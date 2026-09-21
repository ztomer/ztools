//! Filename-quality scorer with leak, shape, and input-relevance gates.
//!
//! Port of `lib/validators/text_validator.py::validate_filename` and
//! `filename_relevance`. Scores length, characters, format, and specificity
//! out of 100, then gates: an instruction leak fails at 0, and a name that
//! shares nothing with the input caps at 40 no matter how well formed.

use crate::units::count;
use regex::Regex;
use std::collections::HashSet;
use std::sync::LazyLock;

use super::faithfulness::detect_instruction_leak;

pub const FILENAME_IRRELEVANT_MAX_SCORE: i64 = 40;
const MAX_SCORE: i64 = 100;
const FILENAME_LENGTH_MIN: usize = 4;
const FILENAME_LENGTH_MAX: usize = 59;
const FILENAME_LENGTH_SCORE: i64 = 30;
const FILENAME_CHARS_SCORE: i64 = 20;
const FILENAME_FORMAT_SCORE: i64 = 25;
const MAX_EXPLANATORY_FILENAME_LEN: usize = 70;
const FILENAME_EXPLANATION_PENALTY_SCORE: i64 = 15;
const FILENAME_SPECIFIC_SCORE: i64 = 25;
const MIN_SPECIFIC_FILENAME_LEN: usize = 4;
const FILENAME_RELEVANCE_FLOOR: f64 = 0.2;
const FILENAME_RELEVANCE_GOOD: f64 = 0.6;
const MIN_FILENAME_LINE_LEN: usize = 3;
const MAX_FILENAME_LINE_LEN: usize = 60;
const DEFAULT_CANDIDATE_FALLBACK_LIMIT: usize = 50;
const FILENAME_SEPARATORS: &[char] = &['_', '-', '.'];

const GENERIC_FILENAMES: &[&str] = &[
    "filename.txt",
    "file.txt",
    "text.txt",
    "output.txt",
    "document.txt",
    "note.txt",
    "image.png",
    "screenshot.png",
    "unnamed",
    "file",
    "filename",
    "output",
    "document",
    "image",
    "photo",
    "screenshot",
];

const FILENAME_STOPWORDS: &[&str] = &[
    "the", "a", "an", "is", "are", "was", "were", "of", "to", "in", "on", "for", "and", "or",
    "with", "that", "this", "it", "its", "please", "try", "again", "showing", "show",
];

static WORD3_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"[a-z0-9]{3,}").expect("word regex is static"));
static WORD_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"[a-z0-9]+").expect("token regex is static"));

fn is_valid_filename_char(c: char) -> bool {
    c.is_alphanumeric() || c == '_' || c == '-' || c == '.'
}

fn has_filename_format(filename: &str) -> bool {
    filename.chars().any(|c| FILENAME_SEPARATORS.contains(&c))
}

/// Strip markdown backtick wrappers: one leading/trailing fence pair, then
/// any remaining backtick edging plus surrounding whitespace.
fn strip_backtick_value(value: &str) -> String {
    let text = value.trim();
    let text = text.strip_prefix("```").unwrap_or(text);
    let text = text.strip_suffix("```").unwrap_or(text);
    text.trim_matches('`').trim().to_string()
}

/// First non-reasoning line of fitting length, else the stripped head.
/// Reasoning fences and `#` comment lines are never candidates.
fn extract_best_filename_candidate(text: &str) -> String {
    let stripped = text.trim();
    if stripped.is_empty() {
        return String::new();
    }
    for line in stripped.lines().map(str::trim).filter(|l| !l.is_empty()) {
        if line.starts_with("```") || line.starts_with('#') {
            continue;
        }
        let len = line.chars().count();
        if MIN_FILENAME_LINE_LEN < len && len < MAX_FILENAME_LINE_LEN {
            return line.to_string();
        }
    }
    stripped
        .chars()
        .take(DEFAULT_CANDIDATE_FALLBACK_LIMIT)
        .collect()
}

/// Fraction of the input's content words that appear in the filename.
///
/// Returns -1.0 for "cannot assess" — no source, or a source with no usable
/// words after stopword stripping. Callers test `coverage >= 0.0` to tell a
/// real zero apart from the sentinel; collapsing them would mark every
/// sourceless filename irrelevant.
#[must_use]
pub fn filename_relevance(name: &str, source_text: &str) -> f64 {
    if source_text.is_empty() {
        return -1.0;
    }
    let stopwords: HashSet<&str> = FILENAME_STOPWORDS.iter().copied().collect();
    let lowered_source = source_text.to_lowercase();
    let words: HashSet<&str> = WORD3_RE
        .find_iter(&lowered_source)
        .map(|m| m.as_str())
        .filter(|w| !stopwords.contains(w))
        .collect();
    if words.is_empty() {
        return -1.0;
    }
    let lowered_name = name.to_lowercase();
    let name_tokens: HashSet<&str> = WORD_RE
        .find_iter(&lowered_name)
        .map(|m| m.as_str())
        .collect();
    let hits = words
        .iter()
        .filter(|w| name_tokens.contains(**w) || name_tokens.iter().any(|t| t.contains(**w)))
        .count();
    count(hits) / count(words.len())
}

/// Score filenames on length, characters, format, and specificity.
#[must_use]
pub fn validate_filename(data: &str, source_text: &str) -> (i64, String) {
    if data.is_empty() {
        return (0, "empty response".to_string());
    }
    let raw = data.trim();

    let leaks = detect_instruction_leak(raw);
    if let Some(first) = leaks.first() {
        return (0, format!("instruction leak: {first}"));
    }

    let mut clean = strip_backtick_value(raw);
    if clean.chars().count() >= FILENAME_LENGTH_MAX || !clean.chars().all(is_valid_filename_char) {
        clean = extract_best_filename_candidate(raw);
    }

    let mut failures = Vec::new();
    let mut score: i64 = 0;

    if GENERIC_FILENAMES.contains(&clean.to_lowercase().as_str())
        || clean.chars().count() < MIN_SPECIFIC_FILENAME_LEN
    {
        return (0, format!("generic: {clean}"));
    }

    let len = clean.chars().count();
    if FILENAME_LENGTH_MIN < len && len < FILENAME_LENGTH_MAX {
        score += FILENAME_LENGTH_SCORE;
    } else {
        failures.push(format!(
            "length {len} not in {}-{}",
            FILENAME_LENGTH_MIN,
            FILENAME_LENGTH_MAX - 1
        ));
    }

    if clean.chars().all(is_valid_filename_char) {
        score += FILENAME_CHARS_SCORE;
    } else {
        failures.push("invalid chars".to_string());
    }

    if has_filename_format(&clean) || clean.contains('_') || clean.contains('-') {
        score += FILENAME_FORMAT_SCORE;
    } else {
        failures.push("no separators/structure".to_string());
    }

    let clean_lower = clean.to_lowercase();
    let has_question_parts = clean.contains('?')
        || clean_lower.contains("please")
        || clean_lower.contains("which")
        || clean_lower.contains("what");
    let has_explanation = len > MAX_EXPLANATORY_FILENAME_LEN
        || clean_lower.starts_with("the ")
        || clean_lower.starts_with("this ")
        || clean_lower.starts_with("a ");
    if has_question_parts {
        failures.push("question-like output".to_string());
    } else if has_explanation {
        score += FILENAME_EXPLANATION_PENALTY_SCORE;
        failures.push("wordy".to_string());
    } else {
        score += FILENAME_SPECIFIC_SCORE;
    }

    let coverage = filename_relevance(&clean, source_text);
    if coverage >= 0.0 {
        if coverage < FILENAME_RELEVANCE_FLOOR {
            failures.push(format!(
                "unrelated to input (coverage {:.0}%)",
                coverage * 100.0
            ));
            return (
                MAX_SCORE.min(FILENAME_IRRELEVANT_MAX_SCORE).min(score),
                failures.join("; "),
            );
        }
        if coverage < FILENAME_RELEVANCE_GOOD {
            failures.push(format!("weak input coverage ({:.0}%)", coverage * 100.0));
            score -= 15;
        }
    }

    (MAX_SCORE.min(score.max(0)), failures.join("; "))
}

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn empty_is_empty() {
        assert_eq!(validate_filename("", ""), (0, "empty response".to_string()));
    }

    #[test]
    fn good_filename_scores_full_marks() {
        assert_eq!(validate_filename("my_great_file", "").0, 100);
    }

    #[test]
    fn too_short_is_generic() {
        let (score, msg) = validate_filename("abc", "");
        assert_eq!(score, 0);
        assert!(msg.contains("generic"), "{msg}");
    }

    #[test]
    fn too_long_falls_back_to_a_candidate() {
        let (score, _) = validate_filename(&"a".repeat(60), "");
        assert!(score < 100, "{score}");
    }

    #[test]
    fn dash_and_dot_separators_score_full_marks() {
        assert_eq!(validate_filename("my-cool-file", "").0, 100);
        assert_eq!(validate_filename("my.file.txt", "").0, 100);
    }

    #[test]
    fn invalid_chars_fall_back_and_keep_the_finding() {
        // 37 chars with junk: the fallback keeps the line (it fits), the
        // "this " prefix trips the wordy arm, and the invalid-chars finding
        // survives: 30 + 0 + 0 + 15 = 45.
        let (score, msg) = validate_filename("this is a long thing @#$%^&*() stuff", "");
        assert_eq!(score, 45);
        assert!(msg.contains("invalid chars"), "{msg}");
    }

    #[test]
    fn question_like_output_is_named() {
        for name in [
            "what_file?",
            "please_what_to_call_this",
            "which_file_is_better",
            "what_should_i_name_this",
        ] {
            let (score, msg) = validate_filename(name, "");
            assert!(msg.contains("question"), "{name}: {msg}");
            assert!(score < 100, "{name}: {score}");
        }
    }

    #[test]
    fn wordy_filenames_are_named() {
        for name in [
            "the quick brown fox jumps",
            "the summer party event",
            "this is some event name",
            "a special event name",
        ] {
            let (_, msg) = validate_filename(name, "");
            assert!(msg.contains("wordy"), "{name}: {msg}");
        }
    }

    #[test]
    fn separatorless_names_keep_length_and_chars_points() {
        // 30 + 20 + 0 (no format) + 25 (specific) = 75.
        let (score, msg) = validate_filename("abcdefghij", "");
        assert_eq!(score, 75);
        assert!(msg.contains("no separators"), "{msg}");
    }

    #[test]
    fn backticks_and_fences_are_stripped() {
        assert_eq!(validate_filename("`my_file`", "").0, 100);
        assert_eq!(validate_filename("```my_file```", "").0, 100);
    }

    #[test]
    fn generic_names_score_zero() {
        for name in ["filename.txt", "file", "image.png"] {
            let (score, msg) = validate_filename(name, "");
            assert_eq!(score, 0);
            assert!(msg.contains("generic"), "{name}: {msg}");
        }
    }

    #[test]
    fn length_boundary_reports() {
        // 59 chars: not inside the exclusive (4, 59) window.
        let (score, msg) = validate_filename(&"a".repeat(59), "");
        assert!(msg.contains("length"), "{score}: {msg}");
        assert!(score < 100);
    }

    const RELEVANCE_SOURCE: &str =
        "Scott Adams essays about failure ambition and navigating corporate life";

    #[test]
    fn zero_overlap_is_assessed_not_skipped() {
        assert_exact!(filename_relevance("zzz_qqq_wwww", RELEVANCE_SOURCE), 0.0);
    }

    #[test]
    fn unrelated_filename_is_capped_and_named() {
        let (score, msg) = validate_filename("zzz_qqq_wwww", RELEVANCE_SOURCE);
        assert!(score <= FILENAME_IRRELEVANT_MAX_SCORE, "{score}");
        assert!(msg.contains("unrelated to input"), "{msg}");
    }

    #[test]
    fn relevant_filename_scores_above_the_cap() {
        let (score, _) = validate_filename("scott_adams_essays", RELEVANCE_SOURCE);
        assert!(score > FILENAME_IRRELEVANT_MAX_SCORE, "{score}");
    }

    #[test]
    fn no_source_returns_the_sentinel_not_zero() {
        assert_exact!(filename_relevance("scott_adams_essays", ""), -1.0);
    }

    #[test]
    fn stopword_only_source_returns_the_sentinel() {
        assert_exact!(
            filename_relevance("scott_adams_essays", "a to the of"),
            -1.0
        );
    }

    #[test]
    fn unassessable_source_does_not_cap() {
        let (score, _) = validate_filename("scott_adams_essays", "");
        assert!(score > FILENAME_IRRELEVANT_MAX_SCORE, "{score}");
    }

    #[test]
    fn partial_coverage_is_reported_without_capping() {
        let (score, msg) = validate_filename("scott_adams_essays", RELEVANCE_SOURCE);
        assert!(msg.contains("weak input coverage"), "{msg}");
        assert!(score > 40, "{score}");
    }

    #[test]
    fn good_filename_outscores_leaked_one() {
        let (good, _) = validate_filename("quarterly_revenue_report", "");
        let (bad, _) = validate_filename("Here is the filename: IMG 1234.PNG", "");
        assert!(good > bad, "{good} vs {bad}");
    }
}
