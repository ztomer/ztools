//! Faithfulness gates: instruction-leak, strict-schema, and contradiction checks.
//!
//! Port of the faithfulness section of
//! `lib/validators/text_validator_mixed.py`. These grade the live eval tasks
//! `filename_leak`, `weekend_transient_schema`, and `summarize_contradiction`,
//! so a model that echoes the instruction instead of answering, wraps JSON in
//! prose, or parrots a planted falsehood scores 0 here rather than passing on
//! shape alone.

use regex::Regex;
use std::sync::LazyLock;

static LEAK_PATTERNS: LazyLock<Vec<Regex>> = LazyLock::new(|| {
    [
        r"here\s+is\s+(the\s+)?filename",
        r"here's\s+(the\s+)?filename",
        r"the\s+filename\s+is",
        r"filename\s*:",
        r"here\s+is\s+your\s+(file|output)",
    ]
    .iter()
    .map(|p| Regex::new(&format!("(?i){p}")).expect("leak pattern must compile"))
    .collect()
});

/// Nemotron-style leakage: the model echoes the instruction ("Here is the
/// filename: ...") instead of emitting the bare value. Empty text has no
/// leak. Shared with the filename scorer, which fails leaked names at 0.
pub(crate) fn detect_instruction_leak(text: &str) -> Vec<String> {
    if text.is_empty() {
        return Vec::new();
    }
    LEAK_PATTERNS
        .iter()
        .filter_map(|p| p.find(text).map(|m| m.as_str().to_string()))
        .collect()
}

/// Score 0 if the output leaks instruction text; 100 if clean.
#[must_use]
pub fn validate_no_leak(text: &str) -> (i64, String) {
    let leaks = detect_instruction_leak(text);
    leaks.first().map_or_else(
        || (100, String::new()),
        |first| (0, format!("instruction leak: {first}")),
    )
}

/// Verify the RAW output is exactly the contracted shape — no prose preamble,
/// no markdown fences, no trailing commentary. Gates the prose-before-JSON
/// failure mode (qwen35b / qwopus).
#[must_use]
pub fn validate_strict_schema(raw: &str, kind: &str) -> (i64, String) {
    if raw.trim().is_empty() {
        return (0, "empty response".to_string());
    }
    let text = raw.trim();
    if text.contains("```") {
        return (0, "contains code fence (not strict schema)".to_string());
    }
    if kind == "json" {
        let brackets: Vec<usize> = text
            .char_indices()
            .filter(|(_, c)| *c == '{' || *c == '[')
            .map(|(i, _)| i)
            .collect();
        let ends: Vec<usize> = text
            .char_indices()
            .filter(|(_, c)| *c == '}' || *c == ']')
            .map(|(i, _)| i)
            .collect();
        let (Some(first), Some(last)) = (brackets.first(), ends.last()) else {
            return (0, "no JSON object/array found".to_string());
        };
        if last < first {
            return (0, "no JSON object/array found".to_string());
        }
        let before = text[..*first].trim();
        let after = &text[last + 1..];
        let after = after.trim();
        if !before.is_empty() || !after.is_empty() {
            return (
                0,
                format!("prose outside JSON (before='{before}', after='{after}')"),
            );
        }
        return (100, String::new());
    }
    if kind == "filename" {
        if !detect_instruction_leak(text).is_empty() {
            return (0, "instruction leak".to_string());
        }
        if text.chars().count() > 80 {
            return (
                0,
                format!("filename too long ({} chars)", text.chars().count()),
            );
        }
        return (100, String::new());
    }
    // markdown / free-text: accept as-is.
    (100, String::new())
}

/// Faithfulness probe: assert the output does NOT parrot a planted falsehood.
///
/// A verbatim hit fails at once; otherwise the distinctive tokens (4+ chars)
/// get a vote, and two of them in the output is still a parrot.
#[must_use]
pub fn validate_no_contradiction(output: &str, contradiction_phrase: &str) -> (i64, String) {
    if output.is_empty() || contradiction_phrase.is_empty() {
        return (100, String::new());
    }
    let out = output.to_lowercase();
    let phrase = contradiction_phrase.to_lowercase();
    let phrase = phrase.trim();
    if !phrase.is_empty() && out.contains(phrase) {
        return (
            0,
            format!("parrots contradiction: '{contradiction_phrase}'"),
        );
    }
    let cleaned: String = phrase
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == ' ' {
                c
            } else {
                ' '
            }
        })
        .collect();
    let tokens: Vec<&str> = cleaned
        .split_whitespace()
        .filter(|t| t.chars().count() >= 4)
        .collect();
    if !tokens.is_empty() && tokens.iter().filter(|t| out.contains(**t)).count() >= 2 {
        return (
            0,
            format!("parrots contradiction: '{contradiction_phrase}'"),
        );
    }
    (100, String::new())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clean_text_has_no_leak() {
        assert_eq!(
            validate_no_leak("how_to_manage_underperformers"),
            (100, String::new())
        );
    }

    #[test]
    fn instruction_echo_is_a_leak() {
        let (score, reason) = validate_no_leak("Here is the filename: how_to_manage");
        assert_eq!(score, 0);
        assert!(reason.contains("leak"), "{reason}");
    }

    #[test]
    fn bare_filename_prefix_is_a_leak() {
        let (score, _) = validate_no_leak("filename: how_to_manage_underperformers");
        assert_eq!(score, 0);
    }

    #[test]
    fn bare_json_is_strict_schema() {
        assert_eq!(
            validate_strict_schema(r#"{"name": "x"}"#, "json"),
            (100, String::new())
        );
    }

    #[test]
    fn prose_before_json_breaks_strict_schema() {
        let (score, reason) =
            validate_strict_schema(r#"Here is the result: {"name": "x"}"#, "json");
        assert_eq!(score, 0);
        assert!(reason.contains("prose"), "{reason}");
    }

    #[test]
    fn fences_break_strict_schema() {
        let (score, reason) = validate_strict_schema("```json\n{\"name\": \"x\"}\n```", "json");
        assert_eq!(score, 0);
        assert!(reason.contains("fence"), "{reason}");
    }

    #[test]
    fn trailing_prose_breaks_strict_schema() {
        let (score, reason) = validate_strict_schema(r#"{"name": "x"} hope that helps"#, "json");
        assert_eq!(score, 0);
        assert!(reason.contains("prose"), "{reason}");
    }

    #[test]
    fn overlong_filename_breaks_strict_schema() {
        let long = "a".repeat(81);
        let (score, reason) = validate_strict_schema(&long, "filename");
        assert_eq!(score, 0);
        assert!(reason.contains("too long"), "{reason}");
    }

    #[test]
    fn parroted_planted_falsehood_is_a_contradiction() {
        let (score, reason) = validate_no_contradiction(
            "Summary: quantum giraffes of Manitoba won the Stanley Cup.",
            "quantum giraffes of Manitoba won the Stanley Cup",
        );
        assert_eq!(score, 0);
        assert!(reason.contains("contradiction"), "{reason}");
    }

    #[test]
    fn paraphrased_parroting_still_counts_through_shared_tokens() {
        let (score, _) = validate_no_contradiction(
            "The Manitoba delegation discussed giraffes at length.",
            "quantum giraffes of Manitoba won the Stanley Cup",
        );
        assert_eq!(score, 0);
    }

    #[test]
    fn unrelated_summary_has_no_contradiction() {
        let (score, _) = validate_no_contradiction(
            "Summary: OpenAI announced GPT-5. CN Tower reopened.",
            "quantum giraffes of Manitoba won the Stanley Cup",
        );
        assert_eq!(score, 100);
    }
}
