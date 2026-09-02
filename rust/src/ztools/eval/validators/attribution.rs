//! Verification that summary bullet points faithfully attribute claims to their actual authors.
//!
//! Port of `lib/validators/attribution.py`.

use regex::Regex;
use serde_json::Value;
use std::collections::{HashMap, HashSet};
use std::sync::LazyLock;

pub const CLAIM_OVERLAP_THRESHOLD: f64 = 0.25;
pub const NO_TAGS_SCORE: i64 = 0;
pub const ATTRIBUTION_POOR_RATIO: f64 = 0.5;

static SOURCE_LINE_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\[@([A-Za-z][\w]*)\s*\|\s*([^\]]+)\]\s*:?\s*(.*)").unwrap());

static BULLET_TAG_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\(@([A-Za-z][\w]*)\s*\|\s*([^)]+)\)[\s.,;:!)\]]*$").unwrap());

static WORD_RE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"[A-Za-z][A-Za-z0-9'-]+").unwrap());

static STOPWORDS: LazyLock<HashSet<&'static str>> = LazyLock::new(|| {
    [
        "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "have", "in", "is",
        "it", "its", "of", "on", "or", "that", "the", "to", "was", "were", "will", "with", "about",
        "after", "all", "also", "been", "more", "new", "now", "other", "over", "than", "their",
        "there", "these", "they", "this", "those", "we", "you", "your", "our", "but", "not", "can",
        "into", "out", "up",
    ]
    .into_iter()
    .collect()
});

pub fn content_words(text: &str) -> HashSet<String> {
    WORD_RE
        .find_iter(&text.to_lowercase())
        .map(|m| m.as_str())
        .filter(|w| !STOPWORDS.contains(w) && w.len() > 2)
        .map(|w| w.to_string())
        .collect()
}

pub fn source_lines_by_author(source_text: &str) -> HashMap<(String, String), String> {
    let mut lines = HashMap::new();
    for raw in source_text.lines() {
        if let Some(caps) = SOURCE_LINE_RE.captures(raw) {
            let handle = caps.get(1).unwrap().as_str().to_lowercase();
            let stamp = caps.get(2).unwrap().as_str().trim().to_string();
            let content = caps
                .get(3)
                .map(|m| m.as_str().to_string())
                .unwrap_or_default();
            lines.insert((handle, stamp), content);
        }
    }
    lines
}

pub fn attribution_faithfulness(text: &str, source_text: &str) -> (usize, usize, Vec<String>) {
    let by_author = source_lines_by_author(source_text);
    let mut faithful = 0;
    let mut total = 0;
    let mut reasons = Vec::new();

    for raw in text.lines() {
        let line = raw.trim_end();
        let trimmed_start = line.trim_start();
        if !trimmed_start.starts_with('-') && !trimmed_start.starts_with('*') {
            continue;
        }
        let tag = match BULLET_TAG_RE.captures(line) {
            Some(c) => c,
            None => continue,
        };
        total += 1;
        let handle = tag.get(1).unwrap().as_str().to_lowercase();
        let stamp = tag.get(2).unwrap().as_str().trim().to_string();

        let said = match by_author.get(&(handle.clone(), stamp.clone())) {
            Some(s) => s,
            None => {
                if by_author.keys().any(|(h, _)| *h == handle) {
                    reasons.push(format!("@{} did not post at {}", handle, stamp));
                } else {
                    reasons.push(format!("@{} is not in the source", handle));
                }
                continue;
            }
        };

        let tag_match = tag.get(0).unwrap();
        let claim_text = &line[..tag_match.start()];
        let claim = content_words(claim_text);
        if claim.is_empty() {
            reasons.push(format!("@{} bullet has no content", handle));
            continue;
        }

        let said_words = content_words(said);
        let overlap = claim.intersection(&said_words).count() as f64 / claim.len() as f64;
        if overlap >= CLAIM_OVERLAP_THRESHOLD {
            faithful += 1;
        } else {
            reasons.push(format!(
                "@{}'s bullet does not match what they posted",
                handle
            ));
        }
    }

    (faithful, total, reasons)
}

pub fn validate_attribution(data: &Value, source_text: &str) -> (i64, String) {
    let text = match data {
        Value::String(s) => s.clone(),
        other => other.to_string(),
    };
    if text.trim().is_empty() {
        return (0, "empty response".to_string());
    }
    if source_text.is_empty() {
        return (0, "no source to check attribution against".to_string());
    }

    let (faithful, total, reasons) = attribution_faithfulness(&text, source_text);
    if total == 0 {
        return (
            NO_TAGS_SCORE,
            "no attributed bullets (every bullet must end with (@handle | time))".to_string(),
        );
    }

    let score = (100.0 * faithful as f64 / total as f64).round() as i64;
    if faithful == total {
        return (score, String::new());
    }

    let mut unique_reasons = Vec::new();
    let mut seen = HashSet::new();
    for r in reasons {
        if seen.insert(r.clone()) {
            unique_reasons.push(r);
        }
    }
    let detail_full = unique_reasons.join("; ");
    let detail = if detail_full.len() > 200 {
        &detail_full[..200]
    } else {
        &detail_full
    };
    let severity = if (faithful as f64 / total as f64) < ATTRIBUTION_POOR_RATIO {
        "misattributed"
    } else {
        "attribution slips"
    };

    (
        score,
        format!("{} {}/{}: {}", severity, total - faithful, total, detail),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_validate_attribution_faithful_bullets() {
        let source = "[@TechCrunch | Aug 10 14:30]: OpenAI releases new model weights\n[@Reuters | Aug 10 15:00]: Markets rally on tech earnings";
        let summary = json!("- OpenAI released weights for their latest model (@TechCrunch | Aug 10 14:30)\n- Tech earnings sparked market rallies (@Reuters | Aug 10 15:00).");
        let (score, reason) = validate_attribution(&summary, source);
        assert_eq!(score, 100);
        assert!(reason.is_empty());
    }

    #[test]
    fn test_validate_attribution_catches_swapped_author() {
        let source = "[@TechCrunch | Aug 10 14:30]: OpenAI releases new model weights\n[@Reuters | Aug 10 15:00]: Markets rally on tech earnings";
        let summary = json!("- Tech earnings sparked market rallies (@TechCrunch | Aug 10 14:30)\n- OpenAI released weights (@Reuters | Aug 10 15:00)");
        let (score, reason) = validate_attribution(&summary, source);
        assert_eq!(score, 0);
        assert!(reason.contains("misattributed 2/2"), "got: {reason}");
    }

    #[test]
    fn source_lines_without_tags_are_not_indexed() {
        let map = source_lines_by_author("a plain line with no tag\n[@alice | Jan 5]: hello world");
        assert_eq!(map.len(), 1);
        assert_eq!(
            map.get(&("alice".to_string(), "Jan 5".to_string())),
            Some(&"hello world".to_string())
        );
    }

    #[test]
    fn faithfulness_skips_unbulleted_untagged_and_empty_claims() {
        let source = "[@TechCrunch | Aug 10 14:30]: OpenAI releases new model weights";
        let text = "plain prose line\n- no attribution tag here\n- (@TechCrunch | Aug 10 14:30)\n- OpenAI releases new model weights (@TechCrunch | Aug 10 14:30)";
        let (faithful, total, reasons) = attribution_faithfulness(text, source);
        // Empty-claim tagged bullets still count toward the total.
        assert_eq!((faithful, total), (1, 2));
        assert!(
            reasons
                .iter()
                .any(|r| r.contains("@techcrunch bullet has no content")),
            "got: {reasons:?}"
        );
    }

    #[test]
    fn unknown_authors_and_wrong_timestamps_report_distinct_reasons() {
        let source = "[@TechCrunch | Aug 10 14:30]: OpenAI releases new model weights";
        let text = "- OpenAI released weights (@GhostWriter | Aug 10 14:30)\n- OpenAI released weights (@TechCrunch | Aug 11 09:00)";
        let (_, total, reasons) = attribution_faithfulness(text, source);
        assert_eq!(total, 2);
        assert!(
            reasons
                .iter()
                .any(|r| r.contains("@ghostwriter is not in the source")),
            "got: {reasons:?}"
        );
        assert!(
            reasons
                .iter()
                .any(|r| r.contains("@techcrunch did not post at Aug 11 09:00")),
            "got: {reasons:?}"
        );
    }

    #[test]
    fn validate_attribution_guard_branches() {
        let source = "[@alice | t]: some words here";

        assert_eq!(
            validate_attribution(&json!("   "), source),
            (0, "empty response".to_string())
        );
        assert_eq!(
            validate_attribution(&json!("some text"), ""),
            (0, "no source to check attribution against".to_string())
        );

        let (score, reason) = validate_attribution(&json!("- bullet one\n- bullet two"), source);
        assert_eq!(score, NO_TAGS_SCORE);
        assert!(
            reason.contains("no attributed bullets"),
            "untagged bullets must be reported: {reason}"
        );

        // Non-string payloads render through Display and find no bullets.
        let (score, reason) = validate_attribution(&json!(["not a bullet"]), source);
        assert_eq!(score, NO_TAGS_SCORE);
        assert!(reason.contains("no attributed bullets"), "got: {reason}");
    }

    #[test]
    fn half_faithful_reports_slips_not_misattribution() {
        let source = "[@TechCrunch | Aug 10 14:30]: OpenAI releases new model weights";
        let text = "- OpenAI releases new model weights (@TechCrunch | Aug 10 14:30)\n- OpenAI released weights (@TechCrunch | Aug 11 09:00)";
        let (score, reason) = validate_attribution(&json!(text), source);
        assert_eq!(score, 50);
        assert!(
            reason.starts_with("attribution slips 1/2: "),
            "got: {reason}"
        );
        assert!(
            reason.contains("did not post at Aug 11 09:00"),
            "got: {reason}"
        );
    }

    #[test]
    fn long_unique_failure_details_are_truncated_to_two_hundred_chars() {
        let source = "[@alice | Jan 01]: note one";
        let mut text = String::new();
        for i in 0..6 {
            text.push_str(&format!(
                "- filler claim number {i} (@alice | TIMESTAMP-{i:06}-PADDINGTEXT)\n"
            ));
        }
        let (score, reason) = validate_attribution(&json!(text), source);
        assert_eq!(score, 0);
        let expected_len = "misattributed 6/6: ".len() + 200;
        assert_eq!(
            reason.len(),
            expected_len,
            "detail must be cut at exactly 200 chars: len={}",
            reason.len()
        );
    }
}
