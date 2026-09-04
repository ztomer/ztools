//! Rubric validators for the taxes_* tasks without a `grounding` block.
//!
//! Ported from `lib/validators/taxes_validator.py`, which itself mirrors
//! `_score_output` from the source Taxes repo. Score is 0-100 from three
//! weighted components: grounding (0-40), no_leak (0-30), substance (0-30).
//! The snapshot's `rubric` block carries `expected_signals` and
//! `gt_forbidden`, so these validators are pure JSON consumers.

use std::path::{Path, PathBuf};

use regex::Regex;
use serde_json::Value;

/// Read the snapshot's rubric block by task short-name.
///
/// A missing rubric silently scores every output against nothing, which reads
/// as a passing grade -- the Python original warns once per task for exactly
/// that reason; here an absent rubric yields an all-empty one.
fn load_rubric(task_name: &str) -> Value {
    let manifest = env!("CARGO_MANIFEST_DIR");
    let candidate = Path::new(manifest)
        .parent()
        .map(|p| {
            p.join("eval_tasks/data/taxes")
                .join(format!("taxes_{task_name}.sanitized.json"))
        })
        .unwrap_or_else(|| {
            PathBuf::from(format!(
                "eval_tasks/data/taxes/taxes_{task_name}.sanitized.json"
            ))
        });

    if let Ok(content) = std::fs::read_to_string(&candidate) {
        if let Ok(val) = serde_json::from_str::<Value>(&content) {
            return val.get("rubric").cloned().unwrap_or(Value::Null);
        }
    }
    Value::Null
}

/// (score 0-40, hits). Signals are matched case-insensitively as substrings.
/// An empty expected list scores full marks -- nothing was promised, so
/// nothing can be missing.
fn grounding_score(output: &str, expected_signals: &[String]) -> (i64, usize) {
    if expected_signals.is_empty() {
        return (40, 0);
    }
    let out_lower = output.to_lowercase();
    let hits = expected_signals
        .iter()
        .filter(|sig| out_lower.contains(&sig.to_lowercase()))
        .count();
    let score = match hits {
        0..=1 => (hits as i64) * 8,
        2 => 16,
        3 => 24,
        4 => 32,
        _ => 40,
    };
    (score, hits)
}

/// 30 unless the output verbatim-quotes a GT-flavored term. Reaching 0 means
/// the eval surface leaked -- a contract failure, not a quality issue, but it
/// still drops the score because the output is dangerous to ship.
fn no_leak_score(output: &str, gt_forbidden: &[String]) -> i64 {
    if output.is_empty() {
        return 30;
    }
    if gt_forbidden
        .iter()
        .any(|term| output.contains(term.as_str()))
    {
        return 0;
    }
    30
}

/// 0-30 depth heuristics: -10 for <600 chars, -10 for no specific dollar
/// amounts cited, -10 for list-form-only. Floor at 0.
fn substance_score(output: &str) -> i64 {
    if output.is_empty() {
        return 0;
    }
    let mut score = 30;
    let chars = output.trim().chars().count();
    if chars < 600 {
        score -= 10;
    }
    let amounts = Regex::new(r"\$([\d,]+(?:\.\d{2})?)").expect("static regex");
    if amounts.find(output).is_none() {
        score -= 10;
    }
    let bullets = Regex::new(r"(?m)^\s*(?:\d+[.)]\s+|[-•*]\s+)").expect("static regex");
    let bullet_count = bullets.find_iter(output).count();
    let non_bullet = Regex::new(r"(?m)^[\s\d.\-•*]+")
        .expect("static regex")
        .replace_all(output, "")
        .split_whitespace()
        .collect::<Vec<&str>>()
        .join(" ");
    if bullet_count >= 3 && non_bullet.chars().count() < bullet_count * 60 {
        score -= 10;
    }
    score.max(0)
}

/// The common rubric: grounding (40) + no_leak (30) + substance (30).
pub fn validate_taxes_task(output: &str, task_name: &str) -> (i64, String) {
    let rubric = load_rubric(task_name);
    let expected: Vec<String> = rubric
        .get("expected_signals")
        .and_then(|v| v.as_array())
        .map(|a| {
            a.iter()
                .filter_map(|s| s.as_str().map(String::from))
                .collect()
        })
        .unwrap_or_default();
    let forbidden: Vec<String> = rubric
        .get("gt_forbidden")
        .and_then(|v| v.as_array())
        .map(|a| {
            a.iter()
                .filter_map(|s| s.as_str().map(String::from))
                .collect()
        })
        .unwrap_or_default();

    let (g_score, g_hits) = grounding_score(output, &expected);
    let l_score = no_leak_score(output, &forbidden);
    let s_score = substance_score(output);
    let total = g_score + l_score + s_score;

    (
        total,
        format!(
            "grounding={g_score}/40 ({g_hits} signals)  no_leak={l_score}/30  substance={s_score}/30"
        ),
    )
}

pub fn validate_taxes_anomalies(output: &Value) -> (i64, String) {
    validate_taxes_task(&value_to_text(output), "anomalies")
}

/// Same rubric + schema check: must be valid JSON with a `risk_items` list.
/// A schema failure HALVES the total.
pub fn validate_taxes_audit_readiness(output: &Value) -> (i64, String) {
    let raw = value_to_text(output);
    let (mut score, mut reason) = validate_taxes_task(&raw, "audit_readiness");
    match serde_json::from_str::<Value>(&raw) {
        Ok(obj) if obj.get("risk_items").map(|r| r.is_array()).unwrap_or(false) => {
            reason.push_str("  schema=ok");
        }
        Ok(_) => {
            score /= 2;
            reason.push_str("  schema=bad-shape (score halved)");
        }
        Err(_) => {
            score /= 2;
            reason.push_str("  schema=not-json (score halved)");
        }
    }
    (score, reason)
}

/// Same rubric + markdown-section check: all but one of the expected
/// `**N. Section**` headings must be present, else -10 floored at 0.
pub fn validate_taxes_synthesis(output: &Value) -> (i64, String) {
    let raw = value_to_text(output);
    let (mut score, mut reason) = validate_taxes_task(&raw, "synthesis");
    let expected_sections: Vec<String> = load_rubric("synthesis")
        .get("expected_sections")
        .and_then(|v| v.as_array())
        .map(|a| {
            a.iter()
                .filter_map(|s| s.as_str().map(String::from))
                .collect()
        })
        .unwrap_or_default();
    if !expected_sections.is_empty() {
        let hits = expected_sections
            .iter()
            .filter(|s| raw.contains(s.as_str()))
            .count();
        if hits < expected_sections.len() - 1 {
            score = (score - 10).max(0);
            reason.push_str(&format!(
                "  sections={hits}/{} (\u{2212}10)",
                expected_sections.len()
            ));
        } else {
            reason.push_str(&format!("  sections={hits}/{}", expected_sections.len()));
        }
    }
    (score, reason)
}

/// Validators consume RAW TEXT in Python (`str(output or "")`); the Rust side
/// carries the cleaned text as a JSON string value when unparsed.
fn value_to_text(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        other => other.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grounding_scores_follow_the_rubric_ladder() {
        assert_eq!(grounding_score("", &[]), (40, 0));
        let sigs: Vec<String> = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        assert_eq!(
            grounding_score("alpha bravo charlie delta echo foxtrot", &sigs),
            (40, 6)
        );
        assert_eq!(grounding_score("alpha bravo charlie", &sigs), (24, 3));
        assert_eq!(grounding_score("alpha bravo", &sigs), (16, 2));
        assert_eq!(grounding_score("only alpha here", &sigs), (8, 1));
        // Case-insensitive.
        assert_eq!(grounding_score("ONLY ALPHA HERE", &sigs), (8, 1));
    }

    #[test]
    fn leak_zeroes_and_substance_penalties_apply() {
        assert_eq!(no_leak_score("", &["x".to_string()]), 30);
        assert_eq!(no_leak_score("mentions x term", &["x".to_string()]), 0);

        assert_eq!(substance_score(""), 0);
        // Short, no $ amount, not list-only: two penalties.
        assert_eq!(substance_score("short prose"), 10);
        // Long enough with an amount and prose: full marks.
        let long = format!("costs $1,234.50 because {}", "word ".repeat(200));
        assert_eq!(substance_score(&long), 30);
    }

    #[test]
    fn audit_readiness_halves_on_bad_schema() {
        // Valid JSON with risk_items keeps the rubric score; without it the
        // score halves even when the prose is fine.
        let good = serde_json::json!({"risk_items": [], "prose": "x"});
        let (score, reason) = validate_taxes_audit_readiness(&good);
        assert!(reason.contains("schema=ok"), "{reason}");
        assert!(score > 0);

        let bad = serde_json::json!("plain prose, no json at all, costs $5.00");
        let (score_bad, reason_bad) = validate_taxes_audit_readiness(&bad);
        assert!(reason_bad.contains("schema=not-json"), "{reason_bad}");
        // Exactly the rubric total, halved with integer division like Python.
        let (base, _) = validate_taxes_task(&value_to_text(&bad), "audit_readiness");
        assert_eq!(score_bad, base / 2);
    }
}
