//! Failure classification for the eval loop: WHY a result failed.
//!
//! Ported from `eval/failures.py::_classify_failure`. The category decides
//! what happens NEXT, so a mislabel is not cosmetic:
//!
//! - INFRA (server 5xx, unreachable, missing model) and TIMEOUT count toward
//!   abandoning a model whose SERVER is failing -- those zeros are not quality
//!   results. This split is why qwen3.6-35b's 34 "HTTP 503" responses are an
//!   outage and not 34 formatting failures.
//! - REASONING (reasoning_content present, content empty) triggers a retry
//!   with MORE room: the model never stopped thinking, and reasoning scales
//!   with the TASK, not the budget. Checked BEFORE any JSON/FORMAT branch --
//!   on a JSON task an endless thinker reads as "no JSON brackets", which used
//!   to be answered by prompt rewrites instead of budget.
//! - PARSE marks a parse failure in the signal store, distinct from a model
//!   that answered wrong.

use serde_json::Value;

pub const FAIL_INFRA: &str = "INFRA";
pub const FAIL_TIMEOUT: &str = "TIMEOUT";
pub const FAIL_PARSE: &str = "PARSE";
pub const FAIL_FORMAT: &str = "FORMAT";
pub const FAIL_CONTENT: &str = "CONTENT";
pub const FAIL_REASONING: &str = "REASONING";
/// No failure. Python uses None; Rust uses an empty category string.
pub const FAIL_NONE: &str = "";

/// Retry budget after a REASONING overrun: MORE room, bounded. Lives next to
/// the classifier because keeping the diagnosis and its remedy apart is how
/// they came to disagree once already.
pub const REASONING_RETRY_MULTIPLIER: f64 = 2.0;
pub const REASONING_RETRY_MAX_TOKENS: u32 = 64_000;

pub fn reasoning_retry_budget(base_budget: u32) -> u32 {
    ((base_budget as f64 * REASONING_RETRY_MULTIPLIER) as u32).min(REASONING_RETRY_MAX_TOKENS)
}

#[derive(Debug, Clone, PartialEq)]
pub struct Diagnosis {
    pub category: &'static str,
    pub reason: String,
    pub evidence: String,
}

impl Diagnosis {
    fn none() -> Self {
        Self {
            category: FAIL_NONE,
            reason: String::new(),
            evidence: String::new(),
        }
    }
}

/// HTTP 5xx | server_overloaded | inference capacity | service unavailable,
/// case-insensitive like the Python regex.
fn is_server_error(error: &str) -> bool {
    let lower = error.to_lowercase();
    lower.contains("http 5")
        || lower.contains("server_overloaded")
        || lower.contains("inference capacity")
        || lower.contains("service unavailable")
}

fn transport_failure(error: &str) -> Option<Diagnosis> {
    if error.contains("Model not found") {
        return Some(Diagnosis {
            category: FAIL_INFRA,
            reason: error.to_string(),
            evidence: "Model not loaded or wrong identifier".to_string(),
        });
    }
    if error.contains("Connection") {
        return Some(Diagnosis {
            category: FAIL_INFRA,
            reason: error.to_string(),
            evidence: "Server unreachable".to_string(),
        });
    }
    if is_server_error(error) {
        return Some(Diagnosis {
            category: FAIL_INFRA,
            reason: error.to_string(),
            evidence: "Server returned 5xx; the model never got to answer".to_string(),
        });
    }
    if error.contains("Timeout") || error.to_lowercase().contains("timed out") {
        return Some(Diagnosis {
            category: FAIL_TIMEOUT,
            reason: error.to_string(),
            evidence: "Model did not respond within the task timeout".to_string(),
        });
    }
    None
}

/// Classify why this attempt failed. Branch order mirrors the Python original
/// exactly, because the branches are ordered by "which mislabel hides the
/// truth best".
pub fn classify_failure(
    error: Option<&str>,
    content: &str,
    reasoning: &str,
    finish_reason: &str,
    parsed: Option<&Value>,
    score: u8,
    parse_json: bool,
) -> Diagnosis {
    let _ = (parsed, finish_reason);
    if score >= 90 {
        return Diagnosis::none();
    }

    // An empty error string matches nothing, exactly like Python's "" checks.
    let error = error.unwrap_or("");
    if let Some(d) = transport_failure(error) {
        return d;
    }

    // Before the parse branch: a reasoning model that never stopped returns
    // empty content, and on a JSON task that reads as "no JSON brackets".
    if content.is_empty() && !reasoning.is_empty() {
        return Diagnosis {
            category: FAIL_REASONING,
            reason: "Reasoned past the token budget".to_string(),
            evidence: format!(
                "{} chars of reasoning_content, empty content, finish_reason={}. \
                 Not a prompt-following failure: the model never stopped thinking, \
                 so raise max_tokens for this task.",
                reasoning.len(),
                finish_reason_or_unknown(finish_reason),
            ),
        };
    }

    if parse_json {
        // The Rust task set carries no parse_json tasks yet; when one arrives,
        // port the FORMAT/PARSE/prose-before-JSON branches from failures.py
        // here rather than letting them fall through to CONTENT.
        return Diagnosis {
            category: FAIL_CONTENT,
            reason: "parse_json task scored below threshold".to_string(),
            evidence: String::new(),
        };
    }

    if content.is_empty() {
        return Diagnosis {
            category: FAIL_FORMAT,
            reason: "Empty content".to_string(),
            evidence: "Model returned empty response".to_string(),
        };
    }

    let reasoning_markers = ["Let me", "I'll", "Wait,", "Actually,", "Here's my", "Thinking"];
    let head: String = content.chars().take(200).collect();
    if reasoning_markers.iter().any(|m| head.contains(m)) && content.chars().count() > 200 {
        return Diagnosis {
            category: FAIL_FORMAT,
            reason: "Output starts with reasoning instead of a direct answer".to_string(),
            evidence: format!(
                "Model output {} chars starting with reasoning instead of a direct answer",
                content.chars().count()
            ),
        };
    }

    Diagnosis {
        category: FAIL_CONTENT,
        reason: "Output failed validation".to_string(),
        evidence: format!(
            "Output was {} chars but failed validation",
            content.chars().count()
        ),
    }
}

fn finish_reason_or_unknown(finish_reason: &str) -> &str {
    if finish_reason.is_empty() {
        "unknown"
    } else {
        finish_reason
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn diag(error: Option<&str>, content: &str, reasoning: &str, score: u8) -> Diagnosis {
        classify_failure(error, content, reasoning, "stop", None, score, false)
    }

    #[test]
    fn infra_timeout_and_reasoning_are_distinguished() {
        assert_eq!(diag(Some("Connection failed - is server running?"), "", "", 0).category, FAIL_INFRA);
        assert_eq!(
            diag(Some("HTTP 503: Server is at inference capacity"), "", "", 0).category,
            FAIL_INFRA
        );
        assert_eq!(diag(Some("Timeout"), "", "", 0).category, FAIL_TIMEOUT);
        assert_eq!(
            diag(None, "", "thinking ".repeat(300).trim(), 0).category,
            FAIL_REASONING
        );
    }

    #[test]
    fn reasoning_beats_format_on_empty_content_with_thinking() {
        // The mislabel one level down: empty content + reasoning must be
        // REASONING even though content is also (trivially) "empty".
        let d = diag(None, "", "a very long chain of thought indeed", 0);
        assert_eq!(d.category, FAIL_REASONING);
        assert!(d.evidence.contains("never stopped thinking"));
    }

    #[test]
    fn empty_and_prosey_outputs_are_format_not_content() {
        assert_eq!(diag(None, "", "", 0).category, FAIL_FORMAT);
        let prosey = format!("Let me think about this. {}", "word ".repeat(60));
        assert_eq!(diag(None, &prosey, "", 0).category, FAIL_FORMAT);
    }

    #[test]
    fn real_output_that_fails_validation_is_content() {
        assert_eq!(diag(None, "a short but wrong answer", "", 0).category, FAIL_CONTENT);
    }

    #[test]
    fn a_passing_score_is_never_a_failure_even_with_an_error_string() {
        assert_eq!(diag(Some("Timeout"), "answer", "", 95).category, FAIL_NONE);
    }

    #[test]
    fn retry_budget_grows_but_is_bounded() {
        assert_eq!(reasoning_retry_budget(32_000), 64_000);
        assert_eq!(reasoning_retry_budget(10_000), 20_000);
        assert_eq!(reasoning_retry_budget(40_000), 64_000);
    }
}
