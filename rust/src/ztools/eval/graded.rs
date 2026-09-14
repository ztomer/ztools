//! Graded checks: the 0-100 validator verdicts the eval task table scores by.
//!
//! The Python `eval/tasks_core.py::TASKS` table gave every task ONE validator,
//! a `source` (usually the prompt itself, so relevance and attribution can be
//! judged against what the model was actually shown) and optional keyword
//! arguments; the task's score was the validator's verdict. This enum is that
//! contract in Rust: each variant names a ported validator and carries the
//! source and arguments the table passed. The runner treats a task whose
//! checks are all graded as scoring the mean of their verdicts, so a
//! one-check task scores exactly what the validator says.
//!
//! Every validator here lives in [`super::validators`] / [`super::validate`] /
//! [`super::vision`]; this file only routes to them.

use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::validators;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum Graded {
    /// `validate_detailed_json(parsed, source)`.
    DetailedJson { source: String },
    /// `validate_mixed_signal(parsed, source)`.
    MixedSignal { source: String },
    /// `validate_filename(text, source)`.
    Filename { source: String },
    /// `validate_mixed_filename(parsed, source)`.
    MixedFilename { source: String },
    /// `validate_summary(text, source)`.
    Summary { source: String },
    /// `validate_mixed_summary(text, source)`.
    MixedSummary { source: String },
    /// `validate_file_summary(text)`.
    FileSummary,
    /// `validate_mixed_file_summary(text, source)`.
    MixedFileSummary { source: String },
    /// `validate_strict_schema(text, kind)`.
    StrictSchema { kind: String },
    /// `validate_no_contradiction(text, phrase)`.
    NoContradiction { phrase: String },
    /// `validate_no_leak(text)`.
    NoLeak,
    /// `validate_factual_accuracy(text, falsehoods)`.
    FactualAccuracy { falsehoods: Vec<String> },
    /// `validate_factual_coverage(text, key_facts)`.
    FactualCoverage { key_facts: Vec<String> },
    /// `validate_image_description(text, fixtures)`; the fixtures are the
    /// shipped `conf/eval_vision.toml`, the same spec the task's images were
    /// drawn from.
    ImageDescription,
    /// `validate_attribution(text, source)`.
    Attribution { source: String },
    /// `validate_no_fabrication(parsed, source, lures)`.
    NoFabrication { source: String, lures: Vec<String> },
    /// `validate_resists_injection(text, source, markers, keywords)`.
    ResistsInjection {
        source: String,
        markers: Vec<String>,
        keywords: Vec<String>,
    },
}

/// The parsed answer when the task asked for JSON, else the raw text as a
/// JSON string — what the Python validators received in each case.
fn value_of(cleaned: &str, parsed: Option<&Value>) -> Value {
    parsed
        .cloned()
        .unwrap_or_else(|| Value::String(cleaned.to_string()))
}

impl Graded {
    /// The validator's `(score, reason)` for one answer.
    #[must_use]
    pub fn score(&self, cleaned: &str, parsed: Option<&Value>) -> (i64, String) {
        match self {
            Self::DetailedJson { source } => {
                validators::validate_detailed_json(&value_of(cleaned, parsed), source)
            }
            Self::MixedSignal { source } => {
                validators::validate_mixed_signal(&value_of(cleaned, parsed), source, None, None)
            }
            Self::Filename { source } => validators::validate_filename(cleaned, source),
            Self::MixedFilename { source } => {
                validators::validate_mixed_filename(&value_of(cleaned, parsed), source)
            }
            Self::Summary { source } => validators::validate_summary(cleaned, source),
            Self::MixedSummary { source } => {
                validators::validate_mixed_summary(&Value::String(cleaned.to_string()), source)
            }
            Self::FileSummary => {
                let (score, reason) = super::validate::validate_file_summary(cleaned);
                (i64::from(score), reason)
            }
            Self::MixedFileSummary { source } => {
                validators::validate_mixed_file_summary(&value_of(cleaned, parsed), source)
            }
            Self::StrictSchema { kind } => validators::validate_strict_schema(cleaned, kind),
            Self::NoContradiction { phrase } => {
                validators::validate_no_contradiction(cleaned, phrase)
            }
            Self::NoLeak => validators::validate_no_leak(cleaned),
            Self::FactualAccuracy { falsehoods } => {
                validators::validate_factual_accuracy(cleaned, falsehoods)
            }
            Self::FactualCoverage { key_facts } => {
                validators::validate_factual_coverage(cleaned, key_facts)
            }
            Self::ImageDescription => match super::vision::shipped_vision_spec() {
                Ok(spec) => super::vision::validate_image_description(cleaned, &spec.fixtures),
                Err(e) => (0, format!("vision fixtures unavailable: {e}")),
            },
            Self::Attribution { source } => {
                validators::validate_attribution(&Value::String(cleaned.to_string()), source)
            }
            Self::NoFabrication { source, lures } => {
                validators::validate_no_fabrication(&value_of(cleaned, parsed), source, lures)
            }
            Self::ResistsInjection {
                source,
                markers,
                keywords,
            } => validators::validate_resists_injection(
                &Value::String(cleaned.to_string()),
                source,
                markers,
                keywords,
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ztools::eval::prompts::{CONTRADICTION_PHRASE, FALSEHOOD_PHRASES};
    use crate::ztools::eval::task_loader::{check_graded_score, run_check, Check};

    fn strs(v: &[&str]) -> Vec<String> {
        v.iter().map(|s| (*s).to_string()).collect()
    }

    #[test]
    fn a_graded_check_scores_the_validator_verdict_and_passes_at_50() {
        let clean = Check::Graded(Graded::NoContradiction {
            phrase: CONTRADICTION_PHRASE.to_string(),
        });
        assert_eq!(
            check_graded_score(&clean, "a calm summary", None),
            Some(100)
        );
        assert!(run_check(&clean, "a calm summary", None));
        let parroted = format!("Breaking: {CONTRADICTION_PHRASE}");
        assert_eq!(check_graded_score(&clean, &parroted, None), Some(0));
        assert!(!run_check(&clean, &parroted, None));
    }

    #[test]
    fn json_variants_prefer_the_parsed_answer_and_fall_back_to_text() {
        let g = Graded::MixedFilename {
            source: "1. scott adams essays\nNOISE\n- spam: buy now click here".to_string(),
        };
        let parsed = serde_json::json!(["scott_adams_essays"]);
        assert_eq!(g.score("ignored", Some(&parsed)).0, 100);
        // Without a parse the raw text is judged as one name.
        assert_eq!(g.score("scott_adams_essays", None).0, 100);
        assert_eq!(g.score("buy_now_click_here", None).0, 0);
    }

    #[test]
    fn factual_accuracy_routes_the_falsehood_list() {
        let g = Graded::FactualAccuracy {
            falsehoods: strs(FALSEHOOD_PHRASES),
        };
        assert_eq!(g.score("nothing false here", None).0, 100);
        assert_eq!(g.score(FALSEHOOD_PHRASES[1], None).0, 67);
    }

    #[test]
    fn image_description_grades_against_the_shipped_fixtures() {
        let g = Graded::ImageDescription;
        assert_eq!(
            g.score("a red circle, a green triangle, a blue square", None)
                .0,
            100
        );
        assert_eq!(g.score("a large brown dog", None).0, 0);
    }

    #[test]
    fn graded_checks_round_trip_through_serde() {
        let check = Check::Graded(Graded::ResistsInjection {
            source: "s".into(),
            markers: strs(&["m"]),
            keywords: strs(&["k"]),
        });
        let json = serde_json::to_string(&check).unwrap();
        assert_eq!(serde_json::from_str::<Check>(&json).unwrap(), check);
    }
}
