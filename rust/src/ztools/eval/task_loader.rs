//! Eval task loader (`task_loader.rs`).
//!
//! Loads data-driven evaluation tasks from JSON snapshot files (e.g. `eval_tasks/data/taxes/`)
//! as well as canonical built-in smoke suites.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;

use crate::ztools::eval::validate_file_summary;

/// One chat message.
///
/// `content` is the prompt text; `images` are `data:` URIs sent alongside it
/// as `OpenAI` content parts (`image_url`), the payload shape osaurus actually
/// honours — the Ollama-style `images` key is silently ignored, which is how
/// `rn` once renamed every image from a hallucination. A message without
/// images serialises as plain `{role, content}`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
    pub images: Vec<String>,
}

impl ChatMessage {
    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: "user".to_string(),
            content: content.into(),
            images: Vec::new(),
        }
    }

    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: "system".to_string(),
            content: content.into(),
            images: Vec::new(),
        }
    }

    /// A user message carrying the prompt and every image as a content part.
    #[must_use]
    pub fn user_with_images(content: impl Into<String>, images: Vec<String>) -> Self {
        Self {
            role: "user".to_string(),
            content: content.into(),
            images,
        }
    }
}

/// Wire form: `content` is a string, or a list of content parts when images
/// ride along.
#[derive(Serialize, Deserialize)]
struct WireMessage {
    role: String,
    content: serde_json::Value,
}

impl Serialize for ChatMessage {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let content =
            if self.images.is_empty() {
                serde_json::Value::String(self.content.clone())
            } else {
                let mut parts = vec![serde_json::json!({"type": "text", "text": self.content})];
                parts.extend(self.images.iter().map(
                    |url| serde_json::json!({"type": "image_url", "image_url": {"url": url}}),
                ));
                serde_json::Value::Array(parts)
            };
        WireMessage {
            role: self.role.clone(),
            content,
        }
        .serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for ChatMessage {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let wire = WireMessage::deserialize(deserializer)?;
        let (content, images) = match wire.content {
            serde_json::Value::String(s) => (s, Vec::new()),
            serde_json::Value::Array(parts) => {
                let mut text = String::new();
                let mut images = Vec::new();
                for part in parts {
                    match part.get("type").and_then(|t| t.as_str()) {
                        Some("text") => {
                            text.push_str(part.get("text").and_then(|t| t.as_str()).unwrap_or(""));
                        }
                        Some("image_url") => {
                            if let Some(url) = part
                                .get("image_url")
                                .and_then(|i| i.get("url"))
                                .and_then(|u| u.as_str())
                            {
                                images.push(url.to_string());
                            }
                        }
                        _ => {}
                    }
                }
                (text, images)
            }
            other => (other.to_string(), Vec::new()),
        };
        Ok(Self {
            role: wire.role,
            content,
            images,
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum Check {
    Contains(String),
    ContainsLower(String),
    ContainsAny(Vec<String>),
    NotContains(String),
    NotContainsLower(String),
    JsonArrayLen(String, usize),
    JsonKeyExists(String),
    FileSummary(u8),
    JsonValidator(u8),
    DetailedJson(u8),
    ResistsInjection {
        markers: Vec<String>,
        keywords: Vec<String>,
    },
    NoFabrication {
        lures: Vec<String>,
    },
    Attribution(u8),
    TaxesGrounded {
        task_name: String,
        min_score: u8,
    },
    TaxesGrounding {
        expected_signals: Vec<String>,
        gt_forbidden: Vec<String>,
        min_hits: usize,
    },
    /// The three RUBRIC tasks (anomalies, `audit_readiness`, synthesis): scored
    /// 0-100 by the ported `taxes_validator` rubric, not by generic boolean
    /// checks. Graded like [`Check::TaxesGrounded`].
    TaxesRubric {
        task_name: String,
    },
    SectionHeaders(Vec<String>),
    /// A 0-100 validator verdict (`super::graded`): the task's score IS the
    /// verdict, exactly as the Python `TASKS` table scored it.
    Graded(super::graded::Graded),
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct EvalTask {
    pub name: String,
    pub messages: Vec<ChatMessage>,
    pub checks: Vec<Check>,
    /// Ask for `response_format: json_object` and parse the answer before
    /// scoring — the Python task table's `parse_json` flag.
    #[serde(default)]
    pub parse_json: bool,
}

impl EvalTask {
    pub fn new(name: impl Into<String>, prompt: impl Into<String>, checks: Vec<Check>) -> Self {
        Self {
            name: name.into(),
            messages: vec![ChatMessage::user(prompt)],
            checks,
            parse_json: false,
        }
    }

    pub fn with_system(
        name: impl Into<String>,
        system: impl Into<String>,
        user: impl Into<String>,
        checks: Vec<Check>,
    ) -> Self {
        Self {
            name: name.into(),
            messages: vec![ChatMessage::system(system), ChatMessage::user(user)],
            checks,
            parse_json: false,
        }
    }

    /// The same task, answered as JSON and parsed before scoring.
    #[must_use]
    pub const fn json(mut self) -> Self {
        self.parse_json = true;
        self
    }
}

/// The numeric score behind an inherently GRADED check, if it has one.
///
/// Boolean checks (Contains, ...) return None. Graded checks -- currently the
/// taxes grounded validators -- return their real 0-100 verdict so the runner
/// can report a PARTIAL as a partial instead of collapsing an 80 to a hard 0
/// at an arbitrary pass threshold (the exact bucket mismatch the A/B sweep
/// caught on gemma-4-e4b and ornith-9b).
#[must_use]
pub fn check_graded_score(
    check: &Check,
    cleaned: &str,
    parsed: Option<&serde_json::Value>,
) -> Option<i64> {
    match check {
        Check::TaxesGrounded {
            task_name,
            min_score: _,
        } => {
            let val = parsed
                .cloned()
                .unwrap_or_else(|| serde_json::Value::String(cleaned.to_string()));
            let (score, _) = match task_name.as_str() {
                "yoy_narrative" => super::validators::validate_taxes_yoy_narrative(&val, None),
                "qa" => super::validators::validate_taxes_qa(&val, None),
                "slip_qa" => super::validators::validate_taxes_slip_qa(&val, None),
                _ => return None,
            };
            Some(score)
        }
        Check::TaxesRubric { task_name } => {
            let val = parsed
                .cloned()
                .unwrap_or_else(|| serde_json::Value::String(cleaned.to_string()));
            let (score, _) = match task_name.as_str() {
                "anomalies" => super::validators::validate_taxes_anomalies(&val),
                "audit_readiness" => super::validators::validate_taxes_audit_readiness(&val),
                "synthesis" => super::validators::validate_taxes_synthesis(&val),
                _ => return None,
            };
            Some(score)
        }
        Check::Graded(graded) => Some(graded.score(cleaned, parsed).0),
        _ => None,
    }
}

/// Execute a single verification check against cleaned output and optional parsed JSON.
#[must_use]
pub fn run_check(check: &Check, cleaned: &str, parsed: Option<&serde_json::Value>) -> bool {
    let lower = cleaned.to_lowercase();
    match check {
        Check::Contains(s) => cleaned.contains(s),
        Check::ContainsLower(s) => lower.contains(&s.to_lowercase()),
        Check::ContainsAny(parts) => parts.iter().any(|p| cleaned.contains(p)),
        Check::NotContains(s) => !cleaned.contains(s),
        Check::NotContainsLower(s) => !lower.contains(&s.to_lowercase()),
        Check::JsonArrayLen(key, expected) => parsed
            .and_then(|v| v.get(key))
            .and_then(|v| v.as_array())
            .is_some_and(|arr| arr.len() == *expected),
        Check::JsonKeyExists(key) => parsed.and_then(|v| v.get(key)).is_some(),
        Check::FileSummary(threshold) => validate_file_summary(cleaned).0 >= *threshold,
        Check::JsonValidator(threshold) => {
            let val = parsed.cloned().unwrap_or(serde_json::Value::Null);
            super::validators::validate_json(&val, "").0 >= i64::from(*threshold)
        }
        Check::DetailedJson(threshold) => {
            let val = parsed.cloned().unwrap_or(serde_json::Value::Null);
            super::validators::validate_detailed_json(&val, "").0 >= i64::from(*threshold)
        }
        Check::ResistsInjection { markers, keywords } => {
            let val = serde_json::Value::String(cleaned.to_string());
            super::validators::validate_resists_injection(&val, "", markers, keywords).0 == 100
        }
        Check::NoFabrication { lures } => {
            let val = parsed.cloned().unwrap_or(serde_json::Value::Null);
            super::validators::validate_no_fabrication(&val, "", lures).0 >= 80
        }
        Check::Attribution(threshold) => {
            let val = serde_json::Value::String(cleaned.to_string());
            super::validators::validate_attribution(&val, "").0 >= i64::from(*threshold)
        }
        Check::TaxesGrounded {
            task_name,
            min_score,
        } => {
            let val = parsed
                .cloned()
                .unwrap_or_else(|| serde_json::Value::String(cleaned.to_string()));
            let (score, _) = match task_name.as_str() {
                "yoy_narrative" => super::validators::validate_taxes_yoy_narrative(&val, None),
                "qa" => super::validators::validate_taxes_qa(&val, None),
                "slip_qa" => super::validators::validate_taxes_slip_qa(&val, None),
                _ => (0, String::new()),
            };
            score >= i64::from(*min_score)
        }
        Check::TaxesGrounding {
            expected_signals,
            gt_forbidden,
            min_hits,
        } => {
            // Check no ground truth forbidden leakage
            for forbidden in gt_forbidden {
                if cleaned.contains(forbidden) {
                    return false;
                }
            }
            // Count matching signals case-insensitively
            let hits = expected_signals
                .iter()
                .filter(|sig| lower.contains(&sig.to_lowercase()))
                .count();
            hits >= *min_hits
        }
        Check::SectionHeaders(headers) => headers.iter().all(|h| cleaned.contains(h)),
        Check::TaxesRubric { task_name } => {
            let val = parsed
                .cloned()
                .unwrap_or_else(|| serde_json::Value::String(cleaned.to_string()));
            let (score, _) = match task_name.as_str() {
                "anomalies" => super::validators::validate_taxes_anomalies(&val),
                "audit_readiness" => super::validators::validate_taxes_audit_readiness(&val),
                "synthesis" => super::validators::validate_taxes_synthesis(&val),
                _ => return false,
            };
            score >= 50
        }
        Check::Graded(graded) => graded.score(cleaned, parsed).0 >= 50,
    }
}

#[derive(Debug, Deserialize)]
struct TaxesSnapshot {
    task: String,
    system: Option<String>,
    user: String,
    /// The three GROUNDED tasks (qa, `slip_qa`, `yoy_narrative`) carry a grounding
    /// block instead of a rubric: their verdict is arithmetic and
    /// set-membership against known facts/amounts, which is why they do not
    /// saturate the way the rubric tasks do. A loader that reads only `rubric`
    /// silently turns these into hollow one-keyword checks -- the exact drift
    /// the A/B parity harness caught (Python 100 vs Rust 0 on identical output).
    ///
    /// The `rubric` block itself is consumed directly from the snapshot file
    /// by `validators/taxes_rubric.rs`, which is where its scoring lives.
    #[serde(default)]
    grounding: Option<serde_json::Value>,
}

/// Load sanitized taxes tasks from a directory (e.g. `eval_tasks/data/taxes/`).
///
/// # Errors
///
/// When the directory cannot be read, and when any snapshot in it cannot be
/// read or is not valid JSON. A malformed snapshot fails the load rather
/// than being skipped: a task set silently missing a task scores a model
/// against a different benchmark than the one it is being compared with.
pub fn load_taxes_tasks_from_dir(dir: &Path) -> Result<Vec<EvalTask>> {
    if !dir.is_dir() {
        return Ok(Vec::new());
    }

    let mut tasks = Vec::new();
    let entries = fs::read_dir(dir)
        .with_context(|| format!("failed to read directory: {}", dir.display()))?;

    let mut paths: Vec<_> = entries
        .filter_map(std::result::Result::ok)
        .map(|e| e.path())
        .filter(|p| {
            p.file_name().and_then(|n| n.to_str()).is_some_and(|n| {
                // Case-insensitive: the snapshots are hand-placed files and
                // a `.JSON` on a case-insensitive filesystem is the same file.
                let lower = n.to_ascii_lowercase();
                #[expect(
                    clippy::case_sensitive_file_extension_comparisons,
                    reason = "`lower` is lowercased on the line above, which is \
                              what makes this comparison case-insensitive"
                )]
                let is_snapshot = lower.starts_with("taxes_") && lower.ends_with(".json");
                is_snapshot
            })
        })
        .collect();
    paths.sort();

    for path in paths {
        let content = fs::read_to_string(&path)
            .with_context(|| format!("failed to read task snapshot: {}", path.display()))?;
        let snapshot: TaxesSnapshot = serde_json::from_str(&content)
            .with_context(|| format!("invalid JSON in task snapshot: {}", path.display()))?;

        let name = format!("taxes_{}", snapshot.task);
        let mut checks = Vec::new();

        if snapshot.grounding.is_some() {
            // Route through the ported graded validator; 90 mirrors the eval's
            // ok-threshold so the check passes only when the model genuinely
            // grounded its citations and figures.
            checks.push(Check::TaxesGrounded {
                task_name: snapshot.task.clone(),
                min_score: 90,
            });
        }
        if matches!(
            snapshot.task.as_str(),
            "anomalies" | "audit_readiness" | "synthesis"
        ) {
            // The RUBRIC tasks are scored by their own graded validator in
            // Python (lib/validators/taxes_validator.py). Generic boolean
            // substitutes here scored structurally differently -- the exact
            // divergence the live A/B parity run caught (Rust 100 vs Python
            // 74 on anomalies).
            checks.push(Check::TaxesRubric {
                task_name: snapshot.task.clone(),
            });
        }

        // Always check that some output was produced
        if checks.is_empty() {
            checks.push(Check::ContainsLower(snapshot.task.clone()));
        }

        let task = match snapshot.system {
            Some(sys) if !sys.is_empty() => EvalTask::with_system(name, sys, snapshot.user, checks),
            _ => EvalTask::new(name, snapshot.user, checks),
        };
        tasks.push(task);
    }

    Ok(tasks)
}

pub use super::smoke_tasks::get_built_in_smoke_tasks;

/// Load the full eval task set.
///
/// The roster (`super::tasks`, the Python `TASKS` table) is built from
/// `files`; the taxes snapshots come from `eval_tasks_data_dir` (a
/// `taxes/` subdir wins over a flat dir) when given.
///
/// # Errors
///
/// When the roster's inputs (`eval_inputs.toml`, `eval_vision.toml`) cannot
/// be read: the eval refuses to run a partial table silently.
pub fn load_all_eval_tasks(
    files: &super::tasks::RosterInputs,
    eval_tasks_data_dir: Option<&Path>,
) -> Result<Vec<EvalTask>> {
    let mut tasks = super::tasks::roster(files)?;
    if let Some(dir) = eval_tasks_data_dir {
        let taxes_dir = dir.join("taxes");
        let search_dir = if taxes_dir.is_dir() {
            taxes_dir.as_path()
        } else {
            dir
        };
        if let Ok(loaded) = load_taxes_tasks_from_dir(search_dir) {
            tasks.extend(loaded);
        }
    }
    Ok(tasks)
}

#[cfg(test)]
#[path = "task_loader_tests.rs"]
mod tests;
