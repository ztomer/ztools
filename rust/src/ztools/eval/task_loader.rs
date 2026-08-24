//! Eval task loader (`task_loader.rs`).
//!
//! Loads data-driven evaluation tasks from JSON snapshot files (e.g. `eval_tasks/data/taxes/`)
//! as well as canonical built-in smoke suites.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;

use crate::ztools::eval::validate_file_summary;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

impl ChatMessage {
    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: "user".to_string(),
            content: content.into(),
        }
    }

    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: "system".to_string(),
            content: content.into(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
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
    /// The three RUBRIC tasks (anomalies, audit_readiness, synthesis): scored
    /// 0-100 by the ported taxes_validator rubric, not by generic boolean
    /// checks. Graded like [`Check::TaxesGrounded`].
    TaxesRubric { task_name: String },
    SectionHeaders(Vec<String>),
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct EvalTask {
    pub name: String,
    pub messages: Vec<ChatMessage>,
    pub checks: Vec<Check>,
}

impl EvalTask {
    pub fn new(name: impl Into<String>, prompt: impl Into<String>, checks: Vec<Check>) -> Self {
        Self {
            name: name.into(),
            messages: vec![ChatMessage::user(prompt)],
            checks,
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
        }
    }
}

/// The numeric score behind an inherently GRADED check, if it has one.
///
/// Boolean checks (Contains, ...) return None. Graded checks -- currently the
/// taxes grounded validators -- return their real 0-100 verdict so the runner
/// can report a PARTIAL as a partial instead of collapsing an 80 to a hard 0
/// at an arbitrary pass threshold (the exact bucket mismatch the A/B sweep
/// caught on gemma-4-e4b and ornith-9b).
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
        _ => None,
    }
}

/// Execute a single verification check against cleaned output and optional parsed JSON.
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
            .map(|arr| arr.len() == *expected)
            .unwrap_or(false),
        Check::JsonKeyExists(key) => parsed.and_then(|v| v.get(key)).is_some(),
        Check::FileSummary(threshold) => validate_file_summary(cleaned).0 >= *threshold,
        Check::JsonValidator(threshold) => {
            let val = parsed.cloned().unwrap_or(serde_json::Value::Null);
            super::validators::validate_json(&val, "").0 >= *threshold as i64
        }
        Check::DetailedJson(threshold) => {
            let val = parsed.cloned().unwrap_or(serde_json::Value::Null);
            super::validators::validate_detailed_json(&val, "").0 >= *threshold as i64
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
            super::validators::validate_attribution(&val, "").0 >= *threshold as i64
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
            score >= *min_score as i64
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
    }
}

#[derive(Debug, Deserialize)]
struct TaxesSnapshot {
    task: String,
    system: Option<String>,
    user: String,
    /// The three GROUNDED tasks (qa, slip_qa, yoy_narrative) carry a grounding
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
pub fn load_taxes_tasks_from_dir(dir: &Path) -> Result<Vec<EvalTask>> {
    if !dir.is_dir() {
        return Ok(Vec::new());
    }

    let mut tasks = Vec::new();
    let entries = fs::read_dir(dir)
        .with_context(|| format!("failed to read directory: {}", dir.display()))?;

    let mut paths: Vec<_> = entries
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            p.file_name()
                .and_then(|n| n.to_str())
                .map(|n| n.starts_with("taxes_") && n.ends_with(".json"))
                .unwrap_or(false)
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

/// Built-in smoke tasks (offline fixtures).
pub fn get_built_in_smoke_tasks() -> Vec<EvalTask> {
    vec![
        EvalTask::new(
            "Weekend Planner (JSON Extraction)",
            "You are an expert family activity planner. Extract up to 10 time-limited events happening STRICTLY this weekend (between 2026-08-07 and 2026-08-09) in Vaughan from the text below.\nOutput JSON now. Use EXACT schema:\n{\"transient_events\": [{\"name\": \"str\", \"location\": \"str\", \"target_ages\": \"str\", \"price\": \"str\", \"start_date\": \"str\", \"end_date\": \"str\", \"duration\": \"str\", \"weather\": \"str\", \"day\": \"str\", \"description\": \"str\"}]}\nRules for every field:\n- Suggest up to 10 specific weekend activities. Do NOT stop after just 1 or 2 events. Find as many as you can.\n- Only extract events that occur within or overlap with the dates 2026-08-07 to 2026-08-09. Discard events from past or future weekends.\n- Copy values from the source text. NEVER invent one.\n\nSearch results:\nEvent 1: Summer Rib Fest at Vaughan Park. August 7 2026. Kids all ages. Free.\nEvent 2: Fall Fair at Markham. August 8 2026. Kids 5-10. $10.\nEvent 3: Food Truck Festival at Toronto. August 9 2026. All ages. Free.\nEvent 4: Magic Show at Vaughan Library. August 7 2026. Kids 4-8. Free.\nEvent 5: Future Festival at Vaughan Park. August 14 2026. All ages. Free.\nOutput ONLY JSON.",
            vec![
                Check::Contains("transient_events".to_string()),
                Check::Contains("Summer Rib Fest".to_string()),
                Check::Contains("Magic Show".to_string()),
                Check::JsonArrayLen("transient_events".to_string(), 2),
            ],
        ),
        EvalTask::new(
            "Twitter Summarizer (Markdown formatting)",
            "Summarize these tweets into a markdown report. Use ## headers and - bullet points.\nTweets:\n- \"New Rust version 1.75 released!\"\n- \"I had a great sandwich today.\"\n- \"Learn about lifetime elision in Rust.\"",
            vec![
                Check::Contains("##".to_string()),
                Check::ContainsAny(vec!["- ".to_string(), "* ".to_string()]),
                Check::ContainsLower("rust".to_string()),
                Check::NotContainsLower("```html".to_string()),
            ],
        ),
        EvalTask::new(
            "Image Renamer (Constraint adherence)",
            "Analyze this image description and output a snake_case filename. End with .jpg.\nDescription: A red sports car parked on a sunny beach.\nRules: Output ONLY the filename. No markdown, no conversational text.",
            vec![
                Check::Contains(".jpg".to_string()),
                Check::Contains("_".to_string()),
                Check::NotContains(" ".to_string()),
                Check::NotContainsLower("here is".to_string()),
            ],
        ),
        EvalTask::new(
            "Twitter Summarizer (Factual Consistency)",
            "Summarize this tweet timeline:\nTweet 1: @john_doe (2026-08-01): Just launched the new API!\nTweet 2: @jane_smith (2026-08-02): The new API is incredibly fast.",
            vec![
                Check::NotContains("@elonmusk".to_string()),
                Check::NotContains("@realDonaldTrump".to_string()),
                Check::ContainsAny(vec!["john_doe".to_string(), "@john_doe".to_string()]),
                Check::ContainsAny(vec!["jane_smith".to_string(), "@jane_smith".to_string()]),
                Check::ContainsAny(vec!["2026-08".to_string(), "August".to_string()]),
                Check::NotContains("2025".to_string()),
                Check::NotContains("2024".to_string()),
            ],
        ),
        EvalTask::new(
            "File Summary (Content detail)",
            "Read the file list below and give one-line summary for each file.\n\nCRITICAL: Rely ONLY on provided content context. DO NOT infer functionality from file names, words, or puns. Describe what each file DOES.\n- Bad: \"a python library\" (infers from .py extension)\n- Good: \"parses web content and extracts metadata\"\n\nFiles:\n- lib/parser.py\n- lib/validator.py\n- lib/fetcher.py\n- lib/reporter.py\n\nOutput a JSON array of {\"path\": \"...\", \"desc\": \"...\"} objects.",
            vec![Check::FileSummary(50)],
        ),
    ]
}

/// Load all eval tasks: smoke tasks plus tasks from data snapshots if found.
pub fn load_all_eval_tasks(eval_tasks_data_dir: Option<&Path>) -> Vec<EvalTask> {
    let mut tasks = get_built_in_smoke_tasks();
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
    tasks
}

#[cfg(test)]
#[path = "task_loader_tests.rs"]
mod tests;
