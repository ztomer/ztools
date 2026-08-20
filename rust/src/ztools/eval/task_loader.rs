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
    TaxesGrounding {
        expected_signals: Vec<String>,
        gt_forbidden: Vec<String>,
        min_hits: usize,
    },
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
    }
}

#[derive(Debug, Deserialize)]
struct TaxesRubric {
    #[serde(default)]
    expected_signals: Vec<String>,
    #[serde(default)]
    gt_forbidden: Vec<String>,
    #[serde(default)]
    schema: Option<String>,
    #[serde(default)]
    schema_key: Option<String>,
    #[serde(default)]
    expected_sections: Option<Vec<String>>,
}

#[derive(Debug, Deserialize)]
struct TaxesSnapshot {
    task: String,
    system: Option<String>,
    user: String,
    rubric: Option<TaxesRubric>,
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

        if let Some(rubric) = snapshot.rubric {
            if !rubric.expected_signals.is_empty() || !rubric.gt_forbidden.is_empty() {
                checks.push(Check::TaxesGrounding {
                    expected_signals: rubric.expected_signals,
                    gt_forbidden: rubric.gt_forbidden,
                    min_hits: 3,
                });
            }
            if let Some(sections) = rubric.expected_sections {
                if !sections.is_empty() {
                    checks.push(Check::SectionHeaders(sections));
                }
            }
            if let Some(key) = rubric.schema_key {
                checks.push(Check::JsonKeyExists(key));
            } else if let Some(schema) = rubric.schema {
                if schema == "json" {
                    checks.push(Check::Contains("{".to_string()));
                }
            }
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
mod tests {
    use super::*;

    #[test]
    fn test_run_check_primitives() {
        assert!(run_check(
            &Check::Contains("hello".to_string()),
            "hello world",
            None
        ));
        assert!(!run_check(
            &Check::Contains("goodbye".to_string()),
            "hello world",
            None
        ));

        assert!(run_check(
            &Check::ContainsLower("HELLO".to_string()),
            "hello world",
            None
        ));
        assert!(run_check(
            &Check::ContainsAny(vec!["cat".to_string(), "dog".to_string()]),
            "my dog is cute",
            None
        ));
        assert!(!run_check(
            &Check::ContainsAny(vec!["cat".to_string(), "bird".to_string()]),
            "my dog is cute",
            None
        ));

        assert!(run_check(
            &Check::NotContains("bad".to_string()),
            "good message",
            None
        ));
        assert!(!run_check(
            &Check::NotContains("good".to_string()),
            "good message",
            None
        ));

        assert!(run_check(
            &Check::NotContainsLower("BAD".to_string()),
            "good message",
            None
        ));
    }

    #[test]
    fn test_run_check_json_and_summary() {
        let val: serde_json::Value = serde_json::json!({
            "items": [1, 2, 3],
            "details": {"title": "doc"}
        });
        assert!(run_check(
            &Check::JsonArrayLen("items".to_string(), 3),
            "",
            Some(&val)
        ));
        assert!(!run_check(
            &Check::JsonArrayLen("items".to_string(), 2),
            "",
            Some(&val)
        ));
        assert!(run_check(
            &Check::JsonKeyExists("details".to_string()),
            "",
            Some(&val)
        ));
        assert!(!run_check(
            &Check::JsonKeyExists("missing".to_string()),
            "",
            Some(&val)
        ));

        // FileSummary check
        let good_summary = r#"[{"path": "lib/parse.py", "desc": "parses incoming data stream and validates headers"}]"#;
        assert!(run_check(&Check::FileSummary(40), good_summary, None));
    }

    #[test]
    fn test_run_check_taxes_grounding_and_sections() {
        let check = Check::TaxesGrounding {
            expected_signals: vec!["T1135".to_string(), "Box 38".to_string(), "RSU".to_string()],
            gt_forbidden: vec!["Filed (GT)".to_string()],
            min_hits: 2,
        };
        let text = "Here is the summary regarding t1135 and rsu income.";
        assert!(run_check(&check, text, None));

        let leaked = "Here is the summary with Filed (GT) and t1135 and rsu.";
        assert!(!run_check(&check, leaked, None));

        let insufficient = "Only t1135 mentioned.";
        assert!(!run_check(&check, insufficient, None));

        let section_check =
            Check::SectionHeaders(vec!["**1. Missing".to_string(), "**2. Impact".to_string()]);
        assert!(run_check(
            &section_check,
            "**1. Missing docs\n**2. Impact is $500",
            None
        ));
        assert!(!run_check(&section_check, "**1. Missing docs only", None));
    }

    #[test]
    fn test_load_taxes_tasks_from_real_data_dir() {
        let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap();
        let taxes_dir = repo_root.join("eval_tasks").join("data").join("taxes");
        if taxes_dir.is_dir() {
            let tasks = load_taxes_tasks_from_dir(&taxes_dir).unwrap();
            assert_eq!(
                tasks.len(),
                6,
                "expected 6 taxes tasks, got {}",
                tasks.len()
            );
            let names: Vec<_> = tasks.iter().map(|t| t.name.as_str()).collect();
            assert!(names.contains(&"taxes_anomalies"));
            assert!(names.contains(&"taxes_audit_readiness"));
            assert!(names.contains(&"taxes_synthesis"));
            assert!(names.contains(&"taxes_qa"));
            assert!(names.contains(&"taxes_slip_qa"));
            assert!(names.contains(&"taxes_yoy_narrative"));
        }
    }

    #[test]
    fn test_load_taxes_tasks_poisoned_file_fails() {
        let temp = tempfile::tempdir().unwrap();
        let bad_file = temp.path().join("taxes_bad.json");
        fs::write(&bad_file, b"{ invalid json").unwrap();

        let result = load_taxes_tasks_from_dir(temp.path());
        assert!(result.is_err(), "expected error on poisoned JSON file");
    }
}
