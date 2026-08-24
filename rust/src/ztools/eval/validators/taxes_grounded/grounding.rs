//! Loading a task's grounding snapshot and unwrapping a model's raw output.
//!
//! Split out of taxes_grounded.rs for the 500-line production cap. The only
//! part of this validator that touches the filesystem.

use serde_json::Value;
use std::path::{Path, PathBuf};

pub(super) fn load_grounding(task_name: &str) -> Value {
    let manifest = env!("CARGO_MANIFEST_DIR");
    let candidate = Path::new(manifest)
        .parent()
        .map(|p| {
            p.join("eval_tasks/data/taxes")
                .join(format!("taxes_{}.sanitized.json", task_name))
        })
        .unwrap_or_else(|| {
            PathBuf::from(format!(
                "eval_tasks/data/taxes/taxes_{}.sanitized.json",
                task_name
            ))
        });

    if let Ok(content) = std::fs::read_to_string(&candidate) {
        if let Ok(val) = serde_json::from_str::<Value>(&content) {
            if let Some(grounding) = val.get("grounding") {
                return grounding.clone();
            }
        }
    }
    Value::Null
}

pub(super) fn parse_output(raw: &Value) -> (Option<Value>, String) {
    if let Value::Object(_) = raw {
        return (Some(raw.clone()), String::new());
    }
    let text = match raw {
        Value::String(s) => s.trim().to_string(),
        other => other.to_string(),
    };
    if text.is_empty() {
        return (None, "empty output".to_string());
    }
    let mut clean_text = text.as_str();
    let mut note = String::new();
    if clean_text.starts_with("```") {
        note = "fenced".to_string();
        clean_text = clean_text
            .trim_start_matches("```json")
            .trim_start_matches("```")
            .trim_end_matches("```")
            .trim();
    }
    if let Ok(val) = serde_json::from_str::<Value>(clean_text) {
        return (Some(val), note);
    }
    if let Some(start) = clean_text.find('{') {
        if let Some(end) = clean_text.rfind('}') {
            if start < end {
                if let Ok(val) = serde_json::from_str::<Value>(&clean_text[start..=end]) {
                    return (Some(val), "extracted-from-prose".to_string());
                }
            }
        }
    }
    (None, "not-json".to_string())
}
