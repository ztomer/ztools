use anyhow::Result;
use reqwest::blocking::Client;
use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};

use crate::ztools::eval::{
    clean_model_output, extract_content_from_code_blocks, validate_file_summary,
};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ModelEvalResult {
    pub model: String,
    pub test_name: String,
    pub score: f64,
    pub passed: usize,
    pub total: usize,
    pub latency_ms: u64,
    pub status: String,
}

/// A data-driven eval check, mirroring the Python validator set. Output is
/// cleaned (content_processing port) and parsed (JSON) before checks run.
#[derive(Debug)]
pub enum Check {
    Contains(&'static str),
    ContainsLower(&'static str),
    ContainsAny(&'static [&'static str]),
    NotContains(&'static str),
    NotContainsLower(&'static str),
    /// Parse the cleaned output as JSON and require `key` to be an array whose
    /// length is exactly `expected`.
    JsonArrayLen(&'static str, usize),
    /// File-summary scoring (validate.py port); score must reach `threshold`.
    FileSummary(u8),
}

#[derive(Debug)]
pub struct EvalTask {
    pub name: &'static str,
    pub prompt: &'static str,
    pub checks: Vec<Check>,
}

fn weekend_prompt() -> &'static str {
    "You are an expert family activity planner. Extract up to 10 time-limited events happening STRICTLY this weekend (between 2026-08-07 and 2026-08-09) in Vaughan from the text below.\nOutput JSON now. Use EXACT schema:\n{\"transient_events\": [{\"name\": \"str\", \"location\": \"str\", \"target_ages\": \"str\", \"price\": \"str\", \"start_date\": \"str\", \"end_date\": \"str\", \"duration\": \"str\", \"weather\": \"str\", \"day\": \"str\", \"description\": \"str\"}]}\nRules for every field:\n- Suggest up to 10 specific weekend activities. Do NOT stop after just 1 or 2 events. Find as many as you can.\n- Only extract events that occur within or overlap with the dates 2026-08-07 to 2026-08-09. Discard events from past or future weekends.\n- Copy values from the source text. NEVER invent one.\n\nSearch results:\nEvent 1: Summer Rib Fest at Vaughan Park. August 7 2026. Kids all ages. Free.\nEvent 2: Fall Fair at Markham. August 8 2026. Kids 5-10. $10.\nEvent 3: Food Truck Festival at Toronto. August 9 2026. All ages. Free.\nEvent 4: Magic Show at Vaughan Library. August 7 2026. Kids 4-8. Free.\nEvent 5: Future Festival at Vaughan Park. August 14 2026. All ages. Free.\nOutput ONLY JSON."
}

fn file_summary_prompt() -> &'static str {
    "Read the file list below and give one-line summary for each file.\n\nCRITICAL: Rely ONLY on provided content context. DO NOT infer functionality from file names, words, or puns. Describe what each file DOES.\n- Bad: \"a python library\" (infers from .py extension)\n- Good: \"parses web content and extracts metadata\"\n\nFiles:\n- lib/parser.py\n- lib/validator.py\n- lib/fetcher.py\n- lib/reporter.py\n\nOutput a JSON array of {\"path\": \"...\", \"desc\": \"...\"} objects."
}

fn get_test_cases() -> Vec<EvalTask> {
    vec![
        EvalTask {
            name: "Weekend Planner (JSON Extraction)",
            prompt: weekend_prompt(),
            checks: vec![
                Check::Contains("transient_events"),
                Check::Contains("Summer Rib Fest"),
                Check::Contains("Magic Show"),
                Check::JsonArrayLen("transient_events", 2),
            ],
        },
        EvalTask {
            name: "Twitter Summarizer (Markdown formatting)",
            prompt: "Summarize these tweets into a markdown report. Use ## headers and - bullet points.\nTweets:\n- \"New Rust version 1.75 released!\"\n- \"I had a great sandwich today.\"\n- \"Learn about lifetime elision in Rust.\"",
            checks: vec![
                Check::Contains("##"),
                Check::ContainsAny(&["- ", "* "]),
                Check::ContainsLower("rust"),
                Check::NotContainsLower("```html"),
            ],
        },
        EvalTask {
            name: "Image Renamer (Constraint adherence)",
            prompt: "Analyze this image description and output a snake_case filename. End with .jpg.\nDescription: A red sports car parked on a sunny beach.\nRules: Output ONLY the filename. No markdown, no conversational text.",
            checks: vec![
                Check::Contains(".jpg"),
                Check::Contains("_"),
                Check::NotContains(" "),
                Check::NotContainsLower("here is"),
            ],
        },
        EvalTask {
            name: "Twitter Summarizer (Factual Consistency)",
            prompt: "Summarize this tweet timeline:\nTweet 1: @john_doe (2026-08-01): Just launched the new API!\nTweet 2: @jane_smith (2026-08-02): The new API is incredibly fast.",
            checks: vec![
                Check::NotContains("@elonmusk"),
                Check::NotContains("@realDonaldTrump"),
                Check::ContainsAny(&["john_doe", "@john_doe"]),
                Check::ContainsAny(&["jane_smith", "@jane_smith"]),
                Check::ContainsAny(&["2026-08", "August"]),
                Check::NotContains("2025"),
                Check::NotContains("2024"),
            ],
        },
        EvalTask {
            name: "File Summary (Content detail)",
            prompt: file_summary_prompt(),
            checks: vec![Check::FileSummary(50)],
        },
    ]
}

/// Extract the largest JSON value from cleaned model output: prefer a markdown
/// code block, else the outermost `[...]`/`{...}` span.
fn extract_json(content: &str) -> Option<serde_json::Value> {
    if let Some(block) = extract_content_from_code_blocks(content) {
        if let Ok(v) = serde_json::from_str(&block) {
            return Some(v);
        }
    }

    let starts: Vec<usize> = content
        .char_indices()
        .filter(|(_, c)| *c == '[' || *c == '{')
        .map(|(i, _)| i)
        .collect();
    let ends: Vec<usize> = content
        .char_indices()
        .filter(|(_, c)| *c == ']' || *c == '}')
        .map(|(i, _)| i)
        .collect();

    for start in &starts {
        for end in ends.iter().rev() {
            if end < start {
                continue;
            }
            if let Ok(v) = serde_json::from_str(&content[*start..=*end]) {
                return Some(v);
            }
        }
    }
    serde_json::from_str(content).ok()
}

fn run_check(check: &Check, cleaned: &str, parsed: Option<&serde_json::Value>) -> bool {
    let lower = cleaned.to_lowercase();
    match check {
        Check::Contains(s) => cleaned.contains(s),
        Check::ContainsLower(s) => lower.contains(s),
        Check::ContainsAny(parts) => parts.iter().any(|p| cleaned.contains(p)),
        Check::NotContains(s) => !cleaned.contains(s),
        Check::NotContainsLower(s) => !lower.contains(s),
        Check::JsonArrayLen(key, expected) => parsed
            .and_then(|v| v.get(*key))
            .and_then(|v| v.as_array())
            .map(|arr| arr.len() == *expected)
            .unwrap_or(false),
        Check::FileSummary(threshold) => validate_file_summary(cleaned).0 >= *threshold,
    }
}

pub fn get_available_models(
    base_url: &str,
    config: &crate::config::ZtoolsConfig,
) -> Result<Vec<String>> {
    let client = Client::builder()
        .timeout(Duration::from_secs(config.llm_quick_timeout_secs))
        .build()?;
    let url = format!("{}/v1/models", base_url.trim_end_matches('/'));
    let resp: serde_json::Value = client.get(&url).send()?.json()?;

    let mut models = Vec::new();
    if let Some(data) = resp.get("data").and_then(|d| d.as_array()) {
        for m in data {
            if let Some(id) = m.get("id").and_then(|id| id.as_str()) {
                if !id.contains("foundation") && !id.contains("diffusion") {
                    models.push(id.to_string());
                }
            }
        }
    }
    Ok(models)
}

pub fn eval_model(
    base_url: &str,
    model_name: &str,
    config: &crate::config::ZtoolsConfig,
) -> Result<Vec<ModelEvalResult>> {
    let defects = crate::ztools::model_health::probe_model_defects(model_name, None);
    if !defects.is_empty() {
        println!(
            "⚠ Skipping broken model '{}': {}",
            model_name,
            defects.join("; ")
        );
        return Ok(vec![ModelEvalResult {
            model: model_name.to_string(),
            test_name: "packaging_health".to_string(),
            score: 0.0,
            passed: 0,
            total: 1,
            latency_ms: 0,
            status: format!("refused: {}", defects[0]),
        }]);
    }

    let client = Client::builder()
        .timeout(Duration::from_secs(config.llm_timeout_secs))
        .build()?;
    let url = format!("{}/v1/chat/completions", base_url.trim_end_matches('/'));

    let mut results = Vec::new();

    let cases = get_test_cases();
    for case in cases {
        let start = Instant::now();
        let payload = serde_json::json!({
            "model": model_name,
            "messages": [{"role": "user", "content": case.prompt}],
            "temperature": 0.0
        });

        let mut passed = 0;
        let total = case.checks.len();

        let resp = client.post(&url).json(&payload).send();
        let elapsed = start.elapsed().as_millis() as u64;

        let mut output_text = String::new();
        if let Ok(r) = resp {
            if r.status().is_success() {
                let json: serde_json::Value = r.json().unwrap_or_default();
                output_text = json["choices"][0]["message"]["content"]
                    .as_str()
                    .unwrap_or("")
                    .to_string();

                // Parity with the Python eval: clean the output (thinking
                // blocks, stats, markdown fences) BEFORE judging it.
                let cleaned = clean_model_output(&output_text);
                let parsed = extract_json(&cleaned);
                for check in &case.checks {
                    if run_check(check, &cleaned, parsed.as_ref()) {
                        passed += 1;
                    }
                }
            }
        }

        if passed != total {
            println!(
                "Test '{}' failed ({}/{}). Model output:\n---\n{}\n---",
                case.name, passed, total, output_text
            );
        }

        let score = (passed as f64 / total as f64) * 100.0;
        let status = if passed == total { "passed" } else { "failed" };

        results.push(ModelEvalResult {
            model: model_name.to_string(),
            test_name: case.name.to_string(),
            score,
            passed,
            total,
            latency_ms: elapsed,
            status: status.to_string(),
        });
    }

    Ok(results)
}

pub fn eval_all_models(
    base_url: &str,
    config: &crate::config::ZtoolsConfig,
) -> Result<Vec<ModelEvalResult>> {
    let models = get_available_models(base_url, config)?;
    let mut all_results = Vec::new();

    for model in models {
        if let Ok(res) = eval_model(base_url, &model, config) {
            all_results.extend(res);
        }
    }
    Ok(all_results)
}

pub fn render_eval_report(results: &[ModelEvalResult]) -> String {
    let mut out = String::new();
    out.push_str("# Model Quality Evaluation Benchmark\n\n");
    out.push_str("| Model | Test | Score | Passed | Latency | Status |\n");
    out.push_str("| :--- | :--- | :--- | :--- | :--- | :--- |\n");

    for r in results {
        out.push_str(&format!(
            "| **{}** | {} | {:.1}% | {}/{} | {}ms | {} |\n",
            r.model, r.test_name, r.score, r.passed, r.total, r.latency_ms, r.status
        ));
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_render_eval_report() {
        let results = vec![ModelEvalResult {
            model: "test_model".to_string(),
            test_name: "test_suite".to_string(),
            score: 100.0,
            passed: 4,
            total: 4,
            latency_ms: 120,
            status: "passed".to_string(),
        }];
        let md = render_eval_report(&results);
        assert!(md.contains("test_model"));
        assert!(md.contains("100.0%"));
        assert!(md.contains("120ms"));
    }

    #[test]
    fn test_get_test_cases_not_empty() {
        let cases = get_test_cases();
        assert_eq!(cases.len(), 5);
    }

    #[test]
    fn json_array_len_check() {
        let cleaned = r#"{"transient_events": [{"name": "a"}, {"name": "b"}]}"#;
        let parsed = extract_json(cleaned).unwrap();
        assert!(run_check(
            &Check::JsonArrayLen("transient_events", 2),
            cleaned,
            Some(&parsed)
        ));
        assert!(!run_check(
            &Check::JsonArrayLen("transient_events", 3),
            cleaned,
            Some(&parsed)
        ));
    }

    #[test]
    fn cleaning_precedes_checks() {
        // Thinking block around the answer must not poison the checks.
        let cleaned = clean_model_output("<think>inner</think> Image: red_car.jpg");
        assert!(run_check(&Check::Contains("red_car.jpg"), &cleaned, None));
    }

    #[test]
    fn file_summary_check_uses_validator() {
        let good = r#"[{"path": "a.py", "desc": "parses config files"},
                        {"path": "b.py", "desc": "validates JSON output"},
                        {"path": "c.py", "desc": "fetches external API data"},
                        {"path": "d.py", "desc": "handles processing logic"}]"#;
        assert!(run_check(&Check::FileSummary(50), good, None));

        let bad = r#"[{"path": "a.py", "desc": "a python script"},
                      {"path": "b.py", "desc": "another file"}]"#;
        assert!(!run_check(&Check::FileSummary(50), bad, None));
    }
}