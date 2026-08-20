use anyhow::Result;
use reqwest::blocking::Client;
use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};

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

struct TestCase {
    name: String,
    prompt: String,
    checks: Vec<fn(&str) -> bool>,
}

fn get_test_cases() -> Vec<TestCase> {
    vec![
        TestCase {
            name: "Weekend Planner (JSON Extraction)".to_string(),
            prompt: "You are an expert family activity planner. Extract up to 10 time-limited events happening STRICTLY this weekend (between 2026-08-07 and 2026-08-09) in Vaughan from the text below.\nOutput JSON now. Use EXACT schema:\n{\"transient_events\": [{\"name\": \"str\", \"location\": \"str\", \"target_ages\": \"str\", \"price\": \"str\", \"start_date\": \"str\", \"end_date\": \"str\", \"duration\": \"str\", \"weather\": \"str\", \"day\": \"str\", \"description\": \"str\"}]}\nRules for every field:\n- Suggest up to 10 specific weekend activities. Do NOT stop after just 1 or 2 events. Find as many as you can.\n- Only extract events that occur within or overlap with the dates 2026-08-07 to 2026-08-09. Discard events from past or future weekends.\n- Copy values from the source text. NEVER invent one.\n\nSearch results:\nEvent 1: Summer Rib Fest at Vaughan Park. August 7 2026. Kids all ages. Free.\nEvent 2: Fall Fair at Markham. August 8 2026. Kids 5-10. $10.\nEvent 3: Food Truck Festival at Toronto. August 9 2026. All ages. Free.\nEvent 4: Magic Show at Vaughan Library. August 7 2026. Kids 4-8. Free.\nEvent 5: Future Festival at Vaughan Park. August 14 2026. All ages. Free.\nOutput ONLY JSON.".to_string(),
            checks: vec![
                |s| s.contains("transient_events"),
                |s| s.contains("Summer Rib Fest"),
                |s| s.contains("Magic Show"),
                |s| {
                    if let Ok(json) = serde_json::from_str::<serde_json::Value>(s.trim().trim_start_matches("```json").trim_end_matches("```").trim()) {
                        json.get("transient_events").and_then(|v| v.as_array()).map(|arr| arr.len() == 2).unwrap_or(false)
                    } else {
                        false
                    }
                },
            ],
        },
        TestCase {
            name: "Twitter Summarizer (Markdown formatting)".to_string(),
            prompt: "Summarize these tweets into a markdown report. Use ## headers and - bullet points.\nTweets:\n- \"New Rust version 1.75 released!\"\n- \"I had a great sandwich today.\"\n- \"Learn about lifetime elision in Rust.\"".to_string(),
            checks: vec![
                |s| s.contains("##"),
                |s| s.contains("- ") || s.contains("* "),
                |s| s.to_lowercase().contains("rust"),
                |s| !s.to_lowercase().contains("```html"),
            ],
        },
        TestCase {
            name: "Image Renamer (Constraint adherence)".to_string(),
            prompt: "Analyze this image description and output a snake_case filename. End with .jpg.\nDescription: A red sports car parked on a sunny beach.\nRules: Output ONLY the filename. No markdown, no conversational text.".to_string(),
            checks: vec![
                |s| s.contains(".jpg"),
                |s| s.contains("_"),
                |s| !s.contains(" "),
                |s| !s.to_lowercase().contains("here is"),
            ],
        },
        TestCase {
            name: "Twitter Summarizer (Factual Consistency)".to_string(),
            prompt: "Summarize this tweet timeline:\nTweet 1: @john_doe (2026-08-01): Just launched the new API!\nTweet 2: @jane_smith (2026-08-02): The new API is incredibly fast.".to_string(),
            checks: vec![
                |s| !s.contains("@elonmusk") && !s.contains("@realDonaldTrump"), // Hallucination checks
                |s| s.contains("john_doe") || s.contains("@john_doe"),
                |s| s.contains("jane_smith") || s.contains("@jane_smith"),
                |s| s.contains("2026-08") || s.contains("August"),
                |s| !s.contains("2025") && !s.contains("2024"), // Wrong dates
            ],
        },
    ]
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

                for check in &case.checks {
                    if check(&output_text) {
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
            test_name: case.name.clone(),
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
        assert_eq!(cases.len(), 4);
    }
}
