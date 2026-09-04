use anyhow::Result;
use reqwest::blocking::Client;
use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};

pub use crate::ztools::eval::{
    clean_model_output, extract_content_from_code_blocks, extract_json, get_built_in_smoke_tasks,
    load_all_eval_tasks, load_taxes_tasks_from_dir, run_check, validate_file_summary, ChatMessage,
    Check, EvalTask, GpuLockGuard, DEFAULT_MAX_IDLE_SECS,
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

pub fn get_test_cases() -> Vec<EvalTask> {
    get_built_in_smoke_tasks()
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

    let lock_label = format!("eval {model_name}");
    let guard = GpuLockGuard::acquire(
        &lock_label,
        Duration::from_secs(config.llm_quick_timeout_secs),
        Duration::from_secs(DEFAULT_MAX_IDLE_SECS),
    );

    let client = Client::builder()
        .timeout(Duration::from_secs(config.llm_timeout_secs))
        .build()?;
    let url = format!("{}/v1/chat/completions", base_url.trim_end_matches('/'));

    let mut results = Vec::new();

    let cases = get_test_cases();
    for case in cases {
        if let Ok(ref g) = guard {
            g.heartbeat();
        }
        let start = Instant::now();
        let payload = serde_json::json!({
            "model": model_name,
            "messages": case.messages,
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

/// Split an osaurus base URL ("http://127.0.0.1:1337") into host and port.
pub fn parse_osaurus_url(url: &str) -> (String, u16) {
    let stripped = url
        .strip_prefix("http://")
        .or_else(|| url.strip_prefix("https://"))
        .unwrap_or(url);
    let trimmed = stripped.trim_end_matches('/');
    match trimmed.rsplit_once(':') {
        Some((host, port)) => match port.parse() {
            Ok(p) => (host.to_string(), p),
            Err(_) => (trimmed.to_string(), 1337),
        },
        None => (trimmed.to_string(), 1337),
    }
}

/// Render full-suite [`TaskOutcome`]s as a markdown table, worst first: the
/// failures are what a reader scans for, so they lead.
pub fn render_task_outcomes(outcomes: &[crate::ztools::eval::TaskOutcome]) -> String {
    use std::fmt::Write;
    let mut report = String::from("# Full Suite Eval\n\n");
    let _ = writeln!(report, "| task | score | status | time | error |");
    let _ = writeln!(report, "|------|-------|--------|------|-------|");
    let mut rows: Vec<_> = outcomes.iter().collect();
    rows.sort_by_key(|o| o.score);
    for o in &rows {
        let _ = writeln!(
            report,
            "| {} | {} | {} | {:.1}s | {} |",
            o.task,
            o.score,
            o.status,
            o.time_secs,
            o.error.as_deref().unwrap_or("-")
        );
    }
    if !rows.is_empty() {
        let mean: f64 = rows.iter().map(|o| o.score as f64).sum::<f64>() / rows.len() as f64;
        let ok = rows.iter().filter(|o| o.status == "ok").count();
        let _ = writeln!(report);
        let _ = writeln!(
            report,
            "{} tasks, {ok} ok, mean score {mean:.1}",
            rows.len()
        );
    }
    report
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
            &Check::JsonArrayLen("transient_events".to_string(), 2),
            cleaned,
            Some(&parsed)
        ));
        assert!(!run_check(
            &Check::JsonArrayLen("transient_events".to_string(), 3),
            cleaned,
            Some(&parsed)
        ));
    }

    #[test]
    fn osaurus_url_parses_to_host_and_port() {
        assert_eq!(
            parse_osaurus_url("http://127.0.0.1:1337"),
            ("127.0.0.1".to_string(), 1337)
        );
        assert_eq!(
            parse_osaurus_url("http://localhost:9999/"),
            ("localhost".to_string(), 9999)
        );
        // Portless and malformed URLs fall back to the default port rather
        // than panicking.
        assert_eq!(
            parse_osaurus_url("http://myhost"),
            ("myhost".to_string(), 1337)
        );
        assert_eq!(parse_osaurus_url("weird"), ("weird".to_string(), 1337));
    }

    #[test]
    fn task_outcomes_render_worst_first_with_summary() {
        use crate::ztools::eval::TaskOutcome;
        let outcomes = vec![
            TaskOutcome {
                task: "good".to_string(),
                score: 100,
                status: "ok".to_string(),
                ..Default::default()
            },
            TaskOutcome {
                task: "bad".to_string(),
                score: 25,
                status: "fail".to_string(),
                error: Some("HTTP 503".to_string()),
                ..Default::default()
            },
        ];
        let report = render_task_outcomes(&outcomes);
        let bad_pos = report.find("| bad ").expect("bad row present");
        let good_pos = report.find("| good ").expect("good row present");
        assert!(bad_pos < good_pos, "worst score must lead:\n{report}");
        assert!(
            report.contains("2 tasks, 1 ok, mean score 62.5"),
            "{report}"
        );
        assert!(report.contains("HTTP 503"), "{report}");
    }

    #[test]
    fn cleaning_precedes_checks() {
        // Thinking block around the answer must not poison the checks.
        let cleaned = clean_model_output("<think>inner</think> Image: red_car.jpg");
        assert!(run_check(
            &Check::Contains("red_car.jpg".to_string()),
            &cleaned,
            None
        ));
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
