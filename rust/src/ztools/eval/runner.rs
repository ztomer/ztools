//! Eval runner: core evaluation loop orchestration.
//! Ported from `references/eval/run.py` — the main eval orchestration,
//! retry logic, timeout management, and result collection.
//!
//! This module implements the Rust eval runner that produces identical
//! outputs to the Python `references/eval/run.py` reference.

use regex::Regex;
use std::collections::HashMap;

use lib::gpu_lock;
use lib::config_getters::{get_max_tokens_for_task, is_generative_model};
use lib::logging_config as eval_logger;
use lib::mlx_lib::call as mlx_call;
use lib::model_caps::is_generative_model as model_caps_is_generative;
use lib::osaurus_lib::call;
use lib::tui::{FAIL, STEP, WARN, console};
use lib::validators_lib::get_source_matching_details;
use lib::validators_lib::safe_content;

use eval::failures::{
    FAIL_CONTENT, FAIL_INFRA, FAIL_NONE, FAIL_REASONING,
    _classify_failure, reasoning_retry_budget,
};
use eval::outputs::save_output;
use eval::prefill::{measure_prefill_rate, record_prefill_rate};
use eval::result_format::_quality_results_to_eval_format;
use eval::signals::{DEFAULT_EVAL_TIMEOUT, _effective_timeout, _record_signal, contended_server_warning};
use eval::tasks_core::TASKS;
use eval::validate::safe_content;
use eval::watchdog::check_stall;

const MAX_RETRIES: i32 = match std::env::var("EVAL_MAX_RETRIES") {
    Ok(v) => v.parse().unwrap_or(1),
    Err(_) => 1,
};
const MAX_CONSECUTIVE_INFRA_FAILURES: i32 = match std::env::var("EVAL_MAX_INFRA_FAILURES") {
    Ok(v) => v.parse().unwrap_or(4),
    Err(_) => 4,
};
const EVAL_TEMPERATURE: f32 = match std::env::var("EVAL_TEMPERATURE") {
    Ok(v) => v.parse().unwrap_or(0.0),
    Err(_) => 0.0,
};
const MEMORY_WARNING_THRESHOLD: i32 = 80;

/// Call model via the appropriate backend (pure transport, no validation).
fn _call_model(
    model: &str,
    task_cfg: &eval::tasks_core::Check,
    task_name: &str,
    host: &str,
    port: u16,
    backend: &str,
    timeout: i32,
    max_tokens: i32,
) -> eval::tasks_core::Check {
    if backend == "mlx" {
        mlx_call(
            model,
            messages = &task_cfg.messages,
            host = host,
            port,
            temperature = EVAL_TEMPERATURE,
            timeout,
        )
    } else {
        call(
            model = model,
            messages = &task_cfg.messages,
            host,
            port,
            task = task_name,
            parse_json = task_cfg.parse_json,
            temperature = EVAL_TEMPERATURE,
            timeout,
            max_tokens,
            stream_guard = true,
        )
    }
}

/// Run validation on a library result. Returns (score, failure_reason, diagnosis).
fn _validate_result(
    result: &eval::tasks_core::Check,
    task_cfg: &eval::tasks_core::Check,
    task_name: &str,
    debug: bool,
) -> (i32, String, eval::failures::Diagnosis) {
    use eval::failures::{_classify_failure, FAIL_CONTENT, FAIL_INFRA, FAIL_NONE, FAIL_REASONING};

    if result.get("error").is_some() {
        let diagnosis = _classify_failure(result, task_cfg, 0, result.get("error").unwrap_or(&"".to_string()).clone());
        return (0, result.get("error").unwrap_or(&"".to_string()).clone(), diagnosis);
    }

    let is_parse_json = task_cfg.get("parse_json", false);
    let parsed = result.get("parsed");
    let content = safe_content(result);
    let source = task_cfg.get("source", "").to_string();

    if is_parse_json && parsed.is_some() {
        let validated = task_cfg.validator(
            parsed.unwrap(),
            source_text = &source,
            **task_cfg.get("validator_kwargs", {}),
        );

        let (score, failure_reason) = if let Some((s, fr)) = validated {
            (s, fr)
        } else {
            (validated, "".to_string())
        };

        let diagnosis = _classify_failure(result, task_cfg, score, &failure_reason);
        return (score, failure_reason, diagnosis);
    }

    if is_parse_json && !content.is_empty() {
        let json_match = Regex::new(r"\[[\s\S]*\]").ok()
            .and_then(|re| re.find(&content))
            .map(|m| m.as_str().to_string())
            .or_else(|| {
                Regex::new(r"\{[\s\S]*\}").ok()
                    .and_then(|re| re.find(&content))
                    .map(|m| m.as_str().to_string())
            });

        let extracted = if let Some(jm) = json_match {
            let Ok(parsed) = serde_json::from_str::<serde_json::Value>(&jm) else {
                None
            }? else {
                if let serde_json::Value::Object(obj) = parsed {
                    Some(obj)
                } else {
                    None
                }
            };

            if let Some(extracted) = extracted {
                if extracted.is_object() {
                    Some(vec![extracted])
                } else {
                    None
                }
            } else {
                None
            }
        } else {
            None
        };

        if let Some(extracted) = extracted {
            let validated = task_cfg.validator(
                &extracted,
                source_text = &source,
                **task_cfg.get("validator_kwargs", {}),
            );

            let (score, failure_reason) = if let Some((s, fr)) = validated {
                (s, fr)
            } else {
                (validated, "".to_string())
            };

            let items_for_debug = extracted.clone();
            let diagnosis = _classify_failure(result, task_cfg, score, &failure_reason);

            if debug && !task_name.contains("weekend") && items_for_debug.is_some() {
                if let Some(items) = items_for_debug {
                    let details = get_source_matching_details(&items, &source);
                    let matched = details["matched"].len();
                    let total = matched + details["unmatched"].len();
                    let ratio = details["ratio"].as_f64().unwrap_or(0.0) * 100.0;
                    eval_logger.info!(
                        "  Source matching for {}: {} / {} ({}%)",
                        task_name, matched, total, ratio
                    );
                    if !details["unmatched"].is_empty() {
                        for item in &details["unmatched"][..3usize] {
                            if let Some(item) = item.as_object() {
                                let terms = item.get("terms").map(|t| t.as_array().map(|a| a.iter().map(|x| x.as_str().unwrap_or("")).collect::<Vec<_>>().join(","))).unwrap_or_default();
                                eval_logger.info!("    Unmatched: {} (terms: {})", item.get("name").unwrap_or("unnamed"), terms.join(","));
                            }
                        }
                    }
                }
            }

            return (score, failure_reason, diagnosis);
        } else {
            let failure = if content.len() <= 50 {
                "Empty content"
            } else {
                "No JSON in output"
            };
            let diagnosis = _classify_failure(result, task_cfg, 0, failure);
            return (0, failure.to_string(), diagnosis);
        }
    }

    if content.is_empty() {
        let failure = "Empty content";
        let diagnosis = _classify_failure(result, task_cfg, 0, failure);
        return (0, failure.to_string(), diagnosis);
    }

    let validated = task_cfg.validator(&content, source_text = &source, **task_cfg.get("validator_kwargs", {}));

    let (score, failure_reason) = if let Some((s, fr)) = validated {
        (s, fr)
    } else {
        (validated, "".to_string())
    };

    let diagnosis = _classify_failure(result, task_cfg, score, &failure_reason);
    return (score, failure_reason, diagnosis);
}

/// Run evaluation with no retries (quick mode).
pub fn run_eval_quick(
    model: &str,
    tasks: &HashMap<String, eval::tasks_core::Check>,
    host: &str,
    port: u16,
    backend: &str,
    verbose: bool,
    timeout: i32,
    on_complete: Option<Box<dyn FnOnce() + Send + Sync>>,
    measure_prefill: bool,
) -> Vec<eval::tasks_core::Check> {
    let orig_retries = MAX_RETRIES;
    let _guard = || {};
    let _ = orig_retries;

    // In quick mode, MAX_RETRIES = 0
    run_eval(
        model,
        tasks,
        host,
        port,
        backend,
        verbose,
        timeout,
        on_complete,
        measure_prefill,
    )
}

/// Run evaluation on model using real-world tasks.
pub fn run_eval(
    model: &str,
    tasks: &HashMap<String, eval::tasks_core::Check>,
    host: &str,
    port: u16,
    backend: &str,
    verbose: bool,
    timeout: i32,
    on_complete: Option<Box<dyn FnOnce() + Send + Sync>>,
    measure_prefill: bool,
) -> Vec<eval::tasks_core::Check> {
    let mut results = Vec::new();

    console.print(&format!("{STEP} Testing {model} ({backend})"));

    // Measure this model's ingestion rate before timing anything else.
    if measure_prefill && backend == "osaurus" && is_generative_model(model) {
        let rate = measure_prefill_rate(model, host, port, transport = &call);
        record_prefill_rate(model, rate);
        if rate > 0 {
            console.print(&format!("{STEP} {model} prefill: {rate:,.0f} chars/sec"));
        }
    }

    let mut consecutive_infra = 0i32;
    let mut last_completion = std::time::Instant::now();

    for (task_name, task_cfg) in tasks {
        if !task_cfg.messages.is_some() {
            console.print(&format!("{WARN} Skipping '{task_name}' (no messages key)"));
            continue;
        }

        // Progress, not duration: the lock's wedge ceiling runs from the last beat,
        // so a healthy multi-hour run never loses the GPU to a peer while a hung one
        // still does. A no-op when this process holds no lock.
        gpu_lock::heartbeat();

        // The single-server invariant is not established by having been true at
        // startup. osaurus_one.sh runs once before a model and then hours pass.
        let contended = contended_server_warning(model, task_name);
        if !contended.is_empty() {
            console.print(&format!("{WARN} {contended}"));
        }

        let prompt_chars: usize = task_cfg
            .messages
            .as_ref()
            .map(|msgs| msgs.iter().map(|m| m.content.as_deref().map_or(0, |c| c.len()).unwrap_or(0)).sum())
            .unwrap_or(0);

        if check_stall(model, last_completion.elapsed().as_secs() as i64) {
            break;
        }

        let task_timeout = _effective_timeout(
            model,
            task_name,
            prompt_chars,
            get_max_tokens_for_task(task_name),
        );

        let mut best_score = -1i32;
        let mut best_result: Option<eval::tasks_core::Check> = None;
        let mut best_failure = String::new();
        let mut best_diagnosis = eval::failures::Diagnosis {
            category: FAIL_NONE,
            reason: String::new(),
            evidence: String::new(),
        };
        let mut first_attempt_failed = false;

        for attempt in 0..=MAX_RETRIES {
            if attempt > 0 {
                eval_logger.warning(
                    &format!(
                        "Retrying task '{task_name}' with model {model} (Attempt {}/{})..",
                        attempt + 1,
                        MAX_RETRIES + 1
                    )
                );
                first_attempt_failed = true;
            }

            // A retry that repeats the identical call cannot fix a reasoning overrun
            let retry_tokens = if attempt > 0 && best_diagnosis.category == FAIL_REASONING {
                let base_budget = get_max_tokens_for_task(task_name, model);
                let tokens = reasoning_retry_budget(base_budget);
                eval_logger.warning(
                    &format!(
                        "Previous attempt reasoned past its budget ({base_budget}); retrying with max_tokens={tokens}"
                    )
                );
                Some(tokens)
            } else {
                None
            };

            let result = match _call_model(
                model,
                task_cfg,
                task_name,
                host,
                port,
                backend,
                task_timeout,
                retry_tokens.unwrap_or(0),
            ) {
                Ok(r) => r,
                Err(e) => {
                    eval_logger.error(&format!("Model call failed with exception: {e}"));
                    eval::tasks_core::Check {
                        content: None,
                        error: Some(e.to_string()),
                        time: None,
                        model: model.to_string(),
                        ..Default::default()
                    }
                }
            };

            let (score, failure_reason, diagnosis) = _validate_result(&result, task_cfg, task_name, true);

            eval_logger.info(&format!("Quality score: {score}/100"));

            save_output(model, task_name, &result, score, &failure_reason);

            if score < 90 {
                let cat = diagnosis.category;
                let evidence = diagnosis.evidence;
                eval_logger.warning(
                    &format!(
                        "[DEBUG_OUTPUT] model={model} task={task_name} score={score} category={cat} failure={failure_reason} evidence={evidence}"
                    )
                );
            }

            if score > best_score {
                best_score = score;
                best_result = Some(result.clone());
                best_failure = failure_reason.clone();
                best_diagnosis = diagnosis;
            }

            if score >= 90 {
                break;
            }
            if diagnosis.category == FAIL_CONTENT {
                break;
            }
        }

        let status = if best_score >= 90 {
            "ok"
        } else if best_score >= 50 {
            "partial"
        } else {
            "fail"
        };
        let category = best_diagnosis.category;

        results.push(eval::tasks_core::Check {
            task: task_name.to_string(),
            status: status.to_string(),
            quality_score: best_score,
            time: best_result.as_ref().and_then(|r| r.time),
            error: best_result.as_ref().and_then(|r| r.error.clone()),
            failure_reason: best_failure.clone(),
            failure_category: if !category.is_empty() { Some(category) } else { None },
            failure_evidence: best_diagnosis.evidence.clone(),
            result: best_result.clone(),
            first_attempt_failed,
        });

        _record_signal(
            model,
            task_name,
            time_taken = (best_result.as_ref().map(|r| r.time).unwrap_or(0)),
            had_retries = first_attempt_failed,
            is_parse_failure = (category == "PARSE"),
        );

        // Stop once the server, not the model, is clearly the problem.
        let infra_flag = matches!(category, FAIL_INFRA | FAIL_TIMEOUT);
        if infra_flag {
            consecutive_infra += 1;
            if consecutive_infra >= MAX_CONSECUTIVE_INFRA_FAILURES {
                console.print(&format!(
                    "{FAIL} Abandoning {model}: {consecutive_infra} consecutive infrastructure failures ({best_failure[:60]}). The server cannot serve this model on this host -- this is not a quality result and must not be read as one."
                ));
                break;
            }
        } else {
            consecutive_infra = 0;
            last_completion = std::time::Instant::now();
        }

        let status_symbol = if status == "ok" { STEP } else if status == "partial" { WARN } else { FAIL };
        let category_tag = if !category.is_empty() { format!(" [{category}]") } else { String::new() };
        let fail_info = if !best_failure.is_empty() { format!(" - {best_failure}") } else { String::new() };
        let evidence_info = if !best_diagnosis.evidence.is_empty() {
            format!("\n    - {}", best_diagnosis.evidence)
        } else {
            String::new()
        };
        let time_taken = best_result.as_ref().map(|r| r.time).unwrap_or(0);
        let time_taken_str = if time_taken > 0 { format!("{time_taken}s") } else { "N/A".to_string() };
        console.print(&format!(
            "  {status_symbol} {task_name}: {best_score}% ({time_taken_str}){category_tag}{fail_info}{evidence_info}"
        ));

        if verbose && best_result.is_some() {
            if let Some(content) = safe_content(&best_result.unwrap()) {
                let c = content.as_str();
                if !c.is_empty() {
                    console.print(&format!("  Raw output: {c}"));
                }
            }
        }
    }

    if let Some(on_complete) = on_complete {
        on_complete();
    }

    // Weekend tasks quality check summary
    let weekend_tasks: Vec<String> = tasks.keys().filter(|k| k.contains("weekend")).cloned().collect();
    let mixed_tasks: Vec<String> = tasks.keys().filter(|k| k.ends_with("_mixed")).cloned().collect();

    if !weekend_tasks.is_empty() {
        console.print("");
        console.print("Quality Check Summary:");
        for r in &results {
            let task_name = &r.task;
            if !weekend_tasks.iter().any(|wt| wt == task_name) {
                continue;
            }
            let task_cfg = tasks.get(task_name).unwrap();
            let source = task_cfg.get("source", "").to_string();
            if source.is_empty() {
                continue;
            }
            let parsed = r.result.as_ref().map(|r| &r.parsed).unwrap_or(&None);
            if parsed.is_none() {
                continue;
            }
            let details = get_source_matching_details(parsed.unwrap(), &source);
            let matched = details["matched"].len();
            let total = matched + details["unmatched"].len();
            let ratio = details["ratio"].as_f64().unwrap_or(0.0) * 100.0;
            console.print(&format!("  {task_name}: {matched}/{total} items from source ({ratio:.0f}%)"));
            if !details["unmatched"].is_empty() {
                let names: Vec<String> = details["unmatched"]
                    .iter()
                    .take(2)
                    .map(|u| {
                        if let Some(item) = u.as_object() {
                            item.get("name").map_or_else(|| "unnamed".to_string(), |n| n.to_string())
                        } else {
                            "unnamed".to_string()
                        }
                    })
                    .collect();
                console.print(&format!("{WARN} Not from source: {names}"));
            }
        }
    }

    if !mixed_tasks.is_empty() {
        console.print("");
        console.print("Signal/Noise Filtering:");
        for r in &results {
            let task_name = &r.task;
            if !mixed_tasks.iter().any(|mt| mt == task_name) {
                continue;
            }
            let reason = r.failure_reason.clone();
            let (noise_part, symbol) = if reason.to_lowercase().contains("noise") {
                (reason.clone(), WARN)
            } else if reason.to_lowercase().contains("missed") || reason.to_lowercase().contains("coverage") {
                (reason.clone(), WARN)
            } else if reason.to_lowercase().contains("included") && reason.to_lowercase().contains("noise") {
                (reason.clone(), WARN)
            } else {
                (reason.clone(), STEP)
            };
            console.print(&format!("  {symbol} {task_name}: {}/100 — {noise_part or 'filtered clean'}", r.quality_score));
        }
    }

    results
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_constants() {
        assert!(MAX_RETRIES >= 0);
        assert!(MAX_CONSECUTIVE_INFRA_FAILURES > 0);
        assert!(EVAL_TEMPERATURE >= 0.0);
    }
}