//! Eval transport: OpenAI-compatible chat-completions client for the eval loop.
//!
//! Ported from `lib/osaurus_lib.py::call` and
//! `lib/llm/streaming.py::stream_with_overrun_guard`. Two entry points:
//!
//! - [`call`] — blocking request; cleans output and optionally parses JSON.
//! - [`stream_with_overrun_guard`] — streamed request that ABORTS a reasoning
//!   overrun as soon as it is certain: a model that has spent
//!   [`REASONING_OVERRUN_FRACTION`] of its token budget thinking while producing
//!   NO content cannot finish an answer, so the remaining spend is wasted.
//!   Below that line, think as long as it likes.
//!
//! Not ported (yet): model-quirk application, missing-model substitution, and
//! the Foundation on-device fallback. A transport error here is reported in the
//! result, never raised — a failed request during a sweep must not end the sweep.

use std::io::{BufRead, BufReader};
use std::time::{Duration, Instant};

use serde_json::{json, Value};

use crate::ztools::eval::clean::{clean_model_output, extract_json};
use crate::ztools::eval::task_loader::ChatMessage;

/// Fraction of the output budget a model may spend on reasoning before an empty
/// `content` is treated as terminal. Below this line a run can still recover;
/// above it the remaining budget cannot hold an answer.
pub const REASONING_OVERRUN_FRACTION: f64 = 0.75;

/// Rough chars-per-token, only used to turn streamed characters into a token
/// estimate for the fraction above. An estimate is adequate because the
/// threshold is a fraction of a budget, not a boundary anything is scored against.
pub const CHARS_PER_TOKEN: u64 = 3;

/// The result shape every caller of the Python `call` receives, carried over:
/// errors are DATA here, not exceptions.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct TransportResult {
    pub model: String,
    pub content: String,
    /// Chain-of-thought text reasoning models stream separately from `content`.
    /// Carrying both lets a caller tell "thought until the budget ran out" from
    /// "returned nothing" -- two failures with different remedies.
    pub reasoning_content: String,
    pub finish_reason: String,
    pub parsed: Option<Value>,
    pub error: Option<String>,
    pub time_secs: f64,
    pub aborted: bool,
    pub abort_reason: String,
}

fn base_url(host: &str, port: u16) -> String {
    if host.contains("://") {
        host.trim_end_matches('/').to_string()
    } else {
        format!("http://{host}:{port}")
    }
}

fn client(timeout_secs: u64) -> reqwest::blocking::Client {
    reqwest::blocking::Client::builder()
        .timeout(Duration::from_secs(timeout_secs))
        .build()
        .expect("reqwest client builds with static config")
}

/// One endpoint description shared by both entry points, so adding a knob
/// cannot mean growing another eight-argument signature.
#[derive(Debug, Clone)]
pub struct RequestSpec<'a> {
    pub model: &'a str,
    pub messages: &'a [ChatMessage],
    pub host: &'a str,
    pub port: u16,
    pub temperature: f64,
    pub max_tokens: u32,
    pub timeout_secs: u64,
}

/// Blocking chat completion. Returns a [`TransportResult`] whose `error` is
/// `Some` on any transport or protocol failure.
///
/// With `parse_json`, the request asks for `response_format: json_object` and
/// the cleaned content is additionally run through [`extract_json`].
pub fn call(spec: &RequestSpec, parse_json: bool) -> TransportResult {
    let mut result = TransportResult {
        model: spec.model.to_string(),
        ..Default::default()
    };
    let start = Instant::now();
    let mut payload = json!({
        "model": spec.model,
        "messages": spec.messages,
        "temperature": spec.temperature,
        "max_tokens": spec.max_tokens,
    });
    if parse_json {
        payload["response_format"] = json!({"type": "json_object"});
    }

    let url = format!("{}/v1/chat/completions", base_url(spec.host, spec.port));
    let response = match client(spec.timeout_secs).post(&url).json(&payload).send() {
        Ok(r) => r,
        Err(e) if e.is_timeout() => {
            result.time_secs = start.elapsed().as_secs_f64();
            result.error = Some("Timeout".to_string());
            return result;
        }
        Err(_) => {
            result.time_secs = start.elapsed().as_secs_f64();
            result.error = Some("Connection failed - is server running?".to_string());
            return result;
        }
    };

    let status = response.status();
    let body = match response.text() {
        Ok(b) => b,
        Err(e) => {
            result.time_secs = start.elapsed().as_secs_f64();
            result.error = Some(format!("Error: read body: {e}"));
            return result;
        }
    };
    result.time_secs = (start.elapsed().as_secs_f64() * 10.0).round() / 10.0;

    if status != reqwest::StatusCode::OK {
        // Truncated like the Python ERROR_TRUNCATE_LEN path: a huge error page
        // must not flood a sweep log.
        result.error = Some(format!(
            "HTTP {}: {}",
            status.as_u16(),
            &body[..body.len().min(500)]
        ));
        return result;
    }

    let data: Value = match serde_json::from_str(&body) {
        Ok(v) => v,
        Err(e) => {
            result.error = Some(format!("Invalid JSON response: {e}"));
            return result;
        }
    };
    let Some(choice) = data.get("choices").and_then(|c| c.get(0)) else {
        result.error = Some("Empty response from API".to_string());
        return result;
    };
    let message = choice.get("message").cloned().unwrap_or(Value::Null);
    let raw_content = message
        .get("content")
        .and_then(|c| c.as_str())
        .unwrap_or("")
        .to_string();
    result.content = clean_model_output(&raw_content);
    result.reasoning_content = message
        .get("reasoning_content")
        .and_then(|c| c.as_str())
        .unwrap_or("")
        .to_string();
    result.finish_reason = choice
        .get("finish_reason")
        .and_then(|c| c.as_str())
        .unwrap_or("")
        .to_string();
    if parse_json && !raw_content.is_empty() {
        result.parsed = extract_json(&raw_content);
    }
    result
}

/// One SSE `data:` line parsed into the choice object it carries, or `None`
/// when the line carries no data (keep-alives, `[DONE]`, unparsable payloads).
fn sse_choice(line: &str) -> Option<Value> {
    let payload = line.strip_prefix("data: ")?.trim();
    if payload.is_empty() || payload == "[DONE]" {
        return None;
    }
    let chunk: Value = serde_json::from_str(payload).ok()?;
    chunk.get("choices")?.get(0).cloned()
}

/// Streamed completion with the reasoning-overrun guard.
///
/// Aborts ONLY when `content` is still empty and `reasoning_content` has passed
/// [`REASONING_OVERRUN_FRACTION`] of the token budget: content having arrived
/// means the think block closed, so however long the model thought, it is
/// answering and must be left alone. An abort is recorded in the result
/// (`aborted`, `abort_reason`, `finish_reason = "aborted_reasoning_overrun"`),
/// never swallowed.
///
/// The wall-clock deadline is enforced HERE rather than trusted to socket
/// timeouts: a model emitting one slow token at a time never trips a per-read
/// gap timeout, and that exact case hung a real sweep for 97 minutes.
pub fn stream_with_overrun_guard(spec: &RequestSpec) -> TransportResult {
    let mut result = TransportResult {
        model: spec.model.to_string(),
        ..Default::default()
    };
    let start = Instant::now();
    let deadline = start + Duration::from_secs(spec.timeout_secs);
    let budget_chars = (spec.max_tokens as f64
        * REASONING_OVERRUN_FRACTION
        * CHARS_PER_TOKEN as f64)
        .max(1.0) as u64;

    let payload = json!({
        "model": spec.model,
        "messages": spec.messages,
        "temperature": spec.temperature,
        "max_tokens": spec.max_tokens,
        "stream": true,
    });
    let url = format!("{}/v1/chat/completions", base_url(spec.host, spec.port));
    let response = match client(spec.timeout_secs)
        .post(&url)
        .header("Accept", "text/event-stream")
        .json(&payload)
        .send()
    {
        Ok(r) => r,
        Err(e) if e.is_timeout() => {
            result.error = Some("Timeout".to_string());
            return result;
        }
        Err(_) => {
            result.error = Some("Connection failed".to_string());
            return result;
        }
    };

    if response.status() != reqwest::StatusCode::OK {
        result.error = Some(format!("HTTP {}", response.status().as_u16()));
        return result;
    }

    let reader = BufReader::new(response);
    for line in reader.lines() {
        if Instant::now() > deadline {
            // Distinct from the overrun abort: this one says "no answer within
            // the time allowed" and is classed INFRA, not a quality failure.
            result.error = Some(format!("Timeout after {}s (streamed)", spec.timeout_secs));
            result.finish_reason = "stream_deadline_exceeded".to_string();
            return result;
        }
        let line = match line {
            Ok(l) => l,
            Err(e) => {
                result.error = Some(format!("Error: read stream: {e}"));
                return result;
            }
        };
        let Some(choice) = sse_choice(&line) else {
            continue;
        };
        let delta = choice.get("delta").cloned().unwrap_or(Value::Null);
        result.content += delta.get("content").and_then(|c| c.as_str()).unwrap_or("");
        result.reasoning_content += delta
            .get("reasoning_content")
            .and_then(|c| c.as_str())
            .unwrap_or("");
        if let Some(fr) = choice.get("finish_reason").and_then(|c| c.as_str()) {
            if !fr.is_empty() {
                result.finish_reason = fr.to_string();
            }
        }

        // The only abort condition.
        if result.content.is_empty() && result.reasoning_content.len() as u64 > budget_chars {
            let spent = result.reasoning_content.len() as u64 / CHARS_PER_TOKEN;
            result.aborted = true;
            result.finish_reason = "aborted_reasoning_overrun".to_string();
            result.abort_reason = format!(
                "~{spent} tokens of reasoning with no content, past \
                 75% of a {}-token budget: the rest cannot hold an answer",
                spec.max_tokens
            );
            break;
        }
    }
    result.time_secs = (start.elapsed().as_secs_f64() * 10.0).round() / 10.0;
    result
}
