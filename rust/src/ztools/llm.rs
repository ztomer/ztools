//! The one chat-completions client the production tools call the server
//! through: streaming, stall-guarded, thinking off, output bounded.
//!
//! Why one, and why these four properties. On 2026-09-19 the weekend planner
//! was measured end to end against a healthy server and produced nothing in
//! 22 minutes. The chain of causes, each of which this module closes:
//!
//! - The request carried no `max_tokens`, and `qwen3.8` REASONS before it
//!   answers: a two-line extract spent 1,720 characters thinking; an eight-line
//!   one thought past the 300s timeout. `enable_thinking: false` (a top-level
//!   field this server honours — measured: `pong` in 3s with zero reasoning
//!   characters, the same extract in 11s) removes the spiral; `max_tokens`
//!   bounds whatever is left server-side.
//! - The client timed out by WALL CLOCK and disconnected. The server does not
//!   free itself when a client leaves: the abandoned generation kept the GPU,
//!   the next request queued behind it, timed out too, and after the third the
//!   server no longer answered `pong` at all (0% CPU, 60s, nothing). A timeout
//!   that abandons a serial server's request does not fail one call, it wedges
//!   every call after it. So this client STREAMS and gives up only on a
//!   STALL — no token for `stall_secs` — while tokens that keep arriving keep
//!   it waiting, up to a hard cap that exists as a backstop, not a budget.
//!
//! The response is read in either wire shape: SSE (`data:` lines, what the
//! real server sends for `stream: true`) or one JSON completion (what every
//! loopback stub in the tests sends, and what a server that ignores `stream`
//! would send). Content only — `reasoning_content` is never the answer.

use std::io::{BufRead, BufReader};
use std::sync::mpsc;
use std::time::Duration;

use anyhow::Result;

/// One request: where, which model, what to say, and whether the answer
/// must be a JSON object.
#[derive(Debug, Clone)]
pub struct ChatRequest<'a> {
    pub base_url: &'a str,
    pub model: &'a str,
    pub system: Option<&'a str>,
    pub user: &'a str,
    pub json: bool,
}

/// How long one call may run, and how large its answer may be.
///
/// `stall_secs` is the guard: no byte of answer for this long means the
/// server is not generating, and waiting longer only delays the verdict.
/// `cap_secs` is the backstop on a call whose tokens keep flowing; it is
/// reached only by a runaway, because `max_tokens` bounds the answer first.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChatBudget {
    pub stall_secs: u64,
    pub cap_secs: u64,
    pub max_tokens: u32,
}

/// What the request thread hands back, in order: the status, then lines.
enum Event {
    SendFailed(String),
    Status(reqwest::StatusCode),
    Line(String),
    ReadFailed(String),
}

/// Send one chat request and return the answer's `content`.
///
/// # Errors
///
/// When the request cannot be sent, the stream stalls past `stall_secs`
/// (or the whole call past `cap_secs`), or the body is neither an SSE stream
/// nor a chat completion. The raw body is included in a parse error, because
/// the usual cause is a server answering something other than the API.
pub fn chat(req: &ChatRequest<'_>, budget: &ChatBudget) -> Result<String> {
    let client = reqwest::blocking::Client::builder()
        .timeout(Duration::from_secs(budget.cap_secs))
        .build()?;
    let url = format!("{}/v1/chat/completions", req.base_url.trim_end_matches('/'));

    let mut messages = Vec::new();
    if let Some(system) = req.system {
        messages.push(serde_json::json!({"role": "system", "content": system}));
    }
    messages.push(serde_json::json!({"role": "user", "content": req.user}));
    let mut payload = serde_json::json!({
        "model": req.model,
        "messages": messages,
        "temperature": 0.0,
        "stream": true,
        "max_tokens": budget.max_tokens,
        "enable_thinking": false,
    });
    if req.json {
        payload["response_format"] = serde_json::json!({"type": "json_object"});
    }

    // The stall guard covers the WHOLE exchange, headers included: a wedged
    // server accepts the connection and then sends nothing, and that silence
    // must be judged by the same clock as a stream that stops mid-answer.
    // The blocking client has no per-read timeout, so the request runs on
    // its own thread and hands over the status and then each line; a wait
    // on the channel longer than `stall_secs` is the verdict. The thread is
    // then abandoned to the cap -- it holds nothing but a socket.
    let (tx, rx) = mpsc::channel::<Event>();
    let request = client.post(&url).json(&payload);
    std::thread::spawn(move || {
        let response = match request.send() {
            Ok(r) => r,
            Err(e) => {
                let _ = tx.send(Event::SendFailed(e.to_string()));
                return;
            }
        };
        if tx.send(Event::Status(response.status())).is_err() {
            return;
        }
        let mut reader = BufReader::new(response);
        loop {
            let mut line = String::new();
            match reader.read_line(&mut line) {
                Ok(0) => break,
                Ok(_) => {
                    if tx.send(Event::Line(line)).is_err() {
                        break;
                    }
                }
                Err(e) => {
                    let _ = tx.send(Event::ReadFailed(e.to_string()));
                    break;
                }
            }
        }
    });
    let stall = Duration::from_secs(budget.stall_secs);
    let next = |rx: &mpsc::Receiver<Event>| -> Result<Option<Event>> {
        match rx.recv_timeout(stall) {
            Ok(event) => Ok(Some(event)),
            Err(mpsc::RecvTimeoutError::Timeout) => anyhow::bail!(
                "Osaurus server stalled: no output for {}s",
                budget.stall_secs
            ),
            Err(mpsc::RecvTimeoutError::Disconnected) => Ok(None),
        }
    };
    let status = match next(&rx)? {
        Some(Event::Status(status)) => status,
        Some(Event::SendFailed(e)) => {
            anyhow::bail!("Failed to send request to Osaurus server: {e}")
        }
        _ => anyhow::bail!("Osaurus server closed the connection before answering"),
    };
    let next_line = |rx: &mpsc::Receiver<Event>| -> Result<Option<String>> {
        match next(rx)? {
            Some(Event::Line(line)) => Ok(Some(line)),
            Some(Event::ReadFailed(e)) => {
                anyhow::bail!("Failed to read Osaurus server response text: {e}")
            }
            _ => Ok(None),
        }
    };

    // The first non-empty line says which wire shape this is.
    let mut first = String::new();
    while let Some(line) = next_line(&rx)? {
        if !line.trim().is_empty() {
            first = line;
            break;
        }
    }
    if first.trim_start().starts_with("data:") {
        return read_sse(first.trim(), &rx, next_line);
    }
    let mut raw = first;
    while let Some(line) = next_line(&rx)? {
        raw.push_str(&line);
    }
    if !status.is_success() {
        anyhow::bail!("Osaurus server answered HTTP {status}: {}", raw.trim());
    }
    parse_completion(&raw)
}

/// Read `data:` events until `[DONE]` or end of stream, gathering `content`.
fn read_sse(
    first: &str,
    rx: &mpsc::Receiver<Event>,
    next_line: impl Fn(&mpsc::Receiver<Event>) -> Result<Option<String>>,
) -> Result<String> {
    let mut content = String::new();
    let mut line = first.to_string();
    loop {
        if let Some(data) = line.trim().strip_prefix("data:") {
            let data = data.trim();
            if data == "[DONE]" {
                break;
            }
            if let Ok(event) = serde_json::from_str::<serde_json::Value>(data) {
                if let Some(delta) = event["choices"][0]["delta"]["content"].as_str() {
                    content.push_str(delta);
                }
                // A server that streams an error mid-way says so here rather
                // than in an HTTP status the client has already sent.
                if let Some(message) = event["error"]["message"].as_str() {
                    anyhow::bail!("Osaurus server streamed an error: {message}");
                }
            }
        }
        match next_line(rx)? {
            Some(next) => line = next,
            None => break,
        }
    }
    Ok(content)
}

/// One non-streamed completion body.
fn parse_completion(raw: &str) -> Result<String> {
    let value: serde_json::Value = serde_json::from_str(raw).map_err(|e| {
        anyhow::anyhow!("Failed to parse Osaurus server response JSON: {e}, raw: {raw}")
    })?;
    if let Some(message) = value["error"]["message"].as_str() {
        anyhow::bail!("Osaurus server answered with an error: {message}");
    }
    let content = value["choices"][0]["message"]["content"]
        .as_str()
        .ok_or_else(|| {
            anyhow::anyhow!("Failed to parse Osaurus server response JSON: no choices[0].message.content, raw: {raw}")
        })?;
    Ok(content.to_string())
}

#[cfg(test)]
#[path = "llm_tests.rs"]
mod tests;
