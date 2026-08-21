//! Integration tests for `eval/transport.rs` against mock HTTP servers.
//!
//! Covers the code a unit test cannot reach: the actual wire format of the
//! blocking call and the SSE stream, the reasoning-overrun abort, and the
//! wall-clock stream deadline.

use std::io::{Read, Write};
use std::net::TcpListener;
use std::thread;

use ztools::eval::task_loader::ChatMessage;
use ztools::eval::transport::{call, stream_with_overrun_guard, RequestSpec};

fn sse_body(deltas: &[&str]) -> String {
    let mut body = String::new();
    for d in deltas {
        body.push_str(&format!("data: {d}\n\n"));
    }
    body.push_str("data: [DONE]\n\n");
    body
}

fn http_response(content_type: &str, body: &str) -> String {
    format!(
        "HTTP/1.1 200 OK\r\nContent-Type: {content_type}\r\nConnection: close\r\nContent-Length: {}\r\n\r\n{body}",
        body.len()
    )
}

/// Server that answers every connection with a canned response.
fn serve(response: String) -> (u16, thread::JoinHandle<()>) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    let handle = thread::spawn(move || {
        for stream in listener.incoming() {
            let mut stream = match stream {
                Ok(s) => s,
                Err(_) => continue,
            };
            let mut buf = [0u8; 8192];
            let _ = stream.read(&mut buf);
            let _ = stream.write_all(response.as_bytes());
            let _ = stream.flush();
        }
    });
    thread::sleep(std::time::Duration::from_millis(50));
    (port, handle)
}

fn msgs() -> Vec<ChatMessage> {
    vec![ChatMessage::user("Output JSON now.")]
}

fn spec(port: u16, max_tokens: u32, timeout_secs: u64) -> RequestSpec<'static> {
    RequestSpec {
        model: "m",
        // Leaked per test invocation: a few messages per test, never freed.
        messages: Box::leak(msgs().into_boxed_slice()),
        host: "127.0.0.1",
        port,
        temperature: 0.0,
        max_tokens,
        timeout_secs,
    }
}

#[test]
fn blocking_call_extracts_content_and_parses_json() {
    let content = r#"{"transient_events":[{"name":"Rib Fest"}]}"#;
    let body = format!(
        r#"{{"choices":[{{"message":{{"content":"{}"}},"finish_reason":"stop"}}]}}"#,
        content.replace('"', "\\\"")
    );
    let (port, _h) = serve(http_response("application/json", &body));
    let spec = spec(port, 100, 10);
    let r = call(&spec, true);
    assert_eq!(r.error, None, "{r:?}");
    assert!(r.content.contains("Rib Fest"), "{r:?}");
    assert_eq!(r.finish_reason, "stop");
    assert!(r.parsed.is_some(), "parse_json should populate parsed: {r:?}");
    assert!(r.time_secs >= 0.0);
}

#[test]
fn blocking_call_reports_http_error_as_data() {
    let body = r#"{"error":"at capacity"}"#;
    let (port, _h) = serve(format!(
        "HTTP/1.1 503 Service Unavailable\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    ));
    let spec = spec(port, 100, 10);
    let r = call(&spec, false);
    assert!(
        r.error.as_deref().unwrap_or("").starts_with("HTTP 503"),
        "{r:?}"
    );
}

#[test]
fn blocking_call_survives_a_dead_server() {
    // Port 1 on localhost is refused; must come back as an error RESULT,
    // never a panic -- a failed request during a sweep must not end it.
    let spec = RequestSpec { model: "m", messages: &msgs(), host: "127.0.0.1", port: 1, temperature: 0.0, max_tokens: 100, timeout_secs: 2 };
    let r = call(&spec, false);
    assert!(r.error.is_some(), "{r:?}");
}

#[test]
fn blocking_call_flags_empty_choices() {
    let (port, _h) = serve(http_response("application/json", r#"{"choices":[]}"#));
    let spec = spec(port, 100, 10);
    let r = call(&spec, false);
    assert_eq!(r.error.as_deref(), Some("Empty response from API"), "{r:?}");
}

#[test]
fn stream_accumulates_content_and_reasoning() {
    let body = sse_body(&[
        r#"{"choices":[{"delta":{"reasoning_content":"thinking"}}]}"#,
        r#"{"choices":[{"delta":{"content":"the answer"}}]}"#,
        r#"{"choices":[{"delta":{},"finish_reason":"stop"}]}"#,
    ]);
    let (port, _h) = serve(http_response("text/event-stream", &body));
    let r = stream_with_overrun_guard(&spec(port, 1000, 10));
    assert_eq!(r.error, None, "{r:?}");
    assert_eq!(r.reasoning_content, "thinking");
    assert_eq!(r.content, "the answer");
    assert_eq!(r.finish_reason, "stop");
    assert!(!r.aborted);
}

#[test]
fn overrun_guard_aborts_reasoning_past_budget_with_no_content() {
    // max_tokens=10 -> budget_chars = 10 * 0.75 * 3 = 22. Stream 200 chars of
    // reasoning with NO content: the run cannot produce an answer, so the
    // guard must stop it and RECORD why.
    let reasoning = "x".repeat(200);
    let body = sse_body(&[&format!(
        r#"{{"choices":[{{"delta":{{"reasoning_content":"{reasoning}"}}}}]}}"#
    )]);
    let (port, _h) = serve(http_response("text/event-stream", &body));
    let r = stream_with_overrun_guard(&spec(port, 10, 10));
    assert!(r.aborted, "{r:?}");
    assert_eq!(r.finish_reason, "aborted_reasoning_overrun");
    assert!(r.abort_reason.contains("cannot hold an answer"), "{r:?}");
    assert!(r.error.is_none(), "an overrun is an eval result, not an error: {r:?}");
}

#[test]
fn overrun_guard_leaves_a_model_that_is_answering_alone() {
    // Same reasoning volume, but content arrived first: the think block closed,
    // so however long it thought, it must NOT be aborted.
    let body = sse_body(&[
        r#"{"choices":[{"delta":{"content":"answer"}}]}"#,
        &format!(
            r#"{{"choices":[{{"delta":{{"reasoning_content":"{}"}}}}]}}"#,
            "x".repeat(500)
        ),
    ]);
    let (port, _h) = serve(http_response("text/event-stream", &body));
    let r = stream_with_overrun_guard(&spec(port, 10, 10));
    assert!(!r.aborted, "{r:?}");
    assert_eq!(r.content, "answer");
}

#[test]
fn stream_deadline_enforced_in_wall_clock() {
    // Server accepts, sends headers, then stalls past the deadline without
    // closing. The guard must return a TIMEOUT error rather than hang.
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    let _h = thread::spawn(move || {
        if let Ok(mut stream) = listener.incoming().next().unwrap() {
            let mut buf = [0u8; 8192];
            let _ = stream.read(&mut buf);
            let _ = stream.write_all(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n\r\n");
            let _ = stream.flush();
            thread::sleep(std::time::Duration::from_secs(5));
        }
    });
    thread::sleep(std::time::Duration::from_millis(50));
    let started = std::time::Instant::now();
    let r = stream_with_overrun_guard(&spec(port, 1000, 1));
    assert!(
        r.error.as_deref().unwrap_or("").contains("Timeout"),
        "{r:?}"
    );
    assert_eq!(r.finish_reason, "stream_deadline_exceeded");
    assert!(started.elapsed().as_secs() < 4, "must not wait out the stall");
}
