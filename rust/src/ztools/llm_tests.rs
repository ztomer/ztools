use super::*;

use std::io::{Read, Write};
use std::net::TcpListener;

/// A loopback server that answers one request with `body` after `delay`,
/// with whatever `content_type` it is told to claim. Returns the base URL
/// and the request it received, once one arrives.
fn serve(
    body: &'static str,
    content_type: &'static str,
    delay: Duration,
) -> (String, std::sync::mpsc::Receiver<String>) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let (mut stream, _) = listener.accept().unwrap();
        let mut buf = vec![0u8; 65536];
        let n = stream.read(&mut buf).unwrap_or(0);
        let _ = tx.send(String::from_utf8_lossy(&buf[..n]).into_owned());
        std::thread::sleep(delay);
        let resp = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\n\r\n{body}",
            body.len()
        );
        let _ = stream.write_all(resp.as_bytes());
    });
    (format!("http://{addr}"), rx)
}

fn budget() -> ChatBudget {
    ChatBudget {
        stall_secs: 2,
        cap_secs: 10,
        max_tokens: 64,
    }
}

fn request(base_url: &str) -> ChatRequest<'_> {
    ChatRequest {
        base_url,
        model: "m",
        system: None,
        user: "hello",
        json: false,
    }
}

const SSE: &str = "data: {\"choices\":[{\"delta\":{\"reasoning_content\":\"thinking...\"}}]}\n\n\
data: {\"choices\":[{\"delta\":{\"content\":\"Hel\"}}]}\n\n\
data: {\"choices\":[{\"delta\":{\"content\":\"lo\"},\"finish_reason\":\"stop\"}]}\n\n\
data: [DONE]\n\n";

#[test]
fn streamed_deltas_are_joined_and_reasoning_is_not_the_answer() {
    let (url, _rx) = serve(SSE, "text/event-stream", Duration::ZERO);
    let answer = chat(&request(&url), &budget()).unwrap();
    assert_eq!(answer, "Hello");
}

#[test]
fn a_plain_completion_body_is_read_too() {
    let (url, _rx) = serve(
        r#"{"choices":[{"message":{"content":"plain answer"}}]}"#,
        "application/json",
        Duration::ZERO,
    );
    assert_eq!(chat(&request(&url), &budget()).unwrap(), "plain answer");
}

/// The wire carries the four properties the module exists for: streaming,
/// thinking off, a bounded answer, and the JSON mode when asked.
#[test]
fn the_request_streams_with_thinking_off_and_a_token_bound() {
    let (url, rx) = serve(SSE, "text/event-stream", Duration::ZERO);
    let req = ChatRequest {
        system: Some("be terse"),
        json: true,
        ..request(&url)
    };
    chat(&req, &budget()).unwrap();
    let wire = rx.recv().unwrap();
    let body = wire.split("\r\n\r\n").nth(1).unwrap_or("");
    let payload: serde_json::Value = serde_json::from_str(body).unwrap();
    assert_eq!(payload["stream"], true);
    assert_eq!(payload["enable_thinking"], false);
    assert_eq!(payload["max_tokens"], 64);
    assert_eq!(payload["response_format"]["type"], "json_object");
    assert_eq!(payload["messages"][0]["role"], "system");
    assert_eq!(payload["messages"][1]["content"], "hello");
}

/// A server that accepts and then says nothing is a stalled server. The
/// verdict arrives after the stall budget, not the cap — and not never.
#[test]
fn a_silent_server_fails_on_the_stall_budget() {
    let (url, _rx) = serve(SSE, "text/event-stream", Duration::from_secs(6));
    let started = std::time::Instant::now();
    let err = chat(&request(&url), &budget()).unwrap_err();
    let elapsed = started.elapsed();
    assert!(
        elapsed >= Duration::from_secs(2) && elapsed < Duration::from_secs(5),
        "expected the 2s stall guard, got {elapsed:?}: {err}"
    );
}

#[test]
fn an_error_body_is_an_error_not_an_empty_answer() {
    let (url, _rx) = serve(
        r#"{"error":{"message":"Model 'm' is not installed"}}"#,
        "application/json",
        Duration::ZERO,
    );
    let err = chat(&request(&url), &budget()).unwrap_err().to_string();
    assert!(err.contains("not installed"), "{err}");
}

#[test]
fn an_unreachable_host_is_an_error() {
    let err = chat(&request("http://127.0.0.1:1"), &budget()).unwrap_err();
    assert!(err.to_string().contains("Failed to send request"), "{err}");
}
