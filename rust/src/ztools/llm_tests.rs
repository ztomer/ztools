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

#[test]
fn a_prompt_that_overflows_documented_window_is_refused_before_wire() {
    let env = crate::test_env::TestEnv::new();
    let shipped_conf = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("CARGO_MANIFEST_DIR is <repo>/rust")
        .join("conf");
    env.set_managed("ZTOOLS_CONF_DIR", shipped_conf.as_os_str());

    let huge_user = "a".repeat(25_000);
    let req = ChatRequest {
        base_url: "http://127.0.0.1:1", // unreachable, must not even be dialed
        model: "foundation",
        system: None,
        user: &huge_user,
        json: false,
    };
    let err = chat(&req, &budget()).unwrap_err();
    assert!(
        err.to_string().contains("prompt does not fit"),
        "expected context refusal, got: {err}"
    );
    assert!(
        err.to_string()
            .contains("foundation's whole context window is 4096"),
        "expected window mentioned, got: {err}"
    );
    drop(env);
}

/// The default call stays greedy and sends no penalty key at all: a penalty is
/// a per-task decision, and a JSON extract must not inherit one.
#[test]
fn the_default_call_is_greedy_with_no_penalty_key() {
    let (url, rx) = serve(SSE, "text/event-stream", Duration::ZERO);
    chat(&request(&url), &budget()).unwrap();
    let wire = rx.recv().unwrap();
    let body = wire.split("\r\n\r\n").nth(1).unwrap_or("");
    let payload: serde_json::Value = serde_json::from_str(body).unwrap();
    assert_eq!(payload["temperature"], 0.0);
    assert!(
        payload.get("frequency_penalty").is_none(),
        "a default call must not carry a penalty: {payload}"
    );
}

/// `chat_with` puts its decoding policy on the wire under the key the server
/// decodes.
#[test]
fn chat_with_sends_its_temperature_and_frequency_penalty() {
    let (url, rx) = serve(SSE, "text/event-stream", Duration::ZERO);
    let sampling = Sampling {
        temperature: 0.25,
        frequency_penalty: Some(0.5),
    };
    chat_with(&request(&url), &budget(), &sampling).unwrap();
    let wire = rx.recv().unwrap();
    let body = wire.split("\r\n\r\n").nth(1).unwrap_or("");
    let payload: serde_json::Value = serde_json::from_str(body).unwrap();
    assert_eq!(payload["temperature"], 0.25);
    assert_eq!(payload["frequency_penalty"], 0.5);
}

/// A cut answer is not an answer. On 2026-10-10 a summary of 45 tweets came
/// back as 8 bullets ending mid-citation (`(@AION2Official | Sat Oct 10
/// 336:51 +...`) and was saved as complete, because nothing here read
/// `finish_reason` or noticed a stream that ended without `[DONE]`.
#[test]
fn an_answer_stopped_by_the_token_limit_is_an_error() {
    const CUT: &str = "data: {\"choices\":[{\"delta\":{\"content\":\"- one (@a | t\"}}]}\n\n\
data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"length\"}]}\n\n\
data: [DONE]\n\n";
    let (url, _rx) = serve(CUT, "text/event-stream", Duration::ZERO);
    let err = chat(&request(&url), &budget()).unwrap_err().to_string();
    assert!(err.contains("max_tokens"), "{err}");

    let (url, _rx) = serve(
        r#"{"choices":[{"message":{"content":"- one (@a | t"},"finish_reason":"length"}]}"#,
        "application/json",
        Duration::ZERO,
    );
    let err = chat(&request(&url), &budget()).unwrap_err().to_string();
    assert!(err.contains("max_tokens"), "{err}");
}

#[test]
fn a_stream_that_ends_without_done_or_a_finish_reason_is_an_error() {
    const DROPPED: &str = "data: {\"choices\":[{\"delta\":{\"content\":\"- one (@a | t\"}}]}\n\n";
    let (url, _rx) = serve(DROPPED, "text/event-stream", Duration::ZERO);
    let err = chat(&request(&url), &budget()).unwrap_err().to_string();
    assert!(err.contains("ended mid-answer"), "{err}");
}
