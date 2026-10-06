//! Integration tests for `eval/transport.rs` against mock HTTP servers.
//!
//! Covers the code a unit test cannot reach: the actual wire format of the
//! blocking call and the SSE stream, the reasoning-overrun abort, and the
//! wall-clock stream deadline.
//!
//! Every `thread::sleep` after a `bind` here used to be a guess about the
//! serving thread's scheduling; they are `support::await_stub`, which waits for
//! the condition a client actually needs -- a completed handshake -- with a
//! deadline that names what never happened. The one remaining sleep was not a
//! wait at all (see `stream_deadline_enforced_in_wall_clock`), which is the only
//! reason it survived this long.

#[path = "support/mod.rs"]
mod support;
// The support module is shared by ten test binaries that between them use every
// item in it; one consumer idling on a helper is not a defect, and this
// re-export at the root is what says so to the dead-code pass.
pub use support::*;

use std::fmt::Write as _;
use std::io::{Read, Write};
use std::net::TcpListener;
use std::thread;

use ztools::eval::task_loader::ChatMessage;
use ztools::eval::transport::{RequestSpec, call, stream_with_overrun_guard};

fn sse_body(deltas: &[&str]) -> String {
    let mut body = String::new();
    for d in deltas {
        let _ = write!(body, "data: {d}\n\n");
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
            let Ok(mut stream) = stream else { continue };
            let mut buf = [0u8; 8192];
            // No probe rule here, deliberately: this stub answers every
            // connection the same way, so `await_stub`'s empty probe costs it
            // nothing (calibrated: 9/9 green without one). The two stubs whose
            // behaviour DOES depend on which connection arrived carry the rule,
            // and there it is load-bearing.
            let _ = stream.read(&mut buf);
            let _ = stream.write_all(response.as_bytes());
            let _ = stream.flush();
        }
    });
    await_stub(port);
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
        allow_substitution: false,
        thinking: false,
        stream_guard: false,
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
    assert!(
        r.parsed.is_some(),
        "parse_json should populate parsed: {r:?}"
    );
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
    let spec = RequestSpec {
        model: "m",
        messages: &msgs(),
        host: "127.0.0.1",
        port: 1,
        temperature: 0.0,
        max_tokens: 100,
        timeout_secs: 2,
        allow_substitution: false,
        thinking: false,
        stream_guard: false,
    };
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
    assert!(
        r.error.is_none(),
        "an overrun is an eval result, not an error: {r:?}"
    );
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
    //
    // WHY THE STALL IS A STALL: the stub sends no `Content-Length` and no
    // `Transfer-Encoding`, so the HTTP/1.1 body is delimited by connection close
    // -- which never comes. Calibrated: give the stub a properly framed, complete
    // SSE answer (`Content-Length`, `[DONE]`, `finish_reason: stop`) and this
    // test goes red on `error: None, finish_reason: "stop"`, which is the proof
    // it measures a stall and not merely a body it could not parse.
    //
    // The stall used to be `thread::sleep(5s)` in the stub, which was not a wait
    // for anything: it was an assertion that the client's own 1s deadline fires
    // first, made by sleeping longer and hoping. Worse, "5s" and "1s" are only
    // ordered while the machine is healthy, and on a slow machine the wait itself
    // becomes the thing being measured. The stub now stalls until the test
    // RELEASES it, so no number has to stay ordered: the client must come back
    // from a server that has said it will say nothing, ever.
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    let (release, released) = std::sync::mpsc::channel::<()>();
    let h = thread::spawn(move || {
        for stream in listener.incoming().flatten() {
            let mut stream = stream;
            let mut buf = [0u8; 8192];
            // `await_stub` connects and closes without sending, and this stub
            // handles exactly ONE connection before it stops: a probe taken as
            // that one leaves the client's real request unanswered in the
            // backlog, so the client times out on a socket nobody is serving and
            // reports `finish_reason: ""` -- a different failure wearing this
            // test's name. Calibrated: removing this rule turns the test red on
            // that exact line.
            if stream.read(&mut buf).unwrap_or(0) == 0 {
                continue;
            }
            let _ = stream.write_all(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n\r\n");
            let _ = stream.flush();
            // Headers are in; from here the response is silent. One receive, no
            // timeout: the ONLY thing that ends the stall is the test below.
            let _ = released.recv();
            return;
        }
    });
    await_stub(port);
    let started = std::time::Instant::now();
    let r = stream_with_overrun_guard(&spec(port, 1000, 1));
    assert!(
        r.error.as_deref().unwrap_or("").contains("Timeout"),
        "{r:?}"
    );
    assert_eq!(r.finish_reason, "stream_deadline_exceeded");
    assert!(
        started.elapsed().as_secs() < 4,
        "must not wait out the stall"
    );
    // Release the stub and WAIT for it. Not what makes the assertions above
    // bite -- calibrated: with the join dropped the test is still green, because
    // the client returns on its own deadline while the stub thread sits parked.
    // It is the teardown half: the thread exits and its socket closes HERE,
    // rather than outliving the test and being killed at process exit.
    drop(release);
    h.join().unwrap();
}

/// The regime switch reaches the wire on BOTH request paths, blocking and
/// streamed, and defaults to the production regime (off). A sweep whose
/// payload silently kept thinking on would rank models the tools never run.
#[test]
fn thinking_flag_reaches_both_wire_shapes() {
    fn serve_recording(response: String) -> (u16, std::sync::mpsc::Receiver<String>) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        let (tx, rx) = std::sync::mpsc::channel();
        thread::spawn(move || {
            for stream in listener.incoming() {
                let Ok(mut stream) = stream else { continue };
                let mut buf = vec![0u8; 65536];
                let n = stream.read(&mut buf).unwrap_or(0);
                // Recording stubs ignore a read of zero bytes, or the readiness
                // probe is recorded as the first request and every assertion
                // below parses an empty payload instead of the real one.
                if n == 0 {
                    continue;
                }
                let _ = tx.send(String::from_utf8_lossy(&buf[..n]).into_owned());
                let _ = stream.write_all(response.as_bytes());
                let _ = stream.flush();
            }
        });
        await_stub(port);
        (port, rx)
    }
    fn payload_of(wire: &str) -> serde_json::Value {
        serde_json::from_str(wire.split("\r\n\r\n").nth(1).unwrap_or("")).unwrap()
    }

    let json_body = r#"{"choices":[{"message":{"content":"ok"}}]}"#;
    let (port, rx) = serve_recording(http_response("application/json", json_body));
    let mut s = spec(port, 100, 5);
    let r = call(&s, false);
    assert!(r.error.is_none(), "{r:?}");
    assert_eq!(payload_of(&rx.recv().unwrap())["enable_thinking"], false);

    s.thinking = true;
    let (port, rx) = serve_recording(http_response(
        "text/event-stream",
        &sse_body(&[r#"{"choices":[{"delta":{"content":"ok"},"finish_reason":"stop"}]}"#]),
    ));
    s.port = port;
    let r = stream_with_overrun_guard(&s);
    assert!(r.error.is_none(), "{r:?}");
    let payload = payload_of(&rx.recv().unwrap());
    assert_eq!(payload["stream"], true, "{payload}");
    assert_eq!(payload["enable_thinking"], true, "{payload}");
}
