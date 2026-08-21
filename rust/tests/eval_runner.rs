//! Integration tests for `eval/runner.rs` against mock HTTP servers.
//!
//! Prove-fail note: the infra-abort and retry tests were verified to fail by
//! breaking the loop conditions before being trusted green.

use std::io::{Read, Write};
use std::net::TcpListener;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread;

use ztools::eval::runner::{run_eval, RunnerConfig};
use ztools::eval::task_loader::{Check, EvalTask};

/// Server that answers every connection with `response`.
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

/// Server that serves `first_response` for the first N requests, then
/// `rest_response` for every later one.
fn serve_then(first: String, rest: String, first_n: usize) -> (u16, thread::JoinHandle<()>) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    let count = Arc::new(AtomicUsize::new(0));
    let handle = thread::spawn(move || {
        for stream in listener.incoming() {
            let mut stream = match stream {
                Ok(s) => s,
                Err(_) => continue,
            };
            let mut buf = [0u8; 8192];
            let _ = stream.read(&mut buf);
            let n = count.fetch_add(1, Ordering::SeqCst);
            let resp = if n < first_n { &first } else { &rest };
            let _ = stream.write_all(resp.as_bytes());
            let _ = stream.flush();
        }
    });
    thread::sleep(std::time::Duration::from_millis(50));
    (port, handle)
}

fn ok_body(content: &str) -> String {
    let escaped = content.replace('\\', "\\\\").replace('"', "\\\"");
    format!(
        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nConnection: close\r\nContent-Length: {}\r\n\r\n{{\"choices\":[{{\"message\":{{\"content\":\"{escaped}\"}},\"finish_reason\":\"stop\"}}]}}",
        body_len(&escaped)
    )
}

fn body_len(escaped: &str) -> usize {
    // The JSON body is built below; compute its length the same way.
    let json = format!(
        "{{\"choices\":[{{\"message\":{{\"content\":\"{escaped}\"}},\"finish_reason\":\"stop\"}}]}}"
    );
    json.len()
}

fn err_response(status_line: &str) -> String {
    format!(
        "{status_line}\r\nContent-Length: 2\r\nConnection: close\r\n\r\n{{}}"
    )
}

fn task(name: &str, content_marker: &str) -> EvalTask {
    EvalTask::new(
        name,
        "answer",
        vec![
            Check::Contains(content_marker.to_string()),
            Check::ContainsLower("ANSWER".to_string()),
        ],
    )
}

fn cfg(port: u16) -> RunnerConfig {
    RunnerConfig {
        host: "127.0.0.1".to_string(),
        port,
        timeout_secs: 5,
        max_retries: 1,
        max_consecutive_infra: 4,
        ..Default::default()
    }
}

#[test]
fn perfect_output_scores_ok_without_retry() {
    let (port, _h) = serve(ok_body("the answer"));
    let tasks = vec![task("t1", "answer")];
    let outcomes = run_eval("m", &tasks, &cfg(port));
    assert_eq!(outcomes.len(), 1);
    let o = &outcomes[0];
    assert_eq!((o.score, o.status.as_str()), (100, "ok"), "{o:?}");
    assert!(o.error.is_none(), "{o:?}");
}

#[test]
fn partial_output_scores_partial() {
    // Only one of two checks passes ("answer" present; "nope" absent from output).
    let (port, _h) = serve(ok_body("answer only"));
    let t = EvalTask::new(
        "t",
        "p",
        vec![Check::Contains("answer".to_string()), Check::Contains("nope".to_string())],
    );
    let outcomes = run_eval("m", &[t], &cfg(port));
    assert_eq!(outcomes[0].score, 50, "{:?}", outcomes[0]);
    assert_eq!(outcomes[0].status, "partial");
}

#[test]
fn zero_score_output_still_carries_a_status() {
    // Output arrives but scores 0: the placeholder-vs-score update must still
    // set the status. It used to leak an EMPTY status when the scored attempt
    // tied the placeholder's 0.
    let (port, _h) = serve(ok_body("unrelated prose"));
    let t = EvalTask::new("t", "p", vec![Check::Contains("never".to_string())]);
    let outcomes = run_eval("m", &[t], &cfg(port));
    assert_eq!(outcomes[0].score, 0);
    assert_eq!(outcomes[0].status, "fail", "{:?}", outcomes[0]);
}

#[test]
fn transport_error_is_retried_then_recorded_as_fail() {
    // First request 503, second succeeds: max_retries=1 must recover.
    let (port, _h) = serve_then(err_response("HTTP/1.1 503 Service Unavailable"), ok_body("the answer"), 1);
    let outcomes = run_eval("m", &[task("t1", "answer")], &cfg(port));
    let o = &outcomes[0];
    assert!(o.error.is_none(), "retry should have recovered: {o:?}");
    assert_eq!(o.score, 100, "{o:?}");
}

#[test]
fn consecutive_infra_failures_abandon_the_model_early() {
    // Every request 503s. The infra counter counts TASKS, not requests
    // (matching the Python loop): with max_consecutive_infra=2 the model is
    // abandoned after tasks a and b, BEFORE task c runs.
    let (port, _h) = serve(err_response("HTTP/1.1 503 Service Unavailable"));
    let tasks = vec![task("a", "x"), task("b", "x"), task("c", "x")];
    let mut c = cfg(port);
    c.max_retries = 1;
    c.max_consecutive_infra = 2;
    let outcomes = run_eval("m", &tasks, &c);
    assert_eq!(outcomes.len(), 2, "must abandon before task c: {outcomes:?}");
    assert!(outcomes.iter().all(|o| o.error.is_some()), "{outcomes:?}");
}

#[test]
fn a_quality_failure_does_not_count_as_infra() {
    // Output arrives but is wrong: that is a model result, not an infra
    // failure, so all three tasks are evaluated.
    let (port, _h) = serve(ok_body("irrelevant prose"));
    let tasks = vec![task("a", "never"), task("b", "never"), task("c", "never")];
    let outcomes = run_eval("m", &tasks, &cfg(port));
    assert_eq!(outcomes.len(), 3, "{outcomes:?}");
    assert!(outcomes.iter().all(|o| o.error.is_none()), "{outcomes:?}");
    assert!(outcomes.iter().all(|o| o.score == 0), "{outcomes:?}");
}
