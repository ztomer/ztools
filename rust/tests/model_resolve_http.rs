//! Integration tests for missing-model substitution over the wire.
//!
//! Ported contract from `references/tests/test_model_substitution_retry.py`:
//! a 404 naming a dead model tag must retry ONCE against a servable stand-in,
//! surface the substitution instead of swallowing it, re-derive quirks for the
//! substitute (a different family), and stay silent when there is no evidence
//! (non-404, unmarked 404) or when the caller disabled substitution.

use std::io::{Read, Write};
use std::net::TcpListener;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;

use ztools::eval::runner::{run_eval, run_eval_with_signals, RunnerConfig};
use ztools::eval::task_loader::{Check, EvalTask};

/// A poisoned mutex must not kill a server thread: that turns one failed
/// request into every subsequent connection hanging out its full timeout.
fn take_lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}

const MISSING_BODY: &str =
    r#"{"error":{"message":"Model 'gone-model' is not installed or registered with any provider."}}"#;

fn ok_body(content: &str) -> String {
    let escaped = content.replace('"', "\\\"");
    let json = format!(
        r#"{{"choices":[{{"message":{{"content":"{escaped}"}},"finish_reason":"stop"}}]}}"#
    );
    format!(
        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nConnection: close\r\nContent-Length: {}\r\n\r\n{json}",
        json.len()
    )
}

/// Routing mock: GET /api/tags answers with `roster`; POST chat completions
/// naming `dead_tag` answer 404-missing for the FIRST `fail_n` OF THOSE and
/// 200 afterwards. Failure is keyed on the MODEL NAME, not a global counter:
/// the learning path's prefill probes hit the same mock before the evaluated
/// task does, and they must not consume the dead tag's failure budget.
/// Every request body is recorded so quirk re-derivation can be asserted
/// against what actually went over the wire.
struct MockServer {
    port: u16,
    _handle: thread::JoinHandle<()>,
    posts: Arc<Mutex<Vec<String>>>,
}

fn serve(
    roster: String,
    dead_tag: &'static str,
    fail_first: usize,
    missing_body: &'static str,
) -> MockServer {
    let dead_marker = format!(r#""model":"{dead_tag}""#);
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    let count = Arc::new(AtomicUsize::new(0));
    let posts: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
    let posts_clone = posts.clone();
    let post_count = count.clone();

    let handle = thread::spawn(move || {
        for stream in listener.incoming() {
            let mut stream = match stream {
                Ok(s) => s,
                Err(_) => continue,
            };
            let mut buf = vec![0u8; 65_536];
            let n = stream.read(&mut buf).unwrap_or(0);
            let request = String::from_utf8_lossy(&buf[..n]).to_string();
            let first_line = request.lines().next().unwrap_or("").to_string();

            let response = if first_line.starts_with("GET /api/tags") {
                format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nConnection: close\r\nContent-Length: {}\r\n\r\n{roster}",
                    roster.len()
                )
            } else {
                // POST /v1/chat/completions: record the body, then decide.
                let mut is_dead_call = false;
                if let Some(body_start) = request.find("\r\n\r\n") {
                    let body = request[body_start + 4..].to_string();
                    is_dead_call = body.contains(&dead_marker);
                    take_lock(&posts_clone).push(body);
                }
                let seen = if is_dead_call {
                    post_count.fetch_add(1, Ordering::SeqCst)
                } else {
                    usize::MAX
                };
                if seen < fail_first {
                    let body_404 = missing_body;
                    format!(
                        "HTTP/1.1 404 Not Found\r\nContent-Type: application/json\r\nConnection: close\r\nContent-Length: {}\r\n\r\n{body_404}",
                        missing_body.len()
                    )
                } else {
                    ok_body("the answer")
                }
            };
            let _ = stream.write_all(response.as_bytes());
            let _ = stream.flush();
        }
    });
    thread::sleep(std::time::Duration::from_millis(50));
    MockServer { port, _handle: handle, posts }
}

impl MockServer {
    fn bodies(&self) -> Vec<String> {
        take_lock(&self.posts).clone()
    }
}

fn roster_json(models: &[(&str, &str)]) -> String {
    let entries: Vec<String> = models
        .iter()
        .map(|(m, size)| {
            format!(
                r#"{{"model":"{m}","details":{{"parameter_size":"{size}"}}}}"#
            )
        })
        .collect();
    format!(r#"{{"models":[{}]}}"#, entries.join(","))
}

fn task(name: &str) -> EvalTask {
    EvalTask::new(
        name,
        "p",
        vec![Check::Contains("answer".to_string())],
    )
}

// A system prompt that carries NO qwen trigger yet: only a correctly
// re-derived qwen-family quirk adds "Output JSON now." in front.
fn task_with_system_prompt(name: &str) -> EvalTask {
    let mut t = EvalTask::new(name, "p", vec![Check::Contains("answer".to_string())]);
    t.messages.insert(0, ztools::eval::task_loader::ChatMessage {
        role: "system".to_string(),
        content: "Extract events from the timeline.".to_string(),
    });
    t
}

#[test]
fn missing_model_404_retries_against_the_substitute_and_says_so() {
    // fail_first=2: one transport::call with stream_guard ON makes TWO requests
    // (streamed attempt, then its blocking fallback); BOTH must see the dead tag.
    let server = serve(
        roster_json(&[("qwen3.8-27b-8bit", "27B")]),
        "gone-model",
        2,
        MISSING_BODY,
    );
    let cfg = RunnerConfig {
        host: "127.0.0.1".into(),
        port: server.port,
        timeout_secs: 5,
        max_retries: 0,
        ..Default::default()
    };
    // gone-model resolves to family "default", so the chain picks the qwen tag.
    let outcomes = run_eval("gone-model", &[task("t1")], &cfg);
    assert_eq!(outcomes.len(), 1);
    let o = &outcomes[0];
    assert_eq!(o.error, None, "substitute should have answered: {o:?}");
    assert_eq!((o.score, o.status.as_str()), (100, "ok"), "{o:?}");
    assert_eq!(o.substituted_from.as_deref(), Some("gone-model"), "{o:?}");
    assert_eq!(o.substituted_to.as_deref(), Some("qwen3.8-27b-8bit"), "{o:?}");
    let reason = o.substitution_reason.as_deref().expect("reason surfaced");
    assert!(reason.contains("Re-derive best_models"), "{reason}");
}

#[test]
fn quirks_are_re_derived_for_the_substitute_not_inherited() {
    let server = serve(
        roster_json(&[("qwen3.8-27b-8bit", "27B")]),
        "gone-model",
        2,
        MISSING_BODY,
    );
    let cfg = RunnerConfig {
        host: "127.0.0.1".into(),
        port: server.port,
        timeout_secs: 5,
        max_retries: 0,
        ..Default::default()
    };
    let outcomes = run_eval("gone-model", &[task_with_system_prompt("t1")], &cfg);
    assert!(outcomes[0].error.is_none(), "{:?}", outcomes[0]);

    // Requests 1-2 went to gone-model (default family): NO qwen trigger.
    // The substitute's requests MUST carry its own trigger exactly once --
    // inheriting nothing and double-applying are both failures.
    let bodies = server.bodies();
    assert_eq!(bodies.len(), 4, "dead stream+blocking, then substitute stream+blocking");
    for dead in &bodies[..2] {
        assert!(
            !dead.contains("Output JSON now."),
            "default family gets no prefix: {dead}"
        );
    }
    assert!(
        bodies[3].contains(r#""model":"qwen3.8-27b-8bit""#),
        "{}",
        bodies[3]
    );
    assert!(
        bodies[3].contains("Output JSON now.\\n\\nExtract events"),
        "substitute's own quirk applied exactly once: {}",
        bodies[3]
    );
}

#[test]
fn substitution_can_be_switched_off_by_the_caller() {
    let server = serve(
        roster_json(&[("qwen3.8-27b-8bit", "27B")]),
        "gone-model",
        999,
        MISSING_BODY,
    );
    let cfg = RunnerConfig {
        host: "127.0.0.1".into(),
        port: server.port,
        timeout_secs: 5,
        max_retries: 0,
        allow_model_substitution: false,
        ..Default::default()
    };
    let outcomes = run_eval("gone-model", &[task("t1")], &cfg);
    let o = &outcomes[0];
    assert!(o.error.as_deref().unwrap_or("").starts_with("HTTP 404"), "{o:?}");
    assert!(o.substituted_from.is_none(), "{o:?}");
}

#[test]
fn an_unmarked_404_is_not_evidence_of_a_missing_model() {
    // Wrong endpoint / proxy noise also returns 404; rewriting the model on
    // that would trade a clear error for a silently wrong answer.
    let server = serve(
        roster_json(&[("qwen3.8-27b-8bit", "27B")]),
        "gone-model",
        999,
        "404 page not found",
    );
    let cfg = RunnerConfig {
        host: "127.0.0.1".into(),
        port: server.port,
        timeout_secs: 5,
        max_retries: 0,
        ..Default::default()
    };
    let outcomes = run_eval("gone-model", &[task("t1")], &cfg);
    let o = &outcomes[0];
    assert!(o.error.as_deref().unwrap_or("").starts_with("HTTP 404"), "{o:?}");
    assert!(o.substitution_reason.is_none(), "{o:?}");
}

// --- production path: signals recording wired into the sweep -----------------

#[test]
#[serial_test::serial]
fn the_learning_path_records_signals_and_answers_through_a_substitute() {
    // Point EVAL_SIGNALS_DIR at a fresh tmp dir so conf/eval_signals.json
    // (the tracked store) is never dirtied by a test.
    let dir = tempfile::tempdir().unwrap();
    let prev = std::env::var_os("EVAL_SIGNALS_DIR");
    std::env::set_var("EVAL_SIGNALS_DIR", dir.path());

    // Every dead-tag request fails forever (the substitute always answers):
    // the learning path's prefill probe hits the dead tag BEFORE the evaluated
    // task does, so a small failure budget would be consumed up there.
    let server = serve(
        roster_json(&[("qwen3.8-27b-8bit", "27B")]),
        "gone-model",
        999,
        MISSING_BODY,
    );
    let cfg = RunnerConfig {
        host: "127.0.0.1".into(),
        port: server.port,
        timeout_secs: 5,
        max_retries: 0,
        record_signals: true,
        ..Default::default()
    };

    let outcome = run_eval_with_signals("gone-model", &[task("t1")], &cfg);

    match prev {
        Some(v) => std::env::set_var("EVAL_SIGNALS_DIR", v),
        None => std::env::remove_var("EVAL_SIGNALS_DIR"),
    }

    let o = &outcome[0];
    assert_eq!(o.error, None, "substitute should have answered: {o:?}");
    assert_eq!(o.substituted_to.as_deref(), Some("qwen3.8-27b-8bit"));

    // The store was written back with this run's observations under the
    // ORIGINAL model name -- the sweep was FOR gone-model; its timings belong
    // to that name even though a stand-in answered (matching Python, which
    // records under the configured key).
    let text =
        std::fs::read_to_string(dir.path().join("eval_signals.json")).unwrap();
    let store: serde_json::Value = serde_json::from_str(&text).unwrap();
    assert!(
        store.get("gone-model").is_some(),
        "signals recorded under the configured name: {store}"
    );
}
