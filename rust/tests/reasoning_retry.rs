//! Integration test: the REASONING diagnosis and its remedy (retry-token
//! escalation) over the wire, against a mock that only answers once the
//! budget has actually been raised.
//!
//! Mirrors the Python loop's behaviour on eval/failures.py FAIL_REASONING:
//! a model that streams reasoning_content with empty content is NOT a format
//! failure -- it never stopped thinking, so the retry gets MORE room (bounded),
//! and grinding at the original budget must not be read as quality.

use std::io::{Read, Write};
use std::net::TcpListener;
use std::sync::Arc;
use std::thread;

use ztools::eval::runner::{run_eval, RunnerConfig};
use ztools::eval::task_loader::{Check, EvalTask};

fn take_lock<T>(m: &std::sync::Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// SSE stream: plenty of reasoning_content, NO content, clean finish.
fn reasoning_only_sse() -> String {
    let reasoning = "x".repeat(500); // below the overrun abort line for 2048 tokens
    let mut body = String::new();
    body.push_str(&format!(
        "data: {{\"choices\":[{{\"delta\":{{\"reasoning_content\":\"{reasoning}\"}}}}]}}\n\n"
    ));
    body.push_str("data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"length\"}]}\n\n");
    body.push_str("data: [DONE]\n\n");
    format!(
        "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\nContent-Length: {}\r\n\r\n{body}",
        body.len()
    )
}

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

/// Answers every request with an endless-thinking stream EXCEPT requests whose
/// max_tokens reached the escalated budget -- those get a real answer.
struct StubbornThinker {
    port: u16,
    _handle: thread::JoinHandle<()>,
    seen_max_tokens: Arc<std::sync::Mutex<Vec<u32>>>,
}

fn serve_stubborn(escalated_budget: u32) -> StubbornThinker {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    let seen: Arc<std::sync::Mutex<Vec<u32>>> = Arc::new(std::sync::Mutex::new(Vec::new()));
    let seen_clone = seen.clone();

    let handle = thread::spawn(move || {
        for stream in listener.incoming() {
            let mut stream = match stream {
                Ok(s) => s,
                Err(_) => continue,
            };
            let mut buf = vec![0u8; 65_536];
            let n = stream.read(&mut buf).unwrap_or(0);
            let request = String::from_utf8_lossy(&buf[..n]).to_string();
            let marker = format!("\"max_tokens\":{escalated_budget}");
            let response = if request.contains(&marker) {
                take_lock(&seen_clone).push(escalated_budget);
                ok_body("the answer")
            } else {
                reasoning_only_sse()
            };
            let _ = stream.write_all(response.as_bytes());
            let _ = stream.flush();
        }
    });
    thread::sleep(std::time::Duration::from_millis(50));
    StubbornThinker { port, _handle: handle, seen_max_tokens: seen }
}

#[test]
fn a_reasoning_overrun_retries_with_a_raised_budget_and_then_passes() {
    let server = serve_stubborn(4_096); // 2x the base 2048, matching the ported multiplier
    let cfg = RunnerConfig {
        host: "127.0.0.1".into(),
        port: server.port,
        timeout_secs: 5,
        max_retries: 1,
        ..Default::default()
    };
    let task = EvalTask::new("t", "p", vec![Check::Contains("answer".to_string())]);
    let outcomes = run_eval("m", &[task], &cfg);

    assert_eq!(outcomes.len(), 1);
    let o = &outcomes[0];
    assert_eq!(o.error, None, "{o:?}");
    assert_eq!((o.score, o.status.as_str()), (100, "ok"), "{o:?}");
    assert!(take_lock(&server.seen_max_tokens).contains(&4_096),
        "the retry must have carried the escalated budget");
}

#[test]
fn an_unresolved_overrun_is_classified_reasoning_not_format() {
    // Nobody ever answers at ANY budget here: the final outcome must carry the
    // REASONING category -- "the model never stopped thinking" -- and not a
    // format/quality label that would send a human rewriting the prompt.
    let server = serve_stubborn(u32::MAX);
    let cfg = RunnerConfig {
        host: "127.0.0.1".into(),
        port: server.port,
        timeout_secs: 5,
        max_retries: 0,
        ..Default::default()
    };
    let task = EvalTask::new("t", "p", vec![Check::Contains("answer".to_string())]);
    let outcomes = run_eval("m", &[task], &cfg);
    let o = &outcomes[0];
    assert_eq!(o.score, 0);
    assert_eq!(o.failure_category, "REASONING", "{o:?}");
}

/// A model whose reasoning EXPANDS to fill whatever budget it is handed, so the
/// stream guard cuts it at every budget and it never answers. Records the
/// max_tokens of every request it sees.
struct FillsWhateverItGets {
    port: u16,
    _handle: thread::JoinHandle<()>,
    seen_max_tokens: Arc<std::sync::Mutex<Vec<u32>>>,
}

fn serve_expands_to_fill() -> FillsWhateverItGets {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    let seen: Arc<std::sync::Mutex<Vec<u32>>> = Arc::new(std::sync::Mutex::new(Vec::new()));
    let seen_clone = seen.clone();

    let handle = thread::spawn(move || {
        for stream in listener.incoming() {
            let mut stream = match stream {
                Ok(s) => s,
                Err(_) => continue,
            };
            let mut buf = vec![0u8; 65_536];
            let n = stream.read(&mut buf).unwrap_or(0);
            let request = String::from_utf8_lossy(&buf[..n]).to_string();
            // Whatever budget it was handed, spend past the guard's line for it.
            let budget: u32 = request
                .split("\"max_tokens\":")
                .nth(1)
                .and_then(|rest| {
                    rest.split(|c: char| !c.is_ascii_digit())
                        .next()
                        .and_then(|d| d.parse().ok())
                })
                .unwrap_or(2048);
            take_lock(&seen_clone).push(budget);
            let chars = (budget as usize) * 3 + 64; // past 0.75 * budget * CHARS_PER_TOKEN
            let reasoning = "x".repeat(chars);
            let mut body = String::new();
            body.push_str(&format!(
                "data: {{\"choices\":[{{\"delta\":{{\"reasoning_content\":\"{reasoning}\"}}}}]}}\n\n"
            ));
            body.push_str("data: [DONE]\n\n");
            let response = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\nContent-Length: {}\r\n\r\n{body}",
                body.len()
            );
            let _ = stream.write_all(response.as_bytes());
            let _ = stream.flush();
        }
    });
    thread::sleep(std::time::Duration::from_millis(50));
    FillsWhateverItGets { port, _handle: handle, seen_max_tokens: seen }
}

/// The escalation is proof-of-shape, bought ONCE per model.
///
/// nemotron wants more room and answers when given it (the test above). ornith-9b
/// expands to fill whatever it gets -- 72,005 chars at a 32,000-token budget,
/// 144,441 at 64,000, guard-aborted at both, scored 0 at both. One attempt cannot
/// tell the shapes apart, so the run escalates once and reads the outcome; a
/// guard abort on the ESCALATED attempt proves more room cannot help, and no later
/// task re-buys that proof.
#[test]
fn a_proven_futile_escalation_is_not_re_bought_on_every_task() {
    let server = serve_expands_to_fill();
    let cfg = RunnerConfig {
        host: "127.0.0.1".into(),
        port: server.port,
        timeout_secs: 5,
        max_retries: 1,
        ..Default::default()
    };
    let tasks: Vec<EvalTask> = ["t1", "t2", "t3"]
        .iter()
        .map(|n| EvalTask::new(*n, "p", vec![Check::Contains("answer".to_string())]))
        .collect();
    let outcomes = run_eval("m", &tasks, &cfg);
    assert_eq!(outcomes.len(), 3);

    let seen = take_lock(&server.seen_max_tokens).clone();
    let escalated: Vec<u32> = seen.iter().copied().filter(|b| *b > 2_048).collect();
    assert_eq!(
        escalated.len(),
        1,
        "escalation must be bought once per model, not once per task; saw {seen:?}"
    );
    // Two calls prove the shape on the first task, then one base call per task.
    assert_eq!(seen.len(), 4, "expected 2 calls then 1 per later task; saw {seen:?}");
}
