//! Integration test for model_eval.rs.
//!
//! Spins up a mock LLM server (Ollama-compatible) on a localhost port and
//! exercises get_available_models, eval_model, eval_all_models, and
//! render_eval_report against it. This covers the HTTP-dependent code that
//! unit tests can't reach.

use std::io::{Read, Write};
use std::net::TcpListener;
use std::thread;

use ztools::config::ZtoolsConfig;

/// A mock LLM server that answers **every** request for the life of the test.
///
/// It has to: `eval_model` sends one request per test case, and `eval_all_models`
/// sends one per model on top of that. A server that answered once and stopped
/// left every later request failing, so the per-case scoring closures never ran
/// and the test looked green while checking almost nothing.
fn mock_llm_server() -> (u16, thread::JoinHandle<()>) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    let handle = thread::spawn(move || {
        for stream in listener.incoming() {
            let mut stream = match stream {
                Ok(s) => s,
                Err(_) => continue,
            };
            let mut buf = [0u8; 8192];
            let n = stream.read(&mut buf).unwrap_or(0);
            let req = String::from_utf8_lossy(&buf[..n]);

            // Route based on path.
            let response = if req.contains("GET /v1/models") {
                r#"{"data":[{"id":"llama-3"},{"id":"foundation-model"},{"id":"diffusion-xl"},{"id":"qwen-7b"}]}"#.to_string()
            } else if req.contains("POST /v1/chat/completions") {
                // Well-formed JSON with exactly the two named events the
                // extraction case looks for, so the JSON-shape check parses
                // rather than bailing out at the first `from_str` error.
                let content = r#"{\"transient_events\":[{\"name\":\"Summer Rib Fest\"},{\"name\":\"Magic Show\"}]}"#;
                format!(r#"{{"choices":[{{"message":{{"content":"{content}"}}}}]}}"#)
            } else {
                "{}".to_string()
            };

            // `Connection: close` keeps the client from holding the socket open
            // for a second request this one-shot-per-connection loop never reads.
            let http = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nConnection: close\r\nContent-Length: {}\r\n\r\n{}",
                response.len(),
                response
            );
            let _ = stream.write_all(http.as_bytes());
            let _ = stream.flush();
        }
    });
    thread::sleep(std::time::Duration::from_millis(50));
    (port, handle)
}

fn default_config() -> ZtoolsConfig {
    ZtoolsConfig::default()
}

#[test]
fn get_available_models_filters_foundation_and_diffusion() {
    let (port, _handle) = mock_llm_server();
    let url = format!("http://127.0.0.1:{port}");
    let models = ztools::model_eval::get_available_models(&url, &default_config()).unwrap();
    // foundation-model and diffusion-xl are filtered out.
    assert!(models.contains(&"llama-3".to_string()), "got: {models:?}");
    assert!(models.contains(&"qwen-7b".to_string()), "got: {models:?}");
    assert!(
        !models.iter().any(|m| m.contains("foundation")),
        "got: {models:?}"
    );
    assert!(
        !models.iter().any(|m| m.contains("diffusion")),
        "got: {models:?}"
    );
}

#[test]
fn eval_model_returns_results_for_each_test_case() {
    let (port, _handle) = mock_llm_server();
    let url = format!("http://127.0.0.1:{port}");
    let results = ztools::model_eval::eval_model(&url, "llama-3", &default_config()).unwrap();
    // There are 5 test cases.
    assert_eq!(
        results.len(),
        5,
        "expected 5 results, got {}",
        results.len()
    );
    for r in &results {
        assert_eq!(r.model, "llama-3");
        assert!(r.total > 0);
    }
    // Every case must have been *scored*, not skipped because the request
    // failed: the extraction case matches the mock payload exactly, so it is
    // the one result that has to come back clean.
    let extraction = results
        .iter()
        .find(|r| r.test_name.contains("JSON Extraction"))
        .expect("extraction case missing");
    assert_eq!(
        (extraction.passed, extraction.status.as_str()),
        (extraction.total, "passed"),
        "the extraction case did not score against the mock: {extraction:?}"
    );
    // The other cases were scored too — a case whose request never landed
    // scores 0, so a non-zero score elsewhere proves the later requests were
    // served rather than dropped.
    assert!(
        results
            .iter()
            .any(|r| r.test_name.contains("Markdown") && r.passed > 0),
        "later cases were not served: {results:?}"
    );
}

#[test]
fn eval_all_models_evaluates_every_available_model() {
    let (port, _handle) = mock_llm_server();
    let url = format!("http://127.0.0.1:{port}");
    let results = ztools::model_eval::eval_all_models(&url, &default_config()).unwrap();
    // 2 models (llama-3, qwen-7b) x 5 test cases = 10 results.
    assert_eq!(results.len(), 10, "expected 10, got {}", results.len());
}

/// Same as `mock_llm_server`, but the image-renamer answer is wrapped in a
/// ` thinking... response` block, so its `NotContains(" ")` check can only pass
/// if `eval_model` cleans the output before judging it.
fn mock_llm_server_with_thinking() -> (u16, thread::JoinHandle<()>) {
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
            let req = String::from_utf8_lossy(&buf);
            let content = if req.contains("red sports car") {
                r#"<think>inner</think> red_car.jpg"#
            } else {
                r#"{"transient_events":[{"name":"Summer Rib Fest"},{"name":"Magic Show"}]}"#
            };
            let response = format!(r#"{{"choices":[{{"message":{{"content":"{content}"}}}}]}}"#);
            let http = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nConnection: close\r\nContent-Length: {}\r\n\r\n{}",
                response.len(),
                response
            );
            let _ = stream.write_all(http.as_bytes());
            let _ = stream.flush();
        }
    });
    thread::sleep(std::time::Duration::from_millis(50));
    (port, handle)
}

#[test]
fn eval_model_cleans_thinking_before_scoring() {
    let (port, _handle) = mock_llm_server_with_thinking();
    let url = format!("http://127.0.0.1:{port}");
    let results = ztools::model_eval::eval_model(&url, "llama-3", &default_config()).unwrap();
    let renaming = results
        .iter()
        .find(|r| r.test_name.contains("Image Renamer"))
        .expect("renamer case missing");
    // The filename is buried in a thinking block; the NotContains(" ") check
    // only passes once the content_processing port strips it.
    assert_eq!(
        (renaming.passed, renaming.status.as_str()),
        (renaming.total, "passed"),
        "thinking block was not cleaned before scoring: {renaming:?}"
    );
}

#[test]
fn render_eval_report_produces_a_markdown_table() {
    let results = vec![
        ztools::model_eval::ModelEvalResult {
            model: "llama-3".into(),
            test_name: "JSON Extraction".into(),
            score: 100.0,
            passed: 4,
            total: 4,
            latency_ms: 1200,
            status: "passed".into(),
        },
        ztools::model_eval::ModelEvalResult {
            model: "qwen".into(),
            test_name: "Markdown".into(),
            score: 75.0,
            passed: 3,
            total: 4,
            latency_ms: 800,
            status: "failed".into(),
        },
    ];
    let report = ztools::model_eval::render_eval_report(&results);
    assert!(report.contains("# Model Quality Evaluation"), "{report}");
    assert!(report.contains("llama-3"), "{report}");
    assert!(report.contains("100.0%"), "{report}");
    assert!(report.contains("passed"), "{report}");
    assert!(report.contains("failed"), "{report}");
}
