//! Weekend model resolution: the configured model or a stated failure.
//!
//! Every scheduled plan through 2026-10-09 was drafted by whatever model the
//! server listed first, because a missing configured model fell back silently.
//! These tests pin the replacement: no substitution of any kind, a failure
//! that names the model and the roster, and no chat call before it.

#[test]
fn resolve_weekend_model_unreachable_endpoint_returns_preferred() {
    let chosen =
        crate::ztools::weekend::resolve_weekend_model("http://127.0.0.1:1", "qwen3.8-27b-8bit");
    assert_eq!(chosen, Ok("qwen3.8-27b-8bit".to_string()));
}

/// A `/v1/models` stub: answers every request with `roster`, recording each
/// request line so a test can prove which calls were (not) made.
fn roster_stub(roster: &'static str) -> (String, std::sync::Arc<std::sync::Mutex<Vec<String>>>) {
    use std::io::{Read, Write};
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let seen = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
    let log = seen.clone();
    std::thread::spawn(move || {
        while let Ok((mut stream, _)) = listener.accept() {
            let mut buf = [0u8; 4096];
            let n = stream.read(&mut buf).unwrap_or(0);
            let head = String::from_utf8_lossy(&buf[..n])
                .lines()
                .next()
                .unwrap_or("")
                .to_string();
            log.lock().unwrap().push(head);
            let resp = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{roster}",
                roster.len()
            );
            let _ = stream.write_all(resp.as_bytes());
        }
    });
    (format!("http://{addr}"), seen)
}

const ROSTER_WITHOUT_CONFIGURED: &str =
    r#"{"data":[{"id":"embed-small"},{"id":"qwen3.8-27b-jang_6d"},{"id":"gemma-4-e2b-it-8bit"}]}"#;

/// The class: a configured model the server does not list is NEVER replaced —
/// not by a same-family tag, not by a preferred family, not by `first()`. It
/// is a stated failure that carries what the server does have. (Calibrated:
/// the old resolver returned "qwen3.8-27b-jang_6d" here, and "embed-small"
/// for a configured name with no family match.)
#[test]
fn a_model_that_is_not_installed_is_named_never_substituted() {
    let (url, _) = roster_stub(ROSTER_WITHOUT_CONFIGURED);
    for configured in ["qwen3.8-27b-8bit", "raptor-v0.5-8b"] {
        assert_eq!(
            crate::ztools::weekend::resolve_weekend_model(&url, configured),
            Err(crate::ztools::weekend::ModelHealth::NotInstalled {
                model: configured.to_string(),
                installed: vec![
                    "embed-small".to_string(),
                    "qwen3.8-27b-jang_6d".to_string(),
                    "gemma-4-e2b-it-8bit".to_string(),
                ],
            })
        );
    }
    let (url, _) = roster_stub(r#"{"data":[{"id":"embed-small"},{"id":"test-model"}]}"#);
    assert_eq!(
        crate::ztools::weekend::resolve_weekend_model(&url, "test-model"),
        Ok("test-model".to_string())
    );
}

/// The warm-up refuses a missing model before it sends a single chat request:
/// the stub sees the roster lookup and nothing else.
#[test]
fn warm_up_makes_no_call_when_the_model_is_not_installed() {
    let (url, seen) = roster_stub(ROSTER_WITHOUT_CONFIGURED);
    // Built under the sandbox: a default config names `~/…` paths.
    let _env = crate::test_env::TestEnv::new();
    let cfg = crate::config::ZtoolsConfig {
        osaurus_url: url,
        weekend_model: "test-model".into(),
        ..crate::config::ZtoolsConfig::default()
    };
    let health = crate::ztools::weekend::warm_model(&cfg);
    assert!(
        matches!(&health, crate::ztools::weekend::ModelHealth::NotInstalled { model, .. } if model == "test-model"),
        "{health:?}"
    );
    let requests = seen.lock().unwrap().clone();
    assert_eq!(requests, vec!["GET /v1/models HTTP/1.1".to_string()]);
}
