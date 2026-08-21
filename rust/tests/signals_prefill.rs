//! Tests for `eval/signals.rs` and `eval/prefill.rs`.
//!
//! Signal-store tests point EVAL_SIGNALS_DIR at a tmp dir so the tracked
//! conf/eval_signals.json is never dirtied. The prefill test uses a mock
//! server that RECORDS the requests it received, so the probe's wire contract
//! (nonce-first filler, max_tokens=1 on the timed call) is verified against
//! what actually went over the wire.

use std::io::{Read, Write};
use std::net::TcpListener;
use std::thread;

use serial_test::serial;

/// Point EVAL_SIGNALS_DIR at a fresh tmp dir for the duration of one test,
/// restoring the previous environment afterwards -- a leaked env var pointing
/// at a deleted dir silently empties every later test's store.
fn signals_dir_guard() -> SignalsDirGuard {
    SignalsDirGuard::new()
}

impl SignalsDirGuard {
    fn path(&self) -> &std::path::Path {
        self._dir.path()
    }
}

struct SignalsDirGuard {
    _dir: tempfile::TempDir,
    prev: Option<std::ffi::OsString>,
}

impl SignalsDirGuard {
    fn new() -> Self {
        let dir = tempfile::tempdir().unwrap();
        let prev = std::env::var_os("EVAL_SIGNALS_DIR");
        std::env::set_var("EVAL_SIGNALS_DIR", dir.path());
        Self { _dir: dir, prev }
    }
}

impl Drop for SignalsDirGuard {
    fn drop(&mut self) {
        match self.prev.take() {
            Some(v) => std::env::set_var("EVAL_SIGNALS_DIR", v),
            None => std::env::remove_var("EVAL_SIGNALS_DIR"),
        }
    }
}

/// The probe sizes its requests from EVAL_DEFAULT_TIMEOUT (default 900s); a
/// mock that fails to answer must fail FAST, not hang a CI run for 15 minutes.
struct BoundedProbeTimeout;

fn bounded_probe_timeout() -> BoundedProbeTimeout {
    std::env::set_var("EVAL_DEFAULT_TIMEOUT", "5");
    BoundedProbeTimeout
}

impl Drop for BoundedProbeTimeout {
    fn drop(&mut self) {
        std::env::remove_var("EVAL_DEFAULT_TIMEOUT");
    }
}

/// A poisoned mutex must not kill a server thread: that turns one failed
/// request into every subsequent connection hanging out its full timeout.
fn take_lock<T>(m: &std::sync::Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}

// --- signal store -----------------------------------------------------------

#[test]
#[serial]
fn record_signal_learns_p95_and_timeout() {
    let mut s: ztools::eval::SignalStore = Default::default();
    // A fast task: p95*1.5 = 150 loses to the documented 900s floor.
    ztools::eval::record_signal(&mut s, "m", "fast", 100.0, false, false);
    assert_eq!(s["m"]["fast"]["p95_latency"], serde_json::json!(100.0));
    assert_eq!(s["m"]["fast"]["timeout"], serde_json::json!(900));
    // A slow task: p95*1.5 = 1500 beats the floor -- the timeout LEARNED.
    ztools::eval::record_signal(&mut s, "m", "slow", 1000.0, false, false);
    assert_eq!(s["m"]["slow"]["timeout"], serde_json::json!(1500));
}

#[test]
#[serial]
fn p95_ema_rises_with_a_later_slow_reading_and_never_shrinks_it() {
    let mut store: ztools::eval::SignalStore = Default::default();
    ztools::eval::record_signal(&mut store, "m", "t", 10.0, false, false);
    ztools::eval::record_signal(&mut store, "m", "t", 1000.0, false, false);
    // EMA: max(1000, 10*0.95 + 1000*0.05) = 1000 -- a spike is not smoothed away.
    let p95 = store["m"]["t"]["p95_latency"].as_f64().unwrap();
    assert!((p95 - 1000.0).abs() < 0.2, "{p95}");
    // And one subsequent fast reading barely moves it.
    ztools::eval::record_signal(&mut store, "m", "t", 11.0, false, false);
    let p95 = store["m"]["t"]["p95_latency"].as_f64().unwrap();
    assert!(p95 > 950.0, "one fast sample must not erase the spike: {p95}");
}

#[test]
#[serial]
fn effective_timeout_never_falls_below_the_documented_floor() {
    let _g = signals_dir_guard();
    let got = ztools::eval::effective_timeout("never-measured-model", "task", 0, 0);
    assert!(
        got >= ztools::eval::default_eval_timeout(),
        "unmeasured model must get the floor, got {got}"
    );
}

#[test]
#[serial]
fn derived_timeout_requires_all_three_clean_terms() {
    let mut store: ztools::eval::SignalStore = Default::default();
    // Only prefill present -> no derivation at all, not a partial guess.
    ztools::eval::record_capability_sample(&mut store, "m", "prefill_chars_per_sec", 5000.0);
    assert_eq!(ztools::eval::derived_timeout("m", 10_000, 16_000), 0);
}

#[test]
#[serial]
fn capability_samples_migrate_scalar_once_then_outvote_it() {
    use ztools::eval::samples::Sample;
    // A legacy scalar seeds history UNCLEAN so clean_estimate returns None...
    let mut history: Vec<Sample> = Vec::new();
    ztools::eval::samples::migrate_sample_history(&mut history, Some(33.0));
    assert_eq!(history.len(), 1);
    assert!(!history[0].clean);
    assert_eq!(history[0].legacy, Some(true));
    // ...and re-seeding is a no-op once real samples exist.
    ztools::eval::samples::migrate_sample_history(&mut history, Some(99.0));
    assert_eq!(history.len(), 1);
    assert_eq!(history[0].v, 33.0);
    // A real CLEAN sample outvotes the legacy scalar in estimate_from.
    ztools::eval::samples::add_sample(&mut history, 100.0, true);
    let est = ztools::eval::samples::estimate_from(&history);
    assert_eq!(est, 100.0, "clean median of [100] beats unclean [33]");
}

#[test]
#[serial]
fn store_roundtrips_through_disk() {
    let g = signals_dir_guard();
    let mut s = ztools::eval::load_signals();
    ztools::eval::record_signal(&mut s, "model-x", "task-y", 42.0, true, false);
    ztools::eval::save_signals(&s);
    let reloaded = ztools::eval::load_signals();
    assert_eq!(reloaded["model-x"]["task-y"]["total_retries"], serde_json::json!(1));
    assert!(g.path().join("eval_signals.json").exists());
    // Sorted, pretty JSON so diffs on the tracked file stay readable.
    let text = std::fs::read_to_string(g.path().join("eval_signals.json")).unwrap();
    assert!(text.starts_with('{'), "{text}");
}

// --- prefill probe ----------------------------------------------------------

/// Server that captures request bodies and answers each with a tiny completion.
fn serve_recording() -> (u16, thread::JoinHandle<()>, std::sync::Arc<std::sync::Mutex<Vec<String>>>) {
    let recorded: std::sync::Arc<std::sync::Mutex<Vec<String>>> =
        std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));

    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    let value = recorded.clone();
    let handle = thread::spawn(move || {
        for stream in listener.incoming() {
            let mut stream = match stream {
                Ok(s) => s,
                Err(_) => continue,
            };
            let mut buf = vec![0u8; 65_536];
            let n = stream.read(&mut buf).unwrap_or(0);
            take_lock(&value)
                .push(String::from_utf8_lossy(&buf[..n]).to_string());
            let body =
                r#"{"choices":[{"message":{"content":"ok"},"finish_reason":"stop"}]}"#;
            let resp = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nConnection: close\r\nContent-Length: {}\r\n\r\n{body}",
                body.len()
            );
            let _ = stream.write_all(resp.as_bytes());
            let _ = stream.flush();
        }
    });
    thread::sleep(std::time::Duration::from_millis(50));
    (port, handle, recorded)
}

#[test]
#[serial]
fn prefill_probe_sends_nonce_led_filler_and_records_capabilities() {
    let _g = signals_dir_guard();
    let _t = bounded_probe_timeout();
    let (port, _h, recorded) = serve_recording();
    let mut store = ztools::eval::load_signals();
    let rate = ztools::eval::measure_prefill_rate(&mut store, "probe-model", "127.0.0.1", port);
    // The mock answers instantly, so the measured rate exceeds the plausibility
    // bound and is DISCARDED rather than enshrined.
    assert_eq!(rate, None, "a microseconds answer is not a measurement");

    // Three calls were made: LOAD(max_tokens=1), DECODE(max_tokens=64), PROBE(max_tokens=1).
    // ONE lock acquisition for all body assertions -- a second take_lock while
    // the first guard is alive deadlocks the non-reentrant mutex.
    let bodies = take_lock(&recorded);
    assert_eq!(bodies.len(), 3, "{}", bodies.len());
    for (i, body) in bodies.iter().enumerate() {
        if i != 1 {
            // LOAD and PREFILL both carry max_tokens=1.
            assert!(body.contains("\"max_tokens\":1"), "call {i}: {body}");
        } else {
            assert!(body.contains(format!("\"max_tokens\":{}", 64).as_str()), "{body}");
        }
    }
    // The timed probe leads with a unique nonce, defeating any prefix cache.
    let probe_body = &bodies[2];
    assert!(probe_body.contains("[run "), "nonce first: {}", probe_body.chars().take(200).collect::<String>());

    // Cold start and decode were recorded from the warmup calls even though the
    // prefill number itself was discarded.
    let caps = &store["probe-model"]["_capabilities"];
    assert!(caps.get("cold_start_seconds").is_some(), "{caps}");
    assert!(caps.get("decode_tokens_per_sec").is_some(), "{caps}");
    assert!(caps.get("prefill_chars_per_sec").is_none(), "discarded rate must not be stored");
}
