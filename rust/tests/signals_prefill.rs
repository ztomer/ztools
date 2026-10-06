//! Tests for `eval/signals.rs` and `eval/prefill.rs`.
//!
//! Signal-store tests point `EVAL_SIGNALS_DIR` at a tmp dir so the tracked
//! `conf/eval_signals.json` is never dirtied. The prefill test uses a mock
//! server that RECORDS the requests it received, so the probe's wire contract
//! (nonce-first filler, `max_tokens=1` on the timed call) is verified against
//! what actually went over the wire.
//!
//! The `thread::sleep` the mock used to take after `bind` was a guess about the
//! serving thread's scheduling; it is now `support::await_stub`, which waits for
//! the condition a client actually needs -- a completed handshake -- under a
//! deadline that names what never happened. The recorder also stops counting a
//! connection that carried no request, which is what makes the probe's
//! three-request count mean three requests.

#[path = "support/mod.rs"]
mod support;
// See `eval_runner.rs`: the shared module's items are reachable API of this
// test binary, so one consumer not needing one is not dead code.
pub use support::*;

use std::collections::BTreeMap;
use std::io::{Read, Write};
use std::net::TcpListener;
use std::thread;

use serial_test::serial;
use ztools::test_env::TestEnv;

// --- signal store -----------------------------------------------------------

#[test]
#[serial]
fn record_signal_learns_p95_and_timeout() {
    let mut s: ztools::eval::SignalStore = BTreeMap::default();
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
    let mut store: ztools::eval::SignalStore = BTreeMap::default();
    ztools::eval::record_signal(&mut store, "m", "t", 10.0, false, false);
    ztools::eval::record_signal(&mut store, "m", "t", 1000.0, false, false);
    // EMA: max(1000, 10*0.95 + 1000*0.05) = 1000 -- a spike is not smoothed away.
    let p95 = store["m"]["t"]["p95_latency"].as_f64().unwrap();
    assert!((p95 - 1000.0).abs() < 0.2, "{p95}");
    // And one subsequent fast reading barely moves it.
    ztools::eval::record_signal(&mut store, "m", "t", 11.0, false, false);
    let p95 = store["m"]["t"]["p95_latency"].as_f64().unwrap();
    assert!(
        p95 > 950.0,
        "one fast sample must not erase the spike: {p95}"
    );
}

#[test]
#[serial]
fn effective_timeout_never_falls_below_the_documented_floor() {
    // The sandbox, not a bare `let _`: it is what makes the policy knobs
    // absent, so the floor asserted here is the documented default rather than
    // whatever the operator's shell exported.
    let _env = TestEnv::new();
    let got = ztools::eval::effective_timeout("never-measured-model", "task", 0, 0);
    assert!(
        got >= ztools::eval::default_eval_timeout(),
        "unmeasured model must get the floor, got {got}"
    );
}

#[test]
#[serial]
fn derived_timeout_requires_all_three_clean_terms() {
    let mut store: ztools::eval::SignalStore = BTreeMap::default();
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
    assert_eq!(
        history[0].v.partial_cmp(&33.0),
        Some(std::cmp::Ordering::Equal),
        "exact: {}",
        history[0].v
    );
    // A real CLEAN sample outvotes the legacy scalar in estimate_from.
    ztools::eval::samples::add_sample(&mut history, 100.0, true);
    let est = ztools::eval::samples::estimate_from(&history);
    assert_eq!(
        est.partial_cmp(&100.0),
        Some(std::cmp::Ordering::Equal),
        "clean median of [100] beats unclean [33], got {est}"
    );
}

#[test]
#[serial]
fn store_roundtrips_through_disk() {
    let (_env, signals_dir) = TestEnv::new().at("EVAL_SIGNALS_DIR");
    let mut s = ztools::eval::load_signals();
    ztools::eval::record_signal(&mut s, "model-x", "task-y", 42.0, true, false);
    ztools::eval::save_signals(&s);
    let reloaded = ztools::eval::load_signals();
    assert_eq!(
        reloaded["model-x"]["task-y"]["total_retries"],
        serde_json::json!(1)
    );
    assert!(signals_dir.join("eval_signals.json").exists());
    // Sorted, pretty JSON so diffs on the tracked file stay readable.
    let text = std::fs::read_to_string(signals_dir.join("eval_signals.json")).unwrap();
    assert!(text.starts_with('{'), "{text}");
}

// --- prefill probe ----------------------------------------------------------

/// Server that captures request bodies and answers each with a tiny completion.
///
/// The probe sizes its requests from `EVAL_DEFAULT_TIMEOUT` (default 900s), so a
/// mock that fails to answer has to fail FAST: the alternative is a CI run that
/// sits on a 15-minute timeout. `TestEnv::policy("EVAL_DEFAULT_TIMEOUT", "5")`
/// in the test below is what buys that.
///
/// A connection that carried no request is NOT recorded. `await_stub` opens one
/// and sends nothing, and recording it would make the probe's three-request
/// count four and the `max_tokens` assertions below read a request that never
/// happened.
fn serve_recording() -> (
    u16,
    thread::JoinHandle<()>,
    std::sync::Arc<std::sync::Mutex<Vec<String>>>,
) {
    let recorded: std::sync::Arc<std::sync::Mutex<Vec<String>>> =
        std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));

    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    let value = recorded.clone();
    let handle = thread::spawn(move || {
        for stream in listener.incoming() {
            let Ok(mut stream) = stream else { continue };
            let mut buf = vec![0u8; 65_536];
            let n = stream.read(&mut buf).unwrap_or(0);
            if n == 0 {
                continue;
            }
            take_lock(&value).push(String::from_utf8_lossy(&buf[..n]).to_string());
            let body = r#"{"choices":[{"message":{"content":"ok"},"finish_reason":"stop"}]}"#;
            let resp = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nConnection: close\r\nContent-Length: {}\r\n\r\n{body}",
                body.len()
            );
            let _ = stream.write_all(resp.as_bytes());
            let _ = stream.flush();
        }
    });
    await_stub(port);
    (port, handle, recorded)
}

#[test]
#[serial]
fn prefill_probe_sends_nonce_led_filler_and_records_capabilities() {
    // `EVAL_DEFAULT_TIMEOUT` is one of the policy knobs `TestEnv` clears on
    // construction, so `policy` is the whole of the old `BoundedProbeTimeout`:
    // a `Drop` that removed the variable outright discarded whatever the
    // operator had exported and left it absent for every later test.
    let _env = TestEnv::new().policy("EVAL_DEFAULT_TIMEOUT", "5");
    let (port, _h, recorded) = serve_recording();
    let mut store = ztools::eval::load_signals();
    let rate = ztools::eval::measure_prefill_rate(&mut store, "probe-model", "127.0.0.1", port);
    // The mock answers instantly, so the measured rate exceeds the plausibility
    // bound and is DISCARDED rather than enshrined.
    assert_eq!(rate, None, "a microseconds answer is not a measurement");

    // Three calls were made: LOAD(max_tokens=1), DECODE(max_tokens=64), PROBE(max_tokens=1).
    // The wait states the condition rather than trusting the ordering: the
    // recorder appends before it answers, so all three are already there, and
    // "already there" is precisely the assumption a fixed sleep was standing in
    // for. ONE lock acquisition per use: a second take_lock while a guard is
    // alive deadlocks the non-reentrant mutex.
    wait_for("the recorder to hold all three probe requests", || {
        take_lock(&recorded).len() == 3
    });
    let bodies: Vec<String> = take_lock(&recorded).clone();
    assert_eq!(
        bodies.len(),
        3,
        "LOAD, DECODE and PROBE, and nothing else: {}",
        bodies.len()
    );
    for (i, body) in bodies.iter().enumerate() {
        if i == 1 {
            assert!(
                body.contains(format!("\"max_tokens\":{}", 64).as_str()),
                "{body}"
            );
        } else {
            // LOAD and PREFILL both carry max_tokens=1.
            assert!(body.contains("\"max_tokens\":1"), "call {i}: {body}");
        }
    }
    // The timed probe leads with a unique nonce, defeating any prefix cache.
    let probe_body = &bodies[2];
    assert!(
        probe_body.contains("[run "),
        "nonce first: {}",
        probe_body.chars().take(200).collect::<String>()
    );

    // Cold start and decode were recorded from the warmup calls even though the
    // prefill number itself was discarded.
    let caps = &store["probe-model"]["_capabilities"];
    assert!(caps.get("cold_start_seconds").is_some(), "{caps}");
    assert!(caps.get("decode_tokens_per_sec").is_some(), "{caps}");
    assert!(
        caps.get("prefill_chars_per_sec").is_none(),
        "discarded rate must not be stored"
    );
}
