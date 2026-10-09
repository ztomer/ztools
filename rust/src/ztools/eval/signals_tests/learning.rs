//! What gets learned: capability samples, p95, retries, and the clean tag.
//!
//! The clean tag is the one thing here that comes from the live machine, so it
//! is asserted AGAINST the public verdict rather than against a hardcoded
//! bool: the test then holds whether or not this box happens to be busy, and
//! stops being a statement about the developer's afternoon.
//!
//! Every case records through `record_task_signal` with a task it built itself,
//! not through the name-based `record_signal`. That is deliberate: the name
//! path resolves the digest from a process-wide registry, so a test using it
//! would pass or fail depending on what another test in this binary registered
//! under that name — and a test whose verdict depends on test ORDER is a test
//! that will one day be wrong. Identity itself (and the name path) is pinned in
//! `identity.rs`, against task names no other test file uses.

use super::super::*;
use crate::test_env::TestEnv;
use crate::ztools::eval::task_loader::EvalTask;
use serial_test::serial;

/// The task whose observations the cases below accumulate. One name per case,
/// so no two cases ever share a series.
fn task(name: &str) -> EvalTask {
    EvalTask::new(name, "the prompt", Vec::new())
}

#[test]
fn nonpositive_or_nonfinite_values_are_never_recorded() {
    let mut signals = SignalStore::new();
    record_capability_sample(&mut signals, "m", "rate", 0.0);
    record_capability_sample(&mut signals, "m", "rate", -3.0);
    record_capability_sample(&mut signals, "m", "rate", f64::NAN);
    record_capability_sample(&mut signals, "m", "rate", f64::INFINITY);
    assert!(signals.is_empty(), "{signals:?}");
}

#[test]
#[serial]
fn recording_a_sample_creates_capabilities_and_rederives_the_estimate() {
    let env = TestEnv::new();
    let mut signals = SignalStore::new();
    record_capability_sample(&mut signals, "m", "rate", 42.5);
    let caps = &signals["m"]["_capabilities"];
    let history: Vec<Sample> = serde_json::from_value(caps["rate_samples"].clone()).unwrap();
    assert_eq!(history.len(), 1);
    assert_exact!(history[0].v, 42.5);
    // The clean tag comes verbatim from the live contention verdict --
    // asserted against the public verdict, not a hardcoded bool, so the
    // test holds whether or not this box happens to be busy.
    assert_eq!(history[0].clean, machine_is_uncontended());
    assert_eq!(caps["rate"], 42.5, "single sample IS the estimate");
    drop(env);
}

#[test]
#[serial]
fn a_legacy_scalar_is_seeded_once_as_an_unclean_sample_then_outvoted() {
    let env = TestEnv::new();
    let mut signals: SignalStore =
        serde_json::from_str(r#"{"m": {"_capabilities": {"rate": 100.0}}}"#).unwrap();
    record_capability_sample(&mut signals, "m", "rate", 42.0);
    let history: Vec<Sample> =
        serde_json::from_value(signals["m"]["_capabilities"]["rate_samples"].clone()).unwrap();
    assert_eq!(history.len(), 2);
    assert_exact!(history[0].v, 100.0);
    assert_eq!(history[0].legacy, Some(true), "scalar seed marked legacy");
    assert!(!history[0].clean, "scalar seed never trusted as clean");
    assert_exact!(history[1].v, 42.0);
    // Estimate depends on the live clean tag: a clean reading outvotes the
    // legacy scalar outright; otherwise both count and the median wins.
    let expected = if history[1].clean { 42.0 } else { 71.0 };
    assert_eq!(signals["m"]["_capabilities"]["rate"], expected);
    drop(env);
}

#[test]
fn time_taken_without_retries_below_one_tick_is_noise() {
    let mut signals = SignalStore::new();
    record_signal(
        &mut signals,
        "m",
        "never-registered-noise-task",
        0.0,
        false,
        false,
    );
    assert_empty!(signals);
}

#[test]
#[serial]
fn first_observation_seeds_p95_and_the_learned_timeout() {
    let env = TestEnv::new();
    env.set("EVAL_DEFAULT_TIMEOUT", "300");
    let mut signals = SignalStore::new();
    let seed_task = task("learned-timeout");
    record_task_signal(&mut signals, "m", &seed_task, 100.0, false, false);
    let row = &signals["m"]["learned-timeout"];
    assert_eq!(row["samples"], 1);
    assert_eq!(row["p95_latency"], 100.0);
    assert_eq!(row["total_retries"], 0);
    assert_eq!(row["parse_failures"], 0);
    // max(documented floor 300, p95 * 1.5 = 150).
    assert_eq!(row["timeout"], 300);

    // The SAME task again: same fingerprint, so the series accumulates.
    record_task_signal(&mut signals, "m", &seed_task, 300.0, true, true);
    let row = &signals["m"]["learned-timeout"];
    assert_eq!(row["samples"], 2);
    assert_eq!(row["total_retries"], 1);
    assert_eq!(row["parse_failures"], 1);
    // p95 EMA upward: max(300, 100*0.95 + 300*0.05) = 300; timeout now
    // max(300, 450) = 450.
    assert_eq!(row["p95_latency"], 300.0);
    assert_eq!(
        row["timeout"], 450,
        "the learned timeout grows past the floor"
    );
    drop(env);
}

#[test]
fn p95_blends_downward_but_never_below_the_new_observation_floor() {
    let seed = task("p95-blend");
    // A series already accumulated against the SAME task, which is the only
    // state an EMA can blend into (see `identity.rs` for the other one).
    let mut signals: SignalStore = serde_json::from_str(&format!(
        r#"{{"m": {{"p95-blend": {{"samples": 5, "p95_latency": 10.0, "fingerprint": "{}"}}}}}}"#,
        task_fingerprint(&seed)
    ))
    .unwrap();
    record_task_signal(&mut signals, "m", &seed, 5.0, false, false);
    // EMA: 10*0.95 + 5*0.05 = 9.75 beats the raw 5; json_p95 rounds to 0.1.
    assert_eq!(signals["m"]["p95-blend"]["p95_latency"], 9.8);
    assert_eq!(signals["m"]["p95-blend"]["samples"], 6);
}

#[test]
fn retries_alone_count_without_touching_p95_or_timeout() {
    let mut signals = SignalStore::new();
    record_task_signal(&mut signals, "m", &task("retries-only"), 0.0, true, false);
    let row = &signals["m"]["retries-only"];
    assert_eq!(row["samples"], 1);
    assert_eq!(row["total_retries"], 1);
    assert!(
        row.get("p95_latency").is_none(),
        "a zero-duration sample sets no p95"
    );
    assert!(
        row.get("timeout").is_none(),
        "no p95 means no learned timeout"
    );
}

#[test]
#[serial]
fn an_unchanged_learned_timeout_is_not_rewritten() {
    let env = TestEnv::new();
    env.set("EVAL_DEFAULT_TIMEOUT", "300");
    let mut signals = SignalStore::new();
    let seed_task = task("timeout-stable");
    record_task_signal(&mut signals, "m", &seed_task, 100.0, false, false);
    let before = signals["m"]["timeout-stable"]["timeout"].clone();
    assert_eq!(before, 300);
    // Same task again: same new_timeout, insert skipped.
    record_task_signal(&mut signals, "m", &seed_task, 100.0, false, false);
    assert_eq!(signals["m"]["timeout-stable"]["timeout"], before);
    assert_eq!(signals["m"]["timeout-stable"]["samples"], 2);
    drop(env);
}
