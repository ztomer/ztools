//! What gets learned: capability samples, p95, retries, and the clean tag.
//!
//! The clean tag is the one thing here that comes from the live machine, so it
//! is asserted AGAINST the public verdict rather than against a hardcoded
//! bool: the test then holds whether or not this box happens to be busy, and
//! stops being a statement about the developer's afternoon.

use super::super::*;
use crate::test_env::TestEnv;
use serial_test::serial;

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
    record_signal(&mut signals, "m", "t", 0.0, false, false);
    assert_empty!(signals);
}

#[test]
#[serial]
fn first_observation_seeds_p95_and_the_learned_timeout() {
    let env = TestEnv::new();
    env.set("EVAL_DEFAULT_TIMEOUT", "300");
    let mut signals = SignalStore::new();
    record_signal(&mut signals, "m", "t", 100.0, false, false);
    let task = &signals["m"]["t"];
    assert_eq!(task["samples"], 1);
    assert_eq!(task["p95_latency"], 100.0);
    assert_eq!(task["total_retries"], 0);
    assert_eq!(task["parse_failures"], 0);
    // max(documented floor 300, p95 * 1.5 = 150).
    assert_eq!(task["timeout"], 300);

    record_signal(&mut signals, "m", "t", 300.0, true, true);
    let task = &signals["m"]["t"];
    assert_eq!(task["samples"], 2);
    assert_eq!(task["total_retries"], 1);
    assert_eq!(task["parse_failures"], 1);
    // p95 EMA upward: max(300, 100*0.95 + 300*0.05) = 300; timeout now
    // max(300, 450) = 450.
    assert_eq!(task["p95_latency"], 300.0);
    assert_eq!(
        task["timeout"], 450,
        "the learned timeout grows past the floor"
    );
    drop(env);
}

#[test]
fn p95_blends_downward_but_never_below_the_new_observation_floor() {
    let mut signals: SignalStore =
        serde_json::from_str(r#"{"m": {"t": {"samples": 5, "p95_latency": 10.0}}}"#).unwrap();
    record_signal(&mut signals, "m", "t", 5.0, false, false);
    // EMA: 10*0.95 + 5*0.05 = 9.75 beats the raw 5; json_p95 rounds to 0.1.
    assert_eq!(signals["m"]["t"]["p95_latency"], 9.8);
    assert_eq!(signals["m"]["t"]["samples"], 6);
}

#[test]
fn retries_alone_count_without_touching_p95_or_timeout() {
    let mut signals = SignalStore::new();
    record_signal(&mut signals, "m", "t", 0.0, true, false);
    let task = &signals["m"]["t"];
    assert_eq!(task["samples"], 1);
    assert_eq!(task["total_retries"], 1);
    assert!(
        task.get("p95_latency").is_none(),
        "a zero-duration sample sets no p95"
    );
    assert!(
        task.get("timeout").is_none(),
        "no p95 means no learned timeout"
    );
}

#[test]
#[serial]
fn an_unchanged_learned_timeout_is_not_rewritten() {
    let env = TestEnv::new();
    env.set("EVAL_DEFAULT_TIMEOUT", "300");
    let mut signals = SignalStore::new();
    record_signal(&mut signals, "m", "t", 100.0, false, false);
    let before = signals["m"]["t"]["timeout"].clone();
    assert_eq!(before, 300);
    // Same observation again: same new_timeout, insert skipped.
    record_signal(&mut signals, "m", "t", 100.0, false, false);
    assert_eq!(signals["m"]["t"]["timeout"], before);
    assert_eq!(signals["m"]["t"]["samples"], 2);
    drop(env);
}
