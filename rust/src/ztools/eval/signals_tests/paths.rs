//! Where the store lives: `signals_path`, the timeout knobs, and the conf root.
//!
//! These are the tests that used to carry a local `EnvGuard` and a local
//! `Fixture`, each with its own variable list.

use serial_test::serial;

use super::super::*;
use crate::test_env::TestEnv;

#[test]
#[serial]
fn timeout_env_overrides_parse_or_fall_back_to_documented_defaults() {
    // The guard CLEARS both knobs and restores whatever they were, so the
    // documented fallbacks below are reached by the variable being genuinely
    // absent -- not by this test having removed it and forgotten to put it back.
    let env = TestEnv::new();
    assert!(std::env::var_os("EVAL_DEFAULT_TIMEOUT").is_none());
    assert!(std::env::var_os("EVAL_MAX_TIMEOUT").is_none());
    assert_eq!(default_eval_timeout(), 900);
    assert_eq!(max_eval_timeout(), 7200);

    env.set("EVAL_DEFAULT_TIMEOUT", "123");
    env.set("EVAL_MAX_TIMEOUT", "456");
    assert_eq!(default_eval_timeout(), 123);
    assert_eq!(max_eval_timeout(), 456);

    env.set("EVAL_DEFAULT_TIMEOUT", "garbage");
    env.set("EVAL_MAX_TIMEOUT", "-7");
    assert_eq!(default_eval_timeout(), 900, "unparsable falls back");
    assert_eq!(max_eval_timeout(), 7200, "unparsable falls back");
    drop(env);
}

#[test]
#[serial]
fn signals_path_honors_the_env_dir_and_the_documented_default() {
    let env = TestEnv::new();
    assert_eq!(
        signals_path(),
        env.path("EVAL_SIGNALS_DIR").join("eval_signals.json")
    );
    env.unset("EVAL_SIGNALS_DIR");
    // Without the env override the path is anchored on the CHECKOUT that built
    // the binary (manifest/../conf), never the process working directory -- a
    // relative conf/ once forked the store into whatever directory a sweep
    // happened to start from.
    let anchored = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("manifest has a parent")
        .join("conf")
        .join("eval_signals.json");
    assert_eq!(signals_path(), anchored);
    drop(env);
}

#[test]
#[serial]
fn load_degrades_to_empty_and_save_roundtrips() {
    let env = TestEnv::new();
    assert_empty!(load_signals(), "missing file is an empty store");

    env.write("EVAL_SIGNALS_DIR", "eval_signals.json", "{not json at all");
    assert_empty!(load_signals(), "malformed file is an empty store");

    let mut store = SignalStore::new();
    store.insert(
        "m".to_string(),
        serde_json::json!({"task": {"timeout": 42}}),
    );
    save_signals(&store);
    let loaded = load_signals();
    assert_eq!(loaded.len(), 1, "saved entry survives the roundtrip");
    assert_eq!(loaded["m"]["task"]["timeout"], 42);
    drop(env);
}

#[test]
#[serial]
fn the_capability_estimator_rederives_only_from_clean_samples() {
    let env = TestEnv::new();
    assert_eq!(
        derived_timeout("no-such-model", 1000, 100),
        0,
        "an unmeasured model has no estimate"
    );
    env.write(
        "EVAL_SIGNALS_DIR",
        "eval_signals.json",
        &capabilities_fixture(),
    );
    assert_eq!(
        derived_timeout("m", 1000, 100),
        13,
        "2s cold + 2s prefill + 5s decode = 9s, x1.5 = 13.5 -> 13"
    );
    drop(env);
}

#[test]
#[serial]
fn derived_timeout_is_zero_unless_all_three_terms_are_measured_clean() {
    let env = TestEnv::new();
    assert_eq!(derived_timeout("m", 1000, 100), 0, "no data at all");

    env.write(
        "EVAL_SIGNALS_DIR",
        "eval_signals.json",
        r#"{"m": {"_capabilities": {
        "prefill_chars_per_sec_samples": [{"v": 500.0, "clean": true}],
        "decode_tokens_per_sec_samples": [{"v": 20.0, "clean": true}]
    }}}"#,
    );
    assert_eq!(derived_timeout("m", 1000, 100), 0, "cold_start missing");

    env.write(
        "EVAL_SIGNALS_DIR",
        "eval_signals.json",
        r#"{"m": {"_capabilities": {
        "prefill_chars_per_sec_samples": [{"v": 500.0, "clean": false}],
        "decode_tokens_per_sec_samples": [{"v": 20.0, "clean": true}],
        "cold_start_seconds_samples": [{"v": 2.0, "clean": true}]
    }}}"#,
    );
    assert_eq!(
        derived_timeout("m", 1000, 100),
        0,
        "unclean prefill disqualifies"
    );

    env.write(
        "EVAL_SIGNALS_DIR",
        "eval_signals.json",
        r#"{"m": {"_capabilities": {
        "prefill_chars_per_sec_samples": [{"v": 0.0, "clean": true}],
        "decode_tokens_per_sec_samples": [{"v": 20.0, "clean": true}],
        "cold_start_seconds_samples": [{"v": 2.0, "clean": true}]
    }}}"#,
    );
    assert_eq!(
        derived_timeout("m", 1000, 100),
        0,
        "a zero estimate is no estimate"
    );
    drop(env);
}

#[test]
#[serial]
fn derived_timeout_caps_at_max_eval_timeout() {
    let env = TestEnv::new();
    env.write(
        "EVAL_SIGNALS_DIR",
        "eval_signals.json",
        &capabilities_fixture(),
    );
    env.set("EVAL_MAX_TIMEOUT", "10");
    assert_eq!(
        derived_timeout("m", 1000, 100),
        max_eval_timeout(),
        "the policy ceiling caps an inflated derivation"
    );
    assert_eq!(
        max_eval_timeout(),
        10,
        "and the ceiling is the knob's value"
    );
    drop(env);
}

#[test]
#[serial]
fn effective_timeout_takes_the_largest_of_learned_configured_derived_and_default() {
    let env = TestEnv::new();
    env.set("EVAL_DEFAULT_TIMEOUT", "300");
    // Empty conf root: no config.toml, so the configured term is the documented
    // 600 fallback; derived is 0 (no capability samples).
    assert_eq!(
        effective_timeout("m", "t1", 0, 0),
        600,
        "documented fallback when neither learned nor configured exists"
    );
    env.write(
        "EVAL_SIGNALS_DIR",
        "eval_signals.json",
        r#"{"m": {"t1": {"timeout": 4000}}}"#,
    );
    assert_eq!(
        effective_timeout("m", "t1", 0, 0),
        4000,
        "learned term wins"
    );

    env.set("EVAL_DEFAULT_TIMEOUT", "8001");
    assert_eq!(
        effective_timeout("m", "t1", 0, 0),
        8001,
        "the default floor participates in the max"
    );
    drop(env);
}

#[test]
#[serial]
fn effective_timeout_reads_the_configured_timeouts_table_via_conf_root() {
    let env = TestEnv::new();
    let conf = env.set_path("ZTOOLS_CONF_DIR", "fixture-conf");
    std::fs::write(
        conf.join("config.toml"),
        "[timeouts]\nmytask = 5000\nzeroed = 0\n",
    )
    .unwrap();
    env.set("EVAL_DEFAULT_TIMEOUT", "300");
    env.write(
        "EVAL_SIGNALS_DIR",
        "eval_signals.json",
        r#"{"m": {"mytask": {"timeout": 100}}}"#,
    );
    assert_eq!(
        effective_timeout("m", "mytask", 0, 0),
        5000,
        "configured table term beats learned, fallback and default"
    );
    assert_eq!(
        effective_timeout("m", "zeroed", 0, 0),
        600,
        "nonpositive table entries are ignored"
    );
    assert_eq!(
        effective_timeout("m", "absent-task", 0, 0),
        600,
        "untabled tasks use the documented fallback"
    );
    drop(env);
}

pub(super) fn capabilities_fixture() -> String {
    r#"{"m": {"_capabilities": {
        "prefill_chars_per_sec": 500,
        "prefill_chars_per_sec_samples": [{"v": 500.0, "clean": true}],
        "decode_tokens_per_sec": 20,
        "decode_tokens_per_sec_samples": [{"v": 20.0, "clean": true}],
        "cold_start_seconds": 2,
        "cold_start_seconds_samples": [{"v": 2.0, "clean": true}]
    }}}"#
        .to_string()
}
