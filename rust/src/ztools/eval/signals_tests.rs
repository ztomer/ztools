use super::*;

#[test]
fn swap_used_gb_reads_the_real_sysctl_output() {
    // REGRESSION: the parser used to grab the literal token "used" and
    // then fail to parse it, so memory_pressure() was None on every
    // machine -- every capability sample UNVERIFIED, derived_timeout
    // permanently 0. The happy path must work against the REAL output.
    let Some((swap, compressor)) = memory_pressure() else {
        panic!("memory_pressure() returned None on a live macOS box");
    };
    assert!(swap >= 0.0 && swap.is_finite(), "swap: {swap}");
    assert!(compressor >= 0.0 && compressor.is_finite(), "compressor: {compressor}");
    // Cross-check the swap figure by parsing sysctl independently.
    let out = Command::new("sysctl").arg("-n").arg("vm.swapusage").output().unwrap();
    let text = String::from_utf8_lossy(&out.stdout);
    let expected: f64 = text
        .split_whitespace()
        .find(|t| t.ends_with('M') || t.ends_with('G'))
        .and_then(|first_total| {
            // first match is total's value; walk to the one after "used"
            let mut seen_used = false;
            for tok in text.split_whitespace() {
                if tok == "used" {
                    seen_used = true;
                    continue;
                }
                if seen_used && (tok.ends_with('M') || tok.ends_with('G')) {
                    return tok.trim_end_matches(['M', 'G']).parse::<f64>().ok();
                }
            }
            first_total.parse::<f64>().ok()
        })
        .expect("swapusage contains a used value") / 1024.0;
    assert!(
        (swap - expected).abs() < 0.01,
        "parsed {swap} GB vs independent parse {expected} GB"
    );
}


use serial_test::serial;

/// Restores every env var a signals test touches, whatever the test did
/// to it -- including the never-set case.
struct EnvGuard {
    saved: Vec<(&'static str, Option<std::ffi::OsString>)>,
}

impl EnvGuard {
    fn capture(keys: &[&'static str]) -> Self {
        let saved = keys
            .iter()
            .map(|k| (*k, std::env::var_os(k)))
            .collect();
        Self { saved }
    }
}

impl Drop for EnvGuard {
    fn drop(&mut self) {
        for (key, prev) in self.saved.drain(..) {
            match prev {
                Some(v) => std::env::set_var(key, v),
                None => std::env::remove_var(key),
            }
        }
    }
}

/// Point EVAL_SIGNALS_DIR at an empty tempdir so neither the operator's
/// real eval_signals.json nor a peer session's GPU lock can decide a
/// test's outcome.
struct Fixture {
    _dir: tempfile::TempDir,
    guard: EnvGuard,
}

impl Fixture {
    fn new(extra_keys: &[&'static str]) -> Self {
        let dir = tempfile::tempdir().unwrap();
        let mut keys: Vec<&'static str> =
            vec!["EVAL_SIGNALS_DIR", "ZTOOLS_CONF_DIR", "ZTOOLS_GPU_LOCK_DIR"];
        keys.extend_from_slice(extra_keys);
        let guard = EnvGuard::capture(&keys);
        std::env::set_var("EVAL_SIGNALS_DIR", dir.path().join("signals"));
        std::fs::create_dir_all(dir.path().join("signals")).unwrap();
        // Empty lock dir: no owner file, so foreign_holder() is None and
        // the contention verdict comes from memory pressure alone.
        std::env::set_var("ZTOOLS_GPU_LOCK_DIR", dir.path().join("lock"));
        // An EMPTY conf root: the real checkout's conf/config.toml must
        // never decide a timeout expectation.
        let conf = dir.path().join("conf");
        std::fs::create_dir_all(&conf).unwrap();
        std::env::set_var("ZTOOLS_CONF_DIR", &conf);
        Self { _dir: dir, guard }
    }

    fn write_signals_file(&self, content: &str) {
        std::fs::write(self._dir.path().join("signals").join("eval_signals.json"), content)
            .unwrap();
    }
}

#[test]
#[serial]
fn timeout_env_overrides_parse_or_fall_back_to_documented_defaults() {
    // Removed BEFORE the guard captures, so teardown exercises the
    // never-set restore arm deterministically.
    std::env::remove_var("EVAL_DEFAULT_TIMEOUT");
    std::env::remove_var("EVAL_MAX_TIMEOUT");
    let dir = Fixture::new(&["EVAL_DEFAULT_TIMEOUT", "EVAL_MAX_TIMEOUT"]);
    let _g = &dir.guard;
    assert_eq!(default_eval_timeout(), 900);
    assert_eq!(max_eval_timeout(), 7200);
    std::env::set_var("EVAL_DEFAULT_TIMEOUT", "123");
    std::env::set_var("EVAL_MAX_TIMEOUT", "456");
    assert_eq!(default_eval_timeout(), 123);
    assert_eq!(max_eval_timeout(), 456);
    std::env::set_var("EVAL_DEFAULT_TIMEOUT", "garbage");
    std::env::set_var("EVAL_MAX_TIMEOUT", "-7");
    assert_eq!(default_eval_timeout(), 900, "unparsable falls back");
    assert_eq!(max_eval_timeout(), 7200, "unparsable falls back");
}

#[test]
#[serial]
fn signals_path_honors_the_env_dir_and_the_documented_default() {
    let dir = Fixture::new(&[]);
    let _g = &dir.guard;
    assert_eq!(
        signals_path(),
        dir._dir.path().join("signals").join("eval_signals.json")
    );
    std::env::remove_var("EVAL_SIGNALS_DIR");
    // Without the env override the path is anchored on the CHECKOUT that
    // built the binary (manifest/../conf), never the process working
    // directory -- a relative conf/ once forked the store into whatever
    // directory a sweep happened to start from.
    let anchored = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("manifest has a parent")
        .join("conf")
        .join("eval_signals.json");
    assert_eq!(signals_path(), anchored);
}

#[test]
#[serial]
fn load_degrades_to_empty_and_save_roundtrips() {
    // A sentinel guarantees the guard's teardown exercises its was-set
    // restore arm for this key.
    std::env::set_var("EVAL_SIGNALS_DIR", "/nonexistent-sentinel");
    let dir = Fixture::new(&[]);
    let _g = &dir.guard;
    assert!(load_signals().is_empty(), "missing file is an empty store");

    dir.write_signals_file("{not json at all");
    assert!(load_signals().is_empty(), "malformed file is an empty store");

    let mut store = SignalStore::new();
    store.insert(
        "m".to_string(),
        serde_json::json!({"task": {"timeout": 42}}),
    );
    save_signals(&store);
    let loaded = load_signals();
    assert_eq!(loaded.len(), 1, "saved entry survives the roundtrip");
    assert_eq!(loaded["m"]["task"]["timeout"], 42);
}

#[test]
#[serial]
fn an_unreadable_pressure_reading_is_never_evidence_of_contention() {
    // Documented contract: None means "cannot tell" and uncontended must
    // then be FALSE -- an unverifiable sample must not masquerade as clean.
    match memory_pressure() {
        None => assert!(!machine_is_uncontended()),
        Some((swap, compressor)) => {
            assert!(swap.is_finite() && swap >= 0.0);
            assert!(compressor.is_finite() && compressor >= 0.0);
            assert_eq!(
                machine_is_uncontended(),
                swap <= MAX_CLEAN_SWAP_GB && compressor <= MAX_CLEAN_COMPRESSOR_GB
            );
        }
    }
    // The same contract through the foreign-holder seam: with no owner
    // file the verdict must come from pressure alone.
    let dir = Fixture::new(&[]);
    let _g = &dir.guard;
    assert!(foreign_holder().is_none());
    let expected = match memory_pressure() {
        None => false,
        Some((swap, compressor)) => {
            swap <= MAX_CLEAN_SWAP_GB && compressor <= MAX_CLEAN_COMPRESSOR_GB
        }
    };
    assert_eq!(machine_is_uncontended(), expected);
}

#[test]
#[serial]
fn a_live_foreign_lock_holder_makes_the_machine_contended() {
    let dir = Fixture::new(&[]);
    let _g = &dir.guard;
    // The parent process is alive, belongs to this user (so kill(pid, 0)
    // succeeds -- signalling launchd would EPERM), and is not this
    // process: a real foreign holder by the lock's own liveness rules.
    let holder_pid = std::os::unix::process::parent_id();
    let start = crate::ztools::eval::gpu_lock::start_time(holder_pid);
    std::fs::create_dir_all(dir._dir.path().join("lock")).unwrap();
    std::fs::write(
        dir._dir.path().join("lock").join("owner"),
        format!("{holder_pid}\n{start}\na concurrent eval run\n"),
    )
    .unwrap();
    assert!(
        foreign_holder().is_some(),
        "pid {holder_pid} holding the fixture lock is foreign"
    );
    assert!(!machine_is_uncontended());
}

#[test]
fn nonpositive_or_nonfinite_values_are_never_recorded() {
    let mut signals = SignalStore::new();
    record_capability_sample(&mut signals, "m", "rate", 0.0);
    record_capability_sample(&mut signals, "m", "rate", -3.0);
    record_capability_sample(&mut signals, "m", "rate", f64::NAN);
    assert!(signals.is_empty(), "{signals:?}");
}

#[test]
#[serial]
fn recording_a_sample_creates_capabilities_and_rederives_the_estimate() {
    let dir = Fixture::new(&[]);
    let _g = &dir.guard;
    let mut signals = SignalStore::new();
    record_capability_sample(&mut signals, "m", "rate", 42.5);
    let caps = &signals["m"]["_capabilities"];
    let history: Vec<Sample> =
        serde_json::from_value(caps["rate_samples"].clone()).unwrap();
    assert_eq!(history.len(), 1);
    assert_eq!(history[0].v, 42.5);
    // The clean tag comes verbatim from the live contention verdict --
    // asserted against the public verdict, not a hardcoded bool, so the
    // test holds whether or not this box happens to be busy.
    assert_eq!(history[0].clean, machine_is_uncontended());
    assert_eq!(caps["rate"], 42.5, "single sample IS the estimate");
}

#[test]
#[serial]
fn a_legacy_scalar_is_seeded_once_as_an_unclean_sample_then_outvoted() {
    let dir = Fixture::new(&[]);
    let _g = &dir.guard;
    let mut signals: SignalStore =
        serde_json::from_str(r#"{"m": {"_capabilities": {"rate": 100.0}}}"#).unwrap();
    record_capability_sample(&mut signals, "m", "rate", 42.0);
    let history: Vec<Sample> = serde_json::from_value(
        signals["m"]["_capabilities"]["rate_samples"].clone(),
    )
    .unwrap();
    assert_eq!(history.len(), 2);
    assert_eq!(history[0].v, 100.0);
    assert_eq!(history[0].legacy, Some(true), "scalar seed marked legacy");
    assert!(!history[0].clean, "scalar seed never trusted as clean");
    assert_eq!(history[1].v, 42.0);
    // Estimate depends on the live clean tag: a clean reading outvotes the
    // legacy scalar outright; otherwise both count and the median wins.
    let expected = if history[1].clean { 42.0 } else { 71.0 };
    assert_eq!(signals["m"]["_capabilities"]["rate"], expected);
}

fn capabilities_fixture() -> String {
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

#[test]
#[serial]
fn derived_timeout_is_zero_unless_all_three_terms_are_measured_clean() {
    let dir = Fixture::new(&[]);
    let _g = &dir.guard;
    assert_eq!(derived_timeout("m", 1000, 100), 0, "no data at all");

    dir.write_signals_file(r#"{"m": {"_capabilities": {
        "prefill_chars_per_sec_samples": [{"v": 500.0, "clean": true}],
        "decode_tokens_per_sec_samples": [{"v": 20.0, "clean": true}]
    }}}"#);
    assert_eq!(derived_timeout("m", 1000, 100), 0, "cold_start missing");

    dir.write_signals_file(r#"{"m": {"_capabilities": {
        "prefill_chars_per_sec_samples": [{"v": 500.0, "clean": false}],
        "decode_tokens_per_sec_samples": [{"v": 20.0, "clean": true}],
        "cold_start_seconds_samples": [{"v": 2.0, "clean": true}]
    }}}"#);
    assert_eq!(derived_timeout("m", 1000, 100), 0, "unclean prefill disqualifies");

    dir.write_signals_file(r#"{"m": {"_capabilities": {
        "prefill_chars_per_sec_samples": [{"v": 0.0, "clean": true}],
        "decode_tokens_per_sec_samples": [{"v": 20.0, "clean": true}],
        "cold_start_seconds_samples": [{"v": 2.0, "clean": true}]
    }}}"#);
    assert_eq!(derived_timeout("m", 1000, 100), 0, "a zero estimate is no estimate");
}

#[test]
#[serial]
fn derived_timeout_caps_at_max_eval_timeout_and_sums_the_terms() {
    let dir = Fixture::new(&["EVAL_MAX_TIMEOUT"]);
    dir.write_signals_file(&capabilities_fixture());
    let _g = &dir.guard;
    // 2s cold + 1000/500 prefill + 100/20 decode = 9s; x1.5 = 13.5 -> 13.
    assert_eq!(derived_timeout("m", 1000, 100), 13);

    std::env::set_var("EVAL_MAX_TIMEOUT", "10");
    assert_eq!(
        derived_timeout("m", 1000, 100),
        max_eval_timeout(),
        "the policy ceiling caps an inflated derivation"
    );
}

#[test]
#[serial]
fn effective_timeout_takes_the_largest_of_learned_configured_derived_and_default() {
    let dir = Fixture::new(&["EVAL_DEFAULT_TIMEOUT"]);
    let _g = &dir.guard;
    std::env::set_var("EVAL_DEFAULT_TIMEOUT", "300");
    // Empty conf root: no config.toml, so the configured term is the
    // documented 600 fallback; derived is 0 (no capability samples).
    assert_eq!(
        effective_timeout("m", "t1", 0, 0),
        600,
        "documented fallback when neither learned nor configured exists"
    );
    dir.write_signals_file(r#"{"m": {"t1": {"timeout": 4000}}}"#);
    assert_eq!(effective_timeout("m", "t1", 0, 0), 4000, "learned term wins");

    // The default floor can be raised above everything via env.
    std::env::set_var("EVAL_DEFAULT_TIMEOUT", "8001");
    assert_eq!(
        effective_timeout("m", "t1", 0, 0),
        8001,
        "the default floor participates in the max"
    );
}

#[test]
#[serial]
fn effective_timeout_reads_the_configured_timeouts_table_via_conf_root() {
    let conf = tempfile::tempdir().unwrap();
    std::fs::write(
        conf.path().join("config.toml"),
        "[timeouts]\nmytask = 5000\nzeroed = 0\n",
    )
    .unwrap();
    let dir = Fixture::new(&["EVAL_DEFAULT_TIMEOUT"]);
    let _g = &dir.guard;
    std::env::set_var("ZTOOLS_CONF_DIR", conf.path());

    std::env::set_var("EVAL_DEFAULT_TIMEOUT", "300");
    dir.write_signals_file(r#"{"m": {"mytask": {"timeout": 100}}}"#);
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
    // ZTOOLS_CONF_DIR is restored by the Fixture's guard on drop.
}

#[test]
fn time_taken_without_retries_below_one_tick_is_noise() {
    let mut signals = SignalStore::new();
    record_signal(&mut signals, "m", "t", 0.0, false, false);
    assert!(signals.is_empty());
}

#[test]
#[serial]
fn first_observation_seeds_p95_and_the_learned_timeout() {
    let _g = EnvGuard::capture(&["EVAL_DEFAULT_TIMEOUT"]);
    std::env::set_var("EVAL_DEFAULT_TIMEOUT", "300");
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
    assert_eq!(task["timeout"], 450, "the learned timeout grows past the floor");
}

#[test]
fn p95_blends_downward_but_never_below_the_new_observation_floor() {
    let mut signals: SignalStore = serde_json::from_str(
        r#"{"m": {"t": {"samples": 5, "p95_latency": 10.0}}}"#,
    )
    .unwrap();
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
    assert!(task.get("p95_latency").is_none(), "a zero-duration sample sets no p95");
    assert!(task.get("timeout").is_none(), "no p95 means no learned timeout");
}

#[test]
#[serial]
fn an_unchanged_learned_timeout_is_not_rewritten() {
    let _g = EnvGuard::capture(&["EVAL_DEFAULT_TIMEOUT"]);
    std::env::set_var("EVAL_DEFAULT_TIMEOUT", "300");
    let mut signals = SignalStore::new();
    record_signal(&mut signals, "m", "t", 100.0, false, false);
    let before = signals["m"]["t"]["timeout"].clone();
    assert_eq!(before, 300);
    // Same observation again: same new_timeout, insert skipped.
    record_signal(&mut signals, "m", "t", 100.0, false, false);
    assert_eq!(signals["m"]["t"]["timeout"], before);
    assert_eq!(signals["m"]["t"]["samples"], 2);
}
