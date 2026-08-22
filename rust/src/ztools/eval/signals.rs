//! Eval signal store: per-model, per-task observations accumulated across runs,
//! plus the learned-timeout arithmetic built on them.
//!
//! Ported from `references/eval/signals.py` and the pressure half of
//! `eval/memory.py`. The store backs three consumers: the eval loop records
//! into it, `_effective_timeout` sizes request timeouts from it, and the
//! capability samples feed the median-of-clean estimator (`samples.rs`).
//!
//! Contention honesty: a sample is tagged CLEAN only when no foreign GPU-lock
//! holder exists AND memory pressure is verifiably low. Pressure that cannot be
//! read marks the sample UNVERIFIED (unclean), never clean -- inventing a
//! healthy reading is exactly how a contended machine's numbers got enshrined
//! in the Python original's history.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::process::Command;

use serde_json::Value;

use crate::ztools::eval::gpu_lock::foreign_holder;
use crate::ztools::eval::samples::{clean_estimate, migrate_sample_history, Sample};

/// A POLICY ceiling, not an estimate: past this a request is assumed wedged.
/// Deliberately not derived -- its job is to bound the damage when the
/// measurements are wrong.
pub const MAX_EVAL_TIMEOUT: u64 = 7200;

const TIMEOUT_SAFETY_FACTOR: f64 = 1.5;
pub const MAX_CLEAN_SWAP_GB: f64 = 8.0;
pub const MAX_CLEAN_COMPRESSOR_GB: f64 = 15.0;
const BYTES_PER_GB: f64 = 1024.0 * 1024.0 * 1024.0;
/// macOS page size on arm64/x86_64.
const PAGE_BYTES: f64 = 16384.0;

fn env_u64(key: &str, default: u64) -> u64 {
    std::env::var(key).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

pub fn default_eval_timeout() -> u64 {
    env_u64("EVAL_DEFAULT_TIMEOUT", 900)
}

pub fn max_eval_timeout() -> u64 {
    env_u64("EVAL_MAX_TIMEOUT", 7200)
}

/// Where eval_signals.json lives. Env-overridable so tests can point it at tmp.
///
/// Anchored on the CHECKOUT that built the binary (`CARGO_MANIFEST_DIR/../conf`)
/// rather than the process working directory: a sweep launched from anywhere
/// else used to read -- and worse, CREATE -- a relative `conf/eval_signals.json`
/// in whatever directory it started from, silently losing every recorded
/// capability and forking the store. The home fallback covers an installed
/// binary with no checkout nearby.
pub fn signals_path() -> PathBuf {
    let dir = if let Ok(d) = std::env::var("EVAL_SIGNALS_DIR") {
        d
    } else {
        let candidates = [
            std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .parent()
                .map(|p| p.join("conf")),
            dirs::home_dir().map(|h| h.join("Projects/ztools/conf")),
        ];
        match candidates.into_iter().flatten().find(|p| p.is_dir()) {
            Some(p) => p.to_string_lossy().to_string(),
            None => "conf".to_string(),
        }
    };
    PathBuf::from(dir).join("eval_signals.json")
}

pub type SignalStore = BTreeMap<String, Value>;

pub fn load_signals() -> SignalStore {
    let path = signals_path();
    match std::fs::read_to_string(&path) {
        Ok(text) => serde_json::from_str(&text).unwrap_or_default(),
        Err(_) => SignalStore::new(),
    }
}

pub fn save_signals(signals: &SignalStore) {
    if let Some(parent) = signals_path().parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    if let Ok(text) = serde_json::to_string_pretty(signals) {
        let _ = std::fs::write(signals_path(), text);
    }
}

/// (swap_used_gb, compressor_gb), or None when they cannot be read -- which
/// every caller must treat as "cannot tell", never as "fine".
pub fn memory_pressure() -> Option<(f64, f64)> {
    let swap_gb = swap_used_gb()?;
    let compressor_gb = compressor_gb()?;
    Some((swap_gb, compressor_gb))
}

fn swap_used_gb() -> Option<f64> {
    // `sysctl -n vm.swapusage` -> "total = 4096.00M  used = 512.25M  free = ..."
    //
    // The VALUE follows the literal token "used" (and an "=" sign). Grabbing
    // the token that STARTS WITH "used" grabs "used" itself, whose
    // `.split('=').nth(1)` is None -- so this function returned None on every
    // machine, every time, which tagged every capability sample UNVERIFIED,
    // zeroed derived_timeout, and silently disabled the median-of-clean
    // estimator's recovery path. Found by coverage work: the happy path was
    // unreachable.
    let out = Command::new("sysctl").arg("-n").arg("vm.swapusage").output().ok()?;
    let text = String::from_utf8_lossy(&out.stdout);
    let mut tokens = text.split_whitespace();
    while let Some(token) = tokens.next() {
        if token != "used" {
            continue;
        }
        for field in tokens.by_ref() {
            if field == "=" {
                continue;
            }
            let value: f64 = field.trim_end_matches(['M', 'G']).parse().ok()?;
            let multiplier = if field.ends_with('G') { 1.0 } else { 1.0 / 1024.0 };
            return Some(value * multiplier);
        }
    }
    None
}

fn compressor_gb() -> Option<f64> {
    let out = Command::new("/usr/bin/vm_stat").output().ok()?;
    let text = String::from_utf8_lossy(&out.stdout);
    let line = text.lines().find(|l| l.starts_with("Pages occupied by compressor"))?;
    let raw = line.split(':').nth(1)?;
    let pages: f64 = raw.trim().trim_end_matches('.').parse().ok()?;
    Some(pages * PAGE_BYTES / BYTES_PER_GB)
}

/// Is the machine quiet enough for a timing to mean anything? False also when
/// it cannot tell -- an unverifiable sample must not masquerade as clean.
pub fn machine_is_uncontended() -> bool {
    if foreign_holder().is_some() {
        return false;
    }
    match memory_pressure() {
        None => false,
        Some((swap_gb, compressor_gb)) => {
            swap_gb <= MAX_CLEAN_SWAP_GB && compressor_gb <= MAX_CLEAN_COMPRESSOR_GB
        }
    }
}

/// Add one observation of `key` under the model's capabilities, re-derive the
/// estimate. Median of recent CLEAN samples outvotes a contaminated reading.
pub fn record_capability_sample(signals: &mut SignalStore, model: &str, key: &str, value: f64) {
    if value <= 0.0 || !value.is_finite() {
        return;
    }
    let caps = signals
        .entry(model.to_string())
        .or_insert_with(|| Value::Object(Default::default()))
        .as_object_mut()
        .expect("model entry is an object");
    let caps_entry = caps
        .entry("_capabilities".to_string())
        .or_insert_with(|| Value::Object(Default::default()));
    let caps_obj = caps_entry.as_object_mut().expect("caps is an object");

    let mut history: Vec<Sample> = caps_obj
        .get(format!("{key}_samples").as_str())
        .and_then(|v| serde_json::from_value(v.clone()).ok())
        .unwrap_or_default();
    migrate_sample_history(
        &mut history,
        caps_obj.get(key).and_then(|v| v.as_f64()),
    );
    let clean = machine_is_uncontended();
    let estimate = crate::ztools::eval::samples::add_sample(&mut history, value, clean);
    caps_obj.insert(format!("{key}_samples"), serde_json::to_value(&history).unwrap_or(Value::Null));
    caps_obj.insert(key.to_string(), serde_json::json!((estimate * 100.0).round() / 100.0));
}

fn caps_clean_estimate(signals: &SignalStore, model: &str, key: &str) -> Option<f64> {
    let caps = signals.get(model)?.get("_capabilities")?;
    let key_samples = format!("{key}_samples");
    let history: Vec<Sample> = caps
        .get(key_samples.as_str())
        .and_then(|v| serde_json::from_value(v.clone()).ok())?;
    clean_estimate(&history).filter(|estimate| *estimate > 0.0)
}

/// How long this model plausibly needs: cold start + ingest + generate, from
/// CLEAN capability samples only. Every term measured, or no answer at all --
/// filling a missing term with a plausible constant is how a guess ends up
/// wearing a measurement's authority. Returns 0 when unmeasurable, and the
/// caller keeps its documented floor.
pub fn derived_timeout(model: &str, prompt_chars: usize, max_tokens: u32) -> u64 {
    let signals = load_signals();
    let (Some(prefill), Some(decode), Some(cold_start)) = (
        caps_clean_estimate(&signals, model, "prefill_chars_per_sec"),
        caps_clean_estimate(&signals, model, "decode_tokens_per_sec"),
        caps_clean_estimate(&signals, model, "cold_start_seconds"),
    ) else {
        return 0;
    };
    let seconds = cold_start + prompt_chars as f64 / prefill + max_tokens as f64 / decode;
    ((seconds * TIMEOUT_SAFETY_FACTOR) as u64).min(max_eval_timeout())
}

/// Timeout actually applied to one request: the largest of the learned
/// per-model/task value, the per-task CONFIGURED timeout from
/// `conf/config.toml [timeouts]` (fallback 600, `lib/llm/constants.py
/// DEFAULT_TIMEOUT`), the documented floor, and the derived estimate.
pub fn effective_timeout(model: &str, task_name: &str, prompt_chars: usize, max_tokens: u32) -> u64 {
    const FALLBACK_CONFIGURED_TIMEOUT: u64 = 600;
    let signals = load_signals();
    let learned = signals
        .get(model)
        .and_then(|m| m.get(task_name))
        .and_then(|t| t.get("timeout"))
        .and_then(|v| v.as_u64())
        .unwrap_or(0);
    let configured = std::fs::read_to_string(
        crate::ztools::eval::budgets::conf_root().join("config.toml"),
    )
    .ok()
    .and_then(|text| toml::from_str::<toml::Value>(&text).ok())
        .and_then(|cfg| {
            cfg.get("timeouts")
                .and_then(|t| t.get(task_name))
                .and_then(|v| v.as_integer())
                .filter(|v| *v > 0)
                .map(|v| v as u64)
        })
        .unwrap_or(FALLBACK_CONFIGURED_TIMEOUT);
    let derived = derived_timeout(model, prompt_chars, max_tokens);
    *[learned, configured, derived, default_eval_timeout()]
        .iter()
        .max()
        .unwrap()
}

/// Record one completed task observation: p95 latency (EMA weighted toward
/// recent), retry/parse counters, and the learned timeout derived from them.
pub fn record_signal(
    signals: &mut SignalStore,
    model: &str,
    task_name: &str,
    time_taken: f64,
    had_retries: bool,
    is_parse_failure: bool,
) {
    if time_taken <= 0.0 && !had_retries {
        return;
    }
    let model_entry = signals
        .entry(model.to_string())
        .or_insert_with(|| Value::Object(Default::default()));
    let obj = model_entry.as_object_mut().expect("model entry is an object");
    let per_task = obj
        .entry(task_name.to_string())
        .or_insert_with(|| Value::Object(Default::default()));
    let task = per_task.as_object_mut().expect("task entry is an object");

    let samples = task.get("samples").and_then(|v| v.as_u64()).unwrap_or(0);
    let old_p95 = task.get("p95_latency").and_then(|v| v.as_f64()).unwrap_or(0.0);

    let mut p95 = old_p95;
    if time_taken > 0.0 {
        p95 = if old_p95 > 0.0 {
            (time_taken).max(old_p95 * 0.95 + time_taken * 0.05)
        } else {
            time_taken
        };
        task.insert("p95_latency".to_string(), json_p95(p95));
    }

    task.insert("samples".to_string(), serde_json::json!(samples + 1));
    let retries = task.get("total_retries").and_then(|v| v.as_u64()).unwrap_or(0);
    task.insert(
        "total_retries".to_string(),
        serde_json::json!(retries + u64::from(had_retries)),
    );
    let parse_failures = task.get("parse_failures").and_then(|v| v.as_u64()).unwrap_or(0);
    task.insert(
        "parse_failures".to_string(),
        serde_json::json!(parse_failures + u64::from(is_parse_failure)),
    );

    if p95 > 0.0 {
        let new_timeout = default_eval_timeout().max((p95 * 1.5) as u64);
        if task.get("timeout").and_then(|v| v.as_u64()) != Some(new_timeout) {
            task.insert("timeout".to_string(), serde_json::json!(new_timeout));
        }
    }
}

fn json_p95(v: f64) -> Value {
    serde_json::json!((v * 10.0).round() / 10.0)
}

#[cfg(test)]
mod tests {
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
}
