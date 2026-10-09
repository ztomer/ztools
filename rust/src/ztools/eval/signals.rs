//! Eval signal store: per-model, per-task observations accumulated across runs,
//! plus the learned-timeout arithmetic built on them.
//!
//! Ported from `references/eval/signals.py` and the pressure half of
//! `eval/memory.py`. The store backs three consumers: the eval loop records
//! into it, `_effective_timeout` sizes request timeouts from it, and the
//! capability samples feed the median-of-clean estimator (`samples.rs`).
//!
//! TWO KINDS OF SERIES LIVE IN ONE FILE, and they take OPPOSITE lessons from a
//! task being replaced. A per-(model, task) series — p95 latency, retries,
//! parse failures — is a running aggregate over how the model answered one
//! question, so it restarts when the question changes (`record_observation`).
//! A `_capabilities` sample is a measurement of the MODEL's speed, which no
//! prompt can change, so it is never fingerprinted and never restarts. Read
//! [`record_task_signal`] for why the distinction is drawn there and not
//! here.
//!
//! WHAT THE MACHINE IS DOING lives in the `platform` submodule: which host
//! published a reading, what each of its fields means, and the macOS/Linux
//! mapping. It is a module of its own because it is a unit of reasoning the
//! store is not, and because `signals.rs` was at 427 of the 500-line cap.
//! Everything the store's consumers need is re-exported below, so
//! `signals::memory_pressure` and friends keep one name regardless of which file
//! defines them.
//!
//! Contention honesty, in one line: a sample is tagged CLEAN only when no
//! foreign GPU-lock holder exists AND memory pressure is verifiably low.
//! Pressure that cannot be read marks the sample UNVERIFIED (unclean), never
//! clean -- inventing a healthy reading is exactly how a contended machine's
//! numbers got enshrined in the Python original's history.

use std::collections::BTreeMap;
use std::path::PathBuf;

use serde_json::Value;

use crate::units::{count, whole_u64};
use crate::ztools::eval::samples::{Sample, clean_estimate, migrate_sample_history};
use crate::ztools::eval::task_fingerprint::{current_task_fingerprint, task_fingerprint};
use crate::ztools::eval::task_loader::EvalTask;

pub use platform::{
    MAX_CLEAN_RECLAIM_GB, MAX_CLEAN_SWAP_GB, MemoryPressure, PROC_MEMINFO, PressureSource, SYSCTL,
    VM_STAT, file_text, machine_is_uncontended, memory_pressure, memory_pressure_from,
    parse_compressor_gb, parse_meminfo_swap_used_gb, parse_swap_used_gb, thrashing_verdict,
    tool_output, uncontended_verdict,
};

#[path = "signals_platform.rs"]
mod platform;

/// A POLICY ceiling, not an estimate: past this a request is assumed wedged.
/// Deliberately not derived -- its job is to bound the damage when the
/// measurements are wrong.
pub const MAX_EVAL_TIMEOUT: u64 = 7200;

const TIMEOUT_SAFETY_FACTOR: f64 = 1.5;

fn env_u64(key: &str, default: u64) -> u64 {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

#[must_use]
pub fn default_eval_timeout() -> u64 {
    env_u64("EVAL_DEFAULT_TIMEOUT", 900)
}

#[must_use]
pub fn max_eval_timeout() -> u64 {
    env_u64("EVAL_MAX_TIMEOUT", 7200)
}

/// Where `eval_signals.json` lives. Env-overridable so tests can point it at tmp.
///
/// Anchored on the CHECKOUT that built the binary (`CARGO_MANIFEST_DIR/../conf`)
/// rather than the process working directory: a sweep launched from anywhere
/// else used to read -- and worse, CREATE -- a relative `conf/eval_signals.json`
/// in whatever directory it started from, silently losing every recorded
/// capability and forking the store. The home fallback covers an installed
/// binary with no checkout nearby.
#[must_use]
pub fn signals_path() -> PathBuf {
    let dir = std::env::var("EVAL_SIGNALS_DIR").unwrap_or_else(|_| {
        let candidates = [
            std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .parent()
                .map(|p| p.join("conf")),
            dirs::home_dir().map(|h| h.join("Projects/ztools/conf")),
        ];
        candidates
            .into_iter()
            .flatten()
            .find(|p| p.is_dir())
            .map_or_else(|| "conf".to_string(), |p| p.to_string_lossy().to_string())
    });
    PathBuf::from(dir).join("eval_signals.json")
}

pub type SignalStore = BTreeMap<String, Value>;

#[must_use]
pub fn load_signals() -> SignalStore {
    let path = signals_path();
    std::fs::read_to_string(&path).map_or_else(
        |_| SignalStore::new(),
        |text| serde_json::from_str(&text).unwrap_or_default(),
    )
}

pub fn save_signals(signals: &SignalStore) {
    if let Some(parent) = signals_path().parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    if let Ok(text) = serde_json::to_string_pretty(signals) {
        let _ = std::fs::write(signals_path(), text);
    }
}

/// Add one observation of `key` under the model's capabilities, re-derive the
/// estimate. Median of recent CLEAN samples outvotes a contaminated reading.
///
/// # Panics
///
/// If an existing signal entry is not a JSON object -- i.e. the signals file
/// was written by something other than this code, or hand-edited. Failing
/// loudly is deliberate: silently replacing it would discard a run's history.
pub fn record_capability_sample(signals: &mut SignalStore, model: &str, key: &str, value: f64) {
    if value <= 0.0 || !value.is_finite() {
        return;
    }
    let caps = signals
        .entry(model.to_string())
        .or_insert_with(|| Value::Object(serde_json::Map::default()))
        .as_object_mut()
        .expect("model entry is an object");
    let caps_entry = caps
        .entry("_capabilities".to_string())
        .or_insert_with(|| Value::Object(serde_json::Map::default()));
    let caps_obj = caps_entry.as_object_mut().expect("caps is an object");

    let mut history: Vec<Sample> = caps_obj
        .get(format!("{key}_samples").as_str())
        .and_then(|v| serde_json::from_value(v.clone()).ok())
        .unwrap_or_default();
    migrate_sample_history(
        &mut history,
        caps_obj.get(key).and_then(serde_json::Value::as_f64),
    );
    let clean = machine_is_uncontended();
    let estimate = crate::ztools::eval::samples::add_sample(&mut history, value, clean);
    caps_obj.insert(
        format!("{key}_samples"),
        serde_json::to_value(&history).unwrap_or(Value::Null),
    );
    caps_obj.insert(
        key.to_string(),
        serde_json::json!((estimate * 100.0).round() / 100.0),
    );
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
/// CLEAN capability samples only.
///
/// Every term measured, or no answer at all -- filling a missing term with a
/// plausible constant is how a guess ends up wearing a measurement's authority.
/// Returns 0 when unmeasurable, and the caller keeps its documented floor.
#[must_use]
pub fn derived_timeout(model: &str, prompt_chars: usize, max_tokens: u32) -> u64 {
    let signals = load_signals();
    let (Some(prefill), Some(decode), Some(cold_start)) = (
        caps_clean_estimate(&signals, model, "prefill_chars_per_sec"),
        caps_clean_estimate(&signals, model, "decode_tokens_per_sec"),
        caps_clean_estimate(&signals, model, "cold_start_seconds"),
    ) else {
        return 0;
    };
    let seconds = cold_start + count(prompt_chars) / prefill + f64::from(max_tokens) / decode;
    whole_u64(seconds * TIMEOUT_SAFETY_FACTOR).min(max_eval_timeout())
}

/// Timeout actually applied to one request.
///
/// The largest of the learned
/// per-model/task value, the per-task CONFIGURED timeout from
/// `conf/config.toml [timeouts]` (fallback 600, `lib/llm/constants.py
/// DEFAULT_TIMEOUT`), the documented floor, and the derived estimate.
///
/// The learned term is a PERFORMANCE estimate and is read across task
/// fingerprints on purpose: how long this model takes on a task of this shape
/// is evidence about the model's speed, not about its answer quality, and a
/// prompt change does not make a previously measured latency wrong. Refusing to
/// learn a timeout here — returning the floor because the fingerprint does not
/// match — would remove the very ceiling that stops a wedged server idling.
/// Quality-shaped fields (p95 of the CURRENT series, retries, parse failures)
/// are the ones that restart with the task, in [`record_observation`].
///
/// # Panics
///
/// If the configured timeouts cannot be ordered -- which requires a NaN in
/// the signal store, and so the same hand-edited or foreign file as above.
pub fn effective_timeout(
    model: &str,
    task_name: &str,
    prompt_chars: usize,
    max_tokens: u32,
) -> u64 {
    const FALLBACK_CONFIGURED_TIMEOUT: u64 = 600;
    let signals = load_signals();
    let learned = signals
        .get(model)
        .and_then(|m| m.get(task_name))
        .and_then(|t| t.get("timeout"))
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(0);
    let configured =
        std::fs::read_to_string(crate::ztools::eval::budgets::conf_root().join("config.toml"))
            .ok()
            .and_then(|text| toml::from_str::<toml::Value>(&text).ok())
            .and_then(|cfg| {
                cfg.get("timeouts")
                    .and_then(|t| t.get(task_name))
                    .and_then(toml::Value::as_integer)
                    .filter(|v| *v > 0)
                    .and_then(|v| u64::try_from(v).ok())
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
///
/// The task's fingerprint comes from the registry — what this process loaded
/// under `task_name` — because a caller that only has a NAME cannot compute the
/// digest. A caller holding the [`EvalTask`] itself should prefer
/// [`record_task_signal`].
///
/// # Panics
///
/// If an existing signal or per-task entry is not a JSON object; see
/// [`record_capability_sample`].
pub fn record_signal(
    signals: &mut SignalStore,
    model: &str,
    task_name: &str,
    time_taken: f64,
    had_retries: bool,
    is_parse_failure: bool,
) {
    let fingerprint = current_task_fingerprint(task_name);
    record_observation(
        signals,
        model,
        task_name,
        fingerprint.as_deref(),
        time_taken,
        had_retries,
        is_parse_failure,
    );
}

/// [`record_signal`] for a caller holding the TASK, which is the honest
/// direction: the digest comes from the task that was actually run rather than
/// from whatever a loader registered under its name.
///
/// # Panics
///
/// If an existing signal or per-task entry is not a JSON object; see
/// [`record_capability_sample`].
pub fn record_task_signal(
    signals: &mut SignalStore,
    model: &str,
    task: &EvalTask,
    time_taken: f64,
    had_retries: bool,
    is_parse_failure: bool,
) {
    record_observation(
        signals,
        model,
        &task.name,
        Some(&task_fingerprint(task)),
        time_taken,
        had_retries,
        is_parse_failure,
    );
}

/// The store holds a RUNNING AGGREGATE per (model, task), not a list of
/// samples, so "do not average samples taken under a different prompt" can only
/// mean one thing here: START A NEW SERIES when the task's identity changes. A
/// p95 read while the model worked through the OLD prompt would otherwise be
/// blended into the new one as though it had answered the new question in that
/// time — and the learned timeout derived from it would then size every later
/// request against a prompt nobody sent. The observations set aside are counted
/// on the series, so the loss is visible rather than silent.
///
/// The same rule is what makes `effective_timeout` and `derived_timeout` safe:
/// they are PERFORMANCE estimates, not quality scores, and a timeout must never
/// refuse to be learned because the fingerprint is unknown or different. A
/// series that restarts has no learned timeout for its first observation, which
/// falls back to the configured value, the DERIVED estimate (still available,
/// because `_capabilities` is model speed and never resets) and the documented
/// floor — so the ceiling that cuts a wedged server off is still there. That is
/// why `_capabilities` is deliberately NOT fingerprinted: prefill rate, decode
/// rate and cold start are properties of the MODEL and the box, and a prompt
/// change does not make a previously measured rate wrong.
///
/// Two UNKNOWN identities accumulate as before (both absent). That is the
/// un-registered path — `record_signal` with a name no loader registered — and
/// there is nothing to compare against; treating it as a change would reset
/// every series on every observation.
fn record_observation(
    signals: &mut SignalStore,
    model: &str,
    task_name: &str,
    fingerprint: Option<&str>,
    time_taken: f64,
    had_retries: bool,
    is_parse_failure: bool,
) {
    if time_taken <= 0.0 && !had_retries {
        return;
    }
    let model_entry = signals
        .entry(model.to_string())
        .or_insert_with(|| Value::Object(serde_json::Map::default()));
    let obj = model_entry
        .as_object_mut()
        .expect("model entry is an object");
    let per_task = obj
        .entry(task_name.to_string())
        .or_insert_with(|| Value::Object(serde_json::Map::default()));
    let task = per_task.as_object_mut().expect("task entry is an object");

    // Absent and null are the same state here: UNKNOWN. Every entry the store
    // holds today is in it, which is why the first fingerprinted observation
    // supersedes the series rather than joining it.
    let stored = task.get(FINGERPRINT_KEY).and_then(Value::as_str);
    if stored != fingerprint {
        let discarded = task.get("samples").and_then(Value::as_u64).unwrap_or(0);
        if discarded > 0 {
            let already = task
                .get(SUPERSEDED_KEY)
                .and_then(Value::as_u64)
                .unwrap_or(0);
            task.insert(
                SUPERSEDED_KEY.to_string(),
                serde_json::json!(already + discarded),
            );
        }
        for key in [
            FINGERPRINT_KEY,
            "p95_latency",
            "samples",
            "total_retries",
            "parse_failures",
            "timeout",
        ] {
            task.remove(key);
        }
    }
    if let Some(fingerprint) = fingerprint {
        task.insert(FINGERPRINT_KEY.to_string(), serde_json::json!(fingerprint));
    }

    let samples = task
        .get("samples")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(0);
    let old_p95 = task
        .get("p95_latency")
        .and_then(serde_json::Value::as_f64)
        .unwrap_or(0.0);

    let mut p95 = old_p95;
    if time_taken > 0.0 {
        p95 = if old_p95 > 0.0 {
            (time_taken).max(time_taken.mul_add(0.05, old_p95 * 0.95))
        } else {
            time_taken
        };
        task.insert("p95_latency".to_string(), json_p95(p95));
    }

    task.insert("samples".to_string(), serde_json::json!(samples + 1));
    let retries = task
        .get("total_retries")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(0);
    task.insert(
        "total_retries".to_string(),
        serde_json::json!(retries + u64::from(had_retries)),
    );
    let parse_failures = task
        .get("parse_failures")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(0);
    task.insert(
        "parse_failures".to_string(),
        serde_json::json!(parse_failures + u64::from(is_parse_failure)),
    );

    if p95 > 0.0 {
        let new_timeout = default_eval_timeout().max(whole_u64(p95 * 1.5));
        if task.get("timeout").and_then(serde_json::Value::as_u64) != Some(new_timeout) {
            task.insert("timeout".to_string(), serde_json::json!(new_timeout));
        }
    }
}

/// The key a per-task series files its identity under. Written only when the
/// identity is known: an absent key is the same UNKNOWN state as a null one,
/// and that is what every entry on disk today is.
const FINGERPRINT_KEY: &str = "fingerprint";

/// The key counting the observations a series discarded because the task under
/// its name was replaced. Visible in `conf/eval_signals.json` itself, which is
/// what makes the set-aside count auditable without a printer.
const SUPERSEDED_KEY: &str = "superseded_samples";

/// How many observations this (model, task) series has set aside because the
/// task was replaced under the same name.
///
/// There is no printer for it in the CLI's report: no table there summarises
/// the store, and inventing one would put a second rendering of the same fact on
/// screen. The count lives on the series, next to the numbers it explains.
#[must_use]
pub fn superseded_samples(signals: &SignalStore, model: &str, task_name: &str) -> u64 {
    signals
        .get(model)
        .and_then(|m| m.get(task_name))
        .and_then(|t| t.get(SUPERSEDED_KEY))
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(0)
}

/// Which task a per-(model, task) series was accumulated against, if any.
#[must_use]
pub fn series_fingerprint(signals: &SignalStore, model: &str, task_name: &str) -> Option<String> {
    signals
        .get(model)
        .and_then(|m| m.get(task_name))
        .and_then(|t| t.get(FINGERPRINT_KEY))
        .and_then(serde_json::Value::as_str)
        .map(str::to_string)
}

fn json_p95(v: f64) -> Value {
    serde_json::json!((v * 10.0).round() / 10.0)
}

#[cfg(test)]
#[path = "signals_tests/mod.rs"]
mod tests;
