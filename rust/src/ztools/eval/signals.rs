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
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
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
    let out = Command::new("sysctl")
        .arg("-n")
        .arg("vm.swapusage")
        .output()
        .ok()?;
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
            let multiplier = if field.ends_with('G') {
                1.0
            } else {
                1.0 / 1024.0
            };
            return Some(value * multiplier);
        }
    }
    None
}

fn compressor_gb() -> Option<f64> {
    let out = Command::new("/usr/bin/vm_stat").output().ok()?;
    let text = String::from_utf8_lossy(&out.stdout);
    let line = text
        .lines()
        .find(|l| l.starts_with("Pages occupied by compressor"))?;
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
    migrate_sample_history(&mut history, caps_obj.get(key).and_then(|v| v.as_f64()));
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
        .and_then(|v| v.as_u64())
        .unwrap_or(0);
    let configured =
        std::fs::read_to_string(crate::ztools::eval::budgets::conf_root().join("config.toml"))
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
    let obj = model_entry
        .as_object_mut()
        .expect("model entry is an object");
    let per_task = obj
        .entry(task_name.to_string())
        .or_insert_with(|| Value::Object(Default::default()));
    let task = per_task.as_object_mut().expect("task entry is an object");

    let samples = task.get("samples").and_then(|v| v.as_u64()).unwrap_or(0);
    let old_p95 = task
        .get("p95_latency")
        .and_then(|v| v.as_f64())
        .unwrap_or(0.0);

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
    let retries = task
        .get("total_retries")
        .and_then(|v| v.as_u64())
        .unwrap_or(0);
    task.insert(
        "total_retries".to_string(),
        serde_json::json!(retries + u64::from(had_retries)),
    );
    let parse_failures = task
        .get("parse_failures")
        .and_then(|v| v.as_u64())
        .unwrap_or(0);
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
#[path = "signals_tests.rs"]
mod tests;
