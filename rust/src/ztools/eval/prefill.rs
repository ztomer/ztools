//! Measure how fast a model ingests a prompt, per model, on this host.
//!
//! Ported from `references/eval/prefill.py`. Used to size request TIMEOUTS so a
//! large prompt is not killed mid-flight; explicitly NOT used to decide how much
//! context to send.
//!
//! Measuring this honestly took four attempts in the Python original, three of
//! which produced a confident wrong number. The guards below each encode one:
//! warm the model FIRST (an unwarmed probe times 27GB of weights loading as
//! throughput), lead the timed prompt with a NONCE (identical filler rides the
//! server's prefix cache and measured 130x too fast), use max_tokens=1 for the
//! timed call only (whole-call timing blames prefill for decode), and discard
//! any rate above [`MAX_PLAUSIBLE_PREFILL_RATE`] (a mock or cache hit returning
//! in microseconds is not a measurement).

use std::time::Instant;

use crate::ztools::eval::signals::{
    default_eval_timeout, record_capability_sample, SignalStore,
};
use crate::ztools::eval::task_loader::ChatMessage;
use crate::ztools::eval::transport::{call, RequestSpec};

pub const PREFILL_PROBE_CHARS: usize = 20_000;
/// Sits above every genuine reading measured on the host (fastest real model
/// was 23,063 chars/sec) and below the slowest known prefix-cache hit (65,000+).
/// It only ever DISCARDS a measurement -- it never invents one.
pub const MAX_PLAUSIBLE_PREFILL_RATE: f64 = 40_000.0;
const WARMUP_TOKENS: u32 = 64;
const WARMUP_PROMPT: &str = "Count from one to twenty in words, one per line.";
const PROBE_LINE: &str =
    "[@SomeHandle | 08:15]: A reasonably typical sentence about a launch today.\n";

fn spec<'a>(
    model: &'a str,
    messages: &'a [ChatMessage],
    host: &'a str,
    port: u16,
    max_tokens: u32,
) -> RequestSpec<'a> {
    RequestSpec {
        model,
        messages,
        host,
        port,
        temperature: 0.0,
        max_tokens,
        timeout_secs: default_eval_timeout(),
        allow_substitution: true,
        stream_guard: false,
    }
}

/// Characters per second this model ingests, measured with max_tokens=1.
///
/// THREE calls, one quantity each -- sharing a call between two measurements is
/// how every previous version got a wrong number:
/// 1. LOAD (max_tokens=1): pays cold start; timed as cold_start_seconds.
/// 2. DECODE (max_tokens=WARMUP_TOKENS): weights resident, tiny prompt, so
///    elapsed time is generation -> decode_tokens_per_sec.
/// 3. PREFILL (nonce-led filler, max_tokens=1): isolated ingestion rate.
///
/// Returns None when the probe cannot run, so "not measured" stays distinct
/// from a measurement.
pub fn measure_prefill_rate(
    signals: &mut SignalStore,
    model: &str,
    host: &str,
    port: u16,
) -> Option<f64> {
    // 1. LOAD.
    let load_msgs = vec![ChatMessage::user(WARMUP_PROMPT)];
    let load_started = Instant::now();
    let loaded = call(&spec(model, &load_msgs, host, port, 1), false);
    let load_elapsed = load_started.elapsed().as_secs_f64();
    if loaded.error.is_some() {
        return None;
    }
    record_capability_sample(signals, model, "cold_start_seconds", load_elapsed);

    // 2. DECODE.
    let decode_started = Instant::now();
    let generated = call(&spec(model, &load_msgs, host, port, WARMUP_TOKENS), false);
    let decode_elapsed = decode_started.elapsed().as_secs_f64();
    if generated.error.is_none() && decode_elapsed > 0.0 {
        record_capability_sample(
            signals,
            model,
            "decode_tokens_per_sec",
            WARMUP_TOKENS as f64 / decode_elapsed,
        );
    }

    // 3. PREFILL. Nonce goes FIRST: a prefix cache matches from the start of
    // the prompt, so a leading unique token makes every byte after it new work.
    let nonce = format!("[run {}]\n", uuid_hex());
    let filler_len = PREFILL_PROBE_CHARS.saturating_sub(nonce.len());
    // PROBE_LINE is ASCII, so truncating at a byte boundary cannot split a char.
    let body = PROBE_LINE.repeat(filler_len / PROBE_LINE.len() + 1);
    let filler = format!("{}{}", nonce, &body[..filler_len]);
    let probe_msgs = vec![ChatMessage::user(filler)];
    let started = Instant::now();
    let result = call(&spec(model, &probe_msgs, host, port, 1), false);
    let elapsed = started.elapsed().as_secs_f64();
    if result.error.is_some() || elapsed <= 0.0 {
        return None;
    }
    let rate = PREFILL_PROBE_CHARS as f64 / elapsed;
    if rate > MAX_PLAUSIBLE_PREFILL_RATE {
        // An instant answer is not a measurement.
        return None;
    }
    Some((rate * 10.0).round() / 10.0)
}

/// Store a measured prefill rate as a per-model capability. Kept at the model
/// level, not per task: it is a property of the model and the host.
pub fn record_prefill_rate(signals: &mut SignalStore, model: &str, rate: Option<f64>) {
    let Some(rate) = rate else { return };
    if rate <= 0.0 {
        return;
    }
    record_capability_sample(signals, model, "prefill_chars_per_sec", rate);
    let caps = signals
        .get_mut(model)
        .and_then(|m| m.as_object_mut())
        .expect("model entry exists from record_capability_sample");
    let caps_obj = caps
        .entry("_capabilities")
        .or_insert_with(|| serde_json::Value::Object(Default::default()));
    if let Some(obj) = caps_obj.as_object_mut() {
        let n = obj.get("prefill_samples").and_then(|v| v.as_u64()).unwrap_or(0);
        obj.insert("prefill_samples".to_string(), serde_json::json!(n + 1));
    }
}

/// Random-enough hex tag so no two probes share a prefix. Not cryptographic --
/// it only needs to defeat a prefix cache within one machine's lifetime.
fn uuid_hex() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let pid = std::process::id();
    format!("{nanos:032x}{pid:08x}")
}
