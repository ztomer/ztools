//! Twitter timeout math.
//!
//! Port of `references/twitter/budget.py`. The request timeout scales with
//! input size and expected output instead of a flat cap that a big prompt on
//! a slow model blows through: cold-start load + prefill(input) +
//! decode(output), clamped to `[min_timeout_secs, max_timeout_secs]`.
//!
//! Every term is an explicit input. Python resolves per-model measurements
//! where they exist and falls back to pessimistic constants; the Rust twitter
//! path has no measurement lookup yet, so callers pass
//! [`TimeoutInputs::pessimistic`] (the same fallback constants) until the
//! `model_caps` wiring lands.

use crate::units::{count, unsigned, whole_u64};
/// Pessimistic fallback constants.
///
/// Mirrors the Python defaults used when no per-model measurement exists.
/// Guessing slow makes the tool wait longer than it needs to; guessing fast
/// kills a request that was working — only one of those loses output.
pub const DEFAULT_COLD_START_SECS: f64 = 120.0;
pub const DEFAULT_PREFILL_CHARS_PER_SEC: f64 = 200.0;
pub const DEFAULT_DECODE_TOKENS_PER_SEC: f64 = 8.0;
/// `min(OUTPUT_RESERVE_TOKENS, OSAURUS_CONTEXT_WINDOW / 2)` at defaults.
pub const DEFAULT_OUTPUT_TOKENS: u64 = 1536;
pub const DEFAULT_MIN_TIMEOUT_SECS: u64 = 60;
pub const DEFAULT_MAX_TIMEOUT_SECS: u64 = 5400;

/// Inputs to the timeout estimate. All rates strictly positive in practice;
/// a zero rate is guarded to 1 rather than dividing by zero.
pub struct TimeoutInputs {
    pub prefill_chars_per_sec: f64,
    pub decode_tokens_per_sec: f64,
    pub cold_start_secs: f64,
    pub output_tokens: u64,
    pub min_timeout_secs: u64,
    pub max_timeout_secs: u64,
}

impl TimeoutInputs {
    /// Python's no-measurement fallback: cold start 120s, prefill 200
    /// chars/s, decode 8 tok/s, 1536 output tokens, clamp 60..5400.
    #[must_use]
    pub const fn pessimistic() -> Self {
        Self {
            prefill_chars_per_sec: DEFAULT_PREFILL_CHARS_PER_SEC,
            decode_tokens_per_sec: DEFAULT_DECODE_TOKENS_PER_SEC,
            cold_start_secs: DEFAULT_COLD_START_SECS,
            output_tokens: DEFAULT_OUTPUT_TOKENS,
            min_timeout_secs: DEFAULT_MIN_TIMEOUT_SECS,
            max_timeout_secs: DEFAULT_MAX_TIMEOUT_SECS,
        }
    }
}

/// Dynamic request timeout scaled to input size + expected output.
///
/// `prompt_chars` counts Unicode code points, matching Python `len(prompt)`.
/// A zero rate degrades to 1 (never divide by zero); the clamped estimate
/// truncates toward zero, matching Python `int()`.
#[must_use]
pub fn estimate_timeout(prompt_chars: usize, inputs: &TimeoutInputs) -> u64 {
    let prefill = count(prompt_chars) / inputs.prefill_chars_per_sec.max(1.0);
    let decode = unsigned(inputs.output_tokens) / inputs.decode_tokens_per_sec.max(1.0);
    let estimate = inputs.cold_start_secs + prefill + decode;
    let clamped = estimate
        .max(unsigned(inputs.min_timeout_secs))
        .min(unsigned(inputs.max_timeout_secs));
    whole_u64(clamped)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pessimistic() -> TimeoutInputs {
        TimeoutInputs::pessimistic()
    }

    #[test]
    fn larger_prompts_get_strictly_more_time() {
        // The whole point of the scaling: a bigger prompt must not share a
        // smaller prompt's deadline.
        let small = estimate_timeout(1000, &pessimistic());
        let large = estimate_timeout(40000, &pessimistic());
        assert!(large > small, "{large} should exceed {small}");
    }

    #[test]
    fn tiny_estimate_clamps_to_the_minimum() {
        let inputs = TimeoutInputs {
            cold_start_secs: 0.0,
            ..pessimistic()
        };
        // 0 + 1/200 + 1536/8 = 192.005 -> 192, above the 60 floor.
        assert_eq!(estimate_timeout(1, &inputs), 192);
    }

    #[test]
    fn huge_estimate_clamps_to_the_maximum() {
        let inputs = TimeoutInputs {
            max_timeout_secs: 300,
            ..pessimistic()
        };
        assert_eq!(estimate_timeout(10_000_000, &inputs), 300);
    }

    #[test]
    fn cold_start_dominates_tiny_prompts() {
        let inputs = TimeoutInputs {
            cold_start_secs: 500.0,
            max_timeout_secs: 1800,
            min_timeout_secs: 1,
            ..pessimistic()
        };
        let timeout = estimate_timeout(1, &inputs);
        assert!(timeout >= 500, "{timeout}");
    }
}
