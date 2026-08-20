//! Stall watchdog for model eval loops.
//!
//! Port of `eval/watchdog.py`. Tracks wall-clock duration without task completion,
//! serving as a backstop against wedged servers or models that fail to make progress.

use std::time::{Duration, Instant};

pub const DEFAULT_MODEL_STALL_SECONDS: u64 = 2400; // 40 minutes

pub fn model_stall_duration() -> Duration {
    if let Ok(val) = std::env::var("EVAL_MODEL_STALL_SECONDS") {
        if let Ok(secs) = val.parse::<u64>() {
            return Duration::from_secs(secs);
        }
    }
    Duration::from_secs(DEFAULT_MODEL_STALL_SECONDS)
}

/// Duration elapsed since the last task completion.
pub fn stalled_for(last_completion: Instant) -> Duration {
    last_completion.elapsed()
}

/// Check whether the duration since last completion exceeds the watchdog limit.
pub fn is_stalled(last_completion: Instant, limit: Duration) -> bool {
    stalled_for(last_completion) > limit
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_watchdog_detects_stall() {
        let now = Instant::now();
        // A recent timestamp is not stalled
        assert!(!is_stalled(now, Duration::from_secs(10)));

        // An instant in the past exceeding limit
        let old = now.checked_sub(Duration::from_secs(20)).unwrap();
        assert!(is_stalled(old, Duration::from_secs(10)));
    }
}
