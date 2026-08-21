//! Performance samples that can recover from a bad reading via clean-window median sampling.
//!
//! Port of `eval/samples.py`. Stores timing/rate observations with clean tags,
//! deriving estimates from the median of recent clean samples so contaminated
//! readings under system contention are outvoted rather than enshrined.

use serde::{Deserialize, Serialize};

pub const SAMPLE_WINDOW: usize = 5;

fn default_clean() -> bool {
    true
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Sample {
    pub v: f64,
    #[serde(default)]
    pub ts: f64,
    #[serde(default = "default_clean")]
    pub clean: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub legacy: Option<bool>,
}

impl Sample {
    pub fn new(v: f64, clean: bool) -> Self {
        Self {
            v: (v * 10000.0).round() / 10000.0,
            ts: 0.0,
            clean,
            legacy: None,
        }
    }
}

/// Compute statistical median of a slice of floats.
pub fn median(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = sorted.len();
    if n % 2 == 1 {
        sorted[n / 2]
    } else {
        (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
    }
}

/// Derive estimate: median of recent clean samples, falling back to all recent samples.
pub fn estimate_from(history: &[Sample]) -> f64 {
    if history.is_empty() {
        return 0.0;
    }
    let clean_values: Vec<f64> = history.iter().filter(|s| s.clean).map(|s| s.v).collect();

    if !clean_values.is_empty() {
        let window = if clean_values.len() > SAMPLE_WINDOW {
            &clean_values[clean_values.len() - SAMPLE_WINDOW..]
        } else {
            &clean_values[..]
        };
        return median(window);
    }

    let all_values: Vec<f64> = history.iter().map(|s| s.v).collect();
    let window = if all_values.len() > SAMPLE_WINDOW {
        &all_values[all_values.len() - SAMPLE_WINDOW..]
    } else {
        &all_values[..]
    };
    median(window)
}

/// Clean estimate: median of clean samples ONLY, returning None when no clean sample exists.
pub fn clean_estimate(history: &[Sample]) -> Option<f64> {
    let clean_values: Vec<f64> = history.iter().filter(|s| s.clean).map(|s| s.v).collect();

    if clean_values.is_empty() {
        return None;
    }

    let window = if clean_values.len() > SAMPLE_WINDOW {
        &clean_values[clean_values.len() - SAMPLE_WINDOW..]
    } else {
        &clean_values[..]
    };
    Some(median(window))
}

/// Seed a sample history from a pre-existing scalar, ONCE, tagged unclean.
///
/// The scalars on disk predate clean-sample tracking and some were taken under
/// load, so they must not be trusted as clean baselines -- but discarding them
/// would throw away the only reading some models have. Tagged legacy they are
/// used until real clean samples arrive, then outvoted.
pub fn migrate_sample_history(history: &mut Vec<Sample>, scalar: Option<f64>) {
    if !history.is_empty() {
        return;
    }
    let Some(value) = scalar else {
        return;
    };
    if value <= 0.0 {
        return;
    }
    let mut seeded = Sample::new(value, false);
    seeded.ts = 0.0;
    seeded.legacy = Some(true);
    history.push(seeded);
}

/// Append a sample, bound history size, and return the updated estimate.
pub fn add_sample(history: &mut Vec<Sample>, value: f64, clean: bool) -> f64 {
    let s = Sample::new(value, clean);
    history.push(s);
    if history.len() > SAMPLE_WINDOW * 2 {
        let excess = history.len() - SAMPLE_WINDOW * 2;
        history.drain(0..excess);
    }
    estimate_from(history)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_median_odd_and_even() {
        assert_eq!(median(&[5.0]), 5.0);
        assert_eq!(median(&[1.0, 3.0, 2.0]), 2.0);
        assert_eq!(median(&[1.0, 2.0, 3.0, 4.0]), 2.5);
    }

    #[test]
    fn test_estimate_prefers_clean_samples() {
        let history = vec![
            Sample::new(100.0, false), // contaminated
            Sample::new(30.0, true),
            Sample::new(32.0, true),
            Sample::new(31.0, true),
        ];
        // Clean median of [30.0, 32.0, 31.0] is 31.0, ignoring 100.0
        assert_eq!(estimate_from(&history), 31.0);

        // When no clean samples exist, falls back to contaminated
        let dirty_only = vec![Sample::new(50.0, false), Sample::new(60.0, false)];
        assert_eq!(estimate_from(&dirty_only), 55.0);
        assert_eq!(clean_estimate(&dirty_only), None);
    }

    #[test]
    fn test_add_sample_bounding() {
        let mut history = Vec::new();
        for i in 1..=15 {
            add_sample(&mut history, i as f64, true);
        }
        // Retains at most SAMPLE_WINDOW * 2 = 10 items
        assert_eq!(history.len(), 10);
        assert_eq!(history[0].v, 6.0);
        assert_eq!(history.last().unwrap().v, 15.0);
    }
}
