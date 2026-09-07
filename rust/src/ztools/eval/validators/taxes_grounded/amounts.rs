//! Money primitives: parsing amounts out of prose and scoring them as grounded.
//!
//! Split out of `taxes_grounded.rs` for the 500-line production cap. Pure value
//! transformers over numbers -- no I/O, no task knowledge.

use crate::ztools::eval::scoring_math::{ratio, rounded};
use regex::Regex;
use serde_json::Value;
use std::collections::HashSet;
use std::sync::LazyLock;

pub const MAX_SCORE: i64 = 100;
pub const MAX_SUBSET_VALUES: usize = 16;

static MONEY_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r"(?x)
        \$\s*\d[\d,]*(?:\.\d+)?
        | \d[\d,]*(?:\.\d+)?\s*(?:CAD|cad|dollars)
        | \d{1,3}(?:,\d{3})+(?:\.\d{2})?
        | \d+\.\d{2}\b
    ",
    )
    .unwrap()
});

#[must_use]
pub fn cents(value: &Value) -> Option<f64> {
    match value {
        Value::Number(n) => n.as_f64().map(|f| (f * 100.0).round() / 100.0),
        Value::String(s) => s
            .trim()
            .parse::<f64>()
            .ok()
            .map(|f| (f * 100.0).round() / 100.0),
        _ => None,
    }
}

pub fn prose_amounts(prose: &str) -> Vec<f64> {
    let mut found = Vec::new();
    for mat in MONEY_RE.find_iter(prose) {
        let cleaned: String = mat
            .as_str()
            .chars()
            .filter(|c| c.is_ascii_digit() || *c == '.')
            .collect();
        if cleaned.is_empty() || cleaned.matches('.').count() > 1 {
            continue;
        }
        if let Ok(f) = cleaned.parse::<f64>() {
            let val = (f * 100.0).round() / 100.0;
            found.push(val.abs());
        }
    }
    found
}

#[must_use]
pub fn known_set(known_amounts: &[Value]) -> HashSet<i64> {
    let mut out = HashSet::new();
    for val in known_amounts {
        if let Some(c) = cents(val) {
            out.insert(rounded(c.abs() * 100.0));
        }
    }
    out
}

#[must_use]
#[expect(
    clippy::cast_precision_loss,
    reason = "a weight times a grounded fraction. The weight is a rubric constant and the fraction is 0.0..=1.0"
)]
pub fn score_prose_amounts(prose: &str, known: &HashSet<i64>, weight: i64) -> (i64, String) {
    let amounts = prose_amounts(prose);
    if amounts.is_empty() {
        return (weight, format!("prose_amounts=0/0 ({weight}/{weight})"));
    }
    let grounded = amounts
        .iter()
        .filter(|a| known.contains(&(rounded(a.abs() * 100.0))))
        .count();
    let score = rounded(weight as f64 * ratio(grounded, amounts.len()));
    (
        score,
        format!(
            "prose_amounts={}/{} ({}/{})",
            grounded,
            amounts.len(),
            score,
            weight
        ),
    )
}

#[must_use]
pub fn traceable_sums(values: &[f64]) -> HashSet<i64> {
    let mut sums = HashSet::new();
    if values.is_empty() {
        return sums;
    }
    if values.len() > MAX_SUBSET_VALUES {
        for v in values {
            sums.insert(rounded(v * 100.0));
        }
        let total: f64 = values.iter().sum();
        sums.insert(rounded(total * 100.0));
        return sums;
    }

    let n = values.len();
    for mask in 1..(1usize << n) {
        let mut sub_sum = 0.0;
        for (i, v) in values.iter().enumerate() {
            if (mask & (1 << i)) != 0 {
                sub_sum += *v;
            }
        }
        sums.insert(rounded(sub_sum * 100.0));
    }
    sums
}
