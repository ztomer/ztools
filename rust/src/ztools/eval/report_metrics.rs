//! Derived per-run metrics: score stats, failure categories, error rates.
//!
//! Port of the pure-math halves of `references/eval/report_core.py`
//! (`compute_score_stats`, `categorize_failures`) and `report_metrics.py`
//! (`compute_error_rates`). Split out of `report.rs` for the 500-line cap,
//! mirroring the Python split (`report_metrics.py` was split out of
//! `report.py`) for the same reason.
//!
//! Explicit boundary: `compute_token_estimates` and `compute_verbosity` are
//! NOT ported. Both read per-outcome `content`/`messages`, which Rust's
//! `TaskOutcome` does not carry — porting the math would mean extending the
//! outcome record and plumbing content through the runner, a serialization
//! decision, not a math port. If content capture lands, these two follow.

use std::collections::{BTreeMap, BTreeSet};

use super::completeness::record_is_complete;
use super::discrimination::{is_gate, ranking_mean, EvalResult};
use super::report::ModelRun;

/// Aggregate statistics for one model's run.
///
/// `complete` is the truncation verdict carried so the printer can mark
/// partial runs (see `eval/completeness.py`); `ranking_mean` excludes gate
/// tasks because a task every model passes cannot order the models that
/// pass it.
pub struct ScoreStats {
    pub mean: f64,
    pub median: f64,
    pub stdev: f64,
    pub min: u8,
    pub max: u8,
    pub count: usize,
    pub complete: bool,
    pub ranking_mean: f64,
    pub gate_tasks: usize,
}

/// One failure category across runs: how many, and the sorted deduped models
/// and tasks behind it. Sorted (not hash order) so repeated runs print
/// identically.
pub struct FailureGroup {
    pub count: usize,
    pub models: Vec<String>,
    pub tasks: Vec<String>,
}

/// Infra errors vs quality failures vs successes, with rates.
pub struct ErrorRates {
    pub infra: usize,
    pub quality: usize,
    pub success: usize,
    pub infra_rate: f64,
    pub quality_rate: f64,
    pub success_rate: f64,
}

#[expect(
    clippy::cast_precision_loss,
    reason = "counts of outcomes in one run -- tens -- exact in f64, same shape as discrimination::ranking_mean"
)]
fn mean(scores: &[f64]) -> f64 {
    scores.iter().sum::<f64>() / scores.len() as f64
}

#[expect(
    clippy::missing_const_for_fn,
    reason = "cannot be const: calls non-const f64::midpoint and indexes a slice"
)]
fn median_of_sorted(scores: &[f64]) -> f64 {
    let n = scores.len();
    if n % 2 == 1 {
        scores[n / 2]
    } else {
        f64::midpoint(scores[n / 2 - 1], scores[n / 2])
    }
}

/// Sample standard deviation (n-1), matching Python `statistics.stdev`.
/// A single score has no spread: 0, matching the `len > 1` guard.
#[expect(
    clippy::cast_precision_loss,
    reason = "counts of outcomes in one run -- tens -- exact in f64"
)]
fn sample_stdev(scores: &[f64], mean: f64) -> f64 {
    if scores.len() < 2 {
        return 0.0;
    }
    let variance =
        scores.iter().map(|s| (s - mean) * (s - mean)).sum::<f64>() / (scores.len() - 1) as f64;
    variance.sqrt()
}

/// A rate over one run's outcomes; 0 when there is nothing to rate.
#[expect(
    clippy::cast_precision_loss,
    reason = "counts of outcomes in one run -- tens -- exact in f64"
)]
fn rate(n: usize, total: usize) -> f64 {
    if total == 0 {
        0.0
    } else {
        n as f64 / total as f64
    }
}

/// Aggregate statistics per model. Models with no outcomes are skipped, not
/// reported as zeros — a mean over nothing must not print beside real means.
#[must_use]
pub fn compute_score_stats(runs: &[ModelRun]) -> BTreeMap<String, ScoreStats> {
    let mut stats = BTreeMap::new();
    for run in runs {
        if run.outcomes.is_empty() {
            continue;
        }
        let mut scores: Vec<f64> = run.outcomes.iter().map(|o| f64::from(o.score)).collect();
        scores.sort_by(f64::total_cmp);
        let n = scores.len();
        let mean = mean(&scores);
        stats.insert(
            run.model.clone(),
            ScoreStats {
                mean,
                median: median_of_sorted(&scores),
                stdev: sample_stdev(&scores, mean),
                min: run.outcomes.iter().map(|o| o.score).min().unwrap_or(0),
                max: run.outcomes.iter().map(|o| o.score).max().unwrap_or(0),
                count: n,
                complete: record_is_complete(run.completeness.as_ref()),
                ranking_mean: ranking_mean(
                    &run.outcomes
                        .iter()
                        .map(|o| EvalResult {
                            task: o.task.clone(),
                            quality_score: i32::from(o.score),
                        })
                        .collect::<Vec<_>>(),
                ),
                gate_tasks: run.outcomes.iter().filter(|o| is_gate(&o.task)).count(),
            },
        );
    }
    stats
}

/// Group sub-90 outcomes by failure category across runs. A blank category
/// reads as `UNKNOWN`, matching Python's missing-key default.
#[must_use]
pub fn categorize_failures(runs: &[ModelRun]) -> BTreeMap<String, FailureGroup> {
    let mut categories: BTreeMap<String, (usize, BTreeSet<String>, BTreeSet<String>)> =
        BTreeMap::new();
    for run in runs {
        for outcome in &run.outcomes {
            if outcome.score >= 90 {
                continue;
            }
            let category = if outcome.failure_category.is_empty() {
                "UNKNOWN".to_string()
            } else {
                outcome.failure_category.clone()
            };
            let entry = categories
                .entry(category)
                .or_insert((0, BTreeSet::new(), BTreeSet::new()));
            entry.0 += 1;
            entry.1.insert(run.model.clone());
            entry.2.insert(outcome.task.clone());
        }
    }
    categories
        .into_iter()
        .map(|(category, (count, models, tasks))| {
            (
                category,
                FailureGroup {
                    count,
                    models: models.into_iter().collect(),
                    tasks: tasks.into_iter().collect(),
                },
            )
        })
        .collect()
}

/// Split outcomes into infra errors, quality failures, and successes.
///
/// An error message (non-empty) or an `INFRA` category is infra; below 50 is
/// a quality failure; the rest succeeded. Models with no outcomes still get
/// a zero entry, matching Python (rates 0, not missing).
#[must_use]
pub fn compute_error_rates(runs: &[ModelRun]) -> BTreeMap<String, ErrorRates> {
    let mut rates = BTreeMap::new();
    for run in runs {
        let mut infra = 0usize;
        let mut quality = 0usize;
        let mut success = 0usize;
        for outcome in &run.outcomes {
            let has_error = outcome.error.as_deref().is_some_and(|e| !e.is_empty());
            if has_error || outcome.failure_category == "INFRA" {
                infra += 1;
            } else if outcome.score < 50 {
                quality += 1;
            } else {
                success += 1;
            }
        }
        let total = infra + quality + success;
        rates.insert(
            run.model.clone(),
            ErrorRates {
                infra,
                quality,
                success,
                infra_rate: rate(infra, total),
                quality_rate: rate(quality, total),
                success_rate: rate(success, total),
            },
        );
    }
    rates
}

#[cfg(test)]
mod tests {
    #![expect(clippy::float_cmp, reason = "exact; see eval::scoring_math")]

    use super::*;
    use crate::ztools::eval::runner::TaskOutcome;

    fn outcome(task: &str, score: u8, failure_category: &str, error: Option<&str>) -> TaskOutcome {
        TaskOutcome {
            task: task.to_string(),
            score,
            failure_category: failure_category.to_string(),
            error: error.map(str::to_string),
            ..Default::default()
        }
    }

    fn run(model: &str, expected: &[&str], outcomes: Vec<TaskOutcome>) -> ModelRun {
        ModelRun::new(
            model,
            &expected
                .iter()
                .map(|s| (*s).to_string())
                .collect::<Vec<_>>(),
            outcomes,
        )
    }

    #[test]
    fn single_score_stats_are_the_score_itself() {
        let runs = vec![run(
            "model_a",
            &["task1"],
            vec![outcome("task1", 85, "", None)],
        )];
        let stats = compute_score_stats(&runs);
        let s = &stats["model_a"];
        assert_eq!(s.mean, 85.0);
        assert_eq!(s.median, 85.0);
        assert_eq!(s.stdev, 0.0);
        assert_eq!((s.min, s.max, s.count), (85, 85, 1));
        assert!(s.complete);
    }

    #[test]
    fn multiple_scores_aggregate_and_an_empty_run_is_skipped() {
        let runs = vec![
            run(
                "m1",
                &["t1", "t2", "t3"],
                vec![
                    outcome("t1", 80, "", None),
                    outcome("t2", 90, "", None),
                    outcome("t3", 100, "", None),
                ],
            ),
            run("m_empty", &[], vec![]),
        ];
        let stats = compute_score_stats(&runs);
        assert!(!stats.contains_key("m_empty"));
        let s = &stats["m1"];
        assert_eq!(s.mean, 90.0);
        assert_eq!(s.median, 90.0);
        assert_eq!(s.stdev, 10.0);
        assert_eq!((s.min, s.max, s.count), (80, 100, 3));
    }

    #[test]
    fn short_runs_are_marked_incomplete_and_gate_tasks_counted() {
        let runs = vec![run("m1", &["t1", "t2"], vec![outcome("t1", 80, "", None)])];
        let stats = compute_score_stats(&runs);
        assert!(!stats["m1"].complete);

        let gated = vec![run(
            "m1",
            &["image_real"],
            vec![outcome("image_real", 100, "", None)],
        )];
        assert_eq!(compute_score_stats(&gated)["m1"].gate_tasks, 1);
    }

    #[test]
    fn high_scores_never_enter_failure_categories() {
        // 90 is excluded: Python skips `quality_score >= 90`, so the boundary
        // belongs outside the failure set, not inside it.
        let runs = vec![
            run("m1", &["t1"], vec![outcome("t1", 95, "INFRA", None)]),
            run("m2", &["t2"], vec![outcome("t2", 90, "FORMAT", None)]),
        ];
        assert!(categorize_failures(&runs).is_empty());
    }

    #[test]
    fn failures_group_by_category_with_sorted_deduped_members() {
        let runs = vec![
            run("m1", &["t1"], vec![outcome("t1", 30, "FORMAT", None)]),
            run("m2", &["t1"], vec![outcome("t2", 40, "FORMAT", None)]),
            run(
                "m1",
                &["t1", "t2"],
                vec![
                    outcome("t1", 30, "PARSE", None),
                    outcome("t1", 20, "PARSE", None),
                ],
            ),
        ];
        let cats = categorize_failures(&runs);
        assert_eq!(cats["FORMAT"].count, 2);
        assert_eq!(cats["FORMAT"].models, vec!["m1", "m2"]);
        assert_eq!(cats["PARSE"].count, 2);
        assert_eq!(cats["PARSE"].models, vec!["m1"]);
        assert_eq!(cats["PARSE"].tasks, vec!["t1"]);
    }

    #[test]
    fn blank_category_becomes_unknown() {
        let runs = vec![run("m1", &["t1"], vec![outcome("t1", 30, "", None)])];
        let cats = categorize_failures(&runs);
        assert_eq!(cats["UNKNOWN"].count, 1);
    }

    #[test]
    fn error_rates_split_infra_quality_and_success() {
        let ok = vec![run("m1", &["t1"], vec![outcome("t1", 100, "", None)])];
        let rates = compute_error_rates(&ok);
        assert_eq!(
            (rates["m1"].success, rates["m1"].infra, rates["m1"].quality),
            (1, 0, 0)
        );
        assert_eq!(rates["m1"].success_rate, 1.0);

        let infra = vec![run(
            "m1",
            &["t1"],
            vec![outcome("t1", 0, "INFRA", Some("Model not found"))],
        )];
        let rates = compute_error_rates(&infra);
        assert_eq!((rates["m1"].infra, rates["m1"].success), (1, 0));

        let quality = vec![run("m1", &["t1"], vec![outcome("t1", 30, "CONTENT", None)])];
        let rates = compute_error_rates(&quality);
        assert_eq!((rates["m1"].quality, rates["m1"].success), (1, 0));

        // 50 is success, not quality: Python counts `< 50` as failure.
        let boundary = vec![run("m1", &["t1"], vec![outcome("t1", 50, "", None)])];
        assert_eq!(compute_error_rates(&boundary)["m1"].success, 1);

        let mixed = vec![run(
            "m1",
            &["t1", "t2", "t3"],
            vec![
                outcome("t1", 100, "", None),
                outcome("t2", 0, "INFRA", Some("err")),
                outcome("t3", 30, "FORMAT", None),
            ],
        )];
        let rates = compute_error_rates(&mixed);
        let r = &rates["m1"];
        assert_eq!((r.success, r.infra, r.quality), (1, 1, 1));
        assert_eq!(r.success_rate + r.infra_rate + r.quality_rate, 1.0);
    }
}
