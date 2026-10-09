//! Eval result persistence and reporting.
//!
//! Ported from `eval/report_core.py` (default dir), `report_history.py`
//! (per-model history with truncated-run quarantine), and the metric halves of
//! `report_metrics.py` (winners, score stats, CSV export, historical trends).
//!
//! Three rules carried over verbatim:
//!
//! - **Test doubles never enter the production leaderboard**: a `mock-model`
//!   once sat at mean 100 atop the trend table.
//! - **Truncated runs are MARKED, not dropped.** A truncated run's individual
//!   task scores are real -- the task that completed completed -- but any
//!   aggregate over them describes the subset the model found easy. Entries
//!   are written with the verdict attached and [`load_historical_stats`]
//!   refuses to average them; writing them to a separate quarantine FILE was
//!   the first design and was wrong (a second store is a second thing
//!   consumers forget to read).
//! - **A row is identified by its task AND the task's fingerprint.** The name
//!   alone was enough until a task was replaced under it: `file_summary` on
//!   2026-10-08 began asking a different question (M1), and every row stored
//!   under that name before then describes the other one. An entry written
//!   before fingerprints existed carries none, and absence means UNKNOWN --
//!   never current -- so no aggregate averages it with the new ones.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::units::{count, signed};
use crate::ztools::eval::completeness::{Completeness, record_is_complete};
use crate::ztools::eval::runner::TaskOutcome;
use crate::ztools::eval::task_fingerprint::{TaskIdentities, current_task_identities, standing_of};

/// Where eval artefacts live when the caller does not say otherwise.
#[must_use]
pub fn default_eval_dir() -> PathBuf {
    dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".config/ztools")
}

/// Whether a model name is a test double rather than a real served model.
#[must_use]
pub fn is_test_model(model: &str) -> bool {
    let lower = model.trim().to_lowercase();
    lower.starts_with("mock") || lower.starts_with("test-") || lower.starts_with("fake")
}

/// One model's sweep: its outcomes plus the completeness verdict derived by
/// diffing what was asked for against what reported back.
#[derive(Debug, Clone)]
pub struct ModelRun {
    pub model: String,
    pub outcomes: Vec<TaskOutcome>,
    pub completeness: Option<Completeness>,
}

impl ModelRun {
    #[must_use]
    pub fn new(model: &str, expected: &[String], outcomes: Vec<TaskOutcome>) -> Self {
        Self {
            model: model.to_string(),
            completeness: Some(Completeness::derive(expected, &outcomes)),
            outcomes,
        }
    }
}

/// One historical observation.
///
/// `complete` is ABSENT on records written before truncation tracking existed,
/// and absence means COMPLETE -- defaulting old entries to incomplete would
/// retroactively disqualify real measurements. The opposite rule holds for
/// `fingerprint`: it is ABSENT on every record written before 2026-10-08, which
/// is every record on disk, and absence means UNKNOWN. Reading an absent
/// fingerprint as the current one is precisely how the pre-M1 `file_summary`
/// rows would be averaged with the post-M1 rows. Old entries stay on disk
/// untouched; an aggregate that can see they are not the task it is being asked
/// about simply does not average them, and says how many it set aside.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HistoryEntry {
    pub date: String,
    pub timestamp: f64,
    pub task: String,
    pub score: i64,
    #[serde(default)]
    pub time: Option<f64>,
    #[serde(default = "default_true")]
    pub complete: bool,
    // `fingerprint` on the wire, the same key the per-task signal series in
    // `conf/eval_signals.json` files its identity under, so one word means one
    // thing across both stores. `default` is load-bearing: every entry on disk
    // today omits it, and an omitted key deserialises to None rather than
    // failing or defaulting to the current task.
    #[serde(default)]
    pub fingerprint: Option<String>,
}

const fn default_true() -> bool {
    true
}

fn history_path(eval_dir: Option<&Path>) -> PathBuf {
    let base = eval_dir.map_or_else(default_eval_dir, Path::to_path_buf);
    base.join("eval_history.json")
}

/// Append this run's per-task scores to `eval_history.json`, keyed by model.
///
/// Test doubles are skipped entirely; entries from an incomplete run carry
/// `complete: false` so [`load_historical_stats`] can refuse to average them,
/// and every entry carries the fingerprint of the task it was taken against so
/// a run against a REPLACED task is not averaged with the runs before it.
///
/// # Errors
///
/// When the history directory cannot be created, or the history file cannot
/// be written. A run by a test model returns the existing history untouched
/// and cannot fail.
pub fn save_historical_results(
    run: &ModelRun,
    eval_dir: Option<&Path>,
) -> std::io::Result<BTreeMap<String, Vec<HistoryEntry>>> {
    if is_test_model(&run.model) {
        return Ok(load_history(eval_dir));
    }
    let path = history_path(eval_dir);
    let mut history = load_history(eval_dir);

    let dir = path.parent().unwrap_or_else(|| Path::new("."));
    std::fs::create_dir_all(dir)?;
    let entry_model = history.entry(run.model.clone()).or_default();
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    // Which task each name currently IS, as this process loaded it. A measured
    // task absent from the set would write a row no aggregate can ever average,
    // and the run would look perfectly healthy while doing it, so the names are
    // named rather than left to be inferred from a zeroed trend table.
    let identities = current_task_identities();
    let unresolved: Vec<&str> = run
        .outcomes
        .iter()
        .filter(|o| o.was_measured())
        .map(|o| o.task.as_str())
        .filter(|task| !identities.contains_key(*task))
        .collect();
    if !unresolved.is_empty() {
        eprintln!(
            "⚠ no task fingerprint registered for {}, so their history entries will not be averaged: \
             the loader registers every task it hands out",
            unresolved.join(", ")
        );
    }
    // An unmeasured row holds no score. Writing its placeholder 0 made the
    // history average an outage, or a prompt the model was never sent, into
    // the model's mean as though it had answered wrong.
    for outcome in run.outcomes.iter().filter(|o| o.was_measured()) {
        entry_model.push(HistoryEntry {
            date: chrono::Local::now().format("%Y-%m-%d").to_string(),
            timestamp: now.as_secs_f64(),
            task: outcome.task.clone(),
            score: i64::from(outcome.score),
            time: if outcome.time_secs > 0.0 {
                Some(outcome.time_secs)
            } else {
                None
            },
            complete: record_is_complete(run.completeness.as_ref()),
            fingerprint: identities.get(&outcome.task).cloned(),
        });
    }

    if let Ok(text) = serde_json::to_string_pretty(&history) {
        std::fs::write(&path, text)?;
    }
    Ok(history)
}

/// The history file as a map, for the delta table (`report_diff.rs`).
#[must_use]
pub fn load_history_entries(eval_dir: Option<&Path>) -> BTreeMap<String, Vec<HistoryEntry>> {
    load_history(eval_dir)
}

fn load_history(eval_dir: Option<&Path>) -> BTreeMap<String, Vec<HistoryEntry>> {
    let path = history_path(eval_dir);
    std::fs::read_to_string(path)
        .ok()
        .and_then(|text| serde_json::from_str(&text).ok())
        .unwrap_or_default()
}

/// Per-model aggregate over COUNTABLE entries only.
///
/// `if e.get("score")` style filtering is falsy-for-zero in the Python
/// original and once dropped every total failure from the mean -- a model that
/// scored 0 on half its runs looked identical to one that never failed. Here
/// every non-None score counts. Truncated-run entries are excluded at LOAD
/// time rather than write time; `excluded` is surfaced because a model whose
/// history is mostly truncated runs has a `runs` count that no longer matches
/// its entry count, and that discrepancy is itself the finding.
///
/// `superseded` and `incomplete` are the two reasons an entry lands in
/// `excluded`, and they overlap (a truncated run of a replaced task is both).
/// A model whose ENTIRE history is one or the other keeps a row with `runs: 0`
/// rather than vanishing: "no history" and "history none of which is about this
/// task" are different findings, and only one of them is true here.
#[derive(Debug, Clone, Serialize)]
pub struct ModelStats {
    pub mean: f64,
    pub median: f64,
    pub stdev: f64,
    /// 0 when nothing is countable. That is the absence of a value, NOT a score
    /// of zero, and `render_trends` prints a dash for it.
    pub min: i64,
    pub max: i64,
    pub runs: usize,
    /// Every entry not averaged, for any reason.
    pub excluded: usize,
    /// Of those, how many were taken against a task that is not the task it is
    /// now: fingerprint absent or different.
    pub superseded: usize,
    /// Of those, how many came from a truncated run.
    pub incomplete: usize,
}

/// The aggregate against the fingerprints this process loaded.
#[must_use]
pub fn load_historical_stats(eval_dir: Option<&Path>) -> BTreeMap<String, ModelStats> {
    historical_stats(eval_dir, &current_task_identities())
}

/// The aggregate, against an explicit set of current fingerprints.
///
/// Pure, so the rule is pinned without the process-wide registry and without a
/// file the test cannot control. An entry is COUNTABLE when its run was
/// complete AND the task it names is the task it is now; everything else is set
/// aside and counted, never dropped and never rewritten.
#[must_use]
pub fn historical_stats(
    eval_dir: Option<&Path>,
    current: &TaskIdentities,
) -> BTreeMap<String, ModelStats> {
    let mut stats = BTreeMap::new();
    for (model, entries) in load_history(eval_dir) {
        if entries.is_empty() {
            continue;
        }
        // Absent `complete` field deserializes to true (serde default), so
        // legacy entries are trusted exactly like Python's `.get(..., True)`.
        // An absent `fingerprint` deserializes to None, which is the opposite
        // default and for the opposite reason: see `HistoryEntry`.
        let mut scores: Vec<i64> = Vec::new();
        let mut excluded = 0usize;
        let mut superseded = 0usize;
        let mut incomplete = 0usize;
        for entry in &entries {
            let standing = standing_of(entry.fingerprint.as_deref(), current, &entry.task);
            if !entry.complete {
                incomplete += 1;
            }
            if !standing.counts() {
                superseded += 1;
            }
            if entry.complete && standing.counts() {
                scores.push(entry.score);
            } else {
                excluded += 1;
            }
        }
        scores.sort_unstable();
        let n = scores.len();
        let mean = if n == 0 {
            0.0
        } else {
            signed(scores.iter().sum::<i64>()) / count(n)
        };
        let median = if n == 0 {
            0.0
        } else if n % 2 == 1 {
            signed(scores[n / 2])
        } else {
            signed(scores[n / 2 - 1] + scores[n / 2]) / 2.0
        };
        let stdev = if n > 1 {
            let var = scores
                .iter()
                .map(|s| {
                    let d = signed(*s) - mean;
                    d * d
                })
                .sum::<f64>()
                / count(n - 1);
            var.sqrt()
        } else {
            0.0
        };
        stats.insert(
            model,
            ModelStats {
                mean,
                median,
                stdev,
                min: scores.first().copied().unwrap_or(0),
                max: scores.last().copied().unwrap_or(0),
                runs: n,
                excluded,
                superseded,
                incomplete,
            },
        );
    }
    stats
}

/// Which model won each task across this batch of runs. Ties keep the first
/// winner seen, matching Python's strict `>` comparison.
#[must_use]
pub fn compute_task_winners(runs: &[ModelRun]) -> BTreeMap<String, (&String, u8)> {
    let mut winners: BTreeMap<String, (&String, u8)> = BTreeMap::new();
    for run in runs {
        for outcome in &run.outcomes {
            match winners.get(&outcome.task) {
                Some((_, best)) if outcome.score <= *best => {}
                _ => {
                    winners.insert(outcome.task.clone(), (&run.model, outcome.score));
                }
            }
        }
    }
    winners
}

pub(super) const fn status_word(score: u8) -> &'static str {
    if score >= 90 {
        "PASS"
    } else if score >= 50 {
        "WARN"
    } else {
        "FAIL"
    }
}

/// Render the historical trends table: mean/median/stdev/min/max/runs/excluded
/// per model.
///
/// Countable models first and best of those first, matching the Python report's
/// ordering. Empty when no history exists at all.
#[must_use]
pub fn render_historical_trends(eval_dir: Option<&Path>) -> Vec<String> {
    render_trends(eval_dir, &current_task_identities())
}

/// The trend table against an explicit set of current fingerprints: pure, so
/// the set-aside rule is pinned without the registry.
#[must_use]
pub fn render_trends(eval_dir: Option<&Path>, current: &TaskIdentities) -> Vec<String> {
    let stats = historical_stats(eval_dir, current);
    if stats.is_empty() {
        return Vec::new();
    }
    let mut rows: Vec<(&String, &ModelStats)> = stats.iter().collect();
    // Countable models first, best mean of those first. A model with nothing
    // countable sorts LAST rather than by its zeroed mean, where a model whose
    // entire history is superseded would otherwise read as the worst model ever
    // measured instead of the one nobody has measured on this task yet.
    rows.sort_by(|(a_name, a), (b_name, b)| {
        b.runs
            .min(1)
            .cmp(&a.runs.min(1))
            .then_with(|| b.mean.total_cmp(&a.mean))
            .then_with(|| a_name.cmp(b_name))
    });

    let mut lines = vec![
        "Historical Trends (only entries whose task fingerprint is the current task's)".to_string(),
        format!(
            "{:<36} {:>6} {:>6} {:>7} {:>5} {:>5} {:>5} {:>9} {:>9}",
            "Model", "Mean", "Median", "Stdev", "Min", "Max", "Runs", "Excluded", "Superseded"
        ),
    ];
    for (name, s) in rows {
        // A model with nothing countable still gets its row -- with dashes
        // where its statistics would be, because a 0 there would be a score of
        // zero, which is a different claim.
        let row = if s.runs == 0 {
            format!(
                "{:<36} {:>6} {:>6} {:>7} {:>5} {:>5} {:>5} {:>9} {:>9}",
                truncate_name(name),
                "—",
                "—",
                "—",
                "—",
                "—",
                s.runs,
                s.excluded,
                s.superseded
            )
        } else {
            format!(
                "{:<36} {:>6.0} {:>6.0} {:>7.1} {:>5} {:>5} {:>5} {:>9} {:>9}",
                truncate_name(name),
                s.mean,
                s.median,
                s.stdev,
                s.min,
                s.max,
                s.runs,
                s.excluded,
                s.superseded
            )
        };
        lines.push(row);
    }
    // The set-aside count, with its reasons: a mean that quietly ignores half
    // the file is the defect this table exists to make visible. The TOTAL is
    // `excluded` and the reasons are a BREAKDOWN, never a sum -- a truncated run
    // of a replaced task is both, so adding the two clauses would claim more
    // entries were set aside than the file holds. Each clause is stated only
    // when it is non-zero, and the overlap is said out loud when it is possible.
    let excluded: usize = stats.values().map(|s| s.excluded).sum();
    let superseded: usize = stats.values().map(|s| s.superseded).sum();
    let incomplete: usize = stats.values().map(|s| s.incomplete).sum();
    if excluded > 0 {
        let mut why: Vec<String> = Vec::new();
        if superseded > 0 {
            why.push(format!(
                "{superseded} against a superseded task (fingerprint absent or different)"
            ));
        }
        if incomplete > 0 {
            why.push(format!("{incomplete} from a truncated run"));
        }
        let overlap = superseded > 0 && incomplete > 0;
        lines.push(format!(
            "Set aside: {} {} not averaged — {}{}",
            excluded,
            entries_word(excluded),
            why.join(", "),
            if overlap {
                "; a truncated run of a replaced task is both"
            } else {
                ""
            }
        ));
    }
    lines
}

const fn entries_word(n: usize) -> &'static str {
    if n == 1 { "entry" } else { "entries" }
}

pub(super) fn truncate_name(name: &str) -> String {
    if name.len() <= 36 {
        name.to_string()
    } else {
        format!("{}...", &name[..33])
    }
}

#[cfg(test)]
#[path = "report_tests.rs"]
pub(in crate::ztools::eval) mod tests;
