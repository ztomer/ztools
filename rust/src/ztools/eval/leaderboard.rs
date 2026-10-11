//! Comparative multi-model leaderboard output (`M8`).
//!
//! Formats the latest clean run of each model from `eval_history.json` into a
//! comparative markdown table ranking overall means and per-slot scores.
//!
//! A score is a measurement of ONE task, and a task is its fingerprint, not its
//! name (`task_fingerprint.rs`). The leaderboard scores a row only when it
//! carries the current task's fingerprint (`M17`); see `generate_leaderboard_filtered`.

use anyhow::Result;
use serde::Serialize;
use std::collections::BTreeMap;
use std::path::Path;

pub use super::leaderboard_format::{
    format_leaderboard, format_leaderboard_by_family, format_leaderboard_csv,
};
use crate::units::{count, signed};
use crate::ztools::eval::report::HistoryEntry;
use crate::ztools::eval::task_fingerprint::{TaskIdentities, standing_of};

/// Output formats supported by leaderboard reporting.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LeaderboardFormat {
    #[default]
    Markdown,
    Json,
    Csv,
}

/// Options controlling leaderboard generation and filtering.
#[derive(Debug, Clone, Default)]
pub struct LeaderboardOptions<'a> {
    pub min_tasks: Option<usize>,
    pub sort_by: Option<&'a str>,
    pub category: Option<&'a str>,
    pub format: LeaderboardFormat,
    pub group_by_family: bool,
    pub fail_on_regression: Option<f64>,
}

/// A model's ranking row on the leaderboard.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ModelLeaderboardEntry {
    pub model: String,
    pub date: String,
    pub timestamp: f64,
    /// Tasks SCORED: rows of the chosen run taken on the current task.
    pub task_count: usize,
    /// Mean over the scored rows; `None` when the run has none on a current task.
    pub overall_mean: Option<f64>,
    pub delta: Option<f64>,
    pub think_score: Option<f64>,
    pub json_score: Option<f64>,
    pub summarize_score: Option<f64>,
    pub filename_score: Option<f64>,
    pub vlm_score: Option<f64>,
    /// Tasks in the chosen run whose fingerprint is not the current task's:
    /// not measured on the current task, so not scored (`M17`).
    pub not_current: Vec<String>,
}

/// Returns whether a task name belongs to the specified category filter (`M13`).
#[must_use]
pub fn task_matches_category(task: &str, category: &str) -> bool {
    let cat = category.to_ascii_lowercase();
    let t = task.to_ascii_lowercase();
    if cat == "twitter" {
        return t.starts_with("summarize") || t.contains("twitter");
    }
    if cat == "vlm" {
        return t.starts_with("image") || t.contains("vlm");
    }
    t.starts_with(&cat) || t.contains(&cat)
}

/// Detects the model family from a model name string (`M15`).
#[must_use]
pub fn detect_model_family(model: &str) -> &'static str {
    const FAMILIES: &[(&str, &str)] = &[
        ("qwen", "qwen"),
        ("qwopus", "qwen"),
        ("gemma", "gemma"),
        ("raptor", "raptor"),
        ("muse", "muse"),
        ("bonsai", "bonsai"),
        ("ornith", "ornith"),
        ("nemotron", "nemotron"),
        ("lfm", "lfm"),
        ("foundation", "foundation"),
    ];
    let m = model.to_ascii_lowercase();
    for &(prefix, family) in FAMILIES {
        if m.starts_with(prefix) || m.contains(prefix) {
            return family;
        }
    }
    "other"
}

/// Checks if any model's delta regressed below `-threshold_pct` (`M16`).
#[must_use]
pub fn check_regression(
    entries: &[ModelLeaderboardEntry],
    threshold_pct: f64,
) -> Option<(String, f64)> {
    let cutoff = -threshold_pct.abs();
    for entry in entries {
        if let Some(delta) = entry.delta
            && delta < cutoff
        {
            return Some((entry.model.clone(), delta));
        }
    }
    None
}

/// Categorizes an eval task name into its corresponding slot.
///
/// It classifies HISTORY, not the roster: `detailed_json` (and the
/// `weekend_fixed*` rows the prefix catches) were retired in W5 but stay in
/// `eval_history.json`, and an old run's json mean still has to be its json mean.
#[must_use]
pub fn task_slot(task: &str) -> Option<&'static str> {
    if task.starts_with("weekend_") || task == "json" || task == "detailed_json" {
        Some("json")
    } else if task.starts_with("summarize") {
        Some("summarize")
    } else if task.starts_with("filename") || task.starts_with("rename") {
        Some("filename")
    } else if task.starts_with("file_summary") || task.starts_with("taxes") {
        Some("think")
    } else if task.starts_with("image_") {
        Some("vlm")
    } else {
        None
    }
}

fn mean_of(entries: &[&HistoryEntry]) -> Option<f64> {
    if entries.is_empty() {
        None
    } else {
        Some(entries.iter().map(|e| signed(e.score)).sum::<f64>() / count(entries.len()))
    }
}

fn slot_mean(entries: &[&HistoryEntry], slot: &str) -> Option<f64> {
    let in_slot: Vec<&HistoryEntry> = entries
        .iter()
        .copied()
        .filter(|e| task_slot(&e.task) == Some(slot))
        .collect();
    mean_of(&in_slot)
}

/// Whether `entry` was taken on the task as it stands now (`M17`).
fn is_current(entry: &HistoryEntry, current: &TaskIdentities) -> bool {
    standing_of(entry.fingerprint.as_deref(), current, &entry.task).counts()
}

fn group_runs<'a>(entries: &[&'a HistoryEntry]) -> Vec<Vec<&'a HistoryEntry>> {
    let mut runs: Vec<Vec<&'a HistoryEntry>> = Vec::new();
    for entry in entries {
        if let Some(group) = runs
            .iter_mut()
            .find(|g| (g[0].timestamp - entry.timestamp).abs() < 2.0)
        {
            group.push(entry);
        } else {
            runs.push(vec![entry]);
        }
    }
    runs
}

fn filter_eligible_runs<'a, 'b>(
    runs: &'b [Vec<&'a HistoryEntry>],
    threshold: usize,
    has_explicit_min: bool,
) -> Vec<&'b Vec<&'a HistoryEntry>> {
    if has_explicit_min {
        runs.iter().filter(|g| g.len() >= threshold).collect()
    } else {
        let multi: Vec<&'b Vec<&'a HistoryEntry>> =
            runs.iter().filter(|g| g.len() >= threshold).collect();
        if multi.is_empty() {
            runs.iter().collect()
        } else {
            multi
        }
    }
}

/// The change from the previous eligible run, over the tasks BOTH runs scored
/// on the current task (`M17`).
///
/// Before M17 this subtracted the previous run's mean over every row from the
/// latest run's, so a re-fingerprinted task made the delta a comparison of two
/// different questions: the same defect as the ranking, one column over. Only
/// tasks current in both runs enter it; with none shared there is no delta.
fn compute_run_delta(
    eligible_runs: &[&Vec<&HistoryEntry>],
    latest_scored: &[&HistoryEntry],
    current: &TaskIdentities,
) -> Option<f64> {
    if eligible_runs.len() < 2 {
        return None;
    }
    let mut sorted = eligible_runs.to_vec();
    sorted.sort_by(|a, b| {
        b[0].timestamp
            .partial_cmp(&a[0].timestamp)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let prev_scored: Vec<&HistoryEntry> = sorted
        .get(1)?
        .iter()
        .copied()
        .filter(|e| is_current(e, current))
        .collect();
    let latest = mean_of(&shared(latest_scored, &prev_scored))?;
    let prev = mean_of(&shared(&prev_scored, latest_scored))?;
    Some(latest - prev)
}

/// The rows of `side` whose task `other` also holds.
fn shared<'a>(side: &[&'a HistoryEntry], other: &[&HistoryEntry]) -> Vec<&'a HistoryEntry> {
    side.iter()
        .copied()
        .filter(|e| other.iter().any(|o| o.task == e.task))
        .collect()
}

fn cmp_opt_score_desc(a: Option<f64>, b: Option<f64>) -> std::cmp::Ordering {
    match (a, b) {
        (Some(va), Some(vb)) => vb.partial_cmp(&va).unwrap_or(std::cmp::Ordering::Equal),
        (Some(_), None) => std::cmp::Ordering::Less,
        (None, Some(_)) => std::cmp::Ordering::Greater,
        (None, None) => std::cmp::Ordering::Equal,
    }
}

fn sort_leaderboard_rows(rows: &mut [ModelLeaderboardEntry], threshold: usize, sort_slot: &str) {
    rows.sort_by(|a, b| {
        let a_full = a.task_count >= threshold;
        let b_full = b.task_count >= threshold;
        b_full
            .cmp(&a_full)
            .then_with(|| match sort_slot {
                "think" => cmp_opt_score_desc(a.think_score, b.think_score),
                "json" => cmp_opt_score_desc(a.json_score, b.json_score),
                "summarize" => cmp_opt_score_desc(a.summarize_score, b.summarize_score),
                "filename" => cmp_opt_score_desc(a.filename_score, b.filename_score),
                "vlm" => cmp_opt_score_desc(a.vlm_score, b.vlm_score),
                _ => std::cmp::Ordering::Equal,
            })
            .then_with(|| cmp_opt_score_desc(a.overall_mean, b.overall_mean))
            .then_with(|| b.task_count.cmp(&a.task_count))
            .then_with(|| a.model.cmp(&b.model))
    });
}

/// One model's row from its chosen run: only rows on the CURRENT task are
/// scored; the rest are named in `not_current`.
fn build_row(
    model: &str,
    run_entries: &[&HistoryEntry],
    eligible: &[&Vec<&HistoryEntry>],
    current: &TaskIdentities,
) -> ModelLeaderboardEntry {
    let (scored, stale): (Vec<&HistoryEntry>, Vec<&HistoryEntry>) = run_entries
        .iter()
        .copied()
        .partition(|e| is_current(e, current));
    let mut not_current: Vec<String> = stale.iter().map(|e| e.task.clone()).collect();
    not_current.sort();
    not_current.dedup();
    ModelLeaderboardEntry {
        model: model.to_string(),
        date: run_entries[0].date.clone(),
        timestamp: run_entries[0].timestamp,
        task_count: scored.len(),
        overall_mean: mean_of(&scored),
        delta: compute_run_delta(eligible, &scored, current),
        think_score: slot_mean(&scored, "think"),
        json_score: slot_mean(&scored, "json"),
        summarize_score: slot_mean(&scored, "summarize"),
        filename_score: slot_mean(&scored, "filename"),
        vlm_score: slot_mean(&scored, "vlm"),
        not_current,
    }
}

/// Aggregates each model's latest clean run from history and ranks by overall mean or slot descending,
/// optionally filtering by task category (`M13`).
///
/// A row is SCORED only when it carries the fingerprint `current` holds for its
/// task (`M17`): a model whose latest run predates a task's change shows that
/// task in `not_current` instead of a score taken on a different question.
///
/// When `min_tasks` is specified (`Some(min)`), models whose latest clean run holds fewer
/// than `min` tasks are filtered out. When `None`, the default threshold (5, or 1 when filtered by category)
/// is used to prioritize multi-task sweeps over spot checks without excluding spot checks.
///
/// When `sort_by` is specified, entries are sorted by that capability slot (`overall`, `think`,
/// `json`, `summarize`, `filename`, `vlm`) descending, with unrated (`None`) slots placed last.
///
/// # Errors
///
/// Returns an error if `sort_by` is not a recognized slot name.
pub fn generate_leaderboard_filtered(
    history: &BTreeMap<String, Vec<HistoryEntry>>,
    current: &TaskIdentities,
    min_tasks: Option<usize>,
    sort_by: Option<&str>,
    category: Option<&str>,
) -> Result<Vec<ModelLeaderboardEntry>> {
    let sort_slot = sort_by.unwrap_or("overall");
    match sort_slot {
        "overall" | "think" | "json" | "summarize" | "filename" | "vlm" => {}
        other => anyhow::bail!(
            "invalid sort slot '{other}': valid slots are overall, think, json, summarize, filename, vlm"
        ),
    }

    let default_threshold = if category.is_some() { 1 } else { 5 };
    let threshold = min_tasks.unwrap_or(default_threshold);
    let mut rows = Vec::new();

    for (model, entries) in history {
        let complete: Vec<&HistoryEntry> = entries
            .iter()
            .filter(|e| {
                e.complete && category.is_none_or(|cat| task_matches_category(&e.task, cat))
            })
            .collect();
        if complete.is_empty() {
            continue;
        }

        let runs = group_runs(&complete);
        let eligible = filter_eligible_runs(&runs, threshold, min_tasks.is_some());
        let chosen_run = eligible.iter().copied().max_by(|a, b| {
            a[0].timestamp
                .partial_cmp(&b[0].timestamp)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let Some(run_entries) = chosen_run else {
            continue;
        };
        rows.push(build_row(model, run_entries, &eligible, current));
    }

    sort_leaderboard_rows(&mut rows, threshold, sort_slot);
    Ok(rows)
}

/// Aggregates each model's latest clean run from history and ranks by overall mean or slot descending.
///
/// # Errors
///
/// Returns an error if `sort_by` is not a recognized slot name.
pub fn generate_leaderboard(
    history: &BTreeMap<String, Vec<HistoryEntry>>,
    current: &TaskIdentities,
    min_tasks: Option<usize>,
    sort_by: Option<&str>,
) -> Result<Vec<ModelLeaderboardEntry>> {
    generate_leaderboard_filtered(history, current, min_tasks, sort_by, None)
}

/// CLI entrypoint for `ztools model-eval --leaderboard`.
///
/// `current` is the fingerprint of every task on the roster as it stands now;
/// the caller loads the roster to get it, because a leaderboard that cannot
/// tell a current row from a superseded one is the M17 defect.
///
/// # Errors
///
/// Returns an error if loading history fails, sorting slot is invalid, JSON serialization fails,
/// or regression threshold is breached (`M16`).
pub fn cli_leaderboard(
    eval_dir: Option<&Path>,
    opts: &LeaderboardOptions<'_>,
    current: &TaskIdentities,
) -> Result<()> {
    let history = crate::ztools::eval::report::load_history_entries(eval_dir);
    let entries = generate_leaderboard_filtered(
        &history,
        current,
        opts.min_tasks,
        opts.sort_by,
        opts.category,
    )?;

    match opts.format {
        LeaderboardFormat::Json => {
            println!("{}", serde_json::to_string_pretty(&entries)?);
        }
        LeaderboardFormat::Csv => {
            print!("{}", format_leaderboard_csv(&entries));
        }
        LeaderboardFormat::Markdown => {
            if opts.group_by_family {
                print!("{}", format_leaderboard_by_family(&entries));
            } else {
                print!("{}", format_leaderboard(&entries));
            }
        }
    }

    if let Some(threshold) = opts.fail_on_regression
        && let Some((model, delta)) = check_regression(&entries, threshold)
    {
        anyhow::bail!(
            "model evaluation regression detected: '{model}' delta {delta:+.1}% fell below threshold -{:.1}%",
            threshold.abs()
        );
    }

    Ok(())
}

#[cfg(test)]
#[path = "leaderboard_tests.rs"]
mod tests;
