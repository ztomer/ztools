//! Comparative multi-model leaderboard output (`M8`).
//!
//! Formats the latest clean run of each model from `eval_history.json` into a
//! comparative markdown table ranking overall means and per-slot scores.

use anyhow::Result;
use serde::Serialize;
use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::path::Path;

use crate::units::{count, signed};
use crate::ztools::eval::report::HistoryEntry;

/// A model's ranking row on the leaderboard.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ModelLeaderboardEntry {
    pub model: String,
    pub date: String,
    pub timestamp: f64,
    pub task_count: usize,
    pub overall_mean: f64,
    pub delta: Option<f64>,
    pub think_score: Option<f64>,
    pub json_score: Option<f64>,
    pub summarize_score: Option<f64>,
    pub filename_score: Option<f64>,
    pub vlm_score: Option<f64>,
}

/// Categorizes an eval task name into its corresponding slot.
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

fn slot_mean(entries: &[&HistoryEntry], slot: &str) -> Option<f64> {
    let scores: Vec<f64> = entries
        .iter()
        .filter(|e| task_slot(&e.task) == Some(slot))
        .map(|e| signed(e.score))
        .collect();
    if scores.is_empty() {
        None
    } else {
        let sum: f64 = scores.iter().sum();
        Some(sum / count(scores.len()))
    }
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

fn compute_run_delta(eligible_runs: &[&Vec<&HistoryEntry>], latest_mean: f64) -> Option<f64> {
    if eligible_runs.len() < 2 {
        return None;
    }
    let mut sorted = eligible_runs.to_vec();
    sorted.sort_by(|a, b| {
        b[0].timestamp
            .partial_cmp(&a[0].timestamp)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let prev_run = sorted.get(1)?;
    let prev_mean = prev_run.iter().map(|e| signed(e.score)).sum::<f64>() / count(prev_run.len());
    Some(latest_mean - prev_mean)
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
            .then_with(|| {
                b.overall_mean
                    .partial_cmp(&a.overall_mean)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .then_with(|| b.task_count.cmp(&a.task_count))
            .then_with(|| a.model.cmp(&b.model))
    });
}

/// Aggregates each model's latest clean run from history and ranks by overall mean or slot descending.
///
/// When `min_tasks` is specified (`Some(min)`), models whose latest clean run holds fewer
/// than `min` tasks are filtered out. When `None`, the default threshold (5) is used to prioritize
/// multi-task sweeps over spot checks without excluding spot checks.
///
/// When `sort_by` is specified, entries are sorted by that capability slot (`overall`, `think`,
/// `json`, `summarize`, `filename`, `vlm`) descending, with unrated (`None`) slots placed last.
///
/// # Errors
///
/// Returns an error if `sort_by` is not a recognized slot name.
pub fn generate_leaderboard(
    history: &BTreeMap<String, Vec<HistoryEntry>>,
    min_tasks: Option<usize>,
    sort_by: Option<&str>,
) -> Result<Vec<ModelLeaderboardEntry>> {
    let sort_slot = sort_by.unwrap_or("overall");
    match sort_slot {
        "overall" | "think" | "json" | "summarize" | "filename" | "vlm" => {}
        other => anyhow::bail!(
            "invalid sort slot '{other}': valid slots are overall, think, json, summarize, filename, vlm"
        ),
    }

    let threshold = min_tasks.unwrap_or(5);
    let mut rows = Vec::new();

    for (model, entries) in history {
        let complete: Vec<&HistoryEntry> = entries.iter().filter(|e| e.complete).collect();
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

        let latest_ts = run_entries[0].timestamp;
        let overall_mean =
            run_entries.iter().map(|e| signed(e.score)).sum::<f64>() / count(run_entries.len());
        let delta = compute_run_delta(&eligible, overall_mean);

        rows.push(ModelLeaderboardEntry {
            model: model.clone(),
            date: run_entries[0].date.clone(),
            timestamp: latest_ts,
            task_count: run_entries.len(),
            overall_mean,
            delta,
            think_score: slot_mean(run_entries, "think"),
            json_score: slot_mean(run_entries, "json"),
            summarize_score: slot_mean(run_entries, "summarize"),
            filename_score: slot_mean(run_entries, "filename"),
            vlm_score: slot_mean(run_entries, "vlm"),
        });
    }

    sort_leaderboard_rows(&mut rows, threshold, sort_slot);
    Ok(rows)
}

fn fmt_cell(val: Option<f64>) -> String {
    val.map_or_else(|| "—".to_string(), |v| format!("{v:.1}%"))
}

fn fmt_delta(val: Option<f64>) -> String {
    match val {
        None => "—".to_string(),
        Some(d) if d > 0.049 => format!("+{d:.1}%"),
        Some(d) if d < -0.049 => format!("{d:.1}%"),
        Some(_) => "0.0%".to_string(),
    }
}

/// Formats the leaderboard entries into a markdown table.
#[must_use]
pub fn format_leaderboard(entries: &[ModelLeaderboardEntry]) -> String {
    if entries.is_empty() {
        return "No complete model eval runs found in history.\n".to_string();
    }

    let mut out = String::new();
    out.push_str("# Model Evaluation Leaderboard\n\n");
    out.push_str("Latest clean full-roster run for each evaluated model from eval history.\n\n");
    out.push_str(
        "| Rank | Model | Mean | Delta | Think | JSON | Summarize | Filename | VLM | Tasks | Date |\n",
    );
    out.push_str(
        "| ---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |\n",
    );

    for (idx, entry) in entries.iter().enumerate() {
        let _ = writeln!(
            out,
            "| {} | `{}` | {:.1}% | {} | {} | {} | {} | {} | {} | {} | {} |",
            idx + 1,
            entry.model,
            entry.overall_mean,
            fmt_delta(entry.delta),
            fmt_cell(entry.think_score),
            fmt_cell(entry.json_score),
            fmt_cell(entry.summarize_score),
            fmt_cell(entry.filename_score),
            fmt_cell(entry.vlm_score),
            entry.task_count,
            entry.date
        );
    }

    out
}

/// CLI entrypoint for `ztools model-eval --leaderboard`.
///
/// # Errors
///
/// Returns an error if loading history fails, sorting slot is invalid, or JSON serialization fails.
pub fn cli_leaderboard(
    eval_dir: Option<&Path>,
    json_output: bool,
    min_tasks: Option<usize>,
    sort_by: Option<&str>,
) -> Result<()> {
    let history = crate::ztools::eval::report::load_history_entries(eval_dir);
    let entries = generate_leaderboard(&history, min_tasks, sort_by)?;
    if json_output {
        println!("{}", serde_json::to_string_pretty(&entries)?);
    } else {
        println!("{}", format_leaderboard(&entries));
    }
    Ok(())
}
