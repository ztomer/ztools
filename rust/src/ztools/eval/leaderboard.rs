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

/// Aggregates each model's latest clean run from history and ranks by overall mean descending.
///
/// When `min_tasks` is specified (`Some(min)`), models whose latest clean run holds fewer
/// than `min` tasks are filtered out. When `None`, the default threshold (5) is used to prioritize
/// multi-task sweeps over spot checks without excluding spot checks.
#[must_use]
pub fn generate_leaderboard(
    history: &BTreeMap<String, Vec<HistoryEntry>>,
    min_tasks: Option<usize>,
) -> Vec<ModelLeaderboardEntry> {
    let mut rows = Vec::new();
    let threshold = min_tasks.unwrap_or(5);

    for (model, entries) in history {
        // Only clean (complete: true) entries participate in leaderboard
        let complete_entries: Vec<&HistoryEntry> = entries.iter().filter(|e| e.complete).collect();
        if complete_entries.is_empty() {
            continue;
        }

        // Group complete entries into distinct runs by timestamp (within 2 seconds)
        let mut runs: Vec<Vec<&HistoryEntry>> = Vec::new();
        for entry in complete_entries {
            if let Some(group) = runs
                .iter_mut()
                .find(|g| (g[0].timestamp - entry.timestamp).abs() < 2.0)
            {
                group.push(entry);
            } else {
                runs.push(vec![entry]);
            }
        }

        // Prefer multi-task runs (>= threshold tasks) over single-task spot checks,
        // selecting the latest run timestamp in that category. If min_tasks is explicitly
        // set, filter out runs with fewer tasks entirely.
        let eligible_runs: Vec<&Vec<&HistoryEntry>> = if min_tasks.is_some() {
            runs.iter().filter(|g| g.len() >= threshold).collect()
        } else {
            let multi_task: Vec<&Vec<&HistoryEntry>> =
                runs.iter().filter(|g| g.len() >= threshold).collect();
            if multi_task.is_empty() {
                runs.iter().collect()
            } else {
                multi_task
            }
        };

        let chosen_run = eligible_runs.into_iter().max_by(|a, b| {
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
        let think_score = slot_mean(run_entries, "think");
        let json_score = slot_mean(run_entries, "json");
        let summarize_score = slot_mean(run_entries, "summarize");
        let filename_score = slot_mean(run_entries, "filename");
        let vlm_score = slot_mean(run_entries, "vlm");

        rows.push(ModelLeaderboardEntry {
            model: model.clone(),
            date: run_entries[0].date.clone(),
            timestamp: latest_ts,
            task_count: run_entries.len(),
            overall_mean,
            think_score,
            json_score,
            summarize_score,
            filename_score,
            vlm_score,
        });
    }

    // Rank multi-task evaluations first (>= threshold tasks) by overall mean descending,
    // followed by spot checks (< threshold tasks), breaking ties by task count descending, then model name.
    rows.sort_by(|a, b| {
        let a_full = a.task_count >= threshold;
        let b_full = b.task_count >= threshold;
        b_full
            .cmp(&a_full)
            .then_with(|| {
                b.overall_mean
                    .partial_cmp(&a.overall_mean)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .then_with(|| b.task_count.cmp(&a.task_count))
            .then_with(|| a.model.cmp(&b.model))
    });

    rows
}

fn fmt_cell(val: Option<f64>) -> String {
    val.map_or_else(|| "—".to_string(), |v| format!("{v:.1}%"))
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
        "| Rank | Model | Mean | Think | JSON | Summarize | Filename | VLM | Tasks | Date |\n",
    );
    out.push_str("| ---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |\n");

    for (idx, entry) in entries.iter().enumerate() {
        let _ = writeln!(
            out,
            "| {} | `{}` | {:.1}% | {} | {} | {} | {} | {} | {} | {} |",
            idx + 1,
            entry.model,
            entry.overall_mean,
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
/// Returns an error if loading history fails or JSON serialization fails.
pub fn cli_leaderboard(
    eval_dir: Option<&Path>,
    json_output: bool,
    min_tasks: Option<usize>,
) -> Result<()> {
    let history = crate::ztools::eval::report::load_history_entries(eval_dir);
    let entries = generate_leaderboard(&history, min_tasks);
    if json_output {
        println!("{}", serde_json::to_string_pretty(&entries)?);
    } else {
        println!("{}", format_leaderboard(&entries));
    }
    Ok(())
}
