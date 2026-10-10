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
    pub task_count: usize,
    pub overall_mean: f64,
    pub delta: Option<f64>,
    pub think_score: Option<f64>,
    pub json_score: Option<f64>,
    pub summarize_score: Option<f64>,
    pub filename_score: Option<f64>,
    pub vlm_score: Option<f64>,
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

/// Aggregates each model's latest clean run from history and ranks by overall mean or slot descending,
/// optionally filtering by task category (`M13`).
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

/// Aggregates each model's latest clean run from history and ranks by overall mean or slot descending.
///
/// # Errors
///
/// Returns an error if `sort_by` is not a recognized slot name.
pub fn generate_leaderboard(
    history: &BTreeMap<String, Vec<HistoryEntry>>,
    min_tasks: Option<usize>,
    sort_by: Option<&str>,
) -> Result<Vec<ModelLeaderboardEntry>> {
    generate_leaderboard_filtered(history, min_tasks, sort_by, None)
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

fn csv_escape(field: &str) -> String {
    if field.contains(',') || field.contains('"') || field.contains('\n') {
        format!("\"{}\"", field.replace('"', "\"\""))
    } else {
        field.to_string()
    }
}

fn fmt_csv_opt(val: Option<f64>) -> String {
    val.map_or_else(String::new, |v| format!("{v:.1}"))
}

/// Formats the leaderboard entries into CSV (`M14`).
#[must_use]
pub fn format_leaderboard_csv(entries: &[ModelLeaderboardEntry]) -> String {
    let mut out = String::new();
    out.push_str("model,mean,delta,think,json,summarize,filename,vlm,tasks,date\n");
    for entry in entries {
        let _ = writeln!(
            out,
            "{},{:.1},{},{},{},{},{},{},{},{}",
            csv_escape(&entry.model),
            entry.overall_mean,
            fmt_csv_opt(entry.delta),
            fmt_csv_opt(entry.think_score),
            fmt_csv_opt(entry.json_score),
            fmt_csv_opt(entry.summarize_score),
            fmt_csv_opt(entry.filename_score),
            fmt_csv_opt(entry.vlm_score),
            entry.task_count,
            csv_escape(&entry.date)
        );
    }
    out
}

/// Formats the leaderboard entries grouped by detected model family (`M15`).
#[must_use]
pub fn format_leaderboard_by_family(entries: &[ModelLeaderboardEntry]) -> String {
    if entries.is_empty() {
        return "No complete model eval runs found in history.\n".to_string();
    }

    let mut groups: Vec<(&'static str, Vec<&ModelLeaderboardEntry>)> = Vec::new();
    for entry in entries {
        let fam = detect_model_family(&entry.model);
        if let Some((_, list)) = groups.iter_mut().find(|(f, _)| *f == fam) {
            list.push(entry);
        } else {
            groups.push((fam, vec![entry]));
        }
    }

    let mut out = String::new();
    out.push_str("# Model Evaluation Leaderboard (Grouped by Family)\n\n");
    for (fam, fam_entries) in groups {
        let _ = writeln!(out, "## Family: {fam}\n");
        out.push_str(
            "| Rank | Model | Mean | Delta | Think | JSON | Summarize | Filename | VLM | Tasks | Date |\n",
        );
        out.push_str(
            "| ---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |\n",
        );
        for (idx, entry) in fam_entries.iter().enumerate() {
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
        out.push('\n');
    }
    out
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
/// Returns an error if loading history fails, sorting slot is invalid, JSON serialization fails,
/// or regression threshold is breached (`M16`).
pub fn cli_leaderboard(eval_dir: Option<&Path>, opts: &LeaderboardOptions<'_>) -> Result<()> {
    let history = crate::ztools::eval::report::load_history_entries(eval_dir);
    let entries =
        generate_leaderboard_filtered(&history, opts.min_tasks, opts.sort_by, opts.category)?;

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
