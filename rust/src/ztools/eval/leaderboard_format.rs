//! Rendering the leaderboard rows: markdown, family-grouped markdown and CSV.
//!
//! Split from `leaderboard.rs` when M17 added the "not measured on the current
//! task" state, so the ranking and its presentation each stay under the
//! 500-line cap.
//!
//! THREE cell states, never two. A number is a score on the task as it stands
//! now; `—` means the model's run never had a row for that slot; `not current`
//! means it did, but every such row was taken on a task that has since changed
//! (a different fingerprint under the same name). Folding the third into `—`
//! would claim the model was never asked; folding it into a number is the M17
//! defect itself, a json score from retired weekend rows ranked beside scores
//! on the new ones.

use std::fmt::Write as _;

use super::leaderboard::{ModelLeaderboardEntry, task_slot};

/// The cell a slot gets when the model was measured on it, but only on a task
/// that has since changed.
pub const NOT_CURRENT_CELL: &str = "not current";

/// The legend line printed under any table that shows [`NOT_CURRENT_CELL`].
pub const NOT_CURRENT_LEGEND: &str = "`not current`: not measured on the current task. The model's latest \
     result for it was taken on a task that has since changed, so it is not scored beside models \
     measured on the task as it is now.";

/// Whether `entry` holds a not-current row in `slot`.
fn slot_is_stale(entry: &ModelLeaderboardEntry, slot: &str) -> bool {
    entry.not_current.iter().any(|t| task_slot(t) == Some(slot))
}

fn fmt_slot(entry: &ModelLeaderboardEntry, slot: &str, val: Option<f64>) -> String {
    match val {
        Some(v) => format!("{v:.1}%"),
        None if slot_is_stale(entry, slot) => NOT_CURRENT_CELL.to_string(),
        None => "—".to_string(),
    }
}

fn fmt_mean(entry: &ModelLeaderboardEntry) -> String {
    match entry.overall_mean {
        Some(v) => format!("{v:.1}%"),
        None if !entry.not_current.is_empty() => NOT_CURRENT_CELL.to_string(),
        None => "—".to_string(),
    }
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

/// A CSV cell: the score, `not current`, or empty for never measured.
fn fmt_csv_slot(entry: &ModelLeaderboardEntry, slot: &str, val: Option<f64>) -> String {
    match val {
        Some(v) => format!("{v:.1}"),
        None if slot_is_stale(entry, slot) => NOT_CURRENT_CELL.to_string(),
        None => String::new(),
    }
}

fn fmt_csv_opt(val: Option<f64>) -> String {
    val.map_or_else(String::new, |v| format!("{v:.1}"))
}

/// Formats the leaderboard entries into CSV (`M14`).
///
/// `not_current` is the last column: the tasks, `;`-separated, whose latest
/// result carries a fingerprint other than the current task's.
#[must_use]
pub fn format_leaderboard_csv(entries: &[ModelLeaderboardEntry]) -> String {
    let mut out = String::new();
    out.push_str("model,mean,delta,think,json,summarize,filename,vlm,tasks,date,not_current\n");
    for entry in entries {
        let _ = writeln!(
            out,
            "{},{},{},{},{},{},{},{},{},{},{}",
            csv_escape(&entry.model),
            fmt_csv_opt(entry.overall_mean),
            fmt_csv_opt(entry.delta),
            fmt_csv_slot(entry, "think", entry.think_score),
            fmt_csv_slot(entry, "json", entry.json_score),
            fmt_csv_slot(entry, "summarize", entry.summarize_score),
            fmt_csv_slot(entry, "filename", entry.filename_score),
            fmt_csv_slot(entry, "vlm", entry.vlm_score),
            entry.task_count,
            csv_escape(&entry.date),
            csv_escape(&entry.not_current.join(";"))
        );
    }
    out
}

const TABLE_HEADER: &str =
    "| Rank | Model | Mean | Delta | Think | JSON | Summarize | Filename | VLM | Tasks | Date |\n";
const TABLE_RULE: &str =
    "| ---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |\n";

fn push_row(out: &mut String, rank: usize, entry: &ModelLeaderboardEntry) {
    let _ = writeln!(
        out,
        "| {} | `{}` | {} | {} | {} | {} | {} | {} | {} | {} | {} |",
        rank,
        entry.model,
        fmt_mean(entry),
        fmt_delta(entry.delta),
        fmt_slot(entry, "think", entry.think_score),
        fmt_slot(entry, "json", entry.json_score),
        fmt_slot(entry, "summarize", entry.summarize_score),
        fmt_slot(entry, "filename", entry.filename_score),
        fmt_slot(entry, "vlm", entry.vlm_score),
        entry.task_count,
        entry.date
    );
}

/// The legend and the per-model list of not-current tasks, when any exist.
fn push_not_current(out: &mut String, entries: &[&ModelLeaderboardEntry]) {
    let stale: Vec<&&ModelLeaderboardEntry> = entries
        .iter()
        .filter(|e| !e.not_current.is_empty())
        .collect();
    if stale.is_empty() {
        return;
    }
    let _ = writeln!(out, "\n{NOT_CURRENT_LEGEND}\n");
    for entry in stale {
        let _ = writeln!(out, "- `{}`: {}", entry.model, entry.not_current.join(", "));
    }
}

/// Formats the leaderboard entries grouped by detected model family (`M15`).
#[must_use]
pub fn format_leaderboard_by_family(entries: &[ModelLeaderboardEntry]) -> String {
    if entries.is_empty() {
        return "No complete model eval runs found in history.\n".to_string();
    }

    let mut groups: Vec<(&'static str, Vec<&ModelLeaderboardEntry>)> = Vec::new();
    for entry in entries {
        let fam = super::leaderboard::detect_model_family(&entry.model);
        if let Some((_, list)) = groups.iter_mut().find(|(f, _)| *f == fam) {
            list.push(entry);
        } else {
            groups.push((fam, vec![entry]));
        }
    }

    let mut out = String::new();
    out.push_str("# Model Evaluation Leaderboard (Grouped by Family)\n\n");
    for (fam, fam_entries) in &groups {
        let _ = writeln!(out, "## Family: {fam}\n");
        out.push_str(TABLE_HEADER);
        out.push_str(TABLE_RULE);
        for (idx, entry) in fam_entries.iter().enumerate() {
            push_row(&mut out, idx + 1, entry);
        }
        out.push('\n');
    }
    let all: Vec<&ModelLeaderboardEntry> = entries.iter().collect();
    push_not_current(&mut out, &all);
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
    out.push_str(TABLE_HEADER);
    out.push_str(TABLE_RULE);
    for (idx, entry) in entries.iter().enumerate() {
        push_row(&mut out, idx + 1, entry);
    }
    let all: Vec<&ModelLeaderboardEntry> = entries.iter().collect();
    push_not_current(&mut out, &all);
    out
}
