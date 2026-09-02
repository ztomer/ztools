//! Eval result persistence and reporting.
//!
//! Ported from `eval/report_core.py` (default dir), `report_history.py`
//! (per-model history with truncated-run quarantine), and the metric halves of
//! `report_metrics.py` (winners, score stats, CSV export, historical trends).
//!
//! Two rules carried over verbatim:
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

use std::collections::BTreeMap;
use std::io::Write as _;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::ztools::eval::completeness::{record_is_complete, Completeness};
use crate::ztools::eval::runner::TaskOutcome;

/// Where eval artefacts live when the caller does not say otherwise.
pub fn default_eval_dir() -> PathBuf {
    dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".config/ztools")
}

/// Whether a model name is a test double rather than a real served model.
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
    pub fn new(model: &str, expected: &[String], outcomes: Vec<TaskOutcome>) -> Self {
        Self {
            model: model.to_string(),
            completeness: Some(Completeness::derive(expected, &outcomes)),
            outcomes,
        }
    }
}

/// One historical observation. `complete` is ABSENT on records written before
/// truncation tracking existed, and absence means COMPLETE -- defaulting old
/// entries to incomplete would retroactively disqualify real measurements.
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
}

fn default_true() -> bool {
    true
}

fn history_path(eval_dir: Option<&Path>) -> PathBuf {
    let base = match eval_dir {
        Some(d) => d.to_path_buf(),
        None => default_eval_dir(),
    };
    base.join("eval_history.json")
}

/// Append this run's per-task scores to `eval_history.json`, keyed by model.
///
/// Test doubles are skipped entirely; entries from an incomplete run carry
/// `complete: false` so [`load_historical_stats`] can refuse to average them.
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
    for outcome in &run.outcomes {
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
        });
    }

    if let Ok(text) = serde_json::to_string_pretty(&history) {
        std::fs::write(&path, text)?;
    }
    Ok(history)
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
#[derive(Debug, Clone, Serialize)]
pub struct ModelStats {
    pub mean: f64,
    pub median: f64,
    pub stdev: f64,
    pub min: i64,
    pub max: i64,
    pub runs: usize,
    pub excluded: usize,
}

pub fn load_historical_stats(eval_dir: Option<&Path>) -> BTreeMap<String, ModelStats> {
    let mut stats = BTreeMap::new();
    for (model, entries) in load_history(eval_dir) {
        if entries.is_empty() {
            continue;
        }
        // Absent `complete` field deserializes to true (serde default), so
        // legacy entries are trusted exactly like Python's `.get(..., True)`.
        let countable: Vec<&HistoryEntry> = entries.iter().filter(|e| e.complete).collect();
        let excluded = entries.len() - countable.len();
        let mut scores: Vec<i64> = countable.iter().map(|e| e.score).collect();
        if scores.is_empty() {
            continue;
        }
        scores.sort_unstable();
        let n = scores.len();
        let mean = scores.iter().sum::<i64>() as f64 / n as f64;
        let median = if n % 2 == 1 {
            scores[n / 2] as f64
        } else {
            (scores[n / 2 - 1] + scores[n / 2]) as f64 / 2.0
        };
        let stdev = if n > 1 {
            let var = scores
                .iter()
                .map(|s| {
                    let d = *s as f64 - mean;
                    d * d
                })
                .sum::<f64>()
                / (n - 1) as f64;
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
                min: scores[0],
                max: scores[n - 1],
                runs: n,
                excluded,
            },
        );
    }
    stats
}

/// Which model won each task across this batch of runs. Ties keep the first
/// winner seen, matching Python's strict `>` comparison.
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

fn status_word(score: u8) -> &'static str {
    if score >= 90 {
        "PASS"
    } else if score >= 50 {
        "WARN"
    } else {
        "FAIL"
    }
}

/// Export one row per (model, task): the shape downstream sheets expect.
pub fn export_csv(runs: &[ModelRun], output_file: &Path) -> std::io::Result<()> {
    let mut file = std::io::BufWriter::new(std::fs::File::create(output_file)?);
    writeln!(
        file,
        "Model,Task,Score,Status,Time(s),Failure,Failure_Category"
    )?;
    for run in runs {
        for o in &run.outcomes {
            writeln!(
                file,
                "{},{},{},{},{},{},{}",
                csv_escape(&run.model),
                csv_escape(&o.task),
                o.score,
                status_word(o.score),
                o.time_secs,
                csv_escape(o.error.as_deref().unwrap_or("")),
                o.failure_category
            )?;
        }
    }
    Ok(())
}

fn csv_escape(field: &str) -> String {
    if field.contains(',') || field.contains('"') || field.contains('\n') {
        format!("\"{}\"", field.replace('"', "\"\""))
    } else {
        field.to_string()
    }
}

/// Render the historical trends table (mean/median/stdev/runs/excluded per
/// model, best first, matching the Python report's ordering). Empty when no
/// history exists yet.
pub fn render_historical_trends(eval_dir: Option<&Path>) -> Vec<String> {
    let stats = load_historical_stats(eval_dir);
    if stats.is_empty() {
        return Vec::new();
    }
    let mut rows: Vec<(&String, &ModelStats)> = stats.iter().collect();
    rows.sort_by(|(_, a), (_, b)| b.mean.total_cmp(&a.mean));

    let mut lines = vec![
        "Historical Trends (countable runs only; truncated entries excluded)".to_string(),
        format!(
            "{:<36} {:>6} {:>6} {:>7} {:>5} {:>5} {:>5} {:>9}",
            "Model", "Mean", "Median", "Stdev", "Min", "Max", "Runs", "Excluded"
        ),
    ];
    for (name, s) in rows {
        lines.push(format!(
            "{:<36} {:>6.0} {:>6.0} {:>7.1} {:>5} {:>5} {:>5} {:>9}",
            truncate_name(name),
            s.mean,
            s.median,
            s.stdev,
            s.min,
            s.max,
            s.runs,
            s.excluded
        ));
    }
    lines
}

fn truncate_name(name: &str) -> String {
    if name.len() <= 36 {
        name.to_string()
    } else {
        format!("{}...", &name[..33])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn outcome(task: &str, score: u8) -> TaskOutcome {
        TaskOutcome {
            task: task.to_string(),
            score,
            status: if score >= 90 { "ok" } else { "fail" }.to_string(),
            ..Default::default()
        }
    }

    fn run(model: &str, outcomes: Vec<TaskOutcome>, complete: bool) -> ModelRun {
        let mut r = ModelRun::new(model, &[], outcomes);
        if let Some(c) = r.completeness.as_mut() {
            c.complete = complete;
            if !complete {
                c.missing = vec!["never-ran".to_string()];
                c.reason = "test".to_string();
            }
        }
        r
    }

    #[test]
    fn test_models_never_enter_the_leaderboard() {
        let dir = tempfile::tempdir().unwrap();
        save_historical_results(
            &run("mock-model", vec![outcome("t", 100)], true),
            Some(dir.path()),
        )
        .unwrap();
        save_historical_results(
            &run("fake-70b", vec![outcome("t", 100)], true),
            Some(dir.path()),
        )
        .unwrap();
        save_historical_results(
            &run("real-model", vec![outcome("t", 80)], true),
            Some(dir.path()),
        )
        .unwrap();
        let stats = load_historical_stats(Some(dir.path()));
        assert!(!stats.contains_key("mock-model"), "{stats:?}");
        assert!(!stats.contains_key("fake-70b"), "{stats:?}");
        assert!(stats.contains_key("real-model"));
    }

    #[test]
    fn truncated_entries_are_written_marked_and_excluded_from_averages() {
        // MARKED, not dropped: the individual scores exist on disk...
        let dir = tempfile::tempdir().unwrap();
        save_historical_results(
            &run("ornith-test", vec![outcome("easy", 100)], false),
            Some(dir.path()),
        )
        .unwrap();
        let raw: BTreeMap<String, Vec<HistoryEntry>> = serde_json::from_str(
            &std::fs::read_to_string(dir.path().join("eval_history.json")).unwrap(),
        )
        .unwrap();
        assert!(
            !raw["ornith-test"][0].complete,
            "verdict travels with the entry"
        );

        // ...but no aggregate ever averages them, and the discrepancy between
        // runs and entry count is surfaced rather than hidden.
        let stats = load_historical_stats(Some(dir.path()));
        assert!(
            !stats.contains_key("ornith-test"),
            "nothing countable -> no stats: {stats:?}"
        );

        save_historical_results(
            &run("ornith-test", vec![outcome("easy", 60)], true),
            Some(dir.path()),
        )
        .unwrap();
        let stats = load_historical_stats(Some(dir.path()))
            .get("ornith-test")
            .unwrap()
            .clone();
        assert_eq!((stats.runs, stats.excluded), (1, 1));
        assert_eq!(stats.mean, 60.0, "the unclean 100 must not be averaged");
    }

    #[test]
    fn legacy_records_without_the_complete_field_are_trusted() {
        let dir = tempfile::tempdir().unwrap();
        let legacy = json!({
            "old-model": [
                {"date": "2026-01-01", "timestamp": 1_767_225_600.0, "task": "t",
                 "score": 90, "time": 1.0}
            ]
        });
        std::fs::write(dir.path().join("eval_history.json"), legacy.to_string()).unwrap();
        let stats = load_historical_stats(Some(dir.path()));
        assert_eq!(stats["old-model"].runs, 1, "absent complete means complete");
        assert_eq!(stats["old-model"].excluded, 0);
    }

    #[test]
    fn zero_scores_count_toward_the_mean() {
        // `if e.get("score")` falsy-for-zero once made a model that scored 0
        // on half its runs look identical to one that never failed.
        let dir = tempfile::tempdir().unwrap();
        save_historical_results(
            &run("m", vec![outcome("a", 100), outcome("b", 0)], true),
            Some(dir.path()),
        )
        .unwrap();
        let stats = load_historical_stats(Some(dir.path())).remove("m").unwrap();
        assert_eq!((stats.runs, stats.min, stats.max), (2, 0, 100));
        assert_eq!(stats.mean, 50.0);
    }

    #[test]
    fn winners_take_the_best_score_per_task_across_runs() {
        let runs = vec![
            run("a", vec![outcome("t1", 90), outcome("t2", 40)], true),
            run("b", vec![outcome("t1", 95), outcome("t2", 40)], true),
        ];
        let winners = compute_task_winners(&runs);
        assert_eq!(winners["t1"].0, "b");
        assert_eq!(winners["t2"].0, "a", "tie keeps the first winner seen");
    }

    #[test]
    fn csv_export_matches_the_downstream_sheet_shape() {
        let dir = tempfile::tempdir().unwrap();
        let mut o = outcome("t1", 95);
        o.time_secs = 1.5;
        o.error = Some("HTTP 503, at capacity".to_string());
        o.failure_category = "INFRA".to_string();
        let runs = vec![run("model-a", vec![o], true)];
        let out = dir.path().join("results.csv");
        export_csv(&runs, &out).unwrap();
        let text = std::fs::read_to_string(out).unwrap();
        let mut lines = text.lines();
        assert_eq!(
            lines.next(),
            Some("Model,Task,Score,Status,Time(s),Failure,Failure_Category")
        );
        let row = lines.next().unwrap();
        // Quoted because the error contains a comma.
        assert_eq!(
            row,
            "model-a,t1,95,PASS,1.5,\"HTTP 503, at capacity\",INFRA"
        );
    }

    #[test]
    fn trends_render_worst_first_with_an_excluded_column() {
        let dir = tempfile::tempdir().unwrap();
        save_historical_results(
            &run("slow-model", vec![outcome("t", 40)], true),
            Some(dir.path()),
        )
        .unwrap();
        save_historical_results(
            &run("fast-model", vec![outcome("t", 90)], true),
            Some(dir.path()),
        )
        .unwrap();
        let lines = render_historical_trends(Some(dir.path()));
        assert!(lines[0].starts_with("Historical Trends"));
        assert!(lines[2].contains("fast-model"), "{lines:?}");
        assert!(
            lines[3].contains("slow-model"),
            "sorted worst-first: {lines:?}"
        );
    }
}
