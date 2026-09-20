//! "What changed since the last run": the per-task delta table the Python
//! evaluator printed after every sweep (B4 deferral, closed 2026-09-19).
//!
//! Reads the same history the trend table reads. For every model in this
//! run, each task's score is compared with the most recent COMPLETE entry
//! for that task from an earlier timestamp; only the tasks that moved are
//! listed, because a table of thirty unchanged rows says less than one line
//! saying nothing moved.

use std::collections::BTreeMap;
use std::path::Path;

use super::report::{load_history_entries, HistoryEntry, ModelRun};

/// One task that moved.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TaskDelta {
    pub model: String,
    pub task: String,
    pub previous: i64,
    pub current: i64,
}

impl TaskDelta {
    #[must_use]
    pub const fn change(&self) -> i64 {
        self.current - self.previous
    }
}

/// The deltas for `runs` against `history`.
///
/// Pure, so the rule is pinned without a file. `history` is what the file
/// holds AFTER this run was appended, which is when the table is printed;
/// the run's own entries are skipped by timestamp (anything at or after
/// `since` is this run).
#[must_use]
pub fn deltas_since(
    runs: &[ModelRun],
    history: &BTreeMap<String, Vec<HistoryEntry>>,
    since: f64,
) -> Vec<TaskDelta> {
    let mut out = Vec::new();
    for run in runs {
        let Some(entries) = history.get(&run.model) else {
            continue;
        };
        for outcome in &run.outcomes {
            let previous = entries
                .iter()
                .filter(|e| e.task == outcome.task && e.complete && e.timestamp < since)
                .max_by(|a, b| a.timestamp.total_cmp(&b.timestamp));
            if let Some(prev) = previous {
                let current = i64::from(outcome.score);
                if prev.score != current {
                    out.push(TaskDelta {
                        model: run.model.clone(),
                        task: outcome.task.clone(),
                        previous: prev.score,
                        current,
                    });
                }
            }
        }
    }
    out
}

/// Render the table: one line per moved task, sorted by the size of the
/// move, or one line saying nothing moved. Tasks with no earlier entry are
/// not "changes" and are not listed.
#[must_use]
pub fn render_deltas(deltas: &[TaskDelta]) -> Vec<String> {
    if deltas.is_empty() {
        return vec!["Changed since the last run: nothing moved".to_string()];
    }
    let mut sorted: Vec<&TaskDelta> = deltas.iter().collect();
    sorted.sort_by_key(|d| (std::cmp::Reverse(d.change().abs()), &d.model, &d.task));
    let mut lines = vec![
        "Changed since the last run (per task, against each model's previous complete run)"
            .to_string(),
        format!(
            "{:<36} {:<28} {:>4} {:>5} {:>6}",
            "Model", "Task", "Was", "Now", "Delta"
        ),
    ];
    for d in sorted {
        lines.push(format!(
            "{:<36} {:<28} {:>4} {:>5} {:>+6}",
            super::report::truncate_name(&d.model),
            d.task,
            d.previous,
            d.current,
            d.change()
        ));
    }
    lines
}

/// The table for a run that started at `since`, from the history on disk.
#[must_use]
pub fn render_diff_from_last_run(
    runs: &[ModelRun],
    eval_dir: Option<&Path>,
    since: f64,
) -> Vec<String> {
    let history = load_history_entries(eval_dir);
    render_deltas(&deltas_since(runs, &history, since))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ztools::eval::report::tests::outcome;

    fn entry(task: &str, score: i64, ts: f64, complete: bool) -> HistoryEntry {
        HistoryEntry {
            date: "2026-09-19".into(),
            timestamp: ts,
            task: task.into(),
            score,
            time: None,
            complete,
        }
    }

    #[test]
    fn only_moved_tasks_are_listed_against_the_latest_complete_prior_entry() {
        let run = ModelRun::new(
            "m",
            &["a".into(), "b".into(), "c".into(), "d".into()],
            vec![
                outcome("a", 80),
                outcome("b", 100),
                outcome("c", 50),
                outcome("d", 10),
            ],
        );
        let mut history = BTreeMap::new();
        history.insert(
            "m".to_string(),
            vec![
                entry("a", 60, 1.0, true),
                entry("a", 70, 2.0, true), // the latest complete prior: 70 -> 80
                entry("a", 90, 3.0, false), // truncated: ignored
                entry("b", 100, 2.0, true), // unchanged: not listed
                entry("c", 100, 2.0, true), // dropped 50
                // d: no prior entry at all -> not a change
                entry("a", 80, 10.0, true), // this run's own entry, at `since`
            ],
        );
        let deltas = deltas_since(&[run], &history, 10.0);
        assert_eq!(
            deltas,
            vec![
                TaskDelta {
                    model: "m".into(),
                    task: "a".into(),
                    previous: 70,
                    current: 80
                },
                TaskDelta {
                    model: "m".into(),
                    task: "c".into(),
                    previous: 100,
                    current: 50
                },
            ]
        );
        let lines = render_deltas(&deltas);
        assert_eq!(lines.len(), 4);
        assert!(
            lines[2].contains(" c ") && lines[2].ends_with("-50"),
            "{}",
            lines[2]
        );
        assert!(
            lines[3].contains(" a ") && lines[3].ends_with("+10"),
            "{}",
            lines[3]
        );
    }

    #[test]
    fn nothing_moved_is_one_line_and_an_unknown_model_is_no_change() {
        let run = ModelRun::new("new-model", &["a".into()], vec![outcome("a", 80)]);
        let deltas = deltas_since(&[run], &BTreeMap::new(), 1.0);
        assert!(deltas.is_empty());
        assert_eq!(
            render_deltas(&deltas),
            vec!["Changed since the last run: nothing moved".to_string()]
        );
    }
}
