//! "What changed since the last run": the per-task delta table the Python
//! evaluator printed after every sweep (B4 deferral, closed 2026-09-19).
//!
//! Reads the same history the trend table reads. For every model in this
//! run, each task's score is compared with the most recent COMPLETE entry
//! for that task from an earlier timestamp — and only for a task the entry is
//! STILL the identity of, which since 2026-10-08 (`file_summary`, M1) is not
//! every entry under its name. Only the tasks that moved are listed, because a
//! table of thirty unchanged rows says less than one line saying nothing moved;
//! and the comparisons that could not be made are COUNTED, because a task whose
//! every prior entry describes a different question reads as a task with no
//! history otherwise, and the prompt that was replaced would vanish from this
//! table without a word.

use std::collections::BTreeMap;
use std::path::Path;

use super::report::{HistoryEntry, ModelRun, load_history_entries};
use super::task_fingerprint::{TaskIdentities, current_task_identities, standing_of};

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

/// The deltas for `runs` against `history`, plus the comparisons set aside.
#[derive(Debug, Clone, Default)]
pub struct DeltaReport {
    pub deltas: Vec<TaskDelta>,
    /// (model, task) pairs that have earlier COMPLETE entries and none of them
    /// is the current task's, so there was nothing to compare against.
    pub superseded: Vec<(String, String)>,
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
    current: &TaskIdentities,
) -> DeltaReport {
    let mut out = DeltaReport::default();
    for run in runs {
        let Some(entries) = history.get(&run.model) else {
            continue;
        };
        for outcome in &run.outcomes {
            let prior: Vec<&HistoryEntry> = entries
                .iter()
                .filter(|e| e.task == outcome.task && e.complete && e.timestamp < since)
                .collect();
            let previous = prior
                .iter()
                .copied()
                .filter(|e| standing_of(e.fingerprint.as_deref(), current, &e.task).counts())
                .max_by(|a, b| a.timestamp.total_cmp(&b.timestamp));
            match previous {
                Some(prev) => {
                    let current_score = i64::from(outcome.score);
                    if prev.score != current_score {
                        out.deltas.push(TaskDelta {
                            model: run.model.clone(),
                            task: outcome.task.clone(),
                            previous: prev.score,
                            current: current_score,
                        });
                    }
                }
                // Entries exist and none of them describes the task as it is
                // now: the task was replaced, so this comparison is set aside
                // and named rather than read as "nothing moved".
                None if !prior.is_empty() => out
                    .superseded
                    .push((run.model.clone(), outcome.task.clone())),
                None => {}
            }
        }
    }
    out
}

/// Render the table: one line per moved task, sorted by the size of the
/// move, or one line saying nothing moved. Tasks with no earlier entry are
/// not "changes" and are not listed.
#[must_use]
pub fn render_deltas(report: &DeltaReport) -> Vec<String> {
    let mut lines = Vec::new();
    if report.deltas.is_empty() {
        lines.push("Changed since the last run: nothing moved".to_string());
    } else {
        let mut sorted: Vec<&TaskDelta> = report.deltas.iter().collect();
        sorted.sort_by_key(|d| (std::cmp::Reverse(d.change().abs()), &d.model, &d.task));
        lines.push(
            "Changed since the last run (per task, against each model's previous complete run)"
                .to_string(),
        );
        lines.push(format!(
            "{:<36} {:<28} {:>4} {:>5} {:>6}",
            "Model", "Task", "Was", "Now", "Delta"
        ));
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
    }
    if !report.superseded.is_empty() {
        let pairs: Vec<String> = report
            .superseded
            .iter()
            .map(|(model, task)| format!("{model}/{task}"))
            .collect();
        lines.push(format!(
            "Set aside: {} task(s) whose only earlier entries describe a superseded version \
             of the task (fingerprint absent or different): {}",
            report.superseded.len(),
            pairs.join(", ")
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
    render_deltas(&deltas_since(
        runs,
        &history,
        since,
        &current_task_identities(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ztools::eval::report::tests::outcome;
    use crate::ztools::eval::task_fingerprint::TaskIdentities;

    fn entry(task: &str, score: i64, ts: f64, complete: bool) -> HistoryEntry {
        HistoryEntry {
            date: "2026-09-19".into(),
            timestamp: ts,
            task: task.into(),
            score,
            time: None,
            complete,
            fingerprint: Some("current".into()),
        }
    }

    /// An entry written against a task that has since been replaced.
    fn superseded_entry(task: &str, score: i64, ts: f64) -> HistoryEntry {
        HistoryEntry {
            fingerprint: Some("pre-m1".into()),
            ..entry(task, score, ts, true)
        }
    }

    /// An entry written before fingerprints existed.
    fn unfingerprinted_entry(task: &str, score: i64, ts: f64) -> HistoryEntry {
        HistoryEntry {
            fingerprint: None,
            ..entry(task, score, ts, true)
        }
    }

    /// Every task name below, at the identity the `entry` helper stamps.
    fn identities(tasks: &[&str]) -> TaskIdentities {
        tasks
            .iter()
            .map(|t| ((*t).to_string(), "current".to_string()))
            .collect()
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
        let report = deltas_since(&[run], &history, 10.0, &identities(&["a", "b", "c", "d"]));
        assert_eq!(
            report.deltas,
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
        assert_empty!(&report.superseded);
        let lines = render_deltas(&report);
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
        let report = deltas_since(&[run], &BTreeMap::new(), 1.0, &identities(&["a"]));
        assert_empty!(&report.deltas);
        assert_eq!(
            render_deltas(&report),
            vec!["Changed since the last run: nothing moved".to_string()]
        );
    }

    /// A task whose prior entries all describe another version of it has
    /// nothing to compare against. Saying so is the difference between a
    /// replaced prompt and a task with no history.
    #[test]
    fn a_replaced_task_has_no_prior_entry_and_says_so() {
        let run = ModelRun::new(
            "m",
            &["file_summary".into()],
            vec![outcome("file_summary", 60)],
        );
        let mut history = BTreeMap::new();
        history.insert(
            "m".to_string(),
            vec![
                superseded_entry("file_summary", 100, 1.0),
                superseded_entry("file_summary", 90, 2.0),
                unfingerprinted_entry("file_summary", 80, 3.0),
                // A truncated entry is set aside by the completeness rule, not
                // by identity, and is not reported as superseded.
                entry("file_summary", 70, 4.0, false),
            ],
        );
        let report = deltas_since(&[run], &history, 10.0, &identities(&["file_summary"]));
        assert_empty!(&report.deltas, "nothing was comparable: {report:?}");
        assert_eq!(
            report.superseded,
            vec![("m".to_string(), "file_summary".to_string())]
        );
        let lines = render_deltas(&report);
        assert_eq!(lines[0], "Changed since the last run: nothing moved");
        assert!(
            lines[1].starts_with("Set aside: 1 task(s)") && lines[1].contains("m/file_summary"),
            "{lines:?}"
        );
    }

    /// The current fingerprint's entry IS comparable, and only the latest of
    /// them counts.
    #[test]
    fn only_current_fingerprint_entries_are_compared() {
        let run = ModelRun::new("m", &["t".into()], vec![outcome("t", 50)]);
        let mut history = BTreeMap::new();
        history.insert(
            "m".to_string(),
            vec![
                entry("t", 100, 5.0, true),          // current, and the latest current
                superseded_entry("t", 10, 9.0),      // superseded, and the LATEST of all
                unfingerprinted_entry("t", 30, 8.0), // unknown, also later than 5.0
            ],
        );
        let report = deltas_since(&[run], &history, 10.0, &identities(&["t"]));
        assert_eq!(
            report.deltas,
            vec![TaskDelta {
                model: "m".into(),
                task: "t".into(),
                previous: 100,
                current: 50
            }],
            "the latest CURRENT entry wins, not the latest entry: {report:?}"
        );
        assert_empty!(&report.superseded);
    }
}
