//! Did the run FINISH, and does its mean describe what it claims to?
//!
//! Ported from `eval/completeness.py`. A model's task loop has abandon paths
//! (consecutive infra failures, the stall watchdog), and both print the right
//! thing and enforce nothing: everything downstream treats the short list of
//! results exactly like a complete one. A truncated run once reported a model
//! at 62% when its complete score was 79% -- the plausible-looking truncation
//! was the dangerous one.
//!
//! THE CLASS: a warning that exists only on stdout is not a gate.
//!
//! WHY THIS COMPARES SETS RATHER THAN THREADING A FLAG. The obvious fix is a
//! `truncated=True` set at each break. That is the per-knob invalidation hook
//! this repo has a rule about: there are two break paths today, a third is one
//! bug away, and the one that forgets the flag is the one that ships.
//! Completeness is DERIVED by diffing the tasks that were asked for against
//! the tasks that reported back; a future abandon path is covered the day it
//! is written, without knowing this module exists.

use crate::ztools::eval::runner::TaskOutcome;

/// The verdict that travels with a run: `complete` is what every consumer
/// gates on; the counts are what makes an incomplete run legible instead of
/// merely rejected.
#[derive(Debug, Clone, PartialEq)]
pub struct Completeness {
    pub expected: usize,
    pub completed: usize,
    pub missing: Vec<String>,
    pub complete: bool,
    pub reason: String,
}

/// Why the run is short, in the terms the abandon paths already use.
///
/// Derived from the outcomes rather than passed in: a reason threaded from the
/// break site is a reason the next break site forgets. The last outcome's
/// failure category distinguishes a wedged server from anything else.
fn reason(missing: &[String], outcomes: &[TaskOutcome]) -> String {
    if outcomes.is_empty() {
        return format!("no task completed; all {} missing", missing.len());
    }
    let last_category = outcomes
        .last()
        .map(|o| o.failure_category.as_str())
        .filter(|c| !c.is_empty())
        .unwrap_or("unknown");
    let mut head: Vec<String> = missing.iter().take(3).cloned().collect();
    if missing.len() > 3 {
        head.push("...".to_string());
    }
    format!(
        "abandoned after {} task(s); {} not run ({}); last failure category {}",
        outcomes.len(),
        missing.len(),
        head.join(", "),
        last_category
    )
}

/// Compare what was asked for against what reported back.
pub fn assess(expected: &[String], outcomes: &[TaskOutcome]) -> Completeness {
    let reported: std::collections::HashSet<&str> =
        outcomes.iter().map(|o| o.task.as_str()).collect();
    let missing: Vec<String> = expected
        .iter()
        .filter(|name| !reported.contains(name.as_str()))
        .cloned()
        .collect();
    let complete = missing.is_empty();
    Completeness {
        expected: expected.len(),
        completed: reported.len().min(expected.len()),
        missing,
        complete,
        reason: String::new(),
    }
}

impl Completeness {
    /// Convenience mirroring Python's `assess(tasks, results)` two-step.
    pub fn derive(expected: &[String], outcomes: &[TaskOutcome]) -> Completeness {
        let mut c = assess(expected, outcomes);
        if !c.complete {
            c.reason = reason(&c.missing.clone(), outcomes);
        }
        c
    }
}

/// Whether a persisted per-model record came from a finished run.
///
/// Absent metadata reads as COMPLETE on purpose: every historical record
/// written before this existed defaults to trusted, so adopting the gate does
/// not retroactively disqualify every real measurement the repo has taken.
pub fn record_is_complete(completeness: Option<&Completeness>) -> bool {
    match completeness {
        None => true,
        Some(c) => c.complete,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn outcome(task: &str, category: &str) -> TaskOutcome {
        TaskOutcome {
            task: task.to_string(),
            failure_category: category.to_string(),
            ..Default::default()
        }
    }

    #[test]
    fn a_run_that_reported_every_task_is_complete() {
        let expected = vec!["a".to_string(), "b".to_string()];
        let outcomes = vec![outcome("a", ""), outcome("b", "")];
        let c = Completeness::derive(&expected, &outcomes);
        assert!(c.complete);
        assert_eq!(c.reason, "");
        assert_eq!((c.expected, c.completed), (2, 2));
    }

    #[test]
    fn an_abandoned_run_is_derived_incomplete_with_the_abandon_reason() {
        // The loop broke after "a" with infra trouble: "b" never ran. No flag
        // was threaded -- the diff alone sees it.
        let expected = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        let outcomes = vec![outcome("a", ""), outcome("b", "INFRA")];
        let c = Completeness::derive(&expected, &outcomes);
        assert!(!c.complete);
        assert_eq!(c.missing, vec!["c".to_string()]);
        assert!(
            c.reason.contains("abandoned after 2 task(s)"),
            "{}",
            c.reason
        );
        assert!(
            c.reason.contains("last failure category INFRA"),
            "{}",
            c.reason
        );
    }

    #[test]
    fn a_run_that_completed_nothing_says_so() {
        let expected = vec!["a".to_string()];
        let c = Completeness::derive(&expected, &[]);
        assert!(!c.complete);
        assert!(
            c.reason.contains("no task completed; all 1 missing"),
            "{}",
            c.reason
        );
    }

    #[test]
    fn more_than_three_missing_tasks_are_elided() {
        let expected: Vec<String> = ["a", "b", "c", "d", "e"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let outcomes = vec![outcome("a", "TIMEOUT")];
        let c = Completeness::derive(&expected, &outcomes);
        assert!(c.reason.contains("b, c, d, ..."), "{}", c.reason);
        assert!(c.reason.contains("4 not run"), "{}", c.reason);
    }

    #[test]
    fn absent_completeness_metadata_reads_as_complete() {
        // Historical records predate this gate; defaulting them to incomplete
        // would retroactively disqualify real measurements.
        assert!(record_is_complete(None));
        let c = Completeness::derive(&["a".to_string()], &[]);
        assert!(!record_is_complete(Some(&c)));
    }
}
