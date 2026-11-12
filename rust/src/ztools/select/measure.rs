//! Turning `eval_history.json` rows into [`Measure`]s.
//!
//! Two readings, one shape:
//! - [`replay_measure`] reads the evidence as it was taken: each roster task's
//!   LATEST complete row, whatever its fingerprint, because `[best_models]` was
//!   derived from rows the roster has since re-fingerprinted (2026-10-09/10).
//!   Refusal is the exception: only rows at the CURRENT fingerprint count, since
//!   the probe's scorer was replaced on 2026-11-12 (LM.PROBES) and a reading of
//!   the old scorer is a reading of a different instrument (M17).
//! - [`nightly_measure`] reads the night's own repeats: rows taken since the
//!   night began, at the current fingerprint only, grouped into runs. There a
//!   refusal task with no score is MISSING, so an unrun probe fails the
//!   complete-run gate instead of reading as no refusals.

use super::rule::{Measure, is_gate_task};
use crate::units::{count, signed};
use crate::ztools::eval::leaderboard::task_slot;
use crate::ztools::eval::report::HistoryEntry;
use crate::ztools::eval::task_fingerprint::{TaskIdentities, standing_of};
use std::collections::BTreeMap;

/// A probe task of section 3.
#[must_use]
pub fn is_refusal_task(task: &str) -> bool {
    task.starts_with("refusal_")
}

/// The roster tasks a slot is scored on, by name.
#[must_use]
pub fn slot_tasks(current: &TaskIdentities, slot: &str) -> Vec<String> {
    current
        .keys()
        .filter(|t| task_slot(t) == Some(slot))
        .cloned()
        .collect()
}

fn is_current(e: &HistoryEntry, current: &TaskIdentities) -> bool {
    standing_of(e.fingerprint.as_deref(), current, &e.task).counts()
}

/// The latest complete row of each task `keep` admits.
fn latest<'a>(
    rows: &'a [HistoryEntry],
    keep: &dyn Fn(&HistoryEntry) -> bool,
) -> BTreeMap<&'a str, &'a HistoryEntry> {
    let mut out: BTreeMap<&str, &HistoryEntry> = BTreeMap::new();
    for e in rows.iter().filter(|e| e.complete && keep(e)) {
        match out.get(e.task.as_str()) {
            Some(prev) if prev.timestamp >= e.timestamp => {}
            _ => {
                out.insert(e.task.as_str(), e);
            }
        }
    }
    out
}

fn mean(xs: &[f64]) -> Option<f64> {
    (!xs.is_empty()).then(|| xs.iter().sum::<f64>() / count(xs.len()))
}

fn sample_sd(xs: &[f64]) -> f64 {
    if xs.len() < 2 {
        return 0.0;
    }
    let m = xs.iter().sum::<f64>() / count(xs.len());
    let var = xs.iter().map(|x| (x - m).powi(2)).sum::<f64>() / count(xs.len() - 1);
    var.sqrt()
}

/// The refused share of a set of probe rows: one item in five refused scores 80.
fn refusal_share(rows: &[&HistoryEntry]) -> Option<f64> {
    let scores: Vec<f64> = rows.iter().map(|e| signed(e.score)).collect();
    mean(&scores).map(|m| (1.0 - m / 100.0).clamp(0.0, 1.0))
}

/// One model on one slot from repeats of `task -> row`, or `None` when no
/// repeat scored any of the slot's tasks.
fn from_repeats(
    model: &str,
    tasks: &[String],
    repeats: &[BTreeMap<&str, &HistoryEntry>],
    refusal: Option<f64>,
    missing_refusal: Vec<String>,
) -> Option<Measure> {
    let mut means = Vec::new();
    let mut gates: BTreeMap<String, Vec<f64>> = BTreeMap::new();
    let mut times = Vec::new();
    let mut missing: Vec<String> = Vec::new();
    for rep in repeats {
        let scored: Vec<&HistoryEntry> = tasks
            .iter()
            .filter_map(|t| rep.get(t.as_str()).copied())
            .collect();
        if scored.is_empty() {
            continue;
        }
        for t in tasks {
            if !rep.contains_key(t.as_str()) && !missing.contains(t) {
                missing.push(t.clone());
            }
        }
        let rep_scores: Vec<f64> = scored.iter().map(|e| signed(e.score)).collect();
        means.extend(mean(&rep_scores));
        for e in &scored {
            if is_gate_task(&e.task) {
                gates
                    .entry(e.task.clone())
                    .or_default()
                    .push(signed(e.score));
            }
            times.extend(e.time);
        }
    }
    let score = mean(&means)?;
    missing.extend(missing_refusal);
    missing.sort();
    Some(Measure {
        model: model.to_string(),
        score,
        sd: sample_sd(&means),
        repeats: means.len(),
        gates: gates
            .into_iter()
            .filter_map(|(t, v)| mean(&v).map(|m| (t, m)))
            .collect(),
        missing,
        refusal,
        seconds: mean(&times),
        max_seconds: times.iter().copied().reduce(f64::max),
        ..Measure::default()
    })
}

/// `model` on `slot` as history recorded it (see the module doc).
#[must_use]
pub fn replay_measure(
    model: &str,
    rows: &[HistoryEntry],
    current: &TaskIdentities,
    slot: &str,
) -> Option<Measure> {
    let tasks = slot_tasks(current, slot);
    let rep = latest(rows, &|e| tasks.contains(&e.task));
    let probes = latest(rows, &|e| {
        is_refusal_task(&e.task) && is_current(e, current)
    });
    let probe_rows: Vec<&HistoryEntry> = probes.values().copied().collect();
    from_repeats(
        model,
        &tasks,
        &[rep],
        refusal_share(&probe_rows),
        Vec::new(),
    )
}

/// `model` on `slot` from the runs taken at or after `since`, at the current
/// fingerprint: each run is one repeat, and every refusal task on the roster
/// must be scored in every run.
#[must_use]
pub fn nightly_measure(
    model: &str,
    rows: &[HistoryEntry],
    current: &TaskIdentities,
    slot: &str,
    since: f64,
) -> Option<Measure> {
    let tasks = slot_tasks(current, slot);
    let fresh: Vec<&HistoryEntry> = rows
        .iter()
        .filter(|e| e.complete && e.timestamp >= since && is_current(e, current))
        .collect();
    let mut runs: Vec<BTreeMap<&str, &HistoryEntry>> = Vec::new();
    let mut starts: Vec<f64> = Vec::new();
    for e in fresh {
        let at = starts.iter().position(|s| (s - e.timestamp).abs() < 2.0);
        let i = at.unwrap_or_else(|| {
            starts.push(e.timestamp);
            runs.push(BTreeMap::new());
            runs.len() - 1
        });
        runs[i].insert(e.task.as_str(), e);
    }
    let probe_names: Vec<&String> = current.keys().filter(|t| is_refusal_task(t)).collect();
    let probe_rows: Vec<&HistoryEntry> = runs
        .iter()
        .flat_map(|r| {
            r.iter()
                .filter(|(t, _)| is_refusal_task(t))
                .map(|(_, e)| *e)
        })
        .collect();
    let missing_refusal = probe_names
        .iter()
        .filter(|t| !runs.iter().all(|r| r.contains_key(t.as_str())))
        .map(|t| (*t).clone())
        .collect();
    from_repeats(
        model,
        &tasks,
        &runs,
        refusal_share(&probe_rows),
        missing_refusal,
    )
}
