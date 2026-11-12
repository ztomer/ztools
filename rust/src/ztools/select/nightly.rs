//! `ztools models nightly`: one candidate a night, PROPOSE-ONLY (LM.SELECT).
//!
//! The night: take the one staged candidate (a crashed night's), or ask the
//! scout for one and stage it if the download leaves the disk floor (else record
//! `no room`); run the candidate and each incumbent it would replace through the
//! slot's tasks and the refusal probe, 3 repeats each; decide every slot by the
//! rule; delete a rejected candidate that night (only a directory its own ledger
//! names); and write one findings line. It never switches a slot: the automatic
//! mode is phase 5 and the owner's to turn on.

use super::measure::{is_refusal_task, nightly_measure, slot_tasks};
use super::rule::{Decision, Rule, decide};
use crate::ztools::eval::report::HistoryEntry;
use crate::ztools::eval::task_fingerprint::TaskIdentities;
use crate::ztools::scout::rules::{self, File};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Repeats per model per slot (section 4).
pub const REPEATS: usize = 3;

/// A candidate the scout would stage, and the slots whose incumbent it is a
/// candidate for.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Nominee {
    pub repo: String,
    pub slots: Vec<String>,
}

/// One directory this job staged: the only kind it may delete.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Staged {
    pub repo: String,
    /// Short name the eval addresses it by.
    pub model: String,
    pub slots: Vec<String>,
    pub bytes: u64,
    pub date: String,
}

/// The machine, as the night sees it.
pub trait Night {
    fn now(&self) -> f64;
    fn date(&self) -> String;
    fn free_bytes(&self) -> Option<u64>;
    /// The ledger's staged candidates.
    fn staged(&self) -> Vec<Staged>;
    /// The scout's first candidate that passes every filter but disk.
    ///
    /// # Errors
    /// When the scout could not run.
    fn nominate(&mut self) -> Result<Option<Nominee>, String>;
    /// The repo's files.
    ///
    /// # Errors
    /// When the list could not be read.
    fn files(&mut self, repo: &str) -> Result<Vec<File>, String>;
    /// Download into the models directory, recording the ledger first.
    ///
    /// # Errors
    /// When any file fails; the ledger stays so the next night deletes it.
    fn download(&mut self, staged: &Staged, files: &[File]) -> Result<(), String>;
    /// Run `model` on `tasks` once under the GPU lock, writing history.
    ///
    /// # Errors
    /// When the eval could not run.
    fn eval(&mut self, model: &str, tasks: &[String]) -> Result<(), String>;
    fn history(&self) -> BTreeMap<String, Vec<HistoryEntry>>;
    fn current(&self) -> TaskIdentities;
    fn slots(&self) -> BTreeMap<String, String>;
    /// `rm -rf` the staged directory and its ledger.
    ///
    /// # Errors
    /// When it could not be removed.
    fn delete(&mut self, staged: &Staged) -> Result<(), String>;
}

/// How a night ended.
#[derive(Debug, Clone, PartialEq)]
pub enum Outcome {
    Nothing(String),
    NoRoom(String),
    Failed(String),
    Decided {
        candidate: String,
        decisions: Vec<Decision>,
        /// Deleted that night because no slot chose it.
        deleted: bool,
    },
}

/// The findings line: `LM <date>: think kept X (cand Y 86.1, -2.9); ...`.
#[must_use]
pub fn findings_line(date: &str, outcome: &Outcome) -> String {
    let body = match outcome {
        Outcome::Nothing(why) => format!("nothing staged ({why})"),
        Outcome::NoRoom(why) => format!("no room ({why})"),
        Outcome::Failed(why) => format!("✗ {why}"),
        Outcome::Decided {
            candidate,
            decisions,
            deleted,
        } => {
            let mut parts: Vec<String> = decisions
                .iter()
                .map(|d| {
                    let j = d.judged.iter().find(|j| &j.model == candidate);
                    let m = j.map_or_else(String::new, |j| {
                        format!(
                            " (cand {} {:.1}, {:+.1}; {})",
                            j.model, j.score, j.delta, j.why
                        )
                    });
                    if d.switched() {
                        format!("{} -> {} proposed{m}", d.slot, d.chosen)
                    } else {
                        format!("{} kept {}{m}", d.slot, d.incumbent)
                    }
                })
                .collect();
            if *deleted {
                parts.push(format!("{candidate} rejected, deleted"));
            }
            parts.join("; ")
        }
    };
    format!("LM {date}: {body}")
}

fn stage(night: &mut dyn Night) -> Result<Staged, Outcome> {
    let nominee = match night.nominate() {
        Ok(Some(n)) => n,
        Ok(None) => return Err(Outcome::Nothing("the scout nominates nothing".into())),
        Err(why) => return Err(Outcome::Failed(format!("scout: {why}"))),
    };
    let files = night
        .files(&nominee.repo)
        .map_err(|why| Outcome::Failed(format!("{}: {why}", nominee.repo)))?;
    let bytes = rules::download_bytes(&files);
    let disk = rules::disk(bytes, night.free_bytes(), 0);
    match disk {
        rules::Check::Pass(_) => {}
        rules::Check::Fail(why) => return Err(Outcome::NoRoom(format!("{}: {why}", nominee.repo))),
        rules::Check::Unverified(why) => {
            return Err(Outcome::Failed(format!("{}: {why}", nominee.repo)));
        }
    }
    let staged = Staged {
        model: crate::ztools::refcheck::short_name(&nominee.repo),
        repo: nominee.repo,
        slots: nominee.slots,
        bytes,
        date: night.date(),
    };
    night
        .download(&staged, &files)
        .map_err(|why| Outcome::Failed(format!("download {}: {why}", staged.repo)))?;
    Ok(staged)
}

/// The slot's tasks and the whole refusal probe.
fn night_tasks(current: &TaskIdentities, slot: &str) -> Vec<String> {
    let mut tasks = slot_tasks(current, slot);
    tasks.extend(current.keys().filter(|t| is_refusal_task(t)).cloned());
    tasks
}

fn evaluate(night: &mut dyn Night, staged: &Staged, rule: &Rule, since: f64) -> Outcome {
    let current = night.current();
    let slots = night.slots();
    for slot in &staged.slots {
        let Some(inc) = slots.get(slot) else { continue };
        let tasks = night_tasks(&current, slot);
        for model in [inc.as_str(), staged.model.as_str()] {
            for _ in 0..REPEATS {
                if let Err(why) = night.eval(model, &tasks) {
                    return Outcome::Failed(format!("eval {model} on {slot}: {why}"));
                }
            }
        }
    }
    let history = night.history();
    let rows = |m: &str| history.get(m).map_or(&[][..], Vec::as_slice);
    let mut decisions = Vec::new();
    for slot in &staged.slots {
        let Some(inc) = slots.get(slot) else { continue };
        let Some(i) = nightly_measure(inc, rows(inc), &current, slot, since) else {
            return Outcome::Failed(format!("{slot}: {inc} has no rows from tonight"));
        };
        let cands: Vec<_> =
            nightly_measure(&staged.model, rows(&staged.model), &current, slot, since)
                .into_iter()
                .collect();
        decisions.push(decide(slot, &i, &cands, rule));
    }
    Outcome::Decided {
        candidate: staged.model.clone(),
        deleted: !decisions.iter().any(Decision::switched),
        decisions,
    }
}

/// Run one night.
pub fn run(night: &mut dyn Night, rule: &Rule) -> Outcome {
    let since = night.now();
    let staged = match night.staged().as_slice() {
        [] => match stage(night) {
            Ok(s) => s,
            Err(outcome) => return outcome,
        },
        [one] => one.clone(),
        many => {
            return Outcome::Failed(format!("{} candidates staged, the limit is 1", many.len()));
        }
    };
    let outcome = evaluate(night, &staged, rule, since);
    match outcome {
        Outcome::Decided { deleted: true, .. } => match night.delete(&staged) {
            Ok(()) => outcome,
            Err(why) => Outcome::Failed(format!("{} rejected but not deleted: {why}", staged.repo)),
        },
        other => other,
    }
}
