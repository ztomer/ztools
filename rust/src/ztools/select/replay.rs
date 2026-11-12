//! The replay: the decision rule over `eval_history.json` must reproduce
//! today's `[best_models]` exactly before the nightly is trusted (LM.DESIGN
//! section 4, phase 4's check).
//!
//! Each slot's incumbent is today's `[best_models]` model; every other
//! INSTALLED model with rows on the slot is a candidate (a model that is not
//! installed cannot be switched to, which is why muse-glimmer and the plain
//! `jang_6d` are out). The rule must keep every slot. The replay is not vacuous:
//! `gemma-4-e4b-it-8bit` leads the think incumbent by 1.0 and is faster, so only
//! the 3.0 margin keeps think (`a_zero_margin_switches_think`).

use super::measure::replay_measure;
use super::rule::{Decision, Rule, decide};
use crate::ztools::eval::report::HistoryEntry;
use crate::ztools::eval::task_fingerprint::TaskIdentities;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Everything a replay reads, recordable as one fixture.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReplayInput {
    /// `eval_history.json`: model to rows.
    pub history: BTreeMap<String, Vec<HistoryEntry>>,
    /// The roster's task fingerprints now.
    pub current: TaskIdentities,
    /// Installed models by short name (`foundation` included when served).
    pub installed: Vec<String>,
    /// Today's `[best_models]`: slot to short name.
    pub best_models: BTreeMap<String, String>,
}

/// What the replay derived.
#[derive(Debug, Clone)]
pub struct Replay {
    pub decisions: Vec<Decision>,
    /// Slots whose incumbent has no rows to measure.
    pub unmeasured: Vec<String>,
}

impl Replay {
    /// Slot to the model the rule leaves in it.
    #[must_use]
    pub fn derived(&self) -> BTreeMap<String, String> {
        self.decisions
            .iter()
            .map(|d| (d.slot.clone(), d.chosen.clone()))
            .collect()
    }

    /// The derivation equals `best` and every slot was measured.
    #[must_use]
    pub fn reproduces(&self, best: &BTreeMap<String, String>) -> bool {
        self.unmeasured.is_empty() && &self.derived() == best
    }

    /// One line a slot: kept or switched, with the closest candidate's margin.
    #[must_use]
    pub fn lines(&self) -> Vec<String> {
        let mut out = Vec::new();
        for d in &self.decisions {
            let verb = if d.switched() { "->" } else { "kept" };
            let head = if d.switched() {
                format!("{}: {} {verb} {}", d.slot, d.incumbent, d.chosen)
            } else {
                format!("{}: {verb} {}", d.slot, d.incumbent)
            };
            out.push(head);
            for j in &d.judged {
                out.push(format!("  {} {:.1}: {}", j.model, j.score, j.why));
            }
        }
        for s in &self.unmeasured {
            out.push(format!("{s}: ✗ the incumbent has no rows to replay"));
        }
        out
    }
}

/// Replay the rule over `input`.
#[must_use]
pub fn replay(input: &ReplayInput, rule: &Rule) -> Replay {
    let mut decisions = Vec::new();
    let mut unmeasured = Vec::new();
    let rows = |m: &str| input.history.get(m).map_or(&[][..], Vec::as_slice);
    for (slot, incumbent) in &input.best_models {
        let Some(inc) = replay_measure(incumbent, rows(incumbent), &input.current, slot) else {
            unmeasured.push(slot.clone());
            continue;
        };
        let candidates: Vec<_> = input
            .installed
            .iter()
            .filter(|m| *m != incumbent)
            .filter_map(|m| replay_measure(m, rows(m), &input.current, slot))
            .collect();
        decisions.push(decide(slot, &inc, &candidates, rule));
    }
    Replay {
        decisions,
        unmeasured,
    }
}
