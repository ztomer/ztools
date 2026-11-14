//! The renamer's runs as canary runs of the `filename` and `vlm` slots
//! (LM.CANARYFLOORS).
//!
//! Every model call is timed, cold load included, and
//! judged by the renamer's own gate (`is_generic_name` and its length floor
//! for a text name, `acceptable_name` for a vision one).

use crate::ztools::autoswitch::canary::{Run, Verdict};
use std::collections::BTreeMap;

/// One model call the renamer made.
#[derive(Debug, Clone, PartialEq)]
pub struct Ask {
    pub slot: &'static str,
    pub model: String,
    /// The model answered at all (it loaded, the call returned).
    pub answered: bool,
    /// The renamer's own gate took the answer.
    pub accepted: bool,
    /// Wall seconds, the hold (a cold load) included.
    pub seconds: f64,
}

/// Every call of one run.
#[derive(Debug, Clone, Default)]
pub struct Watch {
    asks: Vec<Ask>,
}

impl Watch {
    pub fn push(&mut self, ask: Ask) {
        if !ask.model.is_empty() {
            self.asks.push(ask);
        }
    }

    /// One canary run per slot and model asked. A model that did not answer
    /// once fails it; so does a run whose answers the gate refused more often
    /// than it took. The run's time is its slowest call, the cold load
    /// included, which is what the floor bounds.
    #[must_use]
    pub fn runs(&self) -> Vec<Run> {
        let mut by: BTreeMap<(&str, &str), Vec<&Ask>> = BTreeMap::new();
        for a in &self.asks {
            by.entry((a.slot, &a.model)).or_default().push(a);
        }
        by.into_iter()
            .map(|((slot, model), asks)| {
                let n = asks.len();
                let silent = asks.iter().filter(|a| !a.answered).count();
                let refused = asks.iter().filter(|a| a.answered && !a.accepted).count();
                let verdict = if silent > 0 {
                    Verdict::Failed(format!("the model did not answer {silent} of {n} calls"))
                } else if refused * 2 > n {
                    Verdict::Failed(format!("the renamer's gate refused {refused} of {n} names"))
                } else {
                    Verdict::Passed
                };
                Run {
                    slot: slot.to_string(),
                    model: model.to_string(),
                    verdict,
                    seconds: asks.iter().map(|a| a.seconds).fold(0.0, f64::max),
                }
            })
            .collect()
    }

    /// Count each run against an open canary, one line each on stderr.
    pub fn report(&self) {
        for r in self.runs() {
            eprintln!(
                "→ {} {}: slowest call {:.1}s, cold load included",
                r.slot, r.model, r.seconds
            );
            crate::ztools::autoswitch::canary::observe_live(
                &r.slot, &r.model, r.verdict, r.seconds,
            );
        }
    }
}

#[cfg(test)]
#[path = "watch_tests.rs"]
mod tests;
