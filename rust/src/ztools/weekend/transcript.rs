//! What every model call of one weekend run was asked, and what it answered.
//!
//! On 2026-10-10 a Thanksgiving plan came back with ONE transient event from a
//! corpus that named a harvest market, a farmers' market, a family festival and
//! a pumpkin trail inside the window. The plan's ledger said "1 extracted", and
//! nothing the run kept could say which phase lost the rest: the corpus was on
//! disk, the answers were not. They were recovered only because the model
//! server happened to log request bodies -- each phase's PROMPT carries the
//! previous phase's answer -- and that showed the extractor answering eight
//! results naming four real events with "I cannot extract ... they are all
//! pages that list things". The run now keeps that record itself.
//!
//! One entry per call, in call order, with the prompt verbatim and the raw
//! answer verbatim (or that there was none), so a thin plan is diagnosed by
//! reading, not by replaying the model. Saved beside the corpus by
//! [`crate::ztools::store_corpus::record_phases`] under the same run stamp.

use std::fmt::Write as _;
use std::sync::Mutex;

/// One model call: which phase made it, the prompt it sent, the raw answer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PhaseEntry {
    pub phase: String,
    pub prompt: String,
    /// `None` when the call produced no answer (transport failure, stall,
    /// empty content) after its retries.
    pub answer: Option<String>,
}

/// The calls of one run, appended as they happen.
///
/// A `Mutex` rather than a `RefCell` only so a phase can be handed `&PhaseLog`
/// from any thread; the phases themselves run serially.
#[derive(Debug, Default)]
pub struct PhaseLog {
    entries: Mutex<Vec<PhaseEntry>>,
}

impl PhaseLog {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Append one call. A poisoned lock (a panic mid-append elsewhere) still
    /// records: losing the diagnosis of a broken run would be the wrong way
    /// round.
    pub fn record(&self, phase: &str, prompt: &str, answer: Option<&str>) {
        let entry = PhaseEntry {
            phase: phase.to_string(),
            prompt: prompt.to_string(),
            answer: answer.map(ToString::to_string),
        };
        match self.entries.lock() {
            Ok(mut entries) => entries.push(entry),
            Err(poisoned) => poisoned.into_inner().push(entry),
        }
    }

    /// Every call so far, in call order.
    #[must_use]
    pub fn entries(&self) -> Vec<PhaseEntry> {
        match self.entries.lock() {
            Ok(entries) => entries.clone(),
            Err(poisoned) => poisoned.into_inner().clone(),
        }
    }

    /// The record as text: per call, a `=== phase` line naming the prompt's
    /// size, the prompt, then `--- answer` with the answer's size and the
    /// answer, or `--- no answer`.
    #[must_use]
    pub fn render(&self) -> String {
        let mut out = String::new();
        for e in self.entries() {
            let _ = writeln!(
                out,
                "=== {}: prompt {} chars\n{}",
                e.phase,
                e.prompt.chars().count(),
                e.prompt
            );
            match &e.answer {
                Some(a) => {
                    let _ = writeln!(out, "--- answer {} chars\n{a}", a.chars().count());
                }
                None => out.push_str("--- no answer\n"),
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The record keeps both halves of every call verbatim and in order, and a
    /// call with no answer says so rather than vanishing -- a missing answer is
    /// the most informative line in a thin plan's transcript.
    #[test]
    fn every_call_is_kept_in_order_with_its_prompt_and_raw_answer() {
        let log = PhaseLog::new();
        log.record(
            "extract 1-8 of 200",
            "Search results:\n- a",
            Some("I cannot extract"),
        );
        log.record("draft", "Available events:\n", None);
        let text = log.render();
        assert_eq!(
            text,
            "=== extract 1-8 of 200: prompt 19 chars\nSearch results:\n- a\n\
             --- answer 16 chars\nI cannot extract\n\
             === draft: prompt 18 chars\nAvailable events:\n\n--- no answer\n"
        );
        assert_eq!(log.entries().len(), 2);
        assert_eq!(log.entries()[1].answer, None);
    }
}
