//! What the nightly does with the slots its rule chose to switch.

use super::switch::{Promotion, promote};
use super::{Mode, Paths};
use crate::ztools::select::Decision;
use std::collections::BTreeMap;

/// Promote every switched decision (in auto mode) or propose it, and the
/// findings fragment for each. A fault is said in its fragment, never lost.
#[must_use]
pub fn act(
    paths: &Paths,
    mode: &Mode,
    clock: (f64, &str),
    base: &BTreeMap<String, String>,
    decisions: &[Decision],
) -> Vec<(Option<Promotion>, String)> {
    decisions
        .iter()
        .filter(|d| d.switched())
        .map(
            |d| match promote(paths, mode, clock, base, &d.slot, &d.chosen) {
                Ok(p) => {
                    let line = p.line();
                    (Some(p), line)
                }
                Err(e) => (
                    None,
                    format!("{} -> {} ✗ switch failed: {e:#}", d.slot, d.chosen),
                ),
            },
        )
        .collect()
}

/// Whether any promotion switched a slot to `model`.
#[must_use]
pub fn switched_to(acts: &[(Option<Promotion>, String)], model: &str) -> bool {
    acts.iter()
        .any(|(p, _)| matches!(p, Some(Promotion::Switched { to, .. }) if to == model))
}
