//! LM.SELECT: the per-slot decision rule, its replay, and the nightly.
//!
//! The replay runs over `eval_history.json`; the nightly stages, evaluates and
//! decides one candidate (LM.DESIGN section 4, owner-accepted D3/D4
//! 2026-11-11). PROPOSE-ONLY.

pub mod live;
pub mod measure;
pub mod nightly;
pub mod replay;
pub mod rule;
pub mod unloadable;

#[cfg(test)]
mod nightly_tests;
#[cfg(test)]
mod replay_tests;
#[cfg(test)]
mod rule_tests;
#[cfg(test)]
mod unloadable_tests;

pub use replay::{Replay, ReplayInput, replay};
pub use rule::{Decision, Measure, Rule, decide};
