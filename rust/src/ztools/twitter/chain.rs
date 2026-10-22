//! The summarizer's fallback chain and the provenance it records.
//!
//! Port of the loop half of `references/twitter/summarize.py::summarize_with_llm`
//! (`call_with_fallback` over the model chain) and of `twitter/provenance.py`
//! (class C9): the chain worked, but the artifact recorded nothing about
//! which tier answered, so a summary from a weak fallback was byte-for-byte
//! indistinguishable from the primary model's. House rule (honest
//! placeholders): a degraded state must be visually distinct and say WHY.
//!
//! The degraded state says it in ONE line (`Provenance::banner`): which model
//! failed and why, which model answered instead, what was repaired, and how
//! far to trust the result. The full text of every failure is kept beside it
//! -- in the run record and in the refused answer's file -- so the short line
//! loses no fact.
//!
//! The retired tiers are not here: a server restart is ops-side
//! (`tools/osaurus_one.sh`) and the direct-MLX last resort is replaced by
//! macOS 27's `fm` (`ztools::on_device`). What remains is the server chain: the intended model, then
//! the configured fallbacks, first usable answer wins -- except the on-device
//! model, which `[fallback] on_device` routes around the server (T1).
//!
//! Policy is DATA: `conf/twitter.toml [fallback]` names the extra models and
//! the preference order; `TWITTER_FALLBACK_MODELS` keeps its Python-era
//! override for one run.

use std::path::Path;

use anyhow::Result;

use crate::ztools::on_device::ON_DEVICE_MODEL;

use super::fallback::{fallback_chain, resolve_target_model};
use super::quality::Refusal;

/// Which models to fall back to and how to pick a substitute when the
/// intended model is not served. Read from `conf/twitter.toml [fallback]`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FallbackPolicy {
    /// Extra models to try after the intended one, in order.
    pub models: Vec<String>,
    /// Preference substrings used when the intended model is absent from the
    /// server's roster.
    pub preferred: Vec<String>,
    /// The `fm` binary that answers [`ON_DEVICE_MODEL`] without the server
    /// (`[fallback] on_device`, T1). Unset, every model goes to the server.
    pub on_device: Option<String>,
}

impl FallbackPolicy {
    /// The `fm` binary `model` is answered by, when it is the on-device model
    /// and one is configured; `None` means the server answers it.
    #[must_use]
    pub fn on_device_for(&self, model: &str) -> Option<&Path> {
        (model == ON_DEVICE_MODEL)
            .then_some(self.on_device.as_deref())
            .flatten()
            .map(Path::new)
    }

    /// Where `model` is answered, as the provenance names it.
    #[must_use]
    pub fn backend_of(&self, model: &str) -> &'static str {
        if self.on_device_for(model).is_some() {
            "on-device, fm"
        } else {
            "osaurus"
        }
    }
}

/// Environment override for the extra fallback models (comma-separated),
/// kept from the Python summarizer so an operator can steer one run.
pub const FALLBACK_MODELS_ENV: &str = "TWITTER_FALLBACK_MODELS";

/// Load the policy from the first candidate file carrying a `[fallback]`
/// table. `TWITTER_FALLBACK_MODELS`, when set and non-empty, replaces the
/// `models` list for this process.
///
/// # Errors
///
/// When no candidate yields the table: an absent policy would silently turn
/// the chain into a single shot, which is the failure this module exists to
/// make visible.
pub fn load_fallback_policy(paths: &[String]) -> Result<FallbackPolicy> {
    let mut policy = None;
    for raw in paths {
        let path = crate::manifest::expand_tilde(raw);
        let Ok(content) = std::fs::read_to_string(&path) else {
            continue;
        };
        let Ok(val) = toml::from_str::<toml::Value>(&content) else {
            continue;
        };
        let Some(table) = val.get("fallback") else {
            continue;
        };
        policy = Some(FallbackPolicy {
            models: string_list(table.get("models")),
            preferred: string_list(table.get("preferred")),
            on_device: table
                .get("on_device")
                .and_then(toml::Value::as_str)
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .map(str::to_string),
        });
        break;
    }
    let Some(mut policy) = policy else {
        anyhow::bail!("no twitter.toml [fallback] table found");
    };
    if let Ok(raw) = std::env::var(FALLBACK_MODELS_ENV) {
        let names: Vec<String> = raw
            .split(',')
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(str::to_string)
            .collect();
        if !names.is_empty() {
            policy.models = names;
        }
    }
    Ok(policy)
}

fn string_list(value: Option<&toml::Value>) -> Vec<String> {
    value
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_str())
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .map(str::to_string)
                .collect()
        })
        .unwrap_or_default()
}

/// Which tier answered.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tier {
    /// The intended model answered first time.
    Primary,
    /// A different model answered, or the intended one needed a reason.
    Fallback,
}

/// One model that was tried and did not answer with something usable.
///
/// Both halves are written where the failure happens, never derived from one
/// another: `verdict` is what the banner says, `detail` is what the run
/// record and the refused answer's file keep.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Failure {
    /// The model that was tried.
    pub model: String,
    /// What went wrong, in a few words: "a repetition loop".
    pub verdict: String,
    /// What went wrong, in full: "3 of 7 bullets repeat an earlier bullet".
    pub detail: String,
}

/// How a summary was produced.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Provenance {
    /// The model whose answer was kept.
    pub model: String,
    /// The model the run set out to use.
    pub intended: String,
    pub tier: Tier,
    /// Where the kept answer came from: `osaurus` unless the caller says
    /// otherwise (`FallbackPolicy::backend_of`).
    pub backend: &'static str,
    /// Every model that was tried and did not answer, in the order they were
    /// tried.
    pub failures: Vec<Failure>,
}

/// The advice the degraded line carries: how much to trust what follows.
const ADVICE: &str = "treat this summary as lower quality than a normal run";

impl Provenance {
    #[must_use]
    pub fn degraded(&self) -> bool {
        self.tier != Tier::Primary
    }

    #[must_use]
    pub fn describe(&self) -> String {
        format!("{} ({}, {})", self.model, self.backend, self.tier_name())
    }

    /// Which tier, in the word the banner uses.
    #[must_use]
    pub const fn tier_name(&self) -> &'static str {
        match self.tier {
            Tier::Primary => "primary",
            Tier::Fallback => "fallback",
        }
    }

    /// Every failure, in full: "model raptor: the answer was refused: ...".
    ///
    /// The banner carries the verdicts; this is the run record, and the half
    /// that says exactly what the gate saw.
    #[must_use]
    pub fn reasons(&self) -> Vec<String> {
        self.failures
            .iter()
            .map(|f| format!("model {}: {}", f.model, f.detail))
            .collect()
    }

    /// The line the summary document opens with.
    ///
    /// A normal run gets one quiet line; a degraded run gets ONE line that
    /// cannot be mistaken for a normal one and says everything: which model
    /// failed and why, which model answered instead, what was repaired, and
    /// how far to trust the result. It is one line because a reader who
    /// scrolls past eight lines of banner reads none of them (owner,
    /// 2026-10-21).
    #[must_use]
    pub fn banner(&self, repairs: &[String]) -> String {
        if !self.degraded() {
            return self.core(repairs);
        }
        format!("> ⚠ **DEGRADED** — {}", self.core(repairs))
    }

    /// The same line for the terminal, without the document's markdown.
    ///
    /// The terminal and the document say the same thing because they are
    /// written by the same code: one writer, one line, no drift.
    #[must_use]
    pub fn notice(&self, repairs: &[String]) -> String {
        if !self.degraded() {
            return self.core(repairs);
        }
        format!("Degraded summary: {}", self.core(repairs))
    }

    /// The line's content, without the markdown and without the terminal's
    /// label: what happened, in one sentence.
    fn core(&self, repairs: &[String]) -> String {
        if !self.degraded() {
            let mut line = format!("**Model:** {}", self.describe());
            if !repairs.is_empty() {
                line.push_str("; ");
                line.push_str(&repairs_clause(repairs));
            }
            return line;
        }
        let mut clauses: Vec<String> = self
            .failures
            .iter()
            .map(|f| format!("{} failed: {}", f.model, f.verdict))
            .collect();
        clauses.push(self.winner_clause());
        if !repairs.is_empty() {
            clauses.push(repairs_clause(repairs));
        }
        clauses.push(ADVICE.to_string());
        clauses.join("; ")
    }

    /// Who answered, and instead of what.
    ///
    /// The intended model is named here only when it is not named already, by
    /// the failure that stopped it: a line that says the same model twice is
    /// the length the owner asked to have taken out.
    fn winner_clause(&self) -> String {
        let answered = format!(
            "{} ({}, {}) answered instead",
            self.model,
            self.backend,
            self.tier_name()
        );
        if self.failures.iter().any(|f| f.model == self.intended) {
            answered
        } else {
            format!("{answered} of the intended {}", self.intended)
        }
    }
}

/// What the repair changed, said in the line the reader starts at.
fn repairs_clause(repairs: &[String]) -> String {
    format!("repaired before saving: {}", repairs.join("; "))
}

/// Build the ordered chain for this run: the intended model resolved against
/// the server's roster (empty roster = unknown, keep the intent), then the
/// policy's extras.
#[must_use]
pub fn plan_chain(intended: &str, available: &[String], policy: &FallbackPolicy) -> Vec<String> {
    let target = resolve_target_model(intended, available, &policy.preferred);
    fallback_chain(&target, available, &policy.models)
}

/// Run `attempt` down the chain until one model yields a usable answer.
///
/// `attempt` returns `Ok(Some(answer))` for a usable answer, `Ok(None)` when
/// the model answered but produced nothing usable, and `Err` when the call
/// itself failed or the answer was refused; both non-answers become a
/// recorded failure and the chain moves on. The transport stays with the
/// caller so this is testable without a server.
///
/// # Errors
///
/// When the chain is empty or every model failed; the message carries every
/// reason so an unattended run's log says what was tried.
pub fn run_chain<T>(
    chain: &[String],
    intended: &str,
    mut attempt: impl FnMut(&str) -> Result<Option<T>, Refusal>,
) -> Result<(T, Provenance)> {
    let mut failures: Vec<Failure> = Vec::new();
    for model in chain {
        match attempt(model) {
            Ok(Some(answer)) => {
                let tier = if failures.is_empty() && model == intended {
                    Tier::Primary
                } else {
                    Tier::Fallback
                };
                return Ok((
                    answer,
                    Provenance {
                        model: model.clone(),
                        intended: intended.to_string(),
                        tier,
                        backend: "osaurus",
                        failures,
                    },
                ));
            }
            Ok(None) => failures.push(Failure {
                model: model.clone(),
                verdict: "no usable summary".to_string(),
                detail: "returned no usable summary".to_string(),
            }),
            Err(refusal) => failures.push(Failure {
                model: model.clone(),
                verdict: refusal.verdict,
                detail: refusal.detail,
            }),
        }
    }
    if chain.is_empty() {
        anyhow::bail!("no model to try: the fallback chain is empty");
    }
    let reasons: Vec<String> = failures
        .iter()
        .map(|f| format!("model {}: {}", f.model, f.detail))
        .collect();
    anyhow::bail!(
        "every model in the chain failed ({}): {}",
        chain.join(" → "),
        reasons.join("; ")
    )
}

#[cfg(test)]
#[path = "chain_tests.rs"]
mod tests;
