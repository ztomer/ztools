//! The summarizer's fallback chain and the provenance it records.
//!
//! Port of the loop half of `references/twitter/summarize.py::summarize_with_llm`
//! (`call_with_fallback` over the model chain) and of `twitter/provenance.py`
//! (class C9): the chain worked, but the artifact recorded nothing about
//! which tier answered, so a summary from a weak fallback was byte-for-byte
//! indistinguishable from the primary model's. House rule (honest
//! placeholders): a degraded state must be visually distinct and say WHY.
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
    /// Why the run did not go to plan, in the order things went wrong.
    pub reasons: Vec<String>,
}

impl Provenance {
    #[must_use]
    pub fn degraded(&self) -> bool {
        self.tier != Tier::Primary
    }

    #[must_use]
    pub fn describe(&self) -> String {
        let tier = match self.tier {
            Tier::Primary => "primary",
            Tier::Fallback => "fallback",
        };
        format!("{} ({}, {tier})", self.model, self.backend)
    }

    /// The provenance block for the top of the report. A normal run gets one
    /// quiet line; a degraded run gets a block that cannot be mistaken for a
    /// normal one and states the reason.
    #[must_use]
    pub fn banner(&self) -> String {
        if !self.degraded() {
            return format!("**Model:** {}", self.describe());
        }
        let mut lines = vec![
            "> ⚠ **DEGRADED OUTPUT** — this summary was NOT produced by the primary model."
                .to_string(),
            ">".to_string(),
            format!("> **Backend:** {}", self.describe()),
        ];
        for reason in &self.reasons {
            lines.push(format!("> **Why:** {reason}"));
        }
        // Same idiom as above the Backend line: a `>` on its own ends the
        // paragraph inside the block quote, so the advice is its own paragraph
        // instead of a fourth continuation of `**Why:**`.
        lines.push(">".to_string());
        lines.push(
            "> Treat the content below as lower quality than a normal run: the \
             fallback models have smaller context windows and weaker summarisation."
                .to_string(),
        );
        lines.join("\n")
    }
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
/// itself failed; both non-answers become a recorded reason and the chain
/// moves on. The transport stays with the caller so this is testable without
/// a server.
///
/// # Errors
///
/// When the chain is empty or every model failed; the message carries every
/// reason so an unattended run's log says what was tried.
pub fn run_chain<T>(
    chain: &[String],
    intended: &str,
    mut attempt: impl FnMut(&str) -> Result<Option<T>>,
) -> Result<(T, Provenance)> {
    let mut reasons = Vec::new();
    for model in chain {
        match attempt(model) {
            Ok(Some(answer)) => {
                let mut prov_reasons = reasons.clone();
                if model != intended {
                    prov_reasons.push(format!(
                        "answered by {model} instead of the intended {intended}"
                    ));
                }
                let tier = if prov_reasons.is_empty() {
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
                        reasons: prov_reasons,
                    },
                ));
            }
            Ok(None) => reasons.push(format!("model {model} returned no usable summary")),
            Err(e) => reasons.push(format!("model {model} failed: {e}")),
        }
    }
    if chain.is_empty() {
        anyhow::bail!("no model to try: the fallback chain is empty");
    }
    anyhow::bail!(
        "every model in the chain failed ({}): {}",
        chain.join(" → "),
        reasons.join("; ")
    )
}

#[cfg(test)]
#[path = "chain_tests.rs"]
mod tests;
