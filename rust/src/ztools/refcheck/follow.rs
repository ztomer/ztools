//! Osaurus's `model_defaults.core_model` and `default_agent.model` follow the
//! ztools think slot (LM.DESIGN D2, section 6; LM.CORE).
//!
//! Each key moves through its own minimal one-section document. Before anything
//! is applied, for every key that changes: the rollback document (the key's
//! value in the export taken just before) must plan clean, and the forward
//! document's plan must be exactly that one key, or the run refuses and applies
//! nothing. After the apply, a fresh export must plan no diff for each section.
//! Never `--prune`, never the full export.

use super::Env;
use super::plan;
use super::refs::{Slot, short_name};
use anyhow::Result;
use serde_json::Value;

const SLOTS: [(Slot, &str); 2] = [
    (Slot::CoreModel, "/model_defaults/core_model"),
    (Slot::DefaultAgent, "/default_agent/model"),
];

/// What one follow run did.
#[derive(Debug, Default)]
pub struct FollowReport {
    pub lines: Vec<String>,
    /// The rollback documents, one per changed key, each planned clean.
    pub rollback: Vec<Value>,
    pub failed: bool,
}

impl FollowReport {
    /// 0 done (or nothing to do); 1 refused or failed.
    #[must_use]
    pub const fn exit_code(&self) -> u8 {
        if self.failed { 1 } else { 0 }
    }

    fn refuse(mut self, why: &str) -> Self {
        self.lines.push(format!("✗ follow-think refused: {why}"));
        self.failed = true;
        self
    }
}

/// Whether a plan says the document changes nothing.
fn is_clean(out: &super::CmdOut) -> bool {
    out.code == 0 && out.text.contains("No changes")
}

/// The installed Osaurus id whose short name is the think slot's model.
fn think_target(env: &dyn Env, export: &Value) -> Result<String, String> {
    let think = env
        .best_models()
        .get("think")
        .cloned()
        .ok_or("[best_models] has no think slot")?;
    let mut ids: Vec<String> = export
        .get("models")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(Value::as_str)
        .map(str::to_string)
        .chain(env.disk_models())
        .filter(|id| short_name(id) == think.to_lowercase())
        .collect();
    ids.sort();
    ids.dedup();
    match ids.as_slice() {
        [one] => Ok(one.clone()),
        [] => Err(format!("the think model {think} is not installed")),
        many => Err(format!(
            "the think model {think} names {} installed ids: {}",
            many.len(),
            many.join(", ")
        )),
    }
}

/// Point both keys at the think model; with `apply` false, only plan.
///
/// # Errors
///
/// When the export cannot be read or osaurus cannot be run.
pub fn follow_think(env: &dyn Env, apply: bool) -> Result<FollowReport> {
    let mut report = FollowReport::default();
    let export = env.export()?;
    let target = match think_target(env, &export) {
        Ok(t) => t,
        Err(why) => return Ok(report.refuse(&why)),
    };
    let mut forward: Vec<(String, Value)> = Vec::new();
    for (slot, pointer) in &SLOTS {
        let path = pointer.trim_start_matches('/').replace('/', ".");
        let Some(edit) = plan::edit_for(slot) else {
            continue;
        };
        let old = export.pointer(pointer).and_then(Value::as_str);
        if old == Some(target.as_str()) {
            report
                .lines
                .push(format!("✓ {path} already follows think: {target}"));
            continue;
        }
        let Some(old) = old else {
            return Ok(report.refuse(&format!(
                "{path} is unset, so no rollback document can restore it"
            )));
        };
        let back = plan::minimal_doc(&export, slot, old).map_err(anyhow::Error::msg)?;
        let planned = env.plan(&back)?;
        if !is_clean(&planned) {
            let head: String = planned.text.lines().take(3).collect::<Vec<_>>().join(" | ");
            return Ok(report.refuse(&format!(
                "the rollback for {path} does not plan clean (exit {}): {head}",
                planned.code
            )));
        }
        let doc = plan::minimal_doc(&export, slot, &target).map_err(anyhow::Error::msg)?;
        if let Err(why) = plan::verify_plan(&edit, &target, &env.plan(&doc)?) {
            return Ok(report.refuse(&why));
        }
        report
            .lines
            .push(format!("  → {path}: {old} -> {target} (planned)"));
        report.rollback.push(back);
        forward.push((path, doc));
    }
    if !apply || forward.is_empty() {
        return Ok(report);
    }
    for (path, doc) in &forward {
        let out = env.apply(doc)?;
        if out.code != 0 {
            report.failed = true;
            report
                .lines
                .push(format!("✗ apply of {path} exited {}", out.code));
            return Ok(report);
        }
    }
    let after = env.export()?;
    for ((slot, _), (path, _)) in SLOTS.iter().zip(&forward) {
        let again = plan::minimal_doc(&after, slot, &target).map_err(anyhow::Error::msg)?;
        if is_clean(&env.plan(&again)?) {
            report.lines.push(format!(
                "  ↑ {path} now {target}; the re-export plans clean"
            ));
        } else {
            report.failed = true;
            report.lines.push(format!(
                "✗ {path}: the re-export still plans a diff after apply"
            ));
        }
    }
    Ok(report)
}
