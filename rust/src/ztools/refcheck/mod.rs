//! `ztools models refcheck`: no setting may name a model that is not installed.
//!
//! The fault it closes: the Osaurus agent "Disk Audit" pinned
//! `mlx-community/Qwen3.8-27B-8bit`, which was never installed, and Osaurus
//! accepted it silently. Every model reference in the Osaurus export, and the
//! ztools `[best_models]` table, must resolve to an installed model, `null`,
//! `foundation`, or a model of an enabled remote provider. A miss is an ERROR
//! line naming the exact path. With repair on (the default) an Osaurus
//! reference is repointed to the best installed member of its family through a
//! minimal one-section document: `plan` (whose diff must touch only that
//! reference), `apply`, then a re-check. Never `--prune`, never the full export.
//!
//! Everything impure sits behind [`Env`]: the osaurus CLI, the filesystem, the
//! families cache and the eval history, so the tests run on fixtures.

mod live;
mod plan;
mod refs;
mod repair;

#[cfg(test)]
mod tests;
#[cfg(test)]
mod tests_automation;
#[cfg(test)]
mod tests_live;

pub use live::LiveEnv;
pub use refs::{Installed, Reference, Slot, collect, short_name};
pub use repair::{choose, family_key, is_cracked, stem};

use anyhow::Result;
use serde_json::Value;
use std::collections::BTreeMap;

/// What an `osaurus config plan` or `apply` printed, and how it exited.
#[derive(Debug, Clone)]
pub struct CmdOut {
    pub code: i32,
    pub text: String,
}

/// The seam: everything refcheck reads or does outside its own memory.
pub trait Env {
    /// `osaurus config export` as JSON.
    ///
    /// # Errors
    ///
    /// When osaurus cannot be run or its export is not JSON.
    fn export(&self) -> Result<Value>;
    /// The names `osaurus list` prints.
    ///
    /// # Errors
    ///
    /// When osaurus cannot be run.
    fn osaurus_list(&self) -> Result<Vec<String>>;
    /// `org/name` of every `~/MLXModels/<org>/<name>/config.json`.
    fn disk_models(&self) -> Vec<String>;
    /// The ztools `[best_models]` table: slot to model name.
    fn best_models(&self) -> BTreeMap<String, String>;
    /// The cached Hugging Face base model of `repo`, if known.
    fn base_model(&self, repo: &str) -> Option<String>;
    /// Think-slot scores by lowercased short model name, and which of them
    /// fell back to rows that predate the current task fingerprints.
    fn think_scores(&self) -> ThinkScores;
    /// `osaurus config plan <doc>` (a dry run).
    ///
    /// # Errors
    ///
    /// When osaurus cannot be run.
    fn plan(&self, doc: &Value) -> Result<CmdOut>;
    /// `osaurus config apply <doc>`. Never prunes.
    ///
    /// # Errors
    ///
    /// When osaurus cannot be run.
    fn apply(&self, doc: &Value) -> Result<CmdOut>;
}

/// The think-slot scores a repair ranks a family by, and where they came from.
#[derive(Debug, Clone, Default)]
pub struct ThinkScores {
    /// Score by lowercased short model name.
    pub scores: BTreeMap<String, f64>,
    /// Models whose score is from rows that predate the current task
    /// fingerprints, because they have no current think rows.
    pub stale: Vec<String>,
}

impl ThinkScores {
    /// The line a report carries when the ranking is not on current rows:
    /// `None` when every score was taken on the current tasks.
    #[must_use]
    pub fn fallback_line(&self) -> Option<String> {
        if self.scores.is_empty() {
            return Some(
                "  · think ranking: no think scores on record, so the family was ranked by \
                 cracked build, then by name"
                    .into(),
            );
        }
        if self.stale.is_empty() {
            return None;
        }
        Some(format!(
            "  · think ranking: {} scored from runs that predate the current task \
             fingerprints (`ztools model-eval --leaderboard` has no current think rows for them)",
            self.stale.join(", ")
        ))
    }
}

/// The result of one refcheck.
#[derive(Debug, Default)]
pub struct Report {
    pub lines: Vec<String>,
    /// References found missing on the first look.
    pub errors: usize,
    /// References repointed and confirmed by the re-check.
    pub repaired: usize,
    /// References still missing after the run.
    pub unresolved: usize,
}

impl Report {
    /// 0 clean; 3 errors found and all repaired (so the routines failure line
    /// shows it once); 1 something is still wrong.
    #[must_use]
    pub const fn exit_code(&self) -> u8 {
        if self.unresolved > 0 {
            1
        } else if self.errors > 0 {
            3
        } else {
            0
        }
    }
}

fn look(env: &dyn Env) -> Result<(Value, Vec<Reference>, Installed)> {
    let export = env.export()?;
    let installed = Installed::gather(&export, &env.osaurus_list()?, &env.disk_models());
    let missing = collect(&export, &env.best_models())
        .into_iter()
        .filter(|r| !installed.is_valid(r))
        .collect();
    Ok((export, missing, installed))
}

/// The last resort for a reference with no installed family member: the
/// installed `core_model`. A missing `core_model` has nowhere to fall back to.
fn core_fallback(export: &Value, installed: &Installed, r: &Reference) -> Result<String, String> {
    let none = || "no installed member of the family and no core_model to fall back to".to_string();
    if r.slot == Slot::CoreModel {
        return Err(none());
    }
    let core = export
        .pointer("/model_defaults/core_model")
        .and_then(Value::as_str)
        .ok_or_else(none)?;
    let as_ref = Reference {
        path: String::new(),
        slot: Slot::CoreModel,
        model: core.to_string(),
    };
    if installed.is_valid(&as_ref) {
        Ok(core.to_string())
    } else {
        Err(none())
    }
}

/// Repoint one Osaurus reference. `Ok` is the model it now names and whether
/// that was the last-resort fallback; `Err` is why nothing was applied.
fn repair_one(
    env: &dyn Env,
    export: &Value,
    installed: &Installed,
    r: &Reference,
    scores: &BTreeMap<String, f64>,
) -> Result<(String, bool), String> {
    let edit = plan::edit_for(&r.slot)
        .ok_or_else(|| "[best_models] is a ztools setting, not an Osaurus one".to_string())?;
    let (target, fallback) = if let Some(t) = choose(
        &r.model,
        &installed.local_ids(),
        &|id| env.base_model(id),
        scores,
    ) {
        (t, false)
    } else {
        (core_fallback(export, installed, r)?, true)
    };
    let doc = plan::minimal_doc(export, &r.slot, &target)?;
    let planned = env.plan(&doc).map_err(|e| format!("plan failed: {e:#}"))?;
    plan::verify_plan(&edit, &target, &planned)?;
    let applied = env
        .apply(&doc)
        .map_err(|e| format!("apply failed: {e:#}"))?;
    if applied.code != 0 {
        return Err(format!("apply exited {}", applied.code));
    }
    Ok((target, fallback))
}

/// Check every reference; repair the Osaurus ones when `repair` is set.
///
/// # Errors
///
/// When osaurus cannot be run or its export cannot be read.
pub fn run(env: &dyn Env, repair: bool) -> Result<Report> {
    let (export, missing, installed) = look(env)?;
    let mut report = Report {
        errors: missing.len(),
        ..Report::default()
    };
    let mut repaired: Vec<(String, String, String)> = Vec::new();
    let think = env.think_scores();
    let mut ranked = false;
    for r in &missing {
        report
            .lines
            .push(format!("✗ ERROR {} = {}: not installed", r.path, r.model));
        if !repair {
            continue;
        }
        ranked |= plan::edit_for(&r.slot).is_some();
        match repair_one(env, &export, &installed, r, &think.scores) {
            Ok((to, false)) => repaired.push((r.path.clone(), r.model.clone(), to)),
            Ok((to, true)) => report.lines.push(format!(
                "  ⚠ {}: no installed member of the family; pointed at core_model {to}, \
                 which is not a real repair",
                r.path
            )),
            Err(why) => report
                .lines
                .push(format!("  ⚠ {} not repaired: {why}", r.path)),
        }
    }
    if ranked && let Some(line) = think.fallback_line() {
        report.lines.push(line);
    }
    if repair && !repaired.is_empty() {
        let (_, still, _) = look(env)?;
        for (path, from, to) in repaired {
            if still.iter().any(|r| r.path == path) {
                report.lines.push(format!(
                    "  ⚠ {path}: applied {to} but the re-check still reports {from}"
                ));
            } else {
                report.repaired += 1;
                report
                    .lines
                    .push(format!("  ↑ repaired {path}: {from} -> {to}"));
            }
        }
    }
    report.unresolved = report.errors - report.repaired;
    if report.errors == 0 {
        report
            .lines
            .push("✓ refcheck: every model reference is installed".into());
    }
    Ok(report)
}
