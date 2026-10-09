//! Prune superseded and unrecorded task series from eval signals store (`M7`).
//!
//! When tasks evolve (e.g. prompt adjustments, new requirements), old observations
//! are superseded by new fingerprints. This module removes records whose fingerprints
//! no longer match current active task definitions, ensuring the store only retains
//! active observations.

use anyhow::Result;
use serde_json::Value;
use std::path::Path;

use crate::config::ZtoolsConfig;
use crate::ztools::eval::signals::SignalStore;
use crate::ztools::eval::task_fingerprint::{Standing, TaskIdentities, standing_of};

/// Summary metrics from a signals pruning pass.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct PruneReport {
    pub total_models: usize,
    pub models_affected: usize,
    pub kept_task_records: usize,
    pub pruned_task_records: usize,
}

/// Prunes task records whose fingerprints do not match current active task identities.
///
/// Hardware benchmark capabilities (`_capabilities`) are model-level and never pruned.
#[must_use]
pub fn prune_signals(signals: &mut SignalStore, current: &TaskIdentities) -> PruneReport {
    let mut report = PruneReport {
        total_models: signals.len(),
        ..Default::default()
    };

    for val in signals.values_mut() {
        let Some(obj) = val.as_object_mut() else {
            continue;
        };
        let mut to_remove = Vec::new();

        for (task_name, task_val) in obj.iter() {
            if task_name == "_capabilities" {
                continue;
            }
            let fp = task_val.get("fingerprint").and_then(Value::as_str);
            let standing = standing_of(fp, current, task_name);
            if standing == Standing::Current {
                report.kept_task_records += 1;
            } else {
                to_remove.push(task_name.clone());
            }
        }

        if !to_remove.is_empty() {
            report.models_affected += 1;
            for task_name in to_remove {
                obj.remove(&task_name);
                report.pruned_task_records += 1;
            }
        }
    }

    report
}

/// CLI entrypoint for `ztools eval-signals`.
///
/// # Errors
///
/// Fails if the signals store or task roster cannot be read or parsed.
pub fn cli_eval_signals(
    config: &ZtoolsConfig,
    explicit_path: Option<&Path>,
    tasks_dir: Option<&Path>,
    prune: bool,
    dry_run: bool,
) -> Result<()> {
    let store_path =
        explicit_path.map_or_else(crate::ztools::eval::signals_path, Path::to_path_buf);
    if !store_path.exists() {
        anyhow::bail!("eval signals store not found at {}", store_path.display());
    }

    let default_tasks_dir = config.eval_tasks_dir();
    let search_tasks_dir = tasks_dir.or(default_tasks_dir.as_deref());
    let roster_inputs = config.eval_roster_inputs()?;
    let tasks = crate::ztools::eval::load_all_eval_tasks(&roster_inputs, search_tasks_dir)?;
    crate::ztools::eval::remember_current_tasks(&tasks);
    let current = crate::ztools::eval::current_task_identities();

    let text = std::fs::read_to_string(&store_path)
        .map_err(|e| anyhow::anyhow!("cannot read {}: {e}", store_path.display()))?;
    let mut store: SignalStore = serde_json::from_str(&text)
        .map_err(|e| anyhow::anyhow!("cannot parse {}: {e}", store_path.display()))?;

    let report = prune_signals(&mut store, &current);

    if dry_run {
        println!(
            "Dry run: would prune {} task record(s) across {} model(s) ({} active records kept). Store not modified.",
            report.pruned_task_records, report.models_affected, report.kept_task_records
        );
    } else if prune {
        if let Some(parent) = store_path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let formatted = serde_json::to_string_pretty(&store)?;
        std::fs::write(&store_path, formatted)?;
        println!(
            "✓ Pruned {} task record(s) across {} model(s) ({} active records kept). Saved to {}.",
            report.pruned_task_records,
            report.models_affected,
            report.kept_task_records,
            store_path.display()
        );
    } else {
        println!("Eval signals store: {}", store_path.display());
        println!("Models: {}", report.total_models);
        println!("Active task records: {}", report.kept_task_records);
        println!(
            "Superseded/unrecorded task records: {}",
            report.pruned_task_records
        );
        println!(
            "Pass --prune to strip records whose fingerprints do not match current task definitions."
        );
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    #[test]
    fn prune_signals_strips_stale_fingerprints_and_retains_active_ones() {
        let mut store: SignalStore = serde_json::from_str(
            r#"{
                "model_a": {
                    "_capabilities": { "cold_start_seconds": 1.2 },
                    "task_cur": { "fingerprint": "fp_cur_1", "p95_latency": 5.0 },
                    "task_stale": { "fingerprint": "fp_old", "p95_latency": 10.0 },
                    "task_legacy": { "p95_latency": 15.0 },
                    "task_missing": { "fingerprint": "fp_ghost", "p95_latency": 20.0 }
                },
                "model_b": {
                    "task_cur": { "fingerprint": "fp_cur_1", "p95_latency": 6.0 }
                }
            }"#,
        )
        .unwrap();

        let mut current: TaskIdentities = BTreeMap::new();
        current.insert("task_cur".to_string(), "fp_cur_1".to_string());
        current.insert("task_stale".to_string(), "fp_fresh".to_string());
        current.insert("task_legacy".to_string(), "fp_new".to_string());

        let report = prune_signals(&mut store, &current);

        assert_eq!(report.total_models, 2);
        assert_eq!(report.models_affected, 1);
        assert_eq!(report.pruned_task_records, 3);
        assert_eq!(report.kept_task_records, 2);

        let model_a = store.get("model_a").unwrap().as_object().unwrap();
        assert!(
            model_a.contains_key("_capabilities"),
            "_capabilities preserved"
        );
        assert!(model_a.contains_key("task_cur"), "active task preserved");
        assert!(
            !model_a.contains_key("task_stale"),
            "stale fingerprint stripped"
        );
        assert!(
            !model_a.contains_key("task_legacy"),
            "unfingerprinted record stripped"
        );
        assert!(
            !model_a.contains_key("task_missing"),
            "unknown task stripped"
        );

        let model_b = store.get("model_b").unwrap().as_object().unwrap();
        assert!(
            model_b.contains_key("task_cur"),
            "active task on model_b preserved"
        );
    }

    #[test]
    fn prune_signals_on_clean_store_makes_no_changes() {
        let mut store: SignalStore = serde_json::from_str(
            r#"{
                "model_a": {
                    "task_cur": { "fingerprint": "fp_cur_1", "p95_latency": 5.0 }
                }
            }"#,
        )
        .unwrap();

        let mut current: TaskIdentities = BTreeMap::new();
        current.insert("task_cur".to_string(), "fp_cur_1".to_string());

        let report = prune_signals(&mut store, &current);
        assert_eq!(report.pruned_task_records, 0);
        assert_eq!(report.kept_task_records, 1);
        assert_eq!(report.models_affected, 0);
    }
}
