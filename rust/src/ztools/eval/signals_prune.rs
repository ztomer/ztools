//! Prune superseded and unrecorded task series from eval signals store (`M7`).
//!
//! When tasks evolve (e.g. prompt adjustments, new requirements), old observations
//! are superseded by new fingerprints. This module removes records whose fingerprints
//! no longer match current active task definitions, ensuring the store only retains
//! active observations. A task the roster no longer has is never current, so its
//! series always goes, and so does a model entry the prune leaves empty (M21).
//!
//! Dropping a superseded series of a LIVE task also drops its learned timeout,
//! which `effective_timeout` reads across fingerprints. That is what the next
//! observation would do anyway (`record_observation` restarts the series and
//! removes the timeout), and the derived estimate and the floor still bound the
//! first run, so the prune only does early what the store does itself.

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
    /// Model entries the prune left empty, and so removed (M21).
    pub models_removed: usize,
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

    // A model with no series and no capabilities left holds nothing any reader
    // uses; leaving `{}` behind is the dead record this prune exists to remove.
    let before = signals.len();
    signals.retain(|_, val| val.as_object().is_none_or(|obj| !obj.is_empty()));
    report.models_removed = before - signals.len();

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
            "Dry run: would prune {} task record(s) across {} model(s) ({} active records kept, {} emptied model(s) removed). Store not modified.",
            report.pruned_task_records,
            report.models_affected,
            report.kept_task_records,
            report.models_removed
        );
    } else if prune {
        if let Some(parent) = store_path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let formatted = serde_json::to_string_pretty(&store)?;
        std::fs::write(&store_path, formatted)?;
        println!(
            "✓ Pruned {} task record(s) across {} model(s) ({} active records kept, {} emptied model(s) removed). Saved to {}.",
            report.pruned_task_records,
            report.models_affected,
            report.kept_task_records,
            report.models_removed,
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

    /// M21: a series under a name the roster no longer has is dead whatever it
    /// carries. W5 retired `weekend_fixed`, `weekend_fixed_mixed` and
    /// `detailed_json`; their series held fingerprints that WERE current once,
    /// and no reader can ever average or time against them again.
    #[test]
    fn a_series_for_a_retired_task_is_pruned_even_with_a_fingerprint() {
        let mut store: SignalStore = serde_json::from_str(
            r#"{
                "model_a": {
                    "weekend_transient": { "fingerprint": "fp_now", "samples": 1 },
                    "weekend_fixed": { "fingerprint": "fp_was", "timeout": 10277 },
                    "weekend_fixed_mixed": { "fingerprint": "fp_was2", "samples": 4 },
                    "detailed_json": { "samples": 9 }
                }
            }"#,
        )
        .unwrap();
        let current: TaskIdentities =
            BTreeMap::from([("weekend_transient".to_string(), "fp_now".to_string())]);
        let report = prune_signals(&mut store, &current);
        let left: Vec<&String> = store["model_a"].as_object().unwrap().keys().collect();
        assert_eq!(left, vec!["weekend_transient"]);
        assert_eq!(report.pruned_task_records, 3);
    }

    /// M21: a model the prune leaves with nothing is itself a dead record. Two
    /// of the twelve it emptied in the shipped store were test doubles (`m`,
    /// `mock-model`) from before the test sandbox existed.
    #[test]
    fn a_model_left_with_nothing_is_dropped_and_capabilities_alone_keep_one() {
        let mut store: SignalStore = serde_json::from_str(
            r#"{
                "gone": { "weekend_x": { "samples": 159 } },
                "measured": { "_capabilities": { "cold_start_seconds": 1.0 },
                              "weekend_x": { "samples": 1 } },
                "already_empty": {}
            }"#,
        )
        .unwrap();
        let report = prune_signals(&mut store, &TaskIdentities::new());
        let left: Vec<&String> = store.keys().collect();
        assert_eq!(
            left,
            vec!["measured"],
            "capabilities are kept, empties dropped"
        );
        assert_eq!(report.models_removed, 2);
        assert_eq!(report.pruned_task_records, 2);
    }

    /// M21, the shipped store: `conf/eval_signals.json` holds no series for a
    /// task the roster no longer has. Retiring a task leaves its series behind
    /// until `ztools eval-signals --prune` runs; this makes that step part of
    /// the retirement rather than something to remember.
    #[test]
    fn the_shipped_store_holds_no_series_for_a_retired_task() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .expect("rust/ has a parent")
            .to_path_buf();
        let roster = crate::ztools::eval::load_all_eval_tasks(
            &crate::ztools::eval::tasks::RosterInputs::in_dir(&root.join("conf")),
            Some(&root.join("eval_tasks/data")),
        )
        .expect("the shipped roster loads");
        let names: std::collections::BTreeSet<&str> =
            roster.iter().map(|t| t.name.as_str()).collect();
        assert!(names.contains("taxes_qa"), "control: the taxes rows loaded");
        let text = std::fs::read_to_string(root.join("conf/eval_signals.json")).unwrap();
        let store: SignalStore = serde_json::from_str(&text).unwrap();
        let retired: Vec<String> = store
            .iter()
            .flat_map(|(model, series)| {
                series
                    .as_object()
                    .into_iter()
                    .flat_map(|o| o.keys())
                    .filter(|t| *t != "_capabilities" && !names.contains(t.as_str()))
                    .map(move |t| format!("{model}: {t}"))
            })
            .collect();
        assert!(
            retired.is_empty(),
            "series for tasks the roster no longer has -- run `ztools eval-signals \
             --prune`: {retired:#?}"
        );
    }
}
