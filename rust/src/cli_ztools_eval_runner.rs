//! Execution loop for full-suite model evaluations (`ztools model-eval --suite full`).
//!
//! Split out of `cli_ztools.rs` for the house 500-line cap and single-responsibility.

use anyhow::Result;
use std::path::Path;

use super::EvalOptions;
use super::eval_report::{note_unmeasured, print_outcomes, report_suite};
use crate::config::ZtoolsConfig;
use crate::units::unsigned;

/// One model's full-suite run, recorded and printed.
fn record_run(
    model_name: &str,
    expected_tasks: &[String],
    outcomes: &[crate::ztools::eval::TaskOutcome],
    json_output: bool,
) -> Result<crate::ztools::eval::ModelRun> {
    let run_record =
        crate::ztools::eval::ModelRun::new(model_name, expected_tasks, outcomes.to_vec());
    if let Some(c) = &run_record.completeness
        && !c.complete
    {
        eprintln!("⚠ {} (partial): {}", model_name, c.reason);
    }
    if let Err(e) = crate::ztools::eval::save_historical_results(&run_record, None) {
        eprintln!("⚠ could not write eval history: {e}");
    }
    print_outcomes(outcomes, json_output)?;
    Ok(run_record)
}

/// The full suite's tasks: the roster's inputs from `--tasks-dir` (or the
/// configured directory), narrowed by `--task`.
fn load_suite_tasks(
    config: &ZtoolsConfig,
    tasks_dir: Option<&Path>,
    task_filter: Option<&str>,
) -> Result<Vec<crate::ztools::eval::EvalTask>> {
    let default_tasks_dir = config.eval_tasks_dir();
    let tasks_dir = tasks_dir.or(default_tasks_dir.as_deref());
    let mut tasks =
        crate::ztools::eval::load_all_eval_tasks(&config.eval_roster_inputs()?, tasks_dir)?;
    if let Some(filter) = task_filter {
        tasks.retain(|t| super::task_matches_filter(&t.name, filter));
        if tasks.is_empty() {
            anyhow::bail!("--task filter {filter} matched no loaded tasks");
        }
    }
    if tasks.is_empty() {
        anyhow::bail!("no eval tasks found (pass --tasks-dir pointing at task snapshots)");
    }
    Ok(tasks)
}

use super::resolve_models;

pub(super) fn run_full_suite(
    config: &ZtoolsConfig,
    model: &str,
    opts: &EvalOptions<'_>,
) -> Result<()> {
    let EvalOptions {
        tasks_dir,
        task_filter,
        json_output,
        thinking,
        ..
    } = *opts;
    let url = &config.osaurus_url;
    let tasks = load_suite_tasks(config, tasks_dir, task_filter)?;
    let (host, port) = crate::ztools::model_eval::parse_osaurus_url(url);
    // The GPU and the single healthy server are held under a machine-wide
    // lock: several sessions measure against this box, and a second
    // concurrent measurement corrupts both. Same contract as the Python
    // eval entry point.
    let _gpu = crate::ztools::eval::GpuLockGuard::acquire(
        "ztools model-eval --suite full",
        std::time::Duration::from_secs(5),
        std::time::Duration::from_secs(crate::ztools::eval::DEFAULT_MAX_IDLE_SECS),
    )
    .map_err(|e| anyhow::anyhow!("GPU lock unavailable: {e}"))?;
    let expected_tasks: Vec<String> = tasks.iter().map(|t| t.name.clone()).collect();
    let mut runs: Vec<crate::ztools::eval::ModelRun> = Vec::new();
    let mut not_measured: Vec<String> = Vec::new();
    crate::ztools::eval::drain::install();
    let started_at = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0.0, |d| d.as_secs_f64());
    for model_name in resolve_models(url, model, config)? {
        if crate::ztools::eval::drain::requested() {
            break;
        }
        let model_gb = unsigned(crate::ztools::eval::estimate_model_memory_gb(&model_name));
        let refusal = crate::ztools::eval::oversize_refusal(model_gb, None, false, None);
        if !refusal.is_empty() {
            not_measured.push(format!("{model_name}: {refusal}"));
            note_unmeasured(&model_name, &refusal, json_output);
            continue;
        }
        if json_output {
            eprintln!(
                "Testing {model_name} (full suite, {} tasks)...",
                tasks.len()
            );
        } else {
            println!(
                "Testing {model_name} (full suite, {} tasks)...",
                tasks.len()
            );
        }
        let cfg = crate::ztools::eval::RunnerConfig {
            host: host.clone(),
            port,
            record_signals: true,
            thinking,
            ..Default::default()
        };
        let outcomes = crate::ztools::eval::run_eval_with_signals(&model_name, &tasks, &cfg);

        if let Some(why) = crate::ztools::model_eval::tasks_unmeasured_reason(&outcomes) {
            not_measured.push(format!("{model_name}: {why}"));
            note_unmeasured(&model_name, &why, json_output);
        }

        runs.push(record_run(
            &model_name,
            &expected_tasks,
            &outcomes,
            json_output,
        )?);
    }
    report_suite(&runs, started_at);
    if crate::ztools::eval::drain::requested() {
        anyhow::bail!(
            "interrupted by Ctrl-C after {} model run(s); outcomes recorded as truncated",
            runs.len()
        );
    }
    if !not_measured.is_empty() {
        anyhow::bail!(
            "{} model run(s) were {} — nothing was measured for them: {}",
            not_measured.len(),
            crate::ztools::model_eval::NOT_MEASURED,
            not_measured.join(" | ")
        );
    }
    Ok(())
}
