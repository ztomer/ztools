//! Dispatch for the ztools subcommands: the Twitter summarizer, the weekend
//! planner, the image renamer and the model benchmark.
//!
//! Ported from `routines/src/cli_ztools.rs` when the ztools modules moved into
//! their own crate; the functions now take `&ZtoolsConfig` directly instead of
//! the routines `Config` wrapper they used to be handed.

use anyhow::Result;
use chrono::Local;
use std::path::{Path, PathBuf};

use crate::config::ZtoolsConfig;
use crate::units::unsigned;

/// Whether a task name is selected by a `--task` filter.
///
/// The filter is a comma-separated list; each entry matches the FULL name or
/// its trailing segment, so `--task taxes` picks up `weekend.taxes` without
/// the caller having to know the namespace. Entries are trimmed, because
/// `--task a, b` is what a human types.
///
/// Extracted from the run path so the matching rule is provable on its own.
/// A filter that quietly matches nothing is the failure worth catching here:
/// the caller turns that into a refusal rather than running the full suite as
/// though no filter had been given.
fn task_matches_filter(task_name: &str, filter: &str) -> bool {
    filter
        .split(',')
        .map(str::trim)
        .filter(|n| !n.is_empty())
        .any(|n| task_name == n || task_name.ends_with(n))
}

pub(crate) fn weekend_plan(
    config: &ZtoolsConfig,
    location: &str,
    ages: &str,
    md_out: Option<PathBuf>,
    fetch_latest: bool,
    last_updated: bool,
) -> Result<()> {
    use chrono::Datelike;

    if fetch_latest || last_updated {
        return crate::ztools::store::weekend_latest(last_updated);
    }

    let now = Local::now().naive_local().date();

    // The same window `ztools status` checks the stored plan against.
    let (friday, sunday) = crate::ztools::weekend::plan_window(now);

    let d1 = friday.format("%Y-%m-%d").to_string();
    let d2 = sunday.format("%Y-%m-%d").to_string();
    let dates_str = format!("{} to {}", friday.format("%b %d"), sunday.format("%b %d"));
    let year = friday.year();

    // Weather is needed BEFORE the pipeline: the draft and structure phases
    // condition their suggestions and weather labels on the forecast.
    let raw_weather = crate::ztools::weekend::fetch_weather(&d1, &d2);
    let weather_str = crate::ztools::weekend::format_weather_display(&raw_weather);

    let exclusions = crate::ztools::weekend::load_exclusions(config);
    let exclusions_str = if exclusions.is_empty() {
        "none".to_string()
    } else {
        exclusions.join(", ")
    };
    let ctx = crate::ztools::weekend::PlanContext {
        location: location.to_string(),
        ages: ages.to_string(),
        date_range: dates_str.clone(),
        year,
        exclusions: exclusions_str,
    };

    let (transient, corpus, mut health) = crate::ztools::weekend::fetch_duckduckgo_events(
        location,
        friday,
        sunday,
        &weather_str,
        &ctx,
        config,
    );
    let (mut fixed, mut transient) =
        gate_rows(transient, &corpus, config, friday, sunday, &mut health);
    report_constant_columns(&fixed, &transient, ages);

    crate::ztools::weekend::apply_scores(&mut fixed, &weather_str, ages);
    crate::ztools::weekend::apply_scores(&mut transient, &weather_str, ages);

    let md_str = crate::ztools::weekend::format_weekend_plan(
        &transient,
        &fixed,
        location,
        ages,
        &dates_str,
        &weather_str,
        &health,
    );

    // The dated plan always lands in the store (what `ztools status` and the
    // dashboard tab read); `--md-out` is an extra copy, e.g. a `_latest`
    // pointer the tab's refresh keeps.
    let stored = crate::ztools::store::save_weekend_plan(
        &crate::ztools::store::weekend_output_dir(),
        friday,
        sunday,
        &md_str,
    )?;
    println!("✓ saved to {}", stored.display());
    if let Some(out_path) = md_out {
        std::fs::write(&out_path, &md_str)?;
        println!("✓ saved to {}", out_path.display());
    }

    crate::ztools::weekend::print_weekend_plan_gorgeous(
        &dates_str,
        &weather_str,
        &fixed,
        &transient,
    );
    Ok(())
}

/// The row gates, in order, each one's drop count going into the plan's own
/// ledger line: provenance first (a row that traces to nothing we fetched is
/// invention, and there is no point judging an invented row's dates or
/// weather label), then the exclusion list, then C3 (a dated transient event
/// outside the plan's weekend is dropped and each survivor's `day` reconciled
/// with its own dates), then the weather labels on both lists.
///
/// Returns `(fixed, transient)`.
fn gate_rows(
    transient: Vec<crate::ztools::weekend::WeekendEvent>,
    corpus: &str,
    config: &ZtoolsConfig,
    friday: chrono::NaiveDate,
    sunday: chrono::NaiveDate,
    health: &mut crate::ztools::weekend::PlanHealth,
) -> (
    Vec<crate::ztools::weekend::WeekendEvent>,
    Vec<crate::ztools::weekend::WeekendEvent>,
) {
    health.provenance.extracted = transient.len();
    let (transient, provenance_notes) =
        crate::ztools::weekend::drop_unsourced_rows(transient, corpus);
    health.provenance.unsourced = health.provenance.extracted - transient.len();
    for note in &provenance_notes {
        println!("→ {note}");
    }
    let exclusions = crate::ztools::weekend::load_exclusions(config);
    let before = transient.len();
    let (transient, drop_notes) =
        crate::ztools::weekend::drop_excluded_places(transient, &exclusions);
    health.provenance.excluded = before - transient.len();
    for note in &drop_notes {
        println!("→ {note}");
    }
    let (_, fixed) = crate::ztools::weekend::load_cached_activities(config);

    let before = transient.len();
    let (transient, window_notes) =
        crate::ztools::weekend::drop_events_outside_window(transient, friday, sunday);
    health.provenance.outside_window = before - transient.len();
    let (transient, day_notes) =
        crate::ztools::weekend::reconcile_day_with_dates(transient, friday, sunday);
    for note in window_notes.iter().chain(day_notes.iter()) {
        println!("→ {note}");
    }

    let (fixed, weather_notes) = crate::ztools::weekend::correct_weather_labels(fixed);
    let (transient, weather_notes_t) = crate::ztools::weekend::correct_weather_labels(transient);
    for note in weather_notes.iter().chain(weather_notes_t.iter()) {
        println!("→ {note}");
    }
    (fixed, transient)
}

/// The constant-column check runs LAST, over what survived; it reports and
/// changes nothing. The configured family range is the one suspect that is
/// not a literal (C4).
fn report_constant_columns(
    fixed: &[crate::ztools::weekend::WeekendEvent],
    transient: &[crate::ztools::weekend::WeekendEvent],
    ages: &str,
) {
    let mut suspects: std::collections::HashMap<String, Vec<String>> =
        std::collections::HashMap::new();
    suspects.insert("Target Age(s)".to_string(), vec![ages.to_string()]);
    for (label, values) in crate::ztools::weekend::PROMPT_CONSTANTS {
        suspects.insert(
            label.to_string(),
            values
                .iter()
                .map(std::string::ToString::to_string)
                .collect(),
        );
    }
    for note in crate::ztools::weekend::flag_constant_columns(fixed, &suspects)
        .into_iter()
        .chain(crate::ztools::weekend::flag_constant_columns(
            transient, &suspects,
        ))
    {
        println!("→ {note}");
    }
}

pub(crate) fn image_renamer(config: &ZtoolsConfig, dir: &Path, apply: bool) -> Result<()> {
    let max_len = config.max_image_filename_len;
    let candidates =
        crate::ztools::image_renamer::scan_and_rename(dir, "*", apply, max_len, config)?;
    let mode = if apply { "APPLIED" } else { "DRY-RUN" };
    println!(
        "image-renamer ({mode}): {} file(s) processed",
        candidates.len()
    );
    for c in candidates {
        if c.changed {
            println!("  {} -> {}", c.original.display(), c.proposed_name);
        }
    }
    Ok(())
}

// The `--capabilities` probe lives in its own file for the house 500-line cap,
// along a seam that was already there: it reports what a model IS without
// running a single task, while everything below runs them.
#[path = "cli_ztools_capabilities.rs"]
mod capabilities;
use capabilities::print_capabilities;

/// `ztools status`: the harness's view of this project, as JSON.
///
/// Read-only: reads the newest weekend plan and says whether it covers the
/// upcoming weekend and has transient events. Runs nothing, so a status check
/// never spends minutes scraping or burns a model run.
///
/// # Errors
///
/// Only when the status JSON cannot be written to stdout.
pub(crate) fn status() -> Result<()> {
    crate::ztools::status::run()
}

/// `ztools model-eval`: a native-Rust quality benchmark.
///
/// The `model-eval` switches that are not the model: which suite, where the
/// tasks are, which of them, how to print, and under which regime.
pub(crate) struct EvalOptions<'a> {
    pub suite: &'a str,
    pub tasks_dir: Option<&'a std::path::Path>,
    pub task_filter: Option<&'a str>,
    pub json_output: bool,
    pub capabilities: bool,
    pub thinking: bool,
}

pub(crate) fn model_eval(config: &ZtoolsConfig, model: &str, opts: &EvalOptions<'_>) -> Result<()> {
    let EvalOptions {
        suite,
        tasks_dir,
        task_filter,
        json_output,
        capabilities,
        thinking,
    } = *opts;
    let url = &config.osaurus_url;
    if capabilities {
        return print_capabilities(url, model);
    }
    if suite == "full" {
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
        // Ctrl-C drains rather than kills (eval/drain.rs): the task in flight
        // finishes, the outcomes are recorded, the lock guard releases.
        crate::ztools::eval::drain::install();
        // Everything the history holds from before this instant is "the last
        // run" for the delta table printed at the end.
        let started_at = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0.0, |d| d.as_secs_f64());
        for model_name in resolve_models(url, model, config)? {
            // A drained run stops at the model boundary too: the next model
            // is not started once the operator has asked to stop.
            if crate::ztools::eval::drain::requested() {
                break;
            }
            // Refuse to measure what cannot fit or would thrash: a timing
            // taken under memory pressure describes the pressure, and it
            // hardens into config exactly like a real number. Same gate as
            // the Python eval (eval/cli_runtime.py::oversize_refusal).
            let model_gb = unsigned(crate::ztools::eval::estimate_model_memory_gb(&model_name));
            let refusal = crate::ztools::eval::oversize_refusal(model_gb, None, false, None);
            if !refusal.is_empty() {
                eprintln!("✗ Skipping {model_name}: {refusal}");
                continue;
            }
            // The banner is human progress, not data: under --json-output it
            // must not precede the JSON on stdout.
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
            // The learning path: prefill/cold-start/decode measurement, learned
            // per-task timeouts, p95 signal recording, raw-output archival,
            // stall watchdog. Loaded from and saved back to conf/eval_signals.json.
            let outcomes = crate::ztools::eval::run_eval_with_signals(&model_name, &tasks, &cfg);

            // Completeness is DERIVED by diffing expected vs reported -- no
            // abandon path can forget to set a flag. A truncated run says so
            // out loud here AND carries the verdict into its history entries,
            // which load_historical_stats refuses to average (the bonsai 62%
            // vs 79% class of misread).
            let run_record =
                crate::ztools::eval::ModelRun::new(&model_name, &expected_tasks, outcomes.clone());
            if let Some(c) = &run_record.completeness {
                if !c.complete {
                    eprintln!("⚠ {} (partial): {}", model_name, c.reason);
                }
            }
            if let Err(e) = crate::ztools::eval::save_historical_results(&run_record, None) {
                eprintln!("⚠ could not write eval history: {e}");
            }
            runs.push(run_record);
            print_outcomes(&outcomes, json_output)?;
        }
        report_suite(&runs, started_at);
        // Recorded and reported; now say the run was cut, with the exit code
        // a sweep files as FAILED so --resume runs this model again.
        if crate::ztools::eval::drain::requested() {
            anyhow::bail!(
                "interrupted by Ctrl-C after {} model run(s); outcomes recorded as truncated",
                runs.len()
            );
        }
        return Ok(());
    }
    let results = if model == "all" {
        crate::ztools::model_eval::eval_all_models(url, config)?
    } else {
        crate::ztools::model_eval::eval_model(url, model, config)?
    };
    println!(
        "{}",
        crate::ztools::model_eval::render_eval_report(&results)
    );
    Ok(())
}

/// "all" expands to every servable model on the server; any other value is
/// taken literally.
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
        tasks.retain(|t| task_matches_filter(&t.name, filter));
        if tasks.is_empty() {
            anyhow::bail!("--task filter {filter} matched no loaded tasks");
        }
    }
    if tasks.is_empty() {
        anyhow::bail!("no eval tasks found (pass --tasks-dir pointing at task snapshots)");
    }
    Ok(tasks)
}

/// One model's outcomes: a JSON array under `--json-output` (stdout carries
/// nothing else), otherwise the rendered table with substitutions noted.
fn print_outcomes(outcomes: &[crate::ztools::eval::TaskOutcome], json_output: bool) -> Result<()> {
    if json_output {
        use serde::Serialize;
        #[derive(Serialize)]
        struct OutcomeRow<'a> {
            task: &'a str,
            score: u8,
            status: &'a str,
            time_secs: f64,
            error: Option<&'a String>,
            failure_category: &'a str,
            #[serde(skip_serializing_if = "Option::is_none")]
            substituted_to: Option<&'a String>,
            #[serde(skip_serializing_if = "Option::is_none")]
            substitution_reason: Option<&'a String>,
        }
        let rows: Vec<OutcomeRow> = outcomes
            .iter()
            .map(|o| OutcomeRow {
                task: &o.task,
                score: o.score,
                status: o.status.as_str(),
                time_secs: o.time_secs,
                error: o.error.as_ref(),
                failure_category: o.failure_category.as_str(),
                substituted_to: o.substituted_to.as_ref(),
                substitution_reason: o.substitution_reason.as_ref(),
            })
            .collect();
        println!("{}", serde_json::to_string_pretty(&rows)?);
    } else {
        for note in outcomes
            .iter()
            .filter_map(|o| o.substitution_reason.as_deref())
        {
            eprintln!("⚠ {note}");
        }
        print!(
            "{}",
            crate::ztools::model_eval::render_task_outcomes(outcomes)
        );
    }
    Ok(())
}

/// Persistence + reporting, matching the Python evaluator's exports: the
/// per-(model, task) CSV sheet, the historical trends table, the delta from
/// the last run and the verbosity table. Nothing to do for an empty sweep.
fn report_suite(runs: &[crate::ztools::eval::ModelRun], started_at: f64) {
    if runs.is_empty() {
        return;
    }
    let csv_path = crate::ztools::eval::default_eval_dir().join("eval_results.csv");
    match crate::ztools::eval::export_csv(runs, &csv_path) {
        Ok(()) => println!("→ Exported to {}", csv_path.display()),
        Err(e) => eprintln!("⚠ CSV export failed: {e}"),
    }
    for line in crate::ztools::eval::render_historical_trends(None) {
        println!("{line}");
    }
    for line in crate::ztools::eval::render_diff_from_last_run(runs, None, started_at) {
        println!("{line}");
    }
    for line in crate::ztools::eval::render_verbosity(&crate::ztools::eval::compute_verbosity(runs))
    {
        println!("{line}");
    }
}

fn resolve_models(url: &str, model: &str, config: &ZtoolsConfig) -> Result<Vec<String>> {
    if model != "all" {
        return Ok(vec![model.to_string()]);
    }
    crate::ztools::model_eval::get_available_models(url, config)
}

#[cfg(test)]
#[path = "cli_ztools_tests.rs"]
mod tests;
