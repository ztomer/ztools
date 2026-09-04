//! Dispatch for the ztools subcommands: the Twitter summarizer, the weekend
//! planner, the image renamer and the model benchmark.
//!
//! Ported from `routines/src/cli_ztools.rs` when the ztools modules moved into
//! their own crate; the functions now take `&ZtoolsConfig` directly instead of
//! the routines `Config` wrapper they used to be handed.

use anyhow::Result;
use chrono::Local;
use std::path::PathBuf;

use crate::config::ZtoolsConfig;

/// The `twitter-summarize` flag set, grouped so the function signature stays
/// one stable struct instead of a growing argument list.
pub(crate) struct TwitterSummarizeOpts {
    pub json: Option<String>,
    pub model: Option<String>,
    pub md_out: Option<PathBuf>,
    pub use_cache: bool,
    pub fetch_only: bool,
    pub debug: bool,
    pub since: Option<String>,
    pub login: bool,
    pub fetch_latest: bool,
    pub last_updated: bool,
}

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

/// Parse a tweet array, or nothing.
///
/// Deliberately lossy in one direction only: unparseable input yields an empty
/// list so the caller falls through to its other sources, and never a partial
/// one. Half a timeline read as the whole timeline is a summary that is
/// confidently wrong about what was said.
fn tweets_from_json(text: &str) -> Vec<crate::ztools::twitter::Tweet> {
    serde_json::from_str::<Vec<crate::ztools::twitter::Tweet>>(text).unwrap_or_default()
}

/// Tweets from one file, or nothing when it is absent or unreadable.
fn tweets_from_file(path: &std::path::Path) -> Vec<crate::ztools::twitter::Tweet> {
    match std::fs::read_to_string(path) {
        Ok(content) => tweets_from_json(&content),
        Err(_) => Vec::new(),
    }
}

/// The first cache file that yields a NON-EMPTY tweet list, and where it came
/// from.
///
/// The non-empty condition is the point: an empty cache file is not an answer,
/// and stopping at one would report "0 cached tweets" while a populated
/// candidate sat unread behind it. Returning the path too is what lets the
/// caller say which file it used instead of leaving the operator to guess
/// between two hard-coded locations.
///
/// The candidate list is a parameter rather than being built from `$HOME`
/// inside, so every branch here is provable without writing into the
/// developer's own home directory.
fn tweets_from_cache(
    candidates: &[PathBuf],
) -> Option<(PathBuf, Vec<crate::ztools::twitter::Tweet>)> {
    for candidate in candidates {
        let parsed = tweets_from_file(candidate);
        if !parsed.is_empty() {
            return Some((candidate.clone(), parsed));
        }
    }
    None
}

pub(crate) fn twitter_summarize(config: &ZtoolsConfig, opts: TwitterSummarizeOpts) -> Result<()> {
    let TwitterSummarizeOpts {
        json,
        model,
        md_out,
        use_cache,
        fetch_only,
        debug,
        since,
        login,
        fetch_latest,
        last_updated,
    } = opts;
    if fetch_latest || last_updated {
        return crate::ztools::store::twitter_latest(last_updated);
    }
    if login {
        println!("· Launching browser for x.com sign-in...");
        return crate::ztools::twitter::browser::login_live();
    }

    let mut tweets = Vec::new();
    let mut explicit_source = false;

    if let Some(path_or_dash) = json {
        if path_or_dash == "-" {
            let mut buffer = String::new();
            if std::io::Read::read_to_string(&mut std::io::stdin(), &mut buffer).is_ok() {
                tweets = tweets_from_json(&buffer);
            }
        } else {
            tweets = tweets_from_file(std::path::Path::new(&path_or_dash));
        }
        explicit_source = true;
    }

    if !explicit_source {
        if use_cache {
            let candidates: Vec<PathBuf> = [
                dirs::home_dir().map(|h| h.join(".twitter_summary_debug_cache.json")),
                dirs::home_dir().map(|h| h.join(".cache/twitter/debug_tweets.json")),
            ]
            .into_iter()
            .flatten()
            .collect();
            if let Some((from, cached)) = tweets_from_cache(&candidates) {
                println!(
                    "· Using {} cached tweets from {}",
                    cached.len(),
                    from.display()
                );
                tweets = cached;
            }
            if tweets.is_empty() {
                anyhow::bail!(
                    "No cached tweets found. Run without --use-cache first to scrape live tweets."
                );
            }
        } else {
            tweets = crate::ztools::twitter::browser::collect_tweets_live(since.as_deref(), debug)?;
            if tweets.is_empty() {
                println!("· No tweets found in the timeline window.");
                return Ok(());
            }
            if fetch_only {
                println!(
                    "✓ {} tweets fetched and cached. (--fetch-only provided, exiting)",
                    tweets.len()
                );
                return Ok(());
            }
        }
    }

    let path = crate::ztools::twitter::run_summary(&tweets, None, None, model.as_deref(), config)?;
    if let Ok(doc) = std::fs::read_to_string(&path) {
        println!("{}", doc);
    }
    println!("✓ twitter summary generated at {}", path.display());
    if let Some(out_path) = md_out {
        std::fs::copy(&path, &out_path)?;
        println!("✓ copy saved to {}", out_path.display());
    }
    Ok(())
}

pub(crate) fn weekend_plan(
    config: &ZtoolsConfig,
    location: String,
    ages: String,
    md_out: Option<PathBuf>,
    fetch_latest: bool,
    last_updated: bool,
) -> Result<()> {
    if fetch_latest || last_updated {
        return crate::ztools::store::weekend_latest(last_updated);
    }
    let now = Local::now().naive_local().date();
    use chrono::Datelike;

    // Find upcoming Friday
    let mut friday = now;
    while friday.weekday() != chrono::Weekday::Fri {
        friday = friday.succ_opt().unwrap();
    }
    let sunday = friday.succ_opt().unwrap().succ_opt().unwrap();

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
        location: location.clone(),
        ages: ages.clone(),
        date_range: dates_str.clone(),
        year,
        exclusions: exclusions_str,
    };

    let (transient, corpus) = crate::ztools::weekend::fetch_duckduckgo_events(
        &location,
        friday,
        sunday,
        &weather_str,
        &ctx,
        config,
    );
    // Provenance FIRST: a row that traces to nothing we fetched is invention,
    // and there is no point judging an invented row's dates or weather label.
    let (transient, provenance_notes) =
        crate::ztools::weekend::drop_unsourced_rows(transient, &corpus);
    for note in &provenance_notes {
        println!("→ {note}");
    }
    let exclusions = crate::ztools::weekend::load_exclusions(config);
    let (transient, drop_notes) =
        crate::ztools::weekend::drop_excluded_places(transient, &exclusions);
    for note in &drop_notes {
        println!("→ {note}");
    }
    let (_, fixed) = crate::ztools::weekend::load_cached_activities(config);

    // C3: a dated transient event outside the plan's weekend is dropped; then
    // each surviving row's `day` is reconciled with its own dates.
    let (transient, window_notes) =
        crate::ztools::weekend::drop_events_outside_window(transient, friday, sunday);
    let (transient, day_notes) =
        crate::ztools::weekend::reconcile_day_with_dates(transient, friday, sunday);
    for note in window_notes.iter().chain(day_notes.iter()) {
        println!("→ {note}");
    }

    let (mut fixed, weather_notes) = crate::ztools::weekend::correct_weather_labels(fixed);
    let (mut transient, weather_notes_t) =
        crate::ztools::weekend::correct_weather_labels(transient);
    for note in weather_notes.iter().chain(weather_notes_t.iter()) {
        println!("→ {note}");
    }

    // Constant-column check runs LAST, over what survived; it reports and
    // changes nothing. The configured family range is the one suspect that is
    // not a literal (C4).
    let mut suspects: std::collections::HashMap<String, Vec<String>> =
        std::collections::HashMap::new();
    suspects.insert("Target Age(s)".to_string(), vec![ages.clone()]);
    for (label, values) in crate::ztools::weekend::PROMPT_CONSTANTS {
        suspects.insert(
            label.to_string(),
            values.iter().map(|s| s.to_string()).collect(),
        );
    }
    for note in crate::ztools::weekend::flag_constant_columns(&fixed, &suspects)
        .into_iter()
        .chain(crate::ztools::weekend::flag_constant_columns(
            &transient, &suspects,
        ))
    {
        println!("→ {note}");
    }

    crate::ztools::weekend::apply_scores(&mut fixed, &weather_str, &ages);
    crate::ztools::weekend::apply_scores(&mut transient, &weather_str, &ages);

    let md_str = crate::ztools::weekend::format_weekend_plan(
        &transient,
        &fixed,
        &location,
        &ages,
        &dates_str,
        &weather_str,
    );

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

pub(crate) fn image_renamer(config: &ZtoolsConfig, dir: PathBuf, apply: bool) -> Result<()> {
    let max_len = config.max_image_filename_len;
    let candidates =
        crate::ztools::image_renamer::scan_and_rename(&dir, "*", apply, max_len, config)?;
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

pub(crate) fn model_eval(
    config: &ZtoolsConfig,
    model: String,
    suite: &str,
    tasks_dir: Option<&std::path::Path>,
    task_filter: Option<&str>,
    json_output: bool,
    capabilities: bool,
) -> Result<()> {
    let url = &config.osaurus_url;
    if capabilities {
        return print_capabilities(url, &model);
    }
    if suite == "full" {
        let mut tasks = crate::ztools::eval::load_all_eval_tasks(tasks_dir);
        if let Some(filter) = task_filter {
            tasks.retain(|t| task_matches_filter(&t.name, filter));
            if tasks.is_empty() {
                anyhow::bail!("--task filter {filter} matched no loaded tasks");
            }
        }
        if tasks.is_empty() {
            anyhow::bail!("no eval tasks found (pass --tasks-dir pointing at task snapshots)");
        }
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
        for model_name in resolve_models(url, &model, config)? {
            // Refuse to measure what cannot fit or would thrash: a timing
            // taken under memory pressure describes the pressure, and it
            // hardens into config exactly like a real number. Same gate as
            // the Python eval (eval/cli_runtime.py::oversize_refusal).
            let model_gb = crate::ztools::eval::estimate_model_memory_gb(&model_name) as f64;
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
                    crate::ztools::model_eval::render_task_outcomes(&outcomes)
                );
            }
        }
        // Persistence + reporting, matching the Python evaluator's exports:
        // per-(model, task) CSV sheet and the historical trends table.
        if !runs.is_empty() {
            let csv_path = crate::ztools::eval::default_eval_dir().join("eval_results.csv");
            match crate::ztools::eval::export_csv(&runs, &csv_path) {
                Ok(()) => println!("→ Exported to {}", csv_path.display()),
                Err(e) => eprintln!("⚠ CSV export failed: {e}"),
            }
            for line in crate::ztools::eval::render_historical_trends(None) {
                println!("{line}");
            }
        }
        return Ok(());
    }
    let results = if model == "all" {
        crate::ztools::model_eval::eval_all_models(url, config)?
    } else {
        crate::ztools::model_eval::eval_model(url, &model, config)?
    };
    println!(
        "{}",
        crate::ztools::model_eval::render_eval_report(&results)
    );
    Ok(())
}

/// "all" expands to every servable model on the server; any other value is
/// taken literally.
fn resolve_models(url: &str, model: &str, config: &ZtoolsConfig) -> Result<Vec<String>> {
    if model != "all" {
        return Ok(vec![model.to_string()]);
    }
    crate::ztools::model_eval::get_available_models(url, config)
}

#[cfg(test)]
#[path = "cli_ztools_tests.rs"]
mod tests;
