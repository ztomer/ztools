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

#[allow(clippy::too_many_arguments)]
pub(crate) fn twitter_summarize(
    config: &ZtoolsConfig,
    json: Option<String>,
    model: Option<String>,
    md_out: Option<PathBuf>,
    use_cache: bool,
    fetch_only: bool,
    debug: bool,
    since: Option<String>,
    login: bool,
) -> Result<()> {
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
                if let Ok(parsed) =
                    serde_json::from_str::<Vec<crate::ztools::twitter::Tweet>>(&buffer)
                {
                    tweets = parsed;
                }
            }
        } else if std::path::Path::new(&path_or_dash).exists() {
            if let Ok(content) = std::fs::read_to_string(&path_or_dash) {
                if let Ok(parsed) =
                    serde_json::from_str::<Vec<crate::ztools::twitter::Tweet>>(&content)
                {
                    tweets = parsed;
                }
            }
        }
        explicit_source = true;
    }

    if !explicit_source {
        if use_cache {
            let candidates = [
                dirs::home_dir().map(|h| h.join(".twitter_summary_debug_cache.json")),
                dirs::home_dir().map(|h| h.join(".cache/twitter/debug_tweets.json")),
            ];
            for candidate in candidates.into_iter().flatten() {
                if candidate.exists() {
                    if let Ok(content) = std::fs::read_to_string(&candidate) {
                        if let Ok(parsed) = serde_json::from_str::<Vec<crate::ztools::twitter::Tweet>>(&content) {
                            if !parsed.is_empty() {
                                tweets = parsed;
                                println!("· Using {} cached tweets from {}", tweets.len(), candidate.display());
                                break;
                            }
                        }
                    }
                }
            }
            if tweets.is_empty() {
                anyhow::bail!("No cached tweets found. Run without --use-cache first to scrape live tweets.");
            }
        } else {
            tweets = crate::ztools::twitter::browser::collect_tweets_live(since.as_deref(), debug)?;
            if tweets.is_empty() {
                println!("· No tweets found in the timeline window.");
                return Ok(());
            }
            if fetch_only {
                println!("✓ {} tweets fetched and cached. (--fetch-only provided, exiting)", tweets.len());
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
) -> Result<()> {
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

pub(crate) fn model_eval(
    config: &ZtoolsConfig,
    model: String,
    suite: &str,
    tasks_dir: Option<&std::path::Path>,
    task_filter: Option<&str>,
    json_output: bool,
) -> Result<()> {
    let url = &config.osaurus_url;
    if suite == "full" {
        let mut tasks = crate::ztools::eval::load_all_eval_tasks(tasks_dir);
        if let Some(filter) = task_filter {
            let names: Vec<&str> = filter.split(',').map(str::trim).collect();
            tasks.retain(|t| names.iter().any(|n| t.name.ends_with(n) || *n == t.name));
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
            std::time::Duration::from_secs(
                crate::ztools::eval::DEFAULT_MAX_IDLE_SECS,
            ),
        )
        .map_err(|e| anyhow::anyhow!("GPU lock unavailable: {e}"))?;
        for model_name in resolve_models(url, &model, config)? {
            // The banner is human progress, not data: under --json-output it
            // must not precede the JSON on stdout.
            if json_output {
                eprintln!("Testing {model_name} (full suite, {} tasks)...", tasks.len());
            } else {
                println!("Testing {model_name} (full suite, {} tasks)...", tasks.len());
            }
            let cfg = crate::ztools::eval::RunnerConfig {
                host: host.clone(),
                port,
                ..Default::default()
            };
            let outcomes = crate::ztools::eval::run_eval(&model_name, &tasks, &cfg);
            if json_output {
                use serde::Serialize;
                #[derive(Serialize)]
                struct OutcomeRow<'a> {
                    task: &'a str,
                    score: u8,
                    status: &'a str,
                    time_secs: f64,
                    error: Option<&'a String>,
                }
                let rows: Vec<OutcomeRow> = outcomes
                    .iter()
                    .map(|o| OutcomeRow {
                        task: &o.task,
                        score: o.score,
                        status: &o.status,
                        time_secs: o.time_secs,
                        error: o.error.as_ref(),
                    })
                    .collect();
                println!("{}", serde_json::to_string_pretty(&rows)?);
            } else {
                print!(
                    "{}",
                    crate::ztools::model_eval::render_task_outcomes(&outcomes)
                );
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
fn resolve_models(
    url: &str,
    model: &str,
    config: &ZtoolsConfig,
) -> Result<Vec<String>> {
    if model != "all" {
        return Ok(vec![model.to_string()]);
    }
    crate::ztools::model_eval::get_available_models(url, config)
}
