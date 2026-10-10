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
pub(crate) fn task_matches_filter(task_name: &str, filter: &str) -> bool {
    filter
        .split(',')
        .map(str::trim)
        .filter(|n| !n.is_empty())
        .any(|n| task_name == n || task_name.ends_with(n))
}

pub(crate) fn weekend_plan(
    config: &ZtoolsConfig,
    location: &str,
    md_out: Option<PathBuf>,
    fetch_latest: bool,
    last_updated: bool,
) -> Result<()> {
    use chrono::Datelike;

    if fetch_latest || last_updated {
        return crate::ztools::store::weekend_latest(last_updated);
    }

    let now = Local::now().naive_local().date();

    // The same window `ztools status` checks the stored plan against: Friday
    // to Sunday, or to Monday when the province observes that Monday.
    let province = crate::ztools::weekend::holidays::load_province(&config.weekend_region_paths)
        .map_err(anyhow::Error::msg)?;
    let (friday, sunday) = crate::ztools::weekend::plan_window(now, province);
    if let Some(holiday) = province.holiday_on(sunday) {
        println!(
            "→ {} on {sunday}: the plan covers the long weekend",
            holiday.name
        );
    }

    // The family's ages on the plan's first day, from the config's birthdays
    // -- the one source; there is no default list to drift from them.
    let ages = crate::ztools::weekend::family::family_ages(&config.weekend_region_paths, friday)
        .map_err(anyhow::Error::msg)?;
    let ages_label = crate::ztools::weekend::family::ages_label(&ages);

    let d1 = friday.format("%Y-%m-%d").to_string();
    let d2 = sunday.format("%Y-%m-%d").to_string();
    let dates_str = format!("{} to {}", friday.format("%b %d"), sunday.format("%b %d"));
    let year = friday.year();

    // Weather is needed BEFORE the pipeline: the draft and structure phases
    // condition their suggestions and weather labels on the forecast. The
    // endpoint comes from the config (`weather_url`), so a run can be pointed at
    // a stub and no invocation of this command has to leave the machine.
    let raw_weather = crate::ztools::weekend::fetch_weather(&d1, &d2, config);
    let weather_str = crate::ztools::weekend::format_weather_display(&raw_weather);

    let exclusions = crate::ztools::weekend::load_exclusions(config);
    let exclusions_str = if exclusions.is_empty() {
        "none".to_string()
    } else {
        exclusions.join(", ")
    };
    let ctx = crate::ztools::weekend::PlanContext {
        location: location.to_string(),
        ages: ages_label.clone(),
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
    let (mut fixed, mut transient) = gate_rows(
        transient,
        &corpus,
        config,
        (friday, sunday),
        &ages,
        &mut health,
    );
    report_constant_columns(&fixed, &transient, &ages_label);

    crate::ztools::weekend::apply_scores(&mut fixed, &weather_str, &ages);
    crate::ztools::weekend::apply_scores(&mut transient, &weather_str, &ages);

    let md_str = crate::ztools::weekend::format_weekend_plan(
        &transient,
        &fixed,
        location,
        &ages_label,
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
/// with its own dates), then suitability (`gate_suitability`), then the
/// weather labels on both lists.
///
/// Returns `(fixed, transient)`.
fn gate_rows(
    transient: Vec<crate::ztools::weekend::WeekendEvent>,
    corpus: &str,
    config: &ZtoolsConfig,
    (friday, sunday): (chrono::NaiveDate, chrono::NaiveDate),
    ages: &[u32],
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

    let (fixed, transient) = gate_suitability(fixed, transient, ages, health);

    let (fixed, weather_notes) = crate::ztools::weekend::correct_weather_labels(fixed);
    let (transient, weather_notes_t) = crate::ztools::weekend::correct_weather_labels(transient);
    for note in weather_notes.iter().chain(weather_notes_t.iter()) {
        println!("→ {note}");
    }
    (fixed, transient)
}

/// The suitability gates (`weekend/suitability.rs`), after the window: a
/// listing page's title is no venue and no event, a row that fits none of the
/// children is not recommended in either half, and a transient row that is
/// only a fixed venue repeated is dropped. Every drop is counted as
/// `unsuitable` in the plan's ledger and named on the operator's channel.
fn gate_suitability(
    fixed: Vec<crate::ztools::weekend::WeekendEvent>,
    transient: Vec<crate::ztools::weekend::WeekendEvent>,
    ages: &[u32],
    health: &mut crate::ztools::weekend::PlanHealth,
) -> (
    Vec<crate::ztools::weekend::WeekendEvent>,
    Vec<crate::ztools::weekend::WeekendEvent>,
) {
    use crate::ztools::weekend::suitability as fit;
    let before = transient.len();
    let (transient, listing_notes, _) = fit::reject_listing_page_titles(transient);
    let (transient, age_notes) = fit::drop_unsuitable_for_ages(transient, ages);
    let (fixed, fixed_notes) = fit::drop_unsuitable_for_ages(fixed, ages);
    let (transient, twin_notes) = fit::drop_duplicates_of_fixed(transient, &fixed);
    health.provenance.unsuitable = before - transient.len();
    for note in [listing_notes, age_notes, fixed_notes, twin_notes]
        .iter()
        .flatten()
    {
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

// The other half of that split: what a run PRINTS once it has run them. Kept
// next to the dispatch that decides whether a run happened, because the two are
// one change — a table that renders scores for a run which measured nothing is
// the defect, and it is fixed in both places or in neither.
#[path = "cli_ztools_eval_report.rs"]
pub(crate) mod eval_report;

#[path = "cli_ztools_eval_runner.rs"]
mod eval_runner;
use eval_runner::run_full_suite;

/// `ztools status`: the harness's view of this project, as JSON.
///
/// Read-only: reads the newest weekend plan and says whether it covers the
/// upcoming weekend and has transient events. Runs nothing, so a status check
/// never spends minutes scraping or burns a model run.
///
/// # Errors
///
/// Only when the status JSON cannot be written to stdout.
pub(crate) fn status(config: &ZtoolsConfig) -> Result<()> {
    crate::ztools::status::run(config)
}

/// The action to perform for `ztools model-eval`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum EvalAction {
    Run,
    Capabilities,
    Leaderboard,
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
    pub thinking: bool,
    pub action: EvalAction,
    pub leaderboard_opts: crate::ztools::eval::LeaderboardOptions<'a>,
}

pub(crate) fn model_eval(config: &ZtoolsConfig, model: &str, opts: &EvalOptions<'_>) -> Result<()> {
    match opts.action {
        EvalAction::Leaderboard => {
            crate::ztools::eval::cli_leaderboard(None, &opts.leaderboard_opts)
        }
        EvalAction::Capabilities => print_capabilities(&config.osaurus_url, model),
        EvalAction::Run => {
            if opts.suite == "full" {
                run_full_suite(config, model, opts)
            } else {
                run_spot_eval(config, model)
            }
        }
    }
}

fn run_spot_eval(config: &ZtoolsConfig, model: &str) -> Result<()> {
    let url = &config.osaurus_url;
    let results = if model == "all" {
        crate::ztools::model_eval::eval_all_models(url, config)?
    } else {
        crate::ztools::model_eval::eval_model(url, model, config)?
    };
    println!(
        "{}",
        crate::ztools::model_eval::render_eval_report(&results)
    );
    // Print the table FIRST, then fail: the rows carry the URL and the error for
    // every task that was not measured, and discarding them to save an exit code
    // would throw away the diagnosis the operator needs to fix the server.
    if let Some(reason) = crate::ztools::model_eval::unmeasured_reason(&results) {
        anyhow::bail!(reason);
    }
    Ok(())
}

pub(super) fn resolve_models(url: &str, model: &str, config: &ZtoolsConfig) -> Result<Vec<String>> {
    if model != "all" {
        return Ok(vec![model.to_string()]);
    }
    crate::ztools::model_eval::get_available_models(url, config)
}

#[cfg(test)]
#[path = "cli_ztools_tests.rs"]
mod tests;
