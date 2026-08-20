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

pub(crate) fn twitter_summarize(
    config: &ZtoolsConfig,
    json: String,
    model: Option<String>,
    md_out: Option<PathBuf>,
) -> Result<()> {
    let mut tweets = Vec::new();
    if json == "-" {
        use std::io::IsTerminal;
        if !std::io::stdin().is_terminal() {
            let mut buffer = String::new();
            if std::io::Read::read_to_string(&mut std::io::stdin(), &mut buffer).is_ok() {
                if let Ok(parsed) =
                    serde_json::from_str::<Vec<crate::ztools::twitter::Tweet>>(&buffer)
                {
                    tweets = parsed;
                }
            }
        }
    } else if std::path::Path::new(&json).exists() {
        if let Ok(content) = std::fs::read_to_string(&json) {
            if let Ok(parsed) = serde_json::from_str::<Vec<crate::ztools::twitter::Tweet>>(&content)
            {
                tweets = parsed;
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

    let transient = crate::ztools::weekend::fetch_duckduckgo_events(&location, &d1, &d2, config);
    let exclusions = crate::ztools::weekend::load_exclusions(config);
    let (mut transient, drop_notes) =
        crate::ztools::weekend::drop_excluded_places(transient, &exclusions);
    for note in &drop_notes {
        println!("→ {note}");
    }
    let (_, mut fixed) = crate::ztools::weekend::load_cached_activities(config);
    let raw_weather = crate::ztools::weekend::fetch_weather(&d1, &d2);
    let weather_str = crate::ztools::weekend::format_weather_display(&raw_weather);

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

    crate::ztools::weekend::print_weekend_plan_gorgeous(&dates_str, &weather_str, &fixed, &transient);
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

pub(crate) fn model_eval(config: &ZtoolsConfig, model: String) -> Result<()> {
    let url = &config.osaurus_url;
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