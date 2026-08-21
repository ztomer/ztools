//! CLI surface and dispatch for the four ztools subcommands.
//!
//! Ported from `routines/src/main.rs` + `cli_args.rs` + `cli_run.rs` wiring
//! when the ztools modules moved into their own crate.

use anyhow::Result;
use clap::{Parser, Subcommand};
use std::path::PathBuf;

use crate::config::ZtoolsConfig;

#[derive(Parser)]
#[command(
    name = "ztools",
    version,
    about = "Native Rust ztools: Twitter summarizer, weekend planner, image renamer, model eval."
)]
struct Cli {
    /// Path to a ztools TOML config file (see `ZtoolsConfig`). Without it,
    /// built-in defaults apply, plus `[best_models]` overrides read from the
    /// usual ztools config locations.
    #[arg(long, global = true)]
    config: Option<PathBuf>,

    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Run native Rust Twitter timeline summarizer.
    TwitterSummarize {
        /// Optional path to tweets JSON file or stdin `-`.
        #[arg(long)]
        json: Option<String>,
        /// Model to use on Osaurus server.
        #[arg(long)]
        model: Option<String>,
        /// Optional path to write markdown summary.
        #[arg(long)]
        md_out: Option<PathBuf>,
        /// Use cached tweets from last run instead of fetching new ones.
        #[arg(long)]
        use_cache: bool,
        /// Collect tweets, save them to cache, and exit without summarizing.
        #[arg(long)]
        fetch_only: bool,
        /// Show browser window and verbose output.
        #[arg(long)]
        debug: bool,
        /// Override start time (e.g. '24h' or ISO 8601).
        #[arg(long)]
        since: Option<String>,
        /// Open browser to log in to x.com.
        #[arg(long)]
        login: bool,
    },
    /// Run native Rust Weekend planner.
    WeekendPlan {
        /// Location string (e.g. "Vaughan/Toronto").
        #[arg(long, default_value = "Vaughan/Toronto")]
        location: String,
        /// Target family ages, comma separated (e.g. "13,10,6").
        #[arg(long, default_value = "13,10,6")]
        ages: String,
        /// Optional path to write markdown plan.
        #[arg(long)]
        md_out: Option<PathBuf>,
    },
    /// Run native Rust image renamer.
    ImageRenamer {
        /// Directory containing images to process.
        #[arg(default_value = ".")]
        dir: PathBuf,
        /// Apply rename operations (defaults to dry-run).
        #[arg(long)]
        apply: bool,
    },
    /// Run native Rust model quality benchmark.
    ModelEval {
        /// Which model to evaluate (or 'all').
        #[arg(long, default_value = "all")]
        model: String,
        /// "smoke" (default) runs the built-in smoke suite; "full" runs the
        /// eval-loop runner over smoke + taxes tasks with retries and the
        /// reasoning-overrun guard.
        #[arg(long, default_value = "smoke")]
        suite: String,
        /// Optional directory of task JSON snapshots for --suite full
        /// (e.g. eval_tasks/data). Without it, only built-in tasks run.
        #[arg(long)]
        tasks_dir: Option<PathBuf>,
        /// With --suite full: run only the named task(s), comma-separated
        /// (e.g. --task taxes_qa,taxes_slip_qa). Default: all tasks.
        #[arg(long)]
        task: Option<String>,
        /// With --suite full: print outcomes as JSON instead of markdown.
        #[arg(long)]
        json_output: bool,
    },
}

/// Parse the CLI, resolve config, and dispatch to the tool handlers.
pub fn run() -> Result<()> {
    let mut args: Vec<std::ffi::OsString> = std::env::args_os().collect();
    if let Some(first) = args.first() {
        let prog = std::path::Path::new(first)
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("");
        let subcommand = match prog {
            "weekend" | "weekend-plan" => Some("weekend-plan"),
            "twitter" | "twitter-summarize" => Some("twitter-summarize"),
            "image-renamer" | "rename_images" | "rename-images" => Some("image-renamer"),
            "model-eval" | "oeval" => Some("model-eval"),
            _ => None,
        };
        if let Some(sub) = subcommand {
            if args.len() == 1 || args.get(1).and_then(|s| s.to_str()) != Some(sub) {
                args.insert(1, sub.into());
            }
        }
    }
    let cli = Cli::parse_from(args);
    // An explicit `--config` is authoritative: the file is exactly what runs,
    // so a test (or a CI job) can point the URLs at stubs without the dynamic
    // `[best_models]` override reaching out to the operator's real config.
    // Without it, defaults apply and `[best_models]` is layered on top.
    let config = match cli.config {
        Some(path) => {
            let content = std::fs::read_to_string(&path)
                .map_err(|e| anyhow::anyhow!("cannot read config {}: {e}", path.display()))?;
            toml::from_str::<ZtoolsConfig>(&content)
                .map_err(|e| anyhow::anyhow!("cannot parse config {}: {e}", path.display()))?
        }
        None => ZtoolsConfig::default()
            .with_ztools_best_models()
            .with_shared_prompts(),
    };
    match cli.cmd {
        Cmd::TwitterSummarize {
            json,
            model,
            md_out,
            use_cache,
            fetch_only,
            debug,
            since,
            login,
        } => crate::cli_ztools::twitter_summarize(
            &config,
            crate::cli_ztools::TwitterSummarizeOpts {
                json,
                model,
                md_out,
                use_cache,
                fetch_only,
                debug,
                since,
                login,
            },
        ),
        Cmd::WeekendPlan {
            location,
            ages,
            md_out,
        } => crate::cli_ztools::weekend_plan(&config, location, ages, md_out),
        Cmd::ImageRenamer { dir, apply } => crate::cli_ztools::image_renamer(&config, dir, apply),
        Cmd::ModelEval {
            model,
            suite,
            tasks_dir,
            task,
            json_output,
        } => crate::cli_ztools::model_eval(
            &config,
            model,
            &suite,
            tasks_dir.as_deref(),
            task.as_deref(),
            json_output,
        ),
    }
}
