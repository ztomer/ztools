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
    #[command(version)]
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
        /// Delete stored summary .md files and exit.
        #[arg(long)]
        clean: bool,
        /// Print the stored summary from the newest run. Read-only: runs no
        /// model and touches no network (what a dashboard tab must do).
        #[arg(long)]
        fetch_latest: bool,
        /// Print when the newest stored summary was last updated. Read-only.
        #[arg(long)]
        last_updated: bool,
    },
    /// Run native Rust Weekend planner.
    #[command(version)]
    WeekendPlan {
        /// Location string (e.g. "Vaughan/Toronto").
        #[arg(long, default_value = "Vaughan/Toronto")]
        location: String,
        /// Optional path to write markdown plan.
        #[arg(long)]
        md_out: Option<PathBuf>,
        /// Print the stored plan from the newest run. Read-only: runs no model
        /// and touches no network (what a dashboard tab must do).
        #[arg(long)]
        fetch_latest: bool,
        /// Print when the newest stored plan was last updated. Read-only.
        #[arg(long)]
        last_updated: bool,
    },
    /// Print this project's status as JSON for the routines harness.
    ///
    /// Additive and read-only: reads the newest weekend plan, runs nothing
    /// else. The scheduled `ztools status` in `routine do` and the tabs that
    /// show it need a machine-readable answer, not a summary that reads like
    /// a person wrote it because the two eventually disagree in tense.
    #[command(version)]
    Status,
    /// Print the twitter summarizer's status as JSON for the routines harness.
    ///
    /// Read-only: reads the newest stored summary, runs no browser and no
    /// model. The `[status]` command in `routines-twitter.toml`.
    #[command(version)]
    TwitterStatus,
    /// Run native Rust image renamer.
    #[command(version)]
    ImageRenamer {
        /// Directory containing images to process.
        #[arg(default_value = ".")]
        dir: PathBuf,
        /// Apply rename operations (defaults to dry-run).
        #[arg(long)]
        apply: bool,
    },
    /// Run native Rust model quality benchmark.
    #[command(version)]
    ModelEval {
        /// Which model to evaluate (or 'all').
        #[arg(long, default_value = "all")]
        model: String,
        /// "smoke" (default) runs the built-in smoke suite; "full" runs the
        /// eval-loop runner over the full task roster (the 24 prompt tasks
        /// plus the taxes snapshots) with retries and the reasoning-overrun
        /// guard.
        #[arg(long, default_value = "smoke")]
        suite: String,
        /// Optional directory of task JSON snapshots for --suite full
        /// (e.g. `eval_tasks/data`). Without it, the first existing
        /// `eval_tasks_dirs` entry of the ztools config is used.
        #[arg(long)]
        tasks_dir: Option<PathBuf>,
        /// With --suite full: run only the named task(s), comma-separated
        /// (e.g. --task `taxes_qa,taxes_slip_qa`). Default: all tasks.
        #[arg(long)]
        task: Option<String>,
        /// With --suite full: print outcomes as JSON instead of markdown.
        #[arg(long)]
        json_output: bool,
        /// Probe what each installed model IS (family, generative, disk
        /// footprint, viability) without running any task.
        #[arg(long)]
        capabilities: bool,
        /// Let reasoning models think before answering. Off by default,
        /// because that is how the production tools call them; on measures
        /// the pre-2026-09-19 regime for comparison with older sweeps.
        #[arg(long)]
        thinking: bool,
        /// Format comparative markdown leaderboard of latest clean runs across evaluated models.
        #[arg(long)]
        leaderboard: bool,
        /// Minimum task count required for leaderboard inclusion.
        #[arg(long)]
        min_tasks: Option<usize>,
        /// Slot to sort the leaderboard by (overall, think, json, summarize, filename, vlm).
        #[arg(long)]
        sort_by: Option<String>,
        /// Output leaderboard in comma-separated values (CSV) format.
        #[arg(long)]
        csv_output: bool,
        /// Group leaderboard entries by detected model family.
        #[arg(long)]
        group_by_family: bool,
        /// Filter leaderboard scoring and rankings to tasks belonging to category.
        #[arg(long)]
        category: Option<String>,
        /// Fail with non-zero exit code if any model's score delta falls below threshold percent.
        #[arg(long)]
        fail_on_regression: Option<f64>,
    },
    /// Manage stored eval signals (inspect or prune superseded task observations).
    #[command(version)]
    EvalSignals {
        /// Prune task records whose fingerprints do not match current task definitions.
        #[arg(long)]
        prune: bool,
        /// Explicit path to eval signals JSON file (default: `conf/eval_signals.json`).
        #[arg(long)]
        path: Option<PathBuf>,
        /// Directory containing task definitions (defaults to `conf/` and `eval_tasks/data`).
        #[arg(long)]
        tasks_dir: Option<PathBuf>,
        /// Show what would be pruned without modifying the file.
        #[arg(long)]
        dry_run: bool,
    },
}

/// The subcommand a program's own NAME implies, if any.
///
/// `install.sh` symlinks these names at the one binary, so for an installed
/// user `argv[0]` is the ONLY thing that distinguishes `oeval` from `ztools`:
/// the table below is the whole feature. `cli_tests.rs` pins every name the
/// installer creates, in both directions (a symlink that resolves, and a name
/// that must NOT acquire a subcommand).
fn implied_subcommand(prog: &str) -> Option<&'static str> {
    match prog {
        "weekend" | "weekend-plan" => Some("weekend-plan"),
        "twitter" | "twitter-summarize" => Some("twitter-summarize"),
        "image-renamer" | "rename_images" | "rename-images" => Some("image-renamer"),
        "model-eval" | "oeval" => Some("model-eval"),
        _ => None,
    }
}

/// The argv rewrite as a PURE function, for the tests.
///
/// Nothing here reads the process: `args_with_implied_subcommand` is the only
/// caller that does, and it owns nothing but collecting `argv`.
fn argv_with_implied_subcommand(mut args: Vec<std::ffi::OsString>) -> Vec<std::ffi::OsString> {
    if let Some(first) = args.first() {
        // The file name, not the whole argv[0]: the installer links
        // `/opt/homebrew/bin/weekend`, and only the last component is the name
        // the table is keyed on.
        let prog = std::path::Path::new(first)
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("");
        // Explicit wins and is never duplicated: `weekend weekend-plan` already
        // says which subcommand to run, and `weekend --location X` needs the
        // one inserted in front of the flag.
        if let Some(sub) = implied_subcommand(prog)
            && (args.len() == 1 || args.get(1).and_then(|s| s.to_str()) != Some(sub))
        {
            args.insert(1, sub.into());
        }
    }
    args
}

/// The argv with the subcommand implied by the program's own name inserted:
/// a `weekend` symlink runs `ztools weekend-plan`, and so on.
fn args_with_implied_subcommand() -> Vec<std::ffi::OsString> {
    argv_with_implied_subcommand(std::env::args_os().collect())
}

/// The configuration a run uses.
///
/// An explicit `--config` is authoritative: the file is exactly what runs,
/// so a test (or a CI job) can point the URLs at stubs without the dynamic
/// `[best_models]` override reaching out to the operator's real config.
/// Without it, defaults apply and `[best_models]` is layered on top.
///
/// # Errors
///
/// When the explicit file cannot be read or is not valid `ZtoolsConfig` TOML.
fn load_config(explicit: Option<PathBuf>) -> Result<ZtoolsConfig> {
    explicit.map_or_else(
        || {
            Ok(ZtoolsConfig::default()
                .with_ztools_best_models()
                .with_shared_prompts())
        },
        |path| {
            let content = std::fs::read_to_string(&path)
                .map_err(|e| anyhow::anyhow!("cannot read config {}: {e}", path.display()))?;
            toml::from_str::<ZtoolsConfig>(&content)
                .map_err(|e| anyhow::anyhow!("cannot parse config {}: {e}", path.display()))
        },
    )
}

fn dispatch_twitter(config: &ZtoolsConfig, cmd: Cmd) -> Result<()> {
    use crate::cli_ztools_twitter::{TweetSource, TwitterCommand, TwitterSummarizeOpts};

    let Cmd::TwitterSummarize {
        json,
        model,
        md_out,
        use_cache,
        fetch_only,
        debug,
        since,
        login,
        clean,
        fetch_latest,
        last_updated,
    } = cmd
    else {
        return Ok(());
    };
    // The precedence the command has always applied: the read-only
    // queries, then login, then clean, then a run whose source is
    // `--json`, else the cache, else live.
    let command = if fetch_latest || last_updated {
        TwitterCommand::Latest { last_updated }
    } else if login {
        TwitterCommand::Login
    } else if clean {
        TwitterCommand::Clean
    } else {
        let source = match json {
            Some(path_or_dash) => TweetSource::Json(path_or_dash),
            None if use_cache => TweetSource::Cache,
            None => TweetSource::Live {
                since,
                debug,
                fetch_only,
            },
        };
        TwitterCommand::Summarize(TwitterSummarizeOpts {
            source,
            model,
            md_out,
        })
    };
    crate::cli_ztools_twitter::twitter_summarize(config, command)
}

/// Parse the CLI, resolve config, and dispatch to the tool handlers.
///
/// # Errors
///
/// When an explicitly named config cannot be read or parsed, and from
/// whichever subcommand was dispatched -- each states its own reason.
pub fn run() -> Result<()> {
    let cli = Cli::parse_from(args_with_implied_subcommand());
    let config = load_config(cli.config)?;
    match cli.cmd {
        Cmd::TwitterSummarize { .. } => dispatch_twitter(&config, cli.cmd),
        Cmd::WeekendPlan {
            location,
            md_out,
            fetch_latest,
            last_updated,
        } => {
            crate::cli_ztools::weekend_plan(&config, &location, md_out, fetch_latest, last_updated)
        }
        Cmd::ImageRenamer { dir, apply } => crate::cli_ztools::image_renamer(&config, &dir, apply),
        Cmd::Status => crate::cli_ztools::status(&config),
        Cmd::TwitterStatus => crate::ztools::twitter_status::run(),
        Cmd::ModelEval {
            model,
            suite,
            tasks_dir,
            task,
            json_output,
            capabilities,
            thinking,
            leaderboard,
            min_tasks,
            sort_by,
            csv_output,
            group_by_family,
            category,
            fail_on_regression,
        } => {
            let is_leaderboard = leaderboard
                || sort_by.is_some()
                || csv_output
                || group_by_family
                || category.is_some()
                || fail_on_regression.is_some();
            let action = if is_leaderboard {
                crate::cli_ztools::EvalAction::Leaderboard
            } else if capabilities {
                crate::cli_ztools::EvalAction::Capabilities
            } else {
                crate::cli_ztools::EvalAction::Run
            };
            let format = if json_output {
                crate::ztools::eval::LeaderboardFormat::Json
            } else if csv_output {
                crate::ztools::eval::LeaderboardFormat::Csv
            } else {
                crate::ztools::eval::LeaderboardFormat::Markdown
            };
            crate::cli_ztools::model_eval(
                &config,
                &model,
                &crate::cli_ztools::EvalOptions {
                    suite: &suite,
                    tasks_dir: tasks_dir.as_deref(),
                    task_filter: task.as_deref(),
                    json_output,
                    thinking,
                    action,
                    leaderboard_opts: crate::ztools::eval::LeaderboardOptions {
                        min_tasks,
                        sort_by: sort_by.as_deref(),
                        category: category.as_deref(),
                        format,
                        group_by_family,
                        fail_on_regression,
                    },
                },
            )
        }
        Cmd::EvalSignals {
            prune,
            path,
            tasks_dir,
            dry_run,
        } => crate::ztools::eval::cli_eval_signals(
            &config,
            path.as_deref(),
            tasks_dir.as_deref(),
            prune,
            dry_run,
        ),
    }
}

#[cfg(test)]
#[path = "cli_tests.rs"]
mod tests;
