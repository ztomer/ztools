//! `ztools models ...`: the local-model housekeeping commands.

use anyhow::Result;
use clap::Subcommand;

use crate::cli::Exit;
use crate::config::ZtoolsConfig;
use crate::ztools::refcheck::{self, LiveEnv, Report};

/// `ztools models ...`.
#[derive(Subcommand)]
pub enum ModelsCmd {
    /// Check that every model the Osaurus config and `[best_models]` name is
    /// installed, and repoint a missing Osaurus reference to the best installed
    /// member of its family.
    ///
    /// Exit 0 = clean, 3 = errors found and all repaired, 1 = still wrong.
    Refcheck {
        /// Repair a missing Osaurus reference (the default; accepted so a
        /// scheduled command can say so).
        #[arg(long, conflicts_with = "no_repair")]
        repair: bool,
        /// Only report; change nothing.
        #[arg(long)]
        no_repair: bool,
    },
}

/// The `[best_models]` table as the running config holds it.
fn best_models(config: &ZtoolsConfig) -> std::collections::BTreeMap<String, String> {
    [
        ("summarize", &config.twitter_model),
        ("json", &config.weekend_model),
        ("filename", &config.image_renamer_model),
        ("vlm", &config.image_renamer_vlm_model),
        ("think", &config.think_model),
    ]
    .into_iter()
    .map(|(slot, model)| (slot.to_string(), model.clone()))
    .collect()
}

/// Print a report and turn its exit status into the process's.
///
/// # Errors
///
/// An [`Exit`] carrying the report's non-zero status.
fn finish(report: &Report) -> Result<()> {
    for line in &report.lines {
        println!("{line}");
    }
    match report.exit_code() {
        0 => Ok(()),
        code => Err(Exit {
            code,
            reason: format!(
                "refcheck: {} missing, {} repaired, {} unresolved",
                report.errors, report.repaired, report.unresolved
            ),
        }
        .into()),
    }
}

/// Run a `models` subcommand.
///
/// # Errors
///
/// When osaurus cannot be run, or the check leaves a reference unresolved.
pub fn run(config: &ZtoolsConfig, cmd: &ModelsCmd) -> Result<()> {
    match cmd {
        ModelsCmd::Refcheck { no_repair, .. } => {
            let env = LiveEnv::from_home(best_models(config));
            finish(&refcheck::run(&env, !no_repair)?)
        }
    }
}

#[cfg(test)]
#[path = "cli_models_tests.rs"]
mod tests;
