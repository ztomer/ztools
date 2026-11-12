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
    /// Record the summarize model's Hebrew same-story calibration (LM.EMBEDHE):
    /// judge the Hebrew fixture's pairs under the GPU lock, print the verdict,
    /// and write the record.
    ///
    /// Exit 0 = every labelled pair yes and every trap and unrelated pair no.
    HebrewPairing {
        /// The fixture: `{"tweets": [{"group", "text"}, ...]}`.
        #[arg(long)]
        fixture: std::path::PathBuf,
        /// Where the record goes (JSON).
        #[arg(long)]
        out: std::path::PathBuf,
        /// Judge every pair, not just the prefilter's candidates and the traps.
        #[arg(long)]
        all_pairs: bool,
    },
}

/// Run and score one Hebrew judge calibration.
fn hebrew_pairing(
    config: &ZtoolsConfig,
    fixture: &std::path::Path,
    out: &std::path::Path,
    all_pairs: bool,
) -> Result<()> {
    use crate::ztools::embeddings::hebrew::record;
    let rows = record::hebrew_rows(&std::fs::read_to_string(fixture)?)?;
    let rec = record::run(&config.osaurus_url, &config.twitter_model, &rows, all_pairs)?;
    std::fs::write(out, serde_json::to_string_pretty(&rec)? + "\n")?;
    let v = record::verdict(&rec, &rows);
    println!(
        "{}: same {}/{}, traps {}/{}, unrelated {}/{} right; median {:.3}s a pair \
         (load {}); record {}",
        rec.model,
        v.same.right,
        v.same.pairs,
        v.trap.right,
        v.trap.pairs,
        v.unrelated.right,
        v.unrelated.pairs,
        v.median_seconds,
        rec.load,
        out.display()
    );
    for w in &v.wrong {
        println!("  ✗ {w}");
    }
    if !v.judge_alone.is_empty() {
        let n = v.judge_alone.len();
        let pairs = if n == 1 { "pair" } else { "pairs" };
        println!(
            "  → the model alone would also be wrong on {n} {pairs} the prefilter never sends it:"
        );
        for w in &v.judge_alone {
            println!("    {w}");
        }
    }
    if v.passes(&rows) {
        Ok(())
    } else {
        Err(Exit {
            code: 1,
            reason: format!("{} fails the Hebrew calibration", rec.model),
        }
        .into())
    }
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

/// The roster's task fingerprints as they stand now, the ones
/// `model-eval --leaderboard` scores against. A roster that cannot be loaded
/// leaves none, so every think score is reported as the fallback.
fn current_tasks(config: &ZtoolsConfig) -> crate::ztools::eval::TaskIdentities {
    let tasks_dir = config.eval_tasks_dir();
    config
        .eval_roster_inputs()
        .and_then(|inputs| crate::ztools::eval::load_all_eval_tasks(&inputs, tasks_dir.as_deref()))
        .map(|tasks| {
            tasks
                .iter()
                .map(|t| (t.name.clone(), crate::ztools::eval::task_fingerprint(t)))
                .collect()
        })
        .unwrap_or_default()
}

/// Run a `models` subcommand.
///
/// # Errors
///
/// When osaurus cannot be run, or the check leaves a reference unresolved.
pub fn run(config: &ZtoolsConfig, cmd: &ModelsCmd) -> Result<()> {
    match cmd {
        ModelsCmd::Refcheck { no_repair, .. } => {
            let env = LiveEnv::from_home(best_models(config), &current_tasks(config));
            finish(&refcheck::run(&env, !no_repair)?)
        }
        ModelsCmd::HebrewPairing {
            fixture,
            out,
            all_pairs,
        } => hebrew_pairing(config, fixture, out, *all_pairs),
    }
}

#[cfg(test)]
#[path = "cli_models_tests.rs"]
mod tests;
