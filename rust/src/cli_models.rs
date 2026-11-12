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
    /// Point Osaurus's `model_defaults.core_model` and `default_agent.model` at
    /// the think slot's model: each key through a minimal one-section document
    /// whose plan must be that one key, with a rollback document that plans
    /// clean first (saved under `~/.local/state/ztools/models/`). Never prunes.
    ///
    /// Exit 0 = done or nothing to do, 1 = refused or failed.
    FollowThink {
        /// Plan only; apply nothing.
        #[arg(long)]
        dry_run: bool,
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
    /// Propose who could replace each slot's model (LM.SCOUT): read the HF Hub
    /// and the Osaurus catalog, filter by publisher, format, memory fit and
    /// disk. PROPOSE-ONLY: it never downloads, installs or switches a model.
    ///
    /// Exit 0 = every feed answered, 3 = some read `unverified`.
    Scout {
        /// Record every feed's (reduced) answer and the local facts into this
        /// directory as `feeds.jsonl` and `local.json`.
        #[arg(long, conflicts_with = "fixtures")]
        record: Option<std::path::PathBuf>,
        /// Read a recorded directory instead of the network and this Mac.
        #[arg(long)]
        fixtures: Option<std::path::PathBuf>,
    },
    /// One candidate a night (LM.SELECT), PROPOSE-ONLY: stage the scout's
    /// candidate under the 15 GiB disk floor (or record `no room`), evaluate it
    /// and each incumbent it would replace 3 times on the slot's tasks and the
    /// refusal probe, decide by the rule, and delete a rejected candidate.
    ///
    /// `--replay` instead proves the rule: replaying `eval_history.json` must
    /// reproduce `[best_models]` exactly. Exit 0 = reproduced / decided,
    /// 1 = not reproduced / the night failed.
    Nightly {
        /// Replay the rule over history instead of running a night.
        #[arg(long)]
        replay: bool,
        /// With `--replay`: write the replay's input to this JSON file.
        #[arg(long, requires = "replay", conflicts_with = "fixture")]
        record: Option<std::path::PathBuf>,
        /// With `--replay`: read a recorded input instead of this Mac.
        #[arg(long, requires = "replay")]
        fixture: Option<std::path::PathBuf>,
    },
    /// Restore the selection file (`~/.config/ztools/config.toml
    /// [best_models]`) as it was before the latest automatic switch, bar that
    /// candidate if its canary was still open, and record it in the ledger.
    Rollback,
}

/// Run the scout live, recorded, or over a recorded directory.
fn scout(
    config: &ZtoolsConfig,
    record: Option<&std::path::Path>,
    fixtures: Option<&std::path::Path>,
) -> Result<()> {
    use crate::ztools::scout::{self, Fixture, Live, Local};
    let proposal = if let Some(dir) = fixtures {
        let local: Local = serde_json::from_str(&std::fs::read_to_string(dir.join("local.json"))?)?;
        scout::scout(&Fixture::load(&dir.join("feeds.jsonl"))?, &local)
    } else {
        let local = Local::from_home(best_models(config));
        let record = match record {
            Some(dir) => {
                std::fs::create_dir_all(dir)?;
                std::fs::write(
                    dir.join("local.json"),
                    serde_json::to_string_pretty(&local)? + "\n",
                )?;
                let feeds = dir.join("feeds.jsonl");
                std::fs::write(&feeds, "")?;
                Some(feeds)
            }
            None => None,
        };
        let live = Live {
            hf: scout::feeds::HF.to_string(),
            osaurus_url: config.osaurus_url.clone(),
            record,
        };
        scout::scout(&live, &local)
    };
    for line in scout::render(&proposal) {
        println!("{line}");
    }
    if proposal.refused.is_empty() {
        Ok(())
    } else {
        Err(Exit {
            code: 3,
            reason: format!(
                "scout: {} of {} feeds unverified",
                proposal.refused.len(),
                proposal.asked
            ),
        }
        .into())
    }
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

/// Print a follow-think report, save its rollback documents, and exit by it.
///
/// # Errors
///
/// When the rollback cannot be saved, or an [`Exit`] for a refusal.
fn follow_finish(report: &refcheck::FollowReport) -> Result<()> {
    for line in &report.lines {
        println!("{line}");
    }
    if !report.rollback.is_empty() {
        let dir = dirs::home_dir()
            .unwrap_or_default()
            .join(".local/state/ztools/models");
        std::fs::create_dir_all(&dir)?;
        for (i, doc) in report.rollback.iter().enumerate() {
            let path = dir.join(format!("osaurus_rollback_{i}.json"));
            std::fs::write(&path, serde_json::to_string_pretty(doc)?)?;
            println!("  · rollback: osaurus config plan {}", path.display());
        }
    }
    match report.exit_code() {
        0 => Ok(()),
        code => Err(Exit {
            code,
            reason: "follow-think refused or failed".into(),
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
            let env = LiveEnv::from_home(
                &config.osaurus_url,
                best_models(config),
                &current_tasks(config),
            );
            finish(&refcheck::run(&env, !no_repair)?)
        }
        ModelsCmd::FollowThink { dry_run } => {
            let env = LiveEnv::from_home(
                &config.osaurus_url,
                best_models(config),
                &current_tasks(config),
            );
            follow_finish(&refcheck::follow_think(&env, !dry_run)?)
        }
        ModelsCmd::HebrewPairing {
            fixture,
            out,
            all_pairs,
        } => hebrew_pairing(config, fixture, out, *all_pairs),
        ModelsCmd::Scout { record, fixtures } => {
            scout(config, record.as_deref(), fixtures.as_deref())
        }
        ModelsCmd::Rollback => crate::cli_models_nightly::rollback(),
        ModelsCmd::Nightly {
            replay,
            record,
            fixture,
        } => {
            if *replay {
                crate::cli_models_nightly::replay(
                    record.as_deref(),
                    fixture.as_deref(),
                    &best_models(config),
                    &current_tasks(config),
                )
            } else {
                crate::cli_models_nightly::night(config, best_models(config), current_tasks(config))
            }
        }
    }
}

#[cfg(test)]
#[path = "cli_models_tests.rs"]
mod tests;
