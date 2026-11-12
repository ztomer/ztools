//! `ztools models nightly [--replay]` (LM.SELECT).

use anyhow::Result;
use std::collections::BTreeMap;
use std::path::Path;

use crate::cli::Exit;
use crate::config::ZtoolsConfig;
use crate::ztools::eval::task_fingerprint::TaskIdentities;
use crate::ztools::select::live::LiveNight;
use crate::ztools::select::nightly::{self, Outcome, findings_line};
use crate::ztools::select::{ReplayInput, Rule, replay as run_replay};

fn home() -> std::path::PathBuf {
    dirs::home_dir().unwrap_or_else(|| ".".into())
}

/// The installed models by short name, plus `foundation`, which Osaurus serves
/// without a directory.
fn installed_names(root: &Path) -> Vec<String> {
    let mut out: Vec<String> = crate::ztools::scout::local::installed(root)
        .iter()
        .map(|i| crate::ztools::refcheck::short_name(&i.repo))
        .collect();
    out.push("foundation".into());
    out.sort();
    out.dedup();
    out
}

/// Replay the rule; exit 1 unless it reproduces `[best_models]`.
///
/// # Errors
/// An [`Exit`] when the replay does not reproduce, or a fixture is unreadable.
pub fn replay(
    record: Option<&Path>,
    fixture: Option<&Path>,
    best: &BTreeMap<String, String>,
    current: &TaskIdentities,
) -> Result<()> {
    let input: ReplayInput = if let Some(path) = fixture {
        serde_json::from_str(&std::fs::read_to_string(path)?)?
    } else {
        ReplayInput {
            history: crate::ztools::eval::report::load_history_entries(None),
            current: current.clone(),
            installed: installed_names(&home().join("MLXModels")),
            best_models: best.clone(),
        }
    };
    if let Some(path) = record {
        std::fs::write(path, serde_json::to_string_pretty(&input)? + "\n")?;
    }
    let r = run_replay(&input, &Rule::DESIGN);
    for line in r.lines() {
        println!("{line}");
    }
    if r.reproduces(&input.best_models) {
        println!("✓ the replay reproduces [best_models]");
        Ok(())
    } else {
        Err(Exit {
            code: 1,
            reason: "the replay does not reproduce [best_models]".into(),
        }
        .into())
    }
}

/// Run one night and append its findings line.
///
/// # Errors
/// An [`Exit`] when the night failed.
pub fn night(
    config: &ZtoolsConfig,
    slots: BTreeMap<String, String>,
    current: TaskIdentities,
) -> Result<()> {
    let state_dir = home().join(".local/state/ztools/models");
    let mut live = LiveNight {
        hf: crate::ztools::scout::feeds::HF.to_string(),
        osaurus_url: config.osaurus_url.clone(),
        slots,
        current,
        models_root: home().join("MLXModels"),
        state_dir: state_dir.clone(),
        exe: std::env::current_exe()?,
    };
    finish_night(&mut live, &state_dir)
}

/// Run `live`'s night and append its findings line under `state_dir`.
///
/// # Errors
/// An [`Exit`] when the night failed, or the findings cannot be written.
pub fn finish_night(live: &mut LiveNight, state_dir: &Path) -> Result<()> {
    let outcome = nightly::run(live, &Rule::DESIGN);
    let finding = findings_line(
        &chrono::Local::now().format("%Y-%m-%d").to_string(),
        &outcome,
    );
    println!("{finding}");
    std::fs::create_dir_all(state_dir)?;
    let path = state_dir.join("findings.md");
    let mut text = std::fs::read_to_string(&path).unwrap_or_default();
    text.push_str(&finding);
    text.push('\n');
    std::fs::write(&path, text)?;
    match outcome {
        Outcome::Failed(why) => Err(Exit {
            code: 1,
            reason: why,
        }
        .into()),
        _ => Ok(()),
    }
}

#[cfg(test)]
#[path = "cli_models_nightly_tests.rs"]
mod tests;
