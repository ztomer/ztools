//! The real osaurus CLI, filesystem and Hugging Face behind [`Env`].

use super::{CmdOut, Env};
use crate::ztools::eval::leaderboard::task_slot;
use crate::ztools::eval::report::{HistoryEntry, default_eval_dir, load_history_entries};
use anyhow::{Context, Result};
use serde_json::Value;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Duration;

const HF_BASE: &str = "https://huggingface.co";
const HF_TIMEOUT: Duration = Duration::from_secs(5);

/// Every path and program the check touches, each replaceable by a test.
pub struct LiveEnv {
    pub osaurus: PathBuf,
    pub models_root: PathBuf,
    pub families: PathBuf,
    pub hf_base: String,
    pub best: BTreeMap<String, String>,
    pub scores: BTreeMap<String, f64>,
}

impl LiveEnv {
    /// The machine's own locations: `osaurus` on the PATH, `~/MLXModels`,
    /// `~/.local/state/ztools/models/families.json`, the eval history.
    #[must_use]
    pub fn from_home(best: BTreeMap<String, String>) -> Self {
        let home = dirs::home_dir().unwrap_or_else(|| PathBuf::from("."));
        Self {
            osaurus: PathBuf::from("osaurus"),
            models_root: home.join("MLXModels"),
            families: home.join(".local/state/ztools/models/families.json"),
            hf_base: HF_BASE.to_string(),
            best,
            scores: think_scores(&load_history_entries(Some(&default_eval_dir()))),
        }
    }

    fn osaurus_run(&self, args: &[&str], file: Option<&Path>) -> Result<CmdOut> {
        let mut cmd = Command::new(&self.osaurus);
        cmd.args(args);
        if let Some(f) = file {
            cmd.arg(f);
        }
        let out = cmd
            .output()
            .with_context(|| format!("cannot run {}", self.osaurus.display()))?;
        let mut text = String::from_utf8_lossy(&out.stdout).into_owned();
        text.push_str(&String::from_utf8_lossy(&out.stderr));
        Ok(CmdOut {
            code: out.status.code().unwrap_or(-1),
            text,
        })
    }

    fn with_doc(&self, verb: &str, doc: &Value) -> Result<CmdOut> {
        let file = tempfile::Builder::new()
            .prefix("refcheck-")
            .suffix(".json")
            .tempfile()?;
        std::fs::write(file.path(), serde_json::to_string(doc)?)?;
        self.osaurus_run(&["config", verb], Some(file.path()))
    }

    fn cached(&self) -> BTreeMap<String, String> {
        std::fs::read_to_string(&self.families)
            .ok()
            .and_then(|t| serde_json::from_str(&t).ok())
            .unwrap_or_default()
    }

    fn fetch_base(&self, repo: &str) -> Option<String> {
        let client = reqwest::blocking::Client::builder()
            .timeout(HF_TIMEOUT)
            .build()
            .ok()?;
        let resp = client
            .get(format!("{}/api/models/{repo}", self.hf_base))
            .send()
            .ok()?
            .error_for_status()
            .ok()?;
        let body: Value = resp.json().ok()?;
        body.get("tags")?
            .as_array()?
            .iter()
            .filter_map(Value::as_str)
            .filter_map(|t| t.strip_prefix("base_model:"))
            .filter_map(|t| t.rsplit(':').next())
            .find(|b| !b.is_empty())
            .map(str::to_string)
    }
}

impl Env for LiveEnv {
    fn export(&self) -> Result<Value> {
        let file = tempfile::Builder::new()
            .prefix("refcheck-")
            .suffix(".json")
            .tempfile()?;
        let out = self.osaurus_run(
            &["config", "export", "--format", "json", "-o"],
            Some(file.path()),
        )?;
        anyhow::ensure!(
            out.code == 0,
            "osaurus config export exited {}: {}",
            out.code,
            out.text.trim()
        );
        serde_json::from_str(&std::fs::read_to_string(file.path())?)
            .context("osaurus config export is not JSON")
    }

    fn osaurus_list(&self) -> Result<Vec<String>> {
        let out = self.osaurus_run(&["list"], None)?;
        anyhow::ensure!(out.code == 0, "osaurus list exited {}", out.code);
        Ok(out
            .text
            .lines()
            .map(|l| l.trim().to_string())
            .filter(|l| !l.is_empty())
            .collect())
    }

    fn disk_models(&self) -> Vec<String> {
        let subdirs = |p: &Path| -> Vec<String> {
            std::fs::read_dir(p)
                .into_iter()
                .flatten()
                .flatten()
                .filter(|e| e.path().is_dir())
                .filter_map(|e| e.file_name().into_string().ok())
                .collect()
        };
        let mut found = Vec::new();
        for org in subdirs(&self.models_root) {
            for name in subdirs(&self.models_root.join(&org)) {
                if self
                    .models_root
                    .join(&org)
                    .join(&name)
                    .join("config.json")
                    .is_file()
                {
                    found.push(format!("{org}/{name}"));
                }
            }
        }
        found
    }

    fn best_models(&self) -> BTreeMap<String, String> {
        self.best.clone()
    }

    fn base_model(&self, repo: &str) -> Option<String> {
        let mut cache = self.cached();
        if let Some(b) = cache.get(repo) {
            return Some(b.clone());
        }
        let base = self.fetch_base(repo)?;
        cache.insert(repo.to_string(), base.clone());
        if let Some(dir) = self.families.parent()
            && std::fs::create_dir_all(dir).is_ok()
            && let Ok(text) = serde_json::to_string_pretty(&cache)
        {
            // A cache that cannot be written only costs a refetch.
            let _ = std::fs::write(&self.families, text);
        }
        Some(base)
    }

    fn think_scores(&self) -> BTreeMap<String, f64> {
        self.scores.clone()
    }

    fn plan(&self, doc: &Value) -> Result<CmdOut> {
        self.with_doc("plan", doc)
    }

    fn apply(&self, doc: &Value) -> Result<CmdOut> {
        self.with_doc("apply", doc)
    }
}

/// Each model's think-slot score: the mean of its latest complete run that
/// has think rows. Not gated on task fingerprints: the think rows of every
/// model on this machine predate the current fingerprints, and a ranking of
/// the members of ONE family only needs them comparable to each other.
#[must_use]
pub fn think_scores(history: &BTreeMap<String, Vec<HistoryEntry>>) -> BTreeMap<String, f64> {
    let mut out = BTreeMap::new();
    for (model, entries) in history {
        let think: Vec<&HistoryEntry> = entries
            .iter()
            .filter(|e| e.complete && task_slot(&e.task) == Some("think"))
            .collect();
        let Some(latest) = think.iter().map(|e| e.timestamp).max_by(f64::total_cmp) else {
            continue;
        };
        let run: Vec<f64> = think
            .iter()
            .filter(|e| (e.timestamp - latest).abs() < 2.0)
            .map(|e| crate::units::signed(e.score))
            .collect();
        out.insert(
            model.to_lowercase(),
            run.iter().sum::<f64>() / crate::units::count(run.len()),
        );
    }
    out
}
