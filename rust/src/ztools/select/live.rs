//! The real machine behind [`Night`].
//!
//! The scout over the HF Hub and Osaurus's HTTP `/health`, downloads over HTTPS (never the Osaurus CLI), the eval as
//! `ztools model-eval` (which takes the GPU lock and loads through the
//! residency seam), and a ledger that names every directory this job staged.

use super::nightly::{Night, Nominee, Staged};
use crate::ztools::eval::report::{HistoryEntry, load_history_entries};
use crate::ztools::eval::task_fingerprint::TaskIdentities;
use crate::ztools::scout::rules::File;
use crate::ztools::scout::{self, Feed, Live, Local};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

/// This Mac.
pub struct LiveNight {
    /// The HF Hub base, `https://huggingface.co` live.
    pub hf: String,
    pub osaurus_url: String,
    pub slots: BTreeMap<String, String>,
    pub current: TaskIdentities,
    /// `~/MLXModels`.
    pub models_root: PathBuf,
    /// `~/.local/state/ztools/models`.
    pub state_dir: PathBuf,
    /// The `ztools` binary that runs the eval.
    pub exe: PathBuf,
}

impl LiveNight {
    /// The ledger directory of one staged repo.
    #[must_use]
    pub fn ledger_dir(&self, repo: &str) -> PathBuf {
        self.state_dir.join("staged").join(repo.replace('/', "--"))
    }

    /// Where a repo's weights go.
    #[must_use]
    pub fn model_dir(&self, repo: &str) -> PathBuf {
        self.models_root.join(repo)
    }

    fn partial(&self, repo: &str) -> PathBuf {
        self.models_root.join(format!("{repo}.partial"))
    }

    fn feed(&self) -> Live {
        Live {
            hf: self.hf.clone(),
            osaurus_url: self.osaurus_url.clone(),
            record: None,
        }
    }

    /// Record `staged` before any byte lands, so a crashed night's directory
    /// is still this job's to delete.
    ///
    /// # Errors
    /// When the ledger cannot be written.
    pub fn write_ledger(&self, staged: &Staged) -> std::io::Result<()> {
        let dir = self.ledger_dir(&staged.repo);
        std::fs::create_dir_all(&dir)?;
        let text = serde_json::to_string_pretty(staged).map_err(std::io::Error::other)?;
        std::fs::write(dir.join("ledger.json"), text + "\n")
    }
}

/// The first candidate that passes every filter but disk, and the slots it is
/// a candidate for.
#[must_use]
pub fn first_nominee(p: &scout::Proposal) -> Option<Nominee> {
    let ok = |c: &&scout::Candidate| {
        c.provenance.passed()
            && c.format.as_ref().is_some_and(scout::Check::passed)
            && c.fit.as_ref().is_some_and(scout::Check::passed)
    };
    let repo = p
        .incumbents
        .iter()
        .flat_map(|i| &i.candidates)
        .find(ok)?
        .repo
        .clone();
    let slots = p
        .incumbents
        .iter()
        .filter(|i| i.candidates.iter().any(|c| c.repo == repo))
        .flat_map(|i| i.slots.clone())
        .collect();
    Some(Nominee { repo, slots })
}

fn fetch_to(url: &str, dest: &Path) -> Result<(), String> {
    if let Some(dir) = dest.parent() {
        std::fs::create_dir_all(dir).map_err(|e| e.to_string())?;
    }
    let client = reqwest::blocking::Client::builder()
        .timeout(None)
        .build()
        .map_err(|e| e.to_string())?;
    let mut resp = client
        .get(url)
        .send()
        .and_then(reqwest::blocking::Response::error_for_status)
        .map_err(|e| e.to_string())?;
    let mut out = std::fs::File::create(dest).map_err(|e| e.to_string())?;
    resp.copy_to(&mut out).map_err(|e| e.to_string())?;
    Ok(())
}

impl Night for LiveNight {
    fn now(&self) -> f64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0.0, |d| d.as_secs_f64())
    }

    fn date(&self) -> String {
        chrono::Local::now().format("%Y-%m-%d").to_string()
    }

    fn free_bytes(&self) -> Option<u64> {
        scout::local::free_bytes(&self.models_root)
    }

    fn staged(&self) -> Vec<Staged> {
        read_ledgers(&self.state_dir.join("staged"))
    }

    fn nominate(&mut self) -> Result<Option<Nominee>, String> {
        let p = scout::scout(&self.feed(), &Local::from_home(self.slots.clone()));
        Ok(first_nominee(&p))
    }

    fn files(&mut self, repo: &str) -> Result<Vec<File>, String> {
        let rows = self.feed().get(&scout::feeds::tree_key(repo))?;
        Ok(rows
            .as_array()
            .into_iter()
            .flatten()
            .filter_map(|f| {
                Some(File {
                    path: f.get("path")?.as_str()?.to_string(),
                    size: f.get("size")?.as_u64()?,
                })
            })
            .collect())
    }

    fn download(&mut self, staged: &Staged, files: &[File]) -> Result<(), String> {
        self.write_ledger(staged).map_err(|e| e.to_string())?;
        let partial = self.partial(&staged.repo);
        for f in files {
            let url = format!("{}/{}/resolve/main/{}", self.hf, staged.repo, f.path);
            fetch_to(&url, &partial.join(&f.path))?;
        }
        std::fs::rename(&partial, self.model_dir(&staged.repo)).map_err(|e| e.to_string())
    }

    fn eval(&mut self, model: &str, tasks: &[String]) -> Result<(), String> {
        let status = std::process::Command::new(&self.exe)
            .args(["model-eval", "--model", model, "--suite", "full", "--task"])
            .arg(tasks.join(","))
            .stdin(std::process::Stdio::null())
            .status()
            .map_err(|e| e.to_string())?;
        if status.success() {
            Ok(())
        } else {
            Err(format!("model-eval exited {status}"))
        }
    }

    fn history(&self) -> BTreeMap<String, Vec<HistoryEntry>> {
        load_history_entries(None)
    }

    fn current(&self) -> TaskIdentities {
        self.current.clone()
    }

    fn slots(&self) -> BTreeMap<String, String> {
        self.slots.clone()
    }

    fn delete(&mut self, staged: &Staged) -> Result<(), String> {
        delete_staged(self, staged)
    }
}

/// Every ledger under `dir`.
#[must_use]
pub fn read_ledgers(dir: &Path) -> Vec<Staged> {
    let mut out: Vec<Staged> = std::fs::read_dir(dir)
        .into_iter()
        .flatten()
        .flatten()
        .filter_map(|e| std::fs::read_to_string(e.path().join("ledger.json")).ok())
        .filter_map(|t| serde_json::from_str(&t).ok())
        .collect();
    out.sort_by(|a: &Staged, b| a.repo.cmp(&b.repo));
    out
}

/// `rm -rf` a staged candidate: only when its ledger is on disk, so a model
/// the owner installed is never this job's to delete.
///
/// # Errors
/// When there is no ledger for it, or a removal fails.
pub fn delete_staged(night: &LiveNight, staged: &Staged) -> Result<(), String> {
    let ledger = night.ledger_dir(&staged.repo);
    if !ledger.join("ledger.json").is_file() {
        return Err(format!(
            "{}: no ledger, not staged by this job",
            staged.repo
        ));
    }
    for dir in [night.model_dir(&staged.repo), night.partial(&staged.repo)] {
        if dir.exists() {
            std::fs::remove_dir_all(&dir).map_err(|e| format!("{}: {e}", dir.display()))?;
        }
    }
    std::fs::remove_dir_all(&ledger).map_err(|e| format!("{}: {e}", ledger.display()))
}
