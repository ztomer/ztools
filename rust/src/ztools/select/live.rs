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
    /// The installed Osaurus version: a candidate recorded unloadable on it
    /// is not loaded again until it changes.
    pub osaurus_version: Option<String>,
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

    /// The autoswitch paths whose state is this night's.
    #[must_use]
    pub fn switch_paths(&self) -> crate::ztools::autoswitch::Paths {
        crate::ztools::autoswitch::Paths {
            state: self.state_dir.clone(),
            ..crate::ztools::autoswitch::Paths::under(&self.home())
        }
    }

    /// The home the models root hangs off (`~/MLXModels`).
    fn home(&self) -> PathBuf {
        self.models_root
            .parent()
            .map_or_else(|| PathBuf::from("."), Path::to_path_buf)
    }

    /// A switched-in candidate is no longer staged: its ledger moves to
    /// `switched/`, so the next night neither re-evaluates nor deletes it,
    /// and it stays this job's own directory.
    ///
    /// # Errors
    /// When the ledger cannot be moved.
    pub fn unstage(&self, repo: &str) -> std::io::Result<()> {
        let from = self.ledger_dir(repo);
        let to = self
            .state_dir
            .join("switched")
            .join(repo.replace('/', "--"));
        std::fs::create_dir_all(self.state_dir.join("switched"))?;
        std::fs::rename(from, to)
    }

    /// Why `name` (a repo id or its short name) is not to be loaded: it
    /// failed to load on this Osaurus, or its canary bar is still running.
    #[must_use]
    pub fn excluded(&self, name: &str) -> Option<String> {
        if let Some(why) =
            super::unloadable::why(&self.state_dir, name, self.osaurus_version.as_deref())
        {
            return Some(why);
        }
        let (now, _) = crate::ztools::autoswitch::clock();
        let short = crate::ztools::refcheck::short_name(name);
        crate::ztools::autoswitch::Ledger::load(&self.switch_paths())
            .ok()?
            .bar(&short, now)
    }

    /// The directories this job staged for `model` and then switched in:
    /// its weights and its ledger. `None` when it staged none of them.
    #[must_use]
    pub fn owned_dirs(&self, model: &str) -> Option<Vec<PathBuf>> {
        let switched = self.state_dir.join("switched");
        read_ledgers(&switched)
            .into_iter()
            .find(|s| s.model == model)
            .map(|s| {
                vec![
                    self.model_dir(&s.repo),
                    switched.join(s.repo.replace('/', "--")),
                ]
            })
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

/// The first candidate that passes every filter but disk and is not
/// `excluded` (unloadable here, or barred), and the slots it is a candidate for.
#[must_use]
pub fn first_nominee(p: &scout::Proposal, excluded: &dyn Fn(&str) -> bool) -> Option<Nominee> {
    let ok = |c: &&scout::Candidate| {
        !excluded(&c.repo)
            && c.provenance.passed()
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
        Ok(first_nominee(&p, &|repo| self.excluded(repo).is_some()))
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

    fn barred(&self, model: &str) -> Option<String> {
        self.excluded(model)
    }

    fn reject(&mut self, staged: &Staged, why: &str) -> Result<(), String> {
        let rec = super::unloadable::Unloadable {
            model: staged.model.clone(),
            osaurus: self.osaurus_version.clone(),
            date: self.date(),
            why: why.to_string(),
        };
        super::unloadable::record(&self.state_dir, &staged.repo, rec).map_err(|e| e.to_string())
    }

    fn osaurus_answers(&self) -> Result<(), String> {
        // The scout's own /health read, which takes the GPU lock itself.
        self.feed().get(scout::feeds::HEALTH).map(|_| ())
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
