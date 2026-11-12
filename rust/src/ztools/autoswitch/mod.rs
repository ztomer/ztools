//! LM.AUTOSWITCH (LM.DESIGN section 6, phase 5; owner-accepted D5 2026-11-11):
//! the machine-local selection file, the canary, the rollback and the kill
//! switch.
//!
//! A switch writes one slot of `~/.config/ztools/config.toml [best_models]`
//! (the table that outranks the shipped `conf/config.toml` as a whole,
//! `config.rs` `with_ztools_best_models`), records the previous file in the
//! ledger, and opens a canary: the next [`CANARY_RUNS`] real runs of the
//! slot's consumer must succeed, pass the consumer's own gate and stay within
//! the speed floor. Any failure restores the previous selection in that same
//! run and bars the candidate for [`BAR_DAYS`] days.
//!
//! It SHIPS PROPOSE-ONLY: with no `~/.config/ztools/models_auto` file, a
//! switch the rule chose is only proposed. `on` in that file is the owner's
//! act that turns the automatic mode on; `off` there, or `ZTOOLS_MODEL_AUTO=0`,
//! forces propose-only whatever the file says. The repo is never edited.
//! Embedding models are never switched: a threshold is a measured artifact.

pub mod canary;
pub mod night;
pub mod selection;
pub mod switch;

#[cfg(test)]
mod live_tests;
#[cfg(test)]
mod tests;

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

/// Consumer runs a candidate must pass before it is confirmed.
pub const CANARY_RUNS: u32 = 3;
/// Days a candidate whose canary failed is barred.
pub const BAR_DAYS: f64 = 30.0;
/// Days between two switches of one slot.
pub const SLOT_DAYS: f64 = 7.0;
/// The slots this job may switch: ztools' five. Nothing else, and never an
/// embedding model.
pub const SWITCHABLE: &[&str] = &["think", "json", "summarize", "filename", "vlm"];

const DAY: f64 = 86_400.0;

/// Epoch seconds and the local date, now.
#[must_use]
pub fn clock() -> (f64, String) {
    let at = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0.0, |d| d.as_secs_f64());
    (at, chrono::Local::now().format("%Y-%m-%d").to_string())
}

/// Every path this job reads or writes, under one home.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Paths {
    /// `~/.config/ztools/config.toml`: the selection file.
    pub selection: PathBuf,
    /// `~/.config/ztools/models_auto`: the kill switch.
    pub kill: PathBuf,
    /// `~/.local/state/ztools/models`.
    pub state: PathBuf,
}

impl Paths {
    #[must_use]
    pub fn under(home: &Path) -> Self {
        Self {
            selection: home.join(".config/ztools/config.toml"),
            kill: home.join(".config/ztools/models_auto"),
            state: home.join(".local/state/ztools/models"),
        }
    }

    /// The running user's home.
    #[must_use]
    pub fn from_home() -> Self {
        Self::under(&dirs::home_dir().unwrap_or_else(|| ".".into()))
    }

    #[must_use]
    pub fn ledger(&self) -> PathBuf {
        self.state.join("switch_ledger.json")
    }

    /// The previous selection file, kept beside the ledger for a human.
    #[must_use]
    pub fn previous(&self) -> PathBuf {
        self.state.join("selection.prev.toml")
    }

    #[must_use]
    pub fn findings(&self) -> PathBuf {
        self.state.join("findings.md")
    }

    /// Append one line to the findings file.
    ///
    /// # Errors
    /// When the state directory or the file cannot be written.
    pub fn append_finding(&self, line: &str) -> std::io::Result<()> {
        std::fs::create_dir_all(&self.state)?;
        let mut text = std::fs::read_to_string(self.findings()).unwrap_or_default();
        text.push_str(line);
        text.push('\n');
        std::fs::write(self.findings(), text)
    }
}

/// Whether a switch the rule chose is made or only proposed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Mode {
    Auto,
    /// Why it is propose-only.
    Propose(String),
}

/// The mode, given the kill-switch file and `ZTOOLS_MODEL_AUTO`'s value.
#[must_use]
pub fn mode_from(paths: &Paths, env: Option<&str>) -> Mode {
    if env.map(str::trim) == Some("0") {
        return Mode::Propose("ZTOOLS_MODEL_AUTO=0".into());
    }
    match std::fs::read_to_string(&paths.kill).map(|t| t.trim().to_ascii_lowercase()) {
        Ok(t) if t == "on" => Mode::Auto,
        Ok(t) if t == "off" => Mode::Propose(format!("{} says off", paths.kill.display())),
        Ok(t) => Mode::Propose(format!(
            "{} says {t:?}, not on: propose-only",
            paths.kill.display()
        )),
        Err(_) => Mode::Propose("propose-only until the owner turns it on (D5)".into()),
    }
}

/// The mode of this process.
#[must_use]
pub fn mode(paths: &Paths) -> Mode {
    mode_from(paths, std::env::var("ZTOOLS_MODEL_AUTO").ok().as_deref())
}

/// What happened, in order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Kind {
    Switch,
    CanaryPass,
    Confirm,
    Rollback,
}

/// One ledger row.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Event {
    pub at: f64,
    pub date: String,
    pub slot: String,
    pub kind: Kind,
    pub from: String,
    pub to: String,
    #[serde(default)]
    pub why: String,
    /// On a switch: the selection file before it (`None`: there was none).
    #[serde(default)]
    pub previous: Option<String>,
    /// On a switch: the file this switch wrote, so a rollback knows whether
    /// anything has touched it since.
    #[serde(default)]
    pub written: Option<String>,
}

/// The open canary: at most one at a time (one staged candidate).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Canary {
    pub slot: String,
    pub candidate: String,
    pub incumbent: String,
    pub opened: f64,
    pub passed: u32,
}

/// The history ledger.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Ledger {
    #[serde(default)]
    pub events: Vec<Event>,
    #[serde(default)]
    pub canary: Option<Canary>,
    /// Model to the time its bar lifts.
    #[serde(default)]
    pub barred: BTreeMap<String, f64>,
}

impl Ledger {
    /// The ledger on disk; none is an empty one.
    ///
    /// # Errors
    /// When the file exists but is not a ledger: never read as empty, or a
    /// rollback would forget what it must restore.
    pub fn load(paths: &Paths) -> anyhow::Result<Self> {
        match std::fs::read_to_string(paths.ledger()) {
            Ok(t) => serde_json::from_str(&t)
                .map_err(|e| anyhow::anyhow!("{}: {e}", paths.ledger().display())),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(Self::default()),
            Err(e) => Err(anyhow::anyhow!("{}: {e}", paths.ledger().display())),
        }
    }

    /// # Errors
    /// When the state directory or the file cannot be written.
    pub fn save(&self, paths: &Paths) -> anyhow::Result<()> {
        std::fs::create_dir_all(&paths.state)?;
        let tmp = paths.ledger().with_extension("json.tmp");
        std::fs::write(&tmp, serde_json::to_string_pretty(self)? + "\n")?;
        std::fs::rename(&tmp, paths.ledger())?;
        Ok(())
    }

    /// Why `model` may not be switched in now, if it is barred.
    #[must_use]
    pub fn bar(&self, model: &str, now: f64) -> Option<String> {
        self.barred
            .get(model)
            .filter(|until| **until > now)
            .map(|until| {
                let days = ((until - now) / DAY).ceil();
                let unit = if days <= 1.0 { "day" } else { "days" };
                format!("{model} is barred for {days:.0} more {unit}")
            })
    }

    /// The last switch of `slot` (a rollback is not one).
    #[must_use]
    pub fn last_switch(&self, slot: &str) -> Option<&Event> {
        self.events
            .iter()
            .rev()
            .find(|e| e.slot == slot && e.kind == Kind::Switch)
    }
}
