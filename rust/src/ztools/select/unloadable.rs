//! Candidates that failed to load on this Osaurus (LM.RAPTOR07HANG).
//!
//! On 2026-11-14 the night staged `OsaurusAI/Raptor-0.7-4B-JANG_6M`, whose
//! `model_type` (`spark2_5`) the Osaurus catalog lists and whose sibling
//! Raptor-0.6 loads, and its first completion wedged Osaurus 1.0.3 whole:
//! even `/health` stopped answering until a restart. Nothing in a config
//! tells the two apart, so the fact is learned, not predicted: a candidate
//! whose own eval fails after its incumbent's passed is recorded here, its
//! weights deleted, and neither the scout's nominee nor a staged copy of it
//! is loaded again while the same Osaurus version is installed. A new Osaurus
//! version lifts the record, because the new one may load it.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

/// One candidate that failed to load.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Unloadable {
    /// Short name the eval addresses it by.
    pub model: String,
    /// The Osaurus version it failed on; `None` when it was unknown.
    pub osaurus: Option<String>,
    pub date: String,
    pub why: String,
}

/// `state/unloadable.json`: repo to its record.
#[must_use]
pub fn path(state_dir: &Path) -> PathBuf {
    state_dir.join("unloadable.json")
}

/// The records on disk; a missing or unreadable file is none.
#[must_use]
pub fn load(state_dir: &Path) -> BTreeMap<String, Unloadable> {
    std::fs::read_to_string(path(state_dir))
        .ok()
        .and_then(|t| serde_json::from_str(&t).ok())
        .unwrap_or_default()
}

/// Record `repo` as unloadable.
///
/// # Errors
/// When the state directory or the file cannot be written.
pub fn record(state_dir: &Path, repo: &str, rec: Unloadable) -> std::io::Result<()> {
    let mut all = load(state_dir);
    all.insert(repo.to_string(), rec);
    std::fs::create_dir_all(state_dir)?;
    let text = serde_json::to_string_pretty(&all).map_err(std::io::Error::other)?;
    let tmp = path(state_dir).with_extension("json.tmp");
    std::fs::write(&tmp, text + "\n")?;
    std::fs::rename(tmp, path(state_dir))
}

/// Why `name` (a repo id or its short name) may not be loaded on Osaurus
/// `osaurus`, if a record bars it. A record taken on another known version
/// does not bar it; an unknown version on either side does.
#[must_use]
pub fn why(state_dir: &Path, name: &str, osaurus: Option<&str>) -> Option<String> {
    let lower = name.to_lowercase();
    load(state_dir)
        .into_iter()
        .find(|(repo, r)| {
            (repo.to_lowercase() == lower || r.model.to_lowercase() == lower)
                && match (r.osaurus.as_deref(), osaurus) {
                    (Some(a), Some(b)) => a == b,
                    _ => true,
                }
        })
        .map(|(repo, r)| {
            format!(
                "{repo} failed to load on Osaurus {} ({}: {})",
                r.osaurus.as_deref().unwrap_or("of unknown version"),
                r.date,
                r.why
            )
        })
}
