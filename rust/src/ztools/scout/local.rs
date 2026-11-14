//! What the scout knows of this Mac without asking the network.
//!
//! The slots, the installed models and their `model_type`, free disk, staged candidates,
//! and the Osaurus version. Repo ids only: no path of this machine is kept.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::Path;
use std::process::{Command, Stdio};

/// One installed model.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Installed {
    pub repo: String,
    pub model_type: Option<String>,
}

/// The machine's side of a scout run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Local {
    /// `[best_models]`: slot to short model name.
    pub slots: BTreeMap<String, String>,
    pub installed: Vec<Installed>,
    /// Free bytes on the models volume.
    pub free_bytes: Option<u64>,
    /// Candidates staged and not yet decided.
    pub staged: usize,
    /// The installed Osaurus app's version.
    pub osaurus_version: Option<String>,
}

fn subdirs(p: &Path) -> Vec<String> {
    let mut out: Vec<String> = std::fs::read_dir(p)
        .into_iter()
        .flatten()
        .flatten()
        .filter(|e| e.path().is_dir())
        .filter_map(|e| e.file_name().into_string().ok())
        .collect();
    out.sort();
    out
}

/// `<root>/<org>/<name>/config.json`, with the `model_type` it names.
#[must_use]
pub fn installed(root: &Path) -> Vec<Installed> {
    let mut out = Vec::new();
    for org in subdirs(root) {
        for name in subdirs(&root.join(&org)) {
            let Ok(text) = std::fs::read_to_string(root.join(&org).join(&name).join("config.json"))
            else {
                continue;
            };
            let model_type = serde_json::from_str::<serde_json::Value>(&text)
                .ok()
                .and_then(|v| v.get("model_type")?.as_str().map(str::to_string));
            out.push(Installed {
                repo: format!("{org}/{name}"),
                model_type,
            });
        }
    }
    out
}

/// Free bytes from `df -Pk` output: the fourth field of its second line.
#[must_use]
pub fn parse_df(out: &str) -> Option<u64> {
    let kb: u64 = out
        .lines()
        .nth(1)?
        .split_whitespace()
        .nth(3)?
        .parse()
        .ok()?;
    kb.checked_mul(1024)
}

/// Free bytes on the volume holding `path`.
#[must_use]
pub fn free_bytes(path: &Path) -> Option<u64> {
    let out = Command::new("df")
        .arg("-Pk")
        .arg(path)
        .stdin(Stdio::null())
        .output()
        .ok()?;
    parse_df(&String::from_utf8_lossy(&out.stdout))
}

/// `CFBundleShortVersionString` from an XML `Info.plist`.
#[must_use]
pub fn plist_version(xml: &str) -> Option<String> {
    let (_, rest) = xml.split_once("<key>CFBundleShortVersionString</key>")?;
    let (_, rest) = rest.split_once("<string>")?;
    let (v, _) = rest.split_once("</string>")?;
    Some(v.trim().to_string()).filter(|v| !v.is_empty())
}

/// The installed `/Applications/Osaurus.app`'s version.
#[must_use]
pub fn osaurus_version() -> Option<String> {
    std::fs::read_to_string("/Applications/Osaurus.app/Contents/Info.plist")
        .ok()
        .as_deref()
        .and_then(plist_version)
}

impl Local {
    /// This Mac: `~/MLXModels`, the staging directory under
    /// `~/.local/state/ztools/models/staged`, `/Applications/Osaurus.app`.
    #[must_use]
    pub fn from_home(slots: BTreeMap<String, String>) -> Self {
        let home = dirs::home_dir().unwrap_or_else(|| ".".into());
        let root = home.join("MLXModels");
        Self {
            slots,
            installed: installed(&root),
            free_bytes: free_bytes(&root),
            staged: subdirs(&home.join(".local/state/ztools/models/staged")).len(),
            osaurus_version: osaurus_version(),
        }
    }
}
