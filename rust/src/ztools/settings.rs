//! The operator's toggles, and the retention they govern.
//!
//! ONE list of settings ([`SETTINGS`]), stored in
//! `~/.config/ztools/settings.toml` and changed through `ztools settings`. The
//! routines app renders the same list in its Settings window (the manifest's
//! `[settings]` command), so a toggle is declared here once and never
//! re-described by a UI that could drift from it.
//!
//! RETENTION. Each store keeps the last N days, N chosen per store from
//! [`KEEP_DAYS_OPTIONS`] (default [`DEFAULT_KEEP_DAYS`]): after a run saves its
//! file, every `*.md` in that store last written more than N days ago is
//! deleted -- except the newest, so a store never goes empty under the
//! dashboard. 0 keeps only the newest file.
//!
//! The settings were on/off toggles (`*.rolling_7_days`) until 2026-10-10. A
//! stored `true` still reads as 7 days, which is what it meant; a stored
//! `false` meant "keep everything", which no day count says, so it is refused
//! by name and the store is left alone until a count is chosen.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime};

use anyhow::{Context, Result, bail};

/// The day counts a store can keep. 0 keeps only the newest file.
pub const KEEP_DAYS_OPTIONS: &[u64] = &[0, 1, 3, 7, 14, 30];

/// What a store keeps until the operator chooses.
pub const DEFAULT_KEEP_DAYS: u64 = 7;

/// One setting the operator can choose.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Setting {
    pub key: &'static str,
    /// The sentence the Settings window shows; `{}` is where the choice sits.
    pub title: &'static str,
    /// The pre-2026-10-10 on/off key this setting replaced.
    pub legacy: &'static str,
}

pub const TWITTER_RETENTION: &str = "twitter.keep_days";
pub const WEEKEND_RETENTION: &str = "weekend.keep_days";

/// Every setting there is. A key not listed here is refused, so a typo can
/// never be stored and silently ignored.
pub const SETTINGS: &[Setting] = &[
    Setting {
        key: TWITTER_RETENTION,
        title: "Keep Twitter summaries for {} days",
        legacy: "twitter.rolling_7_days",
    },
    Setting {
        key: WEEKEND_RETENTION,
        title: "Keep weekend plans for {} days",
        legacy: "weekend.rolling_7_days",
    },
];

/// `~/.config/ztools/settings.toml`.
#[must_use]
pub fn settings_path() -> PathBuf {
    dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".config/ztools/settings.toml")
}

fn setting(key: &str) -> Result<&'static Setting> {
    SETTINGS.iter().find(|s| s.key == key).with_context(|| {
        let known: Vec<&str> = SETTINGS.iter().map(|s| s.key).collect();
        format!("no setting `{key}` (known: {})", known.join(", "))
    })
}

fn stored(path: &Path) -> Result<BTreeMap<String, toml::Value>> {
    match std::fs::read_to_string(path) {
        Ok(text) => {
            toml::from_str(&text).with_context(|| format!("{} is not valid", path.display()))
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(BTreeMap::new()),
        Err(e) => Err(e).with_context(|| format!("cannot read {}", path.display())),
    }
}

/// The day count `key` holds in the file at `path`, or its default.
///
/// # Errors
///
/// An unknown key; a file that exists but cannot be read or parsed; a stored
/// count that is not one of [`KEEP_DAYS_OPTIONS`]; or a legacy toggle stored
/// off, which meant "keep everything" and has no day count.
pub fn get_at(path: &Path, key: &str) -> Result<u64> {
    let s = setting(key)?;
    let all = stored(path)?;
    if let Some(v) = all.get(key) {
        let days = v
            .as_integer()
            .and_then(|n| u64::try_from(n).ok())
            .filter(|n| KEEP_DAYS_OPTIONS.contains(n));
        return days.with_context(|| format!("`{key}` = {v} is not one of {KEEP_DAYS_OPTIONS:?}"));
    }
    match all.get(s.legacy).and_then(toml::Value::as_bool) {
        Some(true) | None => Ok(DEFAULT_KEEP_DAYS),
        Some(false) => bail!(
            "`{}` = false (keep everything) has no day count; choose one with \
             `ztools settings set {key} N`",
            s.legacy
        ),
    }
}

/// Store `days` for `key` in the file at `path`, dropping its legacy toggle.
///
/// # Errors
///
/// An unknown key, a count not in [`KEEP_DAYS_OPTIONS`], or the file cannot be
/// read, parsed or written.
pub fn set_at(path: &Path, key: &str, days: u64) -> Result<()> {
    let s = setting(key)?;
    if !KEEP_DAYS_OPTIONS.contains(&days) {
        bail!("{days} is not one of {KEEP_DAYS_OPTIONS:?}");
    }
    let mut all = stored(path)?;
    all.remove(s.legacy);
    all.insert(key.to_string(), toml::Value::Integer(i64::try_from(days)?));
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    std::fs::write(path, toml::to_string(&all)?)
        .with_context(|| format!("cannot write {}", path.display()))
}

/// Every setting with its current value, as the JSON `routines` renders: a
/// `choice` whose options are the day counts.
///
/// # Errors
///
/// Any setting's value cannot be read (see [`get_at`]).
pub fn list_json_at(path: &Path) -> Result<serde_json::Value> {
    let options: Vec<String> = KEEP_DAYS_OPTIONS.iter().map(u64::to_string).collect();
    SETTINGS
        .iter()
        .map(|s| {
            Ok(serde_json::json!({
                "key": s.key, "title": s.title, "kind": "choice",
                "value": get_at(path, s.key)?.to_string(),
                "default": DEFAULT_KEEP_DAYS.to_string(),
                "options": options,
            }))
        })
        .collect::<Result<Vec<_>>>()
        .map(serde_json::Value::Array)
}

/// `ztools settings [--json] | get KEY | set KEY DAYS`.
///
/// # Errors
///
/// A malformed command, an unknown key or count, or the settings file failing.
pub fn cli(args: &[String], json: bool) -> Result<()> {
    let path = settings_path();
    match args {
        [] if json => println!("{}", list_json_at(&path)?),
        [] => {
            for s in SETTINGS {
                let days = get_at(&path, s.key)?;
                println!("· {}  {}", s.key, s.title.replace("{}", &days.to_string()));
            }
        }
        [verb, key] if verb == "get" => println!("{}", get_at(&path, key)?),
        [verb, key, value] if verb == "set" => {
            let days: u64 = value
                .parse()
                .with_context(|| format!("`{value}` is not a day count"))?;
            set_at(&path, key, days)?;
            println!("✓ {key} = {days}");
        }
        _ => bail!("usage: ztools settings [--json] | get KEY | set KEY DAYS"),
    }
    Ok(())
}

/// Delete every `*.md` directly in `dir` last modified before `now - keep`.
///
/// Never the newest one. Returns what was deleted. Subdirectories (`rejected/`
/// and the like) keep their own bounds and are not entered.
#[must_use]
pub fn prune(dir: &Path, keep: Duration, now: SystemTime) -> Vec<PathBuf> {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return Vec::new();
    };
    let mut files: Vec<(SystemTime, PathBuf)> = entries
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.is_file() && p.extension().is_some_and(|e| e == "md"))
        .filter_map(|p| Some((std::fs::metadata(&p).ok()?.modified().ok()?, p)))
        .collect();
    files.sort();
    files.pop(); // the newest stays, whatever its age
    let Some(cutoff) = now.checked_sub(keep) else {
        return Vec::new();
    };
    files
        .into_iter()
        .filter(|(modified, _)| *modified < cutoff)
        .filter_map(|(_, p)| std::fs::remove_file(&p).ok().map(|()| p))
        .collect()
}

/// After a run saved into `dir`: prune it to `key`'s day count, and say what
/// went.
///
/// A setting that cannot be read is said and skipped -- retention must never
/// fail the run whose output it tidies, nor delete on a guess.
pub fn retain(dir: &Path, key: &str) {
    match get_at(&settings_path(), key) {
        Ok(days) => {
            let gone = prune(dir, Duration::from_hours(days * 24), SystemTime::now());
            if !gone.is_empty() {
                println!(
                    "→ removed {} file(s) older than {days} days ({key})",
                    gone.len()
                );
            }
        }
        Err(e) => eprintln!("⚠ retention skipped: {e:#}"),
    }
}

#[cfg(test)]
#[path = "settings_tests.rs"]
mod tests;
