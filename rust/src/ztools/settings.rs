//! The operator's toggles, and the retention they govern.
//!
//! ONE list of settings ([`SETTINGS`]), stored in
//! `~/.config/ztools/settings.toml` and changed through `ztools settings`. The
//! routines app renders the same list in its Settings window (the manifest's
//! `[settings]` command), so a toggle is declared here once and never
//! re-described by a UI that could drift from it.
//!
//! RETENTION. The stores keep a rolling seven days: after a run saves its
//! file, every `*.md` in that store last written more than [`KEEP_DAYS`] days
//! ago is deleted -- except the newest, so a store never goes empty under the
//! dashboard because the tool has not run for a week. Each store has its own
//! toggle, on by default.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime};

use anyhow::{Context, Result, bail};

/// How many days of output a store keeps when its toggle is on.
pub const KEEP_DAYS: u64 = 7;

/// One toggle the operator can flip.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Setting {
    pub key: &'static str,
    pub title: &'static str,
    pub help: &'static str,
    pub default: bool,
}

pub const TWITTER_RETENTION: &str = "twitter.rolling_7_days";
pub const WEEKEND_RETENTION: &str = "weekend.rolling_7_days";

/// Every setting there is. A key not listed here is refused, so a typo can
/// never be stored and silently ignored.
pub const SETTINGS: &[Setting] = &[
    Setting {
        key: TWITTER_RETENTION,
        title: "Keep 7 days of Twitter summaries",
        help: "Delete summaries older than 7 days after each run (the newest is always kept)",
        default: true,
    },
    Setting {
        key: WEEKEND_RETENTION,
        title: "Keep 7 days of weekend plans",
        help: "Delete plans older than 7 days after each run (the newest is always kept)",
        default: true,
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

fn stored(path: &Path) -> Result<BTreeMap<String, bool>> {
    match std::fs::read_to_string(path) {
        Ok(text) => {
            toml::from_str(&text).with_context(|| format!("{} is not valid", path.display()))
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(BTreeMap::new()),
        Err(e) => Err(e).with_context(|| format!("cannot read {}", path.display())),
    }
}

/// The value of `key` in the file at `path`, or its default.
///
/// # Errors
///
/// An unknown key, or a file that exists but cannot be read or parsed.
pub fn get_at(path: &Path, key: &str) -> Result<bool> {
    let s = setting(key)?;
    Ok(stored(path)?.get(key).copied().unwrap_or(s.default))
}

/// Store `value` for `key` in the file at `path`.
///
/// # Errors
///
/// An unknown key, or the file cannot be read, parsed or written.
pub fn set_at(path: &Path, key: &str, value: bool) -> Result<()> {
    setting(key)?;
    let mut all = stored(path)?;
    all.insert(key.to_string(), value);
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    std::fs::write(path, toml::to_string(&all)?)
        .with_context(|| format!("cannot write {}", path.display()))
}

/// Every setting with its current value, as the JSON `routines` renders.
///
/// # Errors
///
/// The file exists but cannot be read or parsed.
pub fn list_json_at(path: &Path) -> Result<serde_json::Value> {
    let all = stored(path)?;
    Ok(serde_json::Value::Array(
        SETTINGS
            .iter()
            .map(|s| {
                serde_json::json!({
                    "key": s.key, "title": s.title, "help": s.help, "kind": "bool",
                    "default": s.default,
                    "value": all.get(s.key).copied().unwrap_or(s.default),
                })
            })
            .collect(),
    ))
}

/// `ztools settings [--json] | get KEY | set KEY true|false`.
///
/// # Errors
///
/// A malformed command, an unknown key, or the settings file failing.
pub fn cli(args: &[String], json: bool) -> Result<()> {
    let path = settings_path();
    match args {
        [] if json => println!("{}", list_json_at(&path)?),
        [] => {
            for s in SETTINGS {
                let on = get_at(&path, s.key)?;
                println!("{} {}  {}", if on { "✓" } else { "·" }, s.key, s.title);
            }
        }
        [verb, key] if verb == "get" => println!("{}", get_at(&path, key)?),
        [verb, key, value] if verb == "set" => {
            let value = match value.as_str() {
                "true" | "on" => true,
                "false" | "off" => false,
                other => bail!("`{other}` is not true or false"),
            };
            set_at(&path, key, value)?;
            println!("✓ {key} = {value}");
        }
        _ => bail!("usage: ztools settings [--json] | get KEY | set KEY true|false"),
    }
    Ok(())
}

/// Delete every `*.md` directly in `dir` last modified before `now - keep`,
/// never the newest one. Returns what was deleted. Subdirectories (`rejected/`
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

/// After a run saved into `dir`: prune it when `key`'s toggle is on, and say
/// what went. A settings file that cannot be read is said and skipped --
/// retention must never fail the run whose output it tidies.
pub fn retain(dir: &Path, key: &str) {
    match get_at(&settings_path(), key) {
        Ok(true) => {
            let gone = prune(dir, Duration::from_hours(KEEP_DAYS * 24), SystemTime::now());
            if !gone.is_empty() {
                println!(
                    "→ removed {} file(s) older than {KEEP_DAYS} days ({key})",
                    gone.len()
                );
            }
        }
        Ok(false) => {}
        Err(e) => eprintln!("⚠ retention skipped: {e:#}"),
    }
}

#[cfg(test)]
#[path = "settings_tests.rs"]
mod tests;
