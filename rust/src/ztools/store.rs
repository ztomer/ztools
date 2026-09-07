//! Read-side access to stored summaries and plans.
//!
//! The summarizer and planner WRITE dated `.md` files into a store directory
//! (the twitter side in its own `default_dir`, the weekend side via `--md-out`);
//! their `--fetch-latest` / `--last-updated` read the newest one back. Read-only
//! by construction: nothing here touches a model or the network, so a dashboard
//! tab can open on it without ever re-running the pipeline.

use anyhow::{bail, Result};
use std::path::{Path, PathBuf};
use std::time::SystemTime;

use chrono::Local;

/// Default directory the twitter summarizer stores its dated summaries in.
#[must_use]
pub fn twitter_store_dir() -> PathBuf {
    dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("Documents/twitter_summaries")
}

/// Default directory weekend plans are stored in (the weekly `--md-out`).
#[must_use]
pub fn weekend_store_dir() -> PathBuf {
    dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("Documents/weekend_plans")
}

/// Newest `*.md` in `dir` by modification time. Errors with a stated reason
/// when the directory is missing or holds no markdown at all, so a caller can
/// say *why* (rather than hand back an empty tab).
///
/// # Errors
///
/// When the store directory cannot be listed, and when it holds no summary
/// at all. "Nothing stored yet" is an error rather than an empty result
/// because every caller's next act is to print the file.
pub fn newest_md(dir: &Path) -> Result<PathBuf> {
    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(e) => bail!(
            "stored output directory {} is not readable: {e}",
            dir.display()
        ),
    };
    let mut newest: Option<(PathBuf, SystemTime)> = None;
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("md") {
            continue;
        }
        let Ok(modified) = std::fs::metadata(&path).and_then(|m| m.modified()) else {
            continue;
        };
        // Strictly-newer wins; on a tie a deterministic lexical fallback keeps
        // the result independent of readdir order.
        let replace = match &newest {
            Some((other, t)) => {
                modified > *t || (modified == *t && other.file_name() < path.file_name())
            }
            None => true,
        };
        if replace {
            newest = Some((path, modified));
        }
    }
    match newest {
        Some((path, _)) => Ok(path),
        None => bail!("no stored summaries found in {}", dir.display()),
    }
}

/// Format a file's modification time as `%Y-%m-%d %H:%M` (local).
///
/// # Errors
///
/// When the file's metadata cannot be read, or the platform does not report
/// a modification time for it.
pub fn last_updated(path: &Path) -> Result<String> {
    let modified = std::fs::metadata(path)?.modified()?;
    let dt: chrono::DateTime<Local> =
        chrono::DateTime::<chrono::Utc>::from(modified).with_timezone(&Local);
    Ok(dt.format("%Y-%m-%d %H:%M").to_string())
}

/// Read-side entry point for `twitter-summarize --fetch-latest` /
/// `--last-updated`. `TWITTER_OUTPUT_DIR` overrides the store dir, the same
/// override the status reader honours.
///
/// # Errors
///
/// As [`print_newest`]: no stored summary, or an unreadable one.
pub fn twitter_latest(show_time: bool) -> Result<()> {
    let dir =
        std::env::var("TWITTER_OUTPUT_DIR").map_or_else(|_| twitter_store_dir(), PathBuf::from);
    print_newest(&dir, show_time)
}

/// Read-side entry point for `weekend-plan --fetch-latest` / `--last-updated`;
/// `WEEKEND_OUTPUT_DIR` overrides the store dir.
///
/// # Errors
///
/// As [`print_newest`]: no stored summary, or an unreadable one.
pub fn weekend_latest(show_time: bool) -> Result<()> {
    let dir =
        std::env::var("WEEKEND_OUTPUT_DIR").map_or_else(|_| weekend_store_dir(), PathBuf::from);
    print_newest(&dir, show_time)
}

/// Resolve the newest stored summary (or its update time) and print it.
/// `store_dir` is the resolved store directory; `show_time` selects the
/// timestamp over the content.
///
/// # Errors
///
/// When the store holds no summary, when its modification time cannot be
/// read (with `show_time`), or when the file itself cannot be read.
pub fn print_newest(store_dir: &Path, show_time: bool) -> Result<()> {
    let path = newest_md(store_dir)?;
    if show_time {
        println!("{}", last_updated(&path)?);
    } else {
        print!("{}", std::fs::read_to_string(&path)?);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dir_with(files: &[(&str, &str)]) -> (tempfile::TempDir, PathBuf) {
        let td = tempfile::tempdir().unwrap();
        let d = td.path().to_path_buf();
        for (name, body) in files {
            std::fs::write(d.join(name), body).unwrap();
        }
        (td, d)
    }

    fn set_mtime(path: &Path, age_secs: u64) {
        let now = std::time::SystemTime::now();
        let t = now
            .checked_sub(std::time::Duration::from_secs(age_secs))
            .unwrap();
        std::fs::OpenOptions::new()
            .write(true)
            .open(path)
            .unwrap()
            .set_modified(t)
            .unwrap();
    }

    #[test]
    fn newest_md_picks_the_latest_by_mtime_not_name() {
        let (_td, d) = dir_with(&[("2026-08-01_0000_summary.md", "old"), ("zzz.md", "newest")]);
        set_mtime(&d.join("zzz.md"), 5);
        set_mtime(&d.join("2026-08-01_0000_summary.md"), 9000);
        let got = newest_md(&d).unwrap();
        assert_eq!(got.file_name().unwrap().to_str().unwrap(), "zzz.md");
    }

    #[test]
    fn newest_md_ignores_non_md_files() {
        let (_td, d) = dir_with(&[("notes.txt", "not md"), ("summary.md", "real")]);
        let got = newest_md(&d).unwrap();
        assert_eq!(got.file_name().unwrap().to_str().unwrap(), "summary.md");
    }

    #[test]
    fn newest_md_states_why_when_directory_is_missing_or_empty() {
        let missing = std::env::temp_dir().join("ztools_store_no_such_dir");
        let e = newest_md(&missing).unwrap_err().to_string();
        assert!(e.contains("not readable"), "got: {e}");

        let (_td, d) = dir_with(&[]);
        let e = newest_md(&d).unwrap_err().to_string();
        assert!(e.contains("no stored summaries"), "got: {e}");
    }

    #[test]
    fn last_updated_formats_a_known_time() {
        let (_td, d) = dir_with(&[("summary.md", "x")]);
        let path = d.join("summary.md");
        let t = chrono::NaiveDate::from_ymd_opt(2026, 8, 29)
            .and_then(|day| day.and_hms_opt(17, 53, 0))
            .unwrap()
            .and_local_timezone(Local)
            .unwrap()
            .into();
        std::fs::File::options()
            .write(true)
            .open(&path)
            .unwrap()
            .set_modified(t)
            .unwrap();
        assert_eq!(last_updated(&path).unwrap(), "2026-08-29 17:53");
    }
}
