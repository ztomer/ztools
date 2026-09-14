//! The store directories, and read-side access to what is in them.
//!
//! The summarizer and planner WRITE dated `.md` files into a store directory;
//! their `--fetch-latest` / `--last-updated` and the routines `status`
//! commands read the newest one back. ONE directory per tool: the weekend
//! planner used to have two (Python `wk` wrote `~/Documents/weekend_plan_*.md`,
//! which the status page read, while the dashboard tab read
//! `~/Documents/weekend_plans/`), so a run the tab showed was one the status
//! page called stale. Every reader and the writer now resolve
//! [`weekend_store_dir`] (or `WEEKEND_OUTPUT_DIR`). Nothing here touches a
//! model or the network, so a dashboard tab can open on it without ever
//! re-running the pipeline.

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

/// Default directory weekend plans are stored in.
#[must_use]
pub fn weekend_store_dir() -> PathBuf {
    dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("Documents/weekend_plans")
}

/// The weekend store, honouring `WEEKEND_OUTPUT_DIR` — the one seam the
/// writer, the status page and the dashboard tab all resolve.
#[must_use]
pub fn weekend_output_dir() -> PathBuf {
    std::env::var("WEEKEND_OUTPUT_DIR").map_or_else(|_| weekend_store_dir(), PathBuf::from)
}

/// File name of the stored plan for one weekend.
///
/// `weekend_plan_August_14_to_August_16_2026.md` — byte-compatible with the
/// Python writer (`f"{PLAN_FILE_PREFIX}{dates_str.replace(' ', '_').replace(',', '')}.md"`
/// over `"%B %d to %B %d, %Y"`), because `weekend::report::parse_window_from_filename`
/// — which the status page uses to decide whether a plan is stale — reads the
/// weekend back out of this name.
#[must_use]
pub fn weekend_plan_filename(friday: chrono::NaiveDate, sunday: chrono::NaiveDate) -> String {
    format!(
        "weekend_plan_{}_to_{}.md",
        friday.format("%B_%d"),
        sunday.format("%B_%d_%Y")
    )
}

/// Write the dated plan into `dir`, creating it, and return the path written.
///
/// # Errors
///
/// When the directory cannot be created or the file cannot be written.
pub fn save_weekend_plan(
    dir: &Path,
    friday: chrono::NaiveDate,
    sunday: chrono::NaiveDate,
    markdown: &str,
) -> Result<PathBuf> {
    std::fs::create_dir_all(dir)?;
    let path = dir.join(weekend_plan_filename(friday, sunday));
    std::fs::write(&path, markdown)?;
    Ok(path)
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
    print_newest(&weekend_output_dir(), show_time)
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

/// What a folder cleanup removed and what it could not.
pub struct CleanReport {
    pub deleted: usize,
    pub warnings: Vec<String>,
}

/// Delete `*.md` files in `dir`, keeping everything else.
///
/// A missing directory and per-file failures are warnings, never errors:
/// cleanup is housekeeping before a run, not the run, and must not fail it.
/// Port of `twitter/output.py::clean_folder` minus the process exit, which
/// belongs to the CLI layer, not a library function.
#[must_use]
pub fn clean_folder(dir: &Path) -> CleanReport {
    let mut report = CleanReport {
        deleted: 0,
        warnings: Vec::new(),
    };
    let Ok(entries) = std::fs::read_dir(dir) else {
        report
            .warnings
            .push(format!("Directory {} does not exist.", dir.display()));
        return report;
    };
    let mut names: Vec<_> = entries.filter_map(Result::ok).collect();
    names.sort_by_key(std::fs::DirEntry::file_name);
    for entry in names {
        let path = entry.path();
        // Suffix match, not extension match: `Path::extension` reports no
        // extension for a dotfile literally named `.md`, while glob `*.md`
        // matches it. The filter must agree with what it replaces.
        //
        // Case-SENSITIVE on purpose, also mirroring glob: `*.md` does not
        // match `X.MD`, and a case-fold here would delete files Python keeps.
        #[expect(
            clippy::case_sensitive_file_extension_comparisons,
            reason = "parity with Python glob '*.md', which is case-sensitive; folding would delete files the reference keeps"
        )]
        let is_md = path
            .file_name()
            .and_then(|n| n.to_str())
            .is_some_and(|n| n.ends_with(".md"));
        if !is_md {
            continue;
        }
        match std::fs::remove_file(&path) {
            Ok(()) => report.deleted += 1,
            Err(e) => report
                .warnings
                .push(format!("Failed to delete {}: {e}", path.display())),
        }
    }
    report
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

    #[test]
    fn clean_folder_reports_a_missing_directory_without_failing() {
        let missing = std::env::temp_dir().join("ztools_store_no_such_dir_for_clean");
        let report = clean_folder(&missing);
        assert_eq!(report.deleted, 0);
        assert_eq!(report.warnings.len(), 1);
        assert!(
            report.warnings[0].contains("does not exist"),
            "{:?}",
            report.warnings
        );
    }

    #[test]
    fn clean_folder_removes_md_files_and_keeps_the_rest() {
        let (_td, d) = dir_with(&[
            ("old1.md", "old1"),
            ("old2.md", "old2"),
            ("other.txt", "other"),
        ]);
        let report = clean_folder(&d);
        assert_eq!(report.deleted, 2);
        assert!(report.warnings.is_empty(), "{:?}", report.warnings);
        assert!(!d.join("old1.md").exists());
        assert!(!d.join("old2.md").exists());
        assert!(d.join("other.txt").exists());
    }

    #[test]
    fn clean_folder_warns_and_continues_when_a_delete_fails() {
        // A subdirectory named *.md cannot be unlinked as a file on any
        // platform or privilege level, so the failure path is deterministic
        // without chmod games.
        let td = tempfile::tempdir().unwrap();
        let d = td.path();
        std::fs::create_dir(d.join("locked.md")).unwrap();
        let report = clean_folder(d);
        assert_eq!(report.deleted, 0);
        assert_eq!(report.warnings.len(), 1);
        assert!(
            report.warnings[0].contains("locked.md"),
            "{:?}",
            report.warnings
        );
        assert!(
            report.warnings[0].contains("Failed"),
            "{:?}",
            report.warnings
        );
    }

    #[test]
    fn weekend_plan_filename_is_the_python_shape_and_round_trips() {
        use chrono::NaiveDate;
        let fri = NaiveDate::from_ymd_opt(2026, 8, 14).unwrap();
        let sun = NaiveDate::from_ymd_opt(2026, 8, 16).unwrap();
        let name = weekend_plan_filename(fri, sun);
        assert_eq!(name, "weekend_plan_August_14_to_August_16_2026.md");
        // The status page reads the weekend back out of the name.
        assert_eq!(
            crate::ztools::weekend::report::parse_window_from_filename(&name),
            Some((fri, sun))
        );
        // Year boundary keeps Python's shape (year printed once, from Sunday).
        let fri = NaiveDate::from_ymd_opt(2026, 12, 31).unwrap();
        let sun = NaiveDate::from_ymd_opt(2027, 1, 2).unwrap();
        assert_eq!(
            weekend_plan_filename(fri, sun),
            "weekend_plan_December_31_to_January_02_2027.md"
        );
    }

    #[test]
    fn save_weekend_plan_creates_the_store_and_the_status_reader_finds_it() {
        use chrono::NaiveDate;
        let td = tempfile::tempdir().unwrap();
        let store = td.path().join("nested/weekend_plans");
        let fri = NaiveDate::from_ymd_opt(2026, 9, 18).unwrap();
        let sun = NaiveDate::from_ymd_opt(2026, 9, 20).unwrap();
        let path = save_weekend_plan(&store, fri, sun, "# plan\n").unwrap();
        assert!(path.ends_with("weekend_plan_September_18_to_September_20_2026.md"));
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "# plan\n");
        assert_eq!(newest_md(&store).unwrap(), path);
    }
}
