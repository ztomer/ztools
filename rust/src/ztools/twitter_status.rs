//! Emit the twitter summarizer's status for the `routines` harness.
//!
//! Port of `routines_twitter_status.py`, honouring
//! `~/Projects/routines/docs/STATUS_CONTRACT.md`. READ-ONLY: it reads back the
//! newest stored summary and makes no network call and runs no model — a
//! status command that summarised would turn every daily report into a
//! browser session plus an inference run.
//!
//! Discovery is by mtime over every `*.md` in the store, through the same
//! [`crate::ztools::store::newest_md`] the `--fetch-latest` tab uses. The
//! summary writer has used two file-name shapes; a status that globbed one of
//! them once answered "no summary found" on a directory full of summaries.

use std::path::{Path, PathBuf};

use crate::ztools::store::{newest_md, twitter_store_dir};

/// Identity reported to the harness.
pub const NAME: &str = "ztools-twitter";

/// The header line the summary writer emits (`twitter/mod.rs`); its remainder
/// is the one-line count the status page shows.
const TWEETS_HEADER: &str = "**Tweets:**";

/// The honest answer when there is nothing to read. Never `ok`: a missing
/// directory and a tool that has not run yet must not read as a clean run.
fn unknown(summary: &str) -> serde_json::Value {
    serde_json::json!({ "name": NAME, "state": "unknown", "summary": summary })
}

/// The summary directory, honouring `TWITTER_OUTPUT_DIR` — the single seam
/// the status page AND the dashboard tab both resolve, so a test redirect
/// moves both at once and they can never disagree about the directory.
fn output_dir() -> PathBuf {
    std::env::var("TWITTER_OUTPUT_DIR").map_or_else(|_| twitter_store_dir(), PathBuf::from)
}

/// The `**Tweets:** ...` remainder, if the summary carries the header.
fn header_summary(text: &str) -> Option<String> {
    text.lines()
        .map(str::trim)
        .find_map(|line| line.strip_prefix(TWEETS_HEADER))
        .map(|rest| rest.trim().to_string())
        .filter(|rest| !rest.is_empty())
}

/// Build the status document for the summaries under `directory`.
fn build_status_in(directory: &Path) -> serde_json::Value {
    if !directory.is_dir() {
        return unknown(&format!("no summary directory at {}", directory.display()));
    }
    let Ok(summary_file) = newest_md(directory) else {
        return unknown(&format!(
            "no twitter summary found in {}",
            directory.display()
        ));
    };
    let Ok(text) = std::fs::read_to_string(&summary_file) else {
        return unknown(&format!(
            "summary {} could not be read",
            summary_file.display()
        ));
    };
    let ran_at = std::fs::metadata(&summary_file)
        .and_then(|m| m.modified())
        .map(|t| {
            chrono::DateTime::<chrono::Local>::from(t)
                .to_rfc3339_opts(chrono::SecondsFormat::Micros, false)
        })
        .unwrap_or_default();

    let stem = summary_file
        .file_stem()
        .unwrap_or_default()
        .to_string_lossy()
        .to_string();
    let name = summary_file
        .file_name()
        .unwrap_or_default()
        .to_string_lossy()
        .to_string();
    let summary_text = header_summary(&text).map_or_else(
        || format!("latest summary {name}"),
        |header| format!("latest summary {stem}: {header}"),
    );

    serde_json::json!({
        "name": NAME,
        "state": "ok",
        "summary": summary_text,
        "ran_at": ran_at,
    })
}

/// Entry point for `ztools twitter-status`: build and print the JSON.
///
/// The harness needs a stated reason, not a traceback, so any failure becomes
/// an `unknown` status rather than an error return.
///
/// # Errors
///
/// Only when stdout cannot be written.
pub fn run() -> anyhow::Result<()> {
    let status = build_status_in(&output_dir());
    println!("{}", serde_json::to_string(&status)?);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dir_with(files: &[(&str, &str)]) -> (tempfile::TempDir, PathBuf) {
        let td = tempfile::tempdir().unwrap();
        for (name, body) in files {
            std::fs::write(td.path().join(name), body).unwrap();
        }
        let p = td.path().to_path_buf();
        (td, p)
    }

    fn set_mtime(path: &Path, age_secs: u64) {
        let t = std::time::SystemTime::now() - std::time::Duration::from_secs(age_secs);
        std::fs::File::options()
            .write(true)
            .open(path)
            .unwrap()
            .set_modified(t)
            .unwrap();
    }

    #[test]
    fn missing_directory_is_unknown_with_the_path() {
        let s = build_status_in(Path::new("/nonexistent/twitter_summaries"));
        assert_eq!(s["state"], "unknown");
        assert_eq!(s["name"], NAME);
        assert!(s["summary"]
            .as_str()
            .unwrap()
            .starts_with("no summary directory at /nonexistent"));
    }

    #[test]
    fn empty_directory_is_unknown_not_ok() {
        let (_td, d) = dir_with(&[("notes.txt", "not a summary")]);
        let s = build_status_in(&d);
        assert_eq!(s["state"], "unknown");
        assert!(s["summary"]
            .as_str()
            .unwrap()
            .starts_with("no twitter summary found in "));
    }

    #[test]
    fn header_count_is_lifted_into_the_summary_line() {
        let (_td, d) = dir_with(&[(
            "2026-09-13_0800_summary.md",
            "# Twitter\n\n  **Tweets:** 46 fetched, 40 processed  \n\n## Topics\n",
        )]);
        let s = build_status_in(&d);
        assert_eq!(s["state"], "ok");
        assert_eq!(
            s["summary"],
            "latest summary 2026-09-13_0800_summary: 46 fetched, 40 processed"
        );
        assert!(s["ran_at"].as_str().unwrap().contains('T'));
    }

    #[test]
    fn headerless_summary_reports_the_file_name() {
        let (_td, d) = dir_with(&[("old_shape_to_newer.md", "# no header here\n")]);
        let s = build_status_in(&d);
        assert_eq!(s["state"], "ok");
        assert_eq!(s["summary"], "latest summary old_shape_to_newer.md");
    }

    #[test]
    fn newest_by_mtime_wins_whatever_the_name() {
        let (_td, d) = dir_with(&[
            ("zzz_summary.md", "**Tweets:** 1 fetched, 1 processed"),
            (
                "2026-01-01_0000_summary.md",
                "**Tweets:** 9 fetched, 9 processed",
            ),
        ]);
        set_mtime(&d.join("zzz_summary.md"), 9000);
        set_mtime(&d.join("2026-01-01_0000_summary.md"), 5);
        let s = build_status_in(&d);
        assert_eq!(
            s["summary"],
            "latest summary 2026-01-01_0000_summary: 9 fetched, 9 processed"
        );
    }

    #[test]
    #[serial_test::serial]
    fn output_dir_honours_the_env_override() {
        let prev = std::env::var("TWITTER_OUTPUT_DIR").ok();
        std::env::set_var("TWITTER_OUTPUT_DIR", "/tmp/tw_status_probe");
        let got = output_dir();
        match prev {
            Some(v) => std::env::set_var("TWITTER_OUTPUT_DIR", v),
            None => std::env::remove_var("TWITTER_OUTPUT_DIR"),
        }
        assert_eq!(got, PathBuf::from("/tmp/tw_status_probe"));
        assert!(twitter_store_dir().ends_with("Documents/twitter_summaries"));
    }
}
