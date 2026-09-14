//! Emit this project's status for the `routines` harness.
//!
//! Port of `references/routines_status.py`, honouring
//! `~/Projects/routines/docs/STATUS_CONTRACT.md`. ADDITIVE AND READ-ONLY: it
//! adds a machine-readable view and changes nothing about how ztools runs
//! standalone.
//!
//! IT DOES NOT PLAN. A `wk` run scrapes the web and drives a 35B model for
//! minutes; a status command that did that would make every daily report a
//! multi-minute job, and would burn a model run just to answer "how did the
//! last one go". It reads back the newest `weekend_plan_*.md` instead.
//!
//! IT REUSES THE CHECKERS' OWN PARSERS. [`crate::ztools::weekend::report`] is
//! the Rust port of `eval.report_classes`, the parsers the G3 checks judge
//! plans with, so the numbers here and the numbers the checks assert on come
//! from one place. A private copy of the parsing is exactly how enforcement
//! and its checker drift apart.
//!
//! WHAT IT WATCHES. Two failure modes, both of which look fine from outside:
//! a plan that is HOLLOW (empty plans pass every content check — no fabricated
//! constant, no stale date, no excluded venue — because there are no rows,
//! the open defect PENDING 5.1), and a plan that is STALE (last week's plan
//! looks identical to this week's unless someone compares its window to the
//! calendar).

use chrono::{Datelike, Local};

use crate::ztools::weekend::report::{fixed_rows, parse_window_from_filename, transient_rows};

/// Identity reported to the harness.
pub const NAME: &str = "ztools-weekend";

/// The honest answer when there is nothing to read.
///
/// Never `ok`: "0 events" from a missing file is indistinguishable from a
/// genuinely empty plan, and both are indistinguishable from a working tool
/// that simply has not run yet.
fn unknown(summary: &str) -> serde_json::Value {
    serde_json::json!({ "name": NAME, "state": "unknown", "summary": summary })
}

/// The Friday-to-Sunday a plan should currently cover.
///
/// During a weekend the answer is *this* one, not the next: a plan for the
/// days you are living through is current, not stale.
///
/// Monday = 0 ... Friday = 4, Saturday = 5, Sunday = 6.
fn upcoming_weekend(today: chrono::NaiveDate) -> (chrono::NaiveDate, chrono::NaiveDate) {
    let weekday = i64::from(today.weekday().num_days_from_monday());
    let friday = today - chrono::Duration::days(weekday - 4);
    (friday, friday + chrono::Duration::days(2))
}

/// Newest `weekend_plan_*.md` in `directory` by modification time.
fn newest_plan(directory: &std::path::Path) -> Option<std::path::PathBuf> {
    let mut plans: Vec<std::path::PathBuf> = std::fs::read_dir(directory)
        .ok()?
        .flatten()
        .map(|e| e.path())
        .filter(|p| {
            let name = p.file_name().and_then(|n| n.to_str()).unwrap_or("");
            name.starts_with("weekend_plan_") && p.extension().is_some_and(|e| e == "md")
        })
        .collect();
    plans.sort_by_key(|p| std::fs::metadata(p).and_then(|m| m.modified()).ok());
    plans.last().cloned()
}

/// The store directory status reads plans from: `$WEEKEND_OUTPUT_DIR` else
/// the home `Documents` dir, matching `weekend.cli.OUTPUT_DIR_PATH`.
fn output_dir() -> std::path::PathBuf {
    std::env::var("WEEKEND_OUTPUT_DIR").map_or_else(
        |_| {
            dirs::home_dir().map_or_else(
                || std::path::PathBuf::from(""),
                |home| home.join("Documents"),
            )
        },
        std::path::PathBuf::from,
    )
}

/// Build the status JSON for `today`, reading the newest plan in the store.
fn build_status(today: chrono::NaiveDate) -> serde_json::Value {
    let directory = output_dir();
    if !directory.is_dir() {
        return unknown(&format!("no plan directory at {}", directory.display()));
    }
    let Some(plan) = newest_plan(&directory) else {
        return unknown(&format!(
            "no weekend plan has ever been written to {}",
            directory.display()
        ));
    };

    let Ok(text) = std::fs::read_to_string(&plan) else {
        return unknown(&format!("plan {} could not be read", plan.display()));
    };
    let window =
        parse_window_from_filename(&plan.file_name().unwrap_or_default().to_string_lossy());
    let fixed = fixed_rows(&text).len();
    let transient = transient_rows(&text).len();

    let (wanted_start, wanted_end) = upcoming_weekend(today);
    let covers_upcoming = window.is_some_and(|(start, _)| start == wanted_start);

    let when = window.map_or_else(
        || "an unreadable date range".to_string(),
        |(start, end)| format!("{}..{}", start.format("%Y-%m-%d"), end.format("%Y-%m-%d")),
    );

    let mut items: Vec<serde_json::Value> = Vec::new();
    let mut state = "ok";
    if !covers_upcoming {
        state = "attention";
        items.push(serde_json::json!({
            "name": format!("plan for {}", wanted_start.format("%Y-%m-%d")),
            "action": "not generated yet",
        }));
    }
    if transient == 0 {
        // The known open defect, and the one an empty plan hides best: every
        // content check passes when there are no rows to be wrong.
        state = "attention";
        items.push(serde_json::json!({
            "name": "transient events",
            "action": "none in the latest plan",
        }));
    }

    let mut summary = format!("latest plan {when}: {fixed} fixed, {transient} transient");
    if !covers_upcoming {
        use std::fmt::Write as _;
        let _ = write!(
            summary,
            " (upcoming weekend {} not planned)",
            wanted_start.format("%Y-%m-%d")
        );
    }

    let modified = std::fs::metadata(&plan)
        .and_then(|m| m.modified())
        .ok()
        .map(|t| {
            let dt: chrono::DateTime<Local> = t.into();
            dt.format("%Y-%m-%dT%H:%M:%S%:z").to_string()
        })
        .unwrap_or_default();

    serde_json::json!({
        "name": NAME,
        "state": state,
        "summary": summary,
        // When the plan was actually written, not when we asked.
        "ran_at": modified,
        "items": items,
        "details": {
            "plan": plan.display().to_string(),
            "window": window.map(|(s, e)| [s.format("%Y-%m-%d").to_string(), e.format("%Y-%m-%d").to_string()]),
            "fixed_rows": fixed,
            "transient_rows": transient,
            "upcoming_weekend": [wanted_start.format("%Y-%m-%d").to_string(), wanted_end.format("%Y-%m-%d").to_string()],
            "covers_upcoming": covers_upcoming,
        },
    })
}

/// Entry point for the `ztools status` subcommand: build and print the JSON.
///
/// The harness needs a stated reason, not a traceback, so any failure becomes
/// an `unknown` status rather than an error return.
///
/// # Errors
///
/// Only when stdout cannot be written.
pub fn run() -> anyhow::Result<()> {
    let status = build_status(Local::now().date_naive());
    println!("{}", serde_json::to_string_pretty(&status)?);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::NaiveDate;

    fn d(y: i32, m: u32, day: u32) -> NaiveDate {
        NaiveDate::from_ymd_opt(y, m, day).unwrap()
    }

    #[test]
    fn upcoming_weekend_is_this_weekend_during_it() {
        // Wednesday 2026-08-12 -> upcoming Fri-Sun.
        assert_eq!(
            upcoming_weekend(d(2026, 8, 12)),
            (d(2026, 8, 14), d(2026, 8, 16))
        );
        // Saturday 2026-08-15 -> this weekend, not next.
        assert_eq!(
            upcoming_weekend(d(2026, 8, 15)),
            (d(2026, 8, 14), d(2026, 8, 16))
        );
        // Monday 2026-08-17 -> the following Friday.
        assert_eq!(
            upcoming_weekend(d(2026, 8, 17)),
            (d(2026, 8, 21), d(2026, 8, 23))
        );
    }

    fn set_mtime(path: &std::path::Path, age_secs: u64) {
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
    fn newest_plan_picks_latest_by_mtime_and_only_plan_files() {
        let td = tempfile::tempdir().unwrap();
        std::fs::write(
            td.path()
                .join("weekend_plan_August_14_to_August_16_2026.md"),
            "x",
        )
        .unwrap();
        std::fs::write(
            td.path()
                .join("weekend_plan_August_07_to_August_09_2026.md"),
            "y",
        )
        .unwrap();
        std::fs::write(td.path().join("notes.txt"), "not a plan").unwrap();
        set_mtime(
            &td.path()
                .join("weekend_plan_August_07_to_August_09_2026.md"),
            9000,
        );
        set_mtime(
            &td.path()
                .join("weekend_plan_August_14_to_August_16_2026.md"),
            5,
        );
        let path = newest_plan(td.path()).unwrap();
        assert_eq!(
            path.file_name().unwrap().to_str().unwrap(),
            "weekend_plan_August_14_to_August_16_2026.md"
        );
    }

    #[test]
    #[serial_test::serial]
    fn status_unknown_when_directory_is_missing() {
        let prev = std::env::var_os("WEEKEND_OUTPUT_DIR");
        std::env::remove_var("WEEKEND_OUTPUT_DIR");
        // Point HOME somewhere empty via the env var so we do not read the
        // user's real Documents.
        std::env::set_var(
            "WEEKEND_OUTPUT_DIR",
            "/tmp/ztools_status_nonexistent_dir_2026",
        );
        let status = build_status(d(2026, 8, 15));
        assert_eq!(status["state"], "unknown");
        assert!(status["summary"]
            .as_str()
            .unwrap()
            .contains("no plan directory"));
        match prev {
            Some(v) => std::env::set_var("WEEKEND_OUTPUT_DIR", v),
            None => std::env::remove_var("WEEKEND_OUTPUT_DIR"),
        }
    }

    #[test]
    #[serial_test::serial]
    fn status_flags_a_stale_plan_and_hollow_transient() {
        let td = tempfile::tempdir().unwrap();
        // A plan covering last weekend, with no transient rows.
        let text = "# Weekend Plan: August 07 to August 09, 2026\n\n\
### Fixed / Year-Round Activities\n\n\
| Score | Activity & Location | Target Age(s) | Price | Why It Fits |\n\
| :--- | :--- | :--- | :--- | :--- |\n\
| 3.0/5 | Air Riderz (Vaughan) | 6-13 | $18 | Good |\n";
        std::fs::write(
            td.path()
                .join("weekend_plan_August_07_to_August_09_2026.md"),
            text,
        )
        .unwrap();

        let prev = std::env::var_os("WEEKEND_OUTPUT_DIR");
        std::env::set_var("WEEKEND_OUTPUT_DIR", td.path());
        let status = build_status(d(2026, 8, 15));
        match prev {
            Some(v) => std::env::set_var("WEEKEND_OUTPUT_DIR", v),
            None => std::env::remove_var("WEEKEND_OUTPUT_DIR"),
        }

        assert_eq!(status["state"], "attention");
        let items = status["items"].as_array().unwrap();
        assert!(items.iter().any(|i| i["name"] == "transient events"));
        assert!(items.iter().any(|i| i["name"]
            .as_str()
            .unwrap_or_default()
            .starts_with("plan for 2026-08-14")));
        assert_eq!(status["details"]["fixed_rows"], 1);
        assert_eq!(status["details"]["transient_rows"], 0);
        assert_eq!(status["details"]["covers_upcoming"], false);
        assert!(status["summary"]
            .as_str()
            .unwrap()
            .contains("(upcoming weekend 2026-08-14 not planned)"));
    }

    #[test]
    #[serial_test::serial]
    fn status_is_ok_for_a_current_full_plan() {
        let td = tempfile::tempdir().unwrap();
        let text = "# Weekend Plan: August 14 to August 16, 2026\n\n\
### Fixed / Year-Round Activities\n\n\
| Score | Activity & Location | Target Age(s) | Price | Why It Fits |\n\
| :--- | :--- | :--- | :--- | :--- |\n\
| 3.0/5 | Air Riderz (Vaughan) | 6-13 | $18 | Good |\n\n\
### Transient / Limited-Time Events\n\n\
| Score | Event & Location | Day & Time | Target Age(s) | Price | Why It Fits |\n\
| :--- | :--- | :--- | :--- | :--- | :--- |\n\
| 4.0/5 | Maple Syrup Festival (Vaughan) | Saturday | 6-13 | By donation | Fresh |\n";
        std::fs::write(
            td.path()
                .join("weekend_plan_August_14_to_August_16_2026.md"),
            text,
        )
        .unwrap();

        let prev = std::env::var_os("WEEKEND_OUTPUT_DIR");
        std::env::set_var("WEEKEND_OUTPUT_DIR", td.path());
        let status = build_status(d(2026, 8, 13));
        match prev {
            Some(v) => std::env::set_var("WEEKEND_OUTPUT_DIR", v),
            None => std::env::remove_var("WEEKEND_OUTPUT_DIR"),
        }

        assert_eq!(status["state"], "ok");
        assert!(status["items"].as_array().unwrap().is_empty());
        assert_eq!(status["details"]["fixed_rows"], 1);
        assert_eq!(status["details"]["transient_rows"], 1);
        assert_eq!(status["details"]["covers_upcoming"], true);
    }
}
