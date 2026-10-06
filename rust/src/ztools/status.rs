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

use chrono::Local;

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

/// The Friday-to-Sunday a plan should currently cover — the planner's own
/// definition, so the two cannot disagree about which weekend is "upcoming".
fn upcoming_weekend(today: chrono::NaiveDate) -> (chrono::NaiveDate, chrono::NaiveDate) {
    crate::ztools::weekend::plan_window(today)
}

/// Newest `weekend_plan_*.md` in `directory` by modification time.
fn newest_plan(directory: &std::path::Path) -> Option<std::path::PathBuf> {
    let mut plans: Vec<std::path::PathBuf> = std::fs::read_dir(directory)
        .ok()?
        .flatten()
        .map(|e| e.path())
        .filter(|p| {
            // Dated plans only. The store also holds `weekend_plan_latest.md`
            // (the tab's pointer copy of a dated plan) and older `*_plan.md`
            // shapes; neither carries the weekend in its name, and a status
            // built on one reports "an unreadable date range" for a plan
            // that exists under its dated name right beside it.
            let name = p.file_name().and_then(|n| n.to_str()).unwrap_or("");
            p.extension().is_some_and(|e| e == "md") && parse_window_from_filename(name).is_some()
        })
        .collect();
    plans.sort_by_key(|p| std::fs::metadata(p).and_then(|m| m.modified()).ok());
    plans.last().cloned()
}

/// The store directory status reads plans from — the same one
/// `weekend-plan` writes and the dashboard tab reads
/// (`store::weekend_output_dir`). The Python original read `~/Documents`
/// while the tab read `~/Documents/weekend_plans/`; two directories for one
/// fact is how a plan the tab showed came to be one the status called stale.
fn output_dir() -> std::path::PathBuf {
    crate::ztools::store::weekend_output_dir()
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
    // The plan's own degraded-warning says WHY it is empty (weekend/health.rs);
    // the wall is the cause worth naming on the dashboard, because it is the
    // one that looks like a quiet weekend and is not.
    let bot_walled = text.contains("blocked by a bot wall");
    let provenance = crate::ztools::weekend::Provenance::parse(&text);
    if transient == 0 {
        // The known open defect, and the one an empty plan hides best: every
        // content check passes when there are no rows to be wrong.
        state = "attention";
        items.push(serde_json::json!({
            "name": "transient events",
            "action": if bot_walled {
                "none — search was blocked by a bot wall"
            } else {
                "none in the latest plan"
            },
        }));
    }

    let mut summary = format!("latest plan {when}: {fixed} fixed, {transient} transient");
    if bot_walled {
        summary.push_str(" (search bot-walled)");
    }
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
            // The plan's own ledger (weekend/health.rs::Provenance); absent
            // for plans written before it existed.
            "provenance": provenance.map(|p| serde_json::json!({
                "extracted": p.extracted,
                "unsourced": p.unsourced,
                "outside_window": p.outside_window,
                "excluded": p.excluded,
            })),
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
    use crate::test_env::TestEnv;
    use crate::ztools::weekend::{
        FIXED_SECTION_HEADING, FIXED_TABLE_HEADER, PlanHealth, TABLE_SEPARATOR,
        TRANSIENT_SECTION_HEADING, TRANSIENT_TABLE_HEADER, WeekendEvent, format_weekend_plan,
    };
    use chrono::NaiveDate;

    fn event(
        name: &str,
        location: &str,
        price: &str,
        ages: &str,
        day: &str,
        why: &str,
        score: f32,
    ) -> WeekendEvent {
        WeekendEvent {
            name: name.to_string(),
            location: location.to_string(),
            price: price.to_string(),
            target_ages: ages.to_string(),
            day: day.to_string(),
            dates: String::new(),
            description: why.to_string(),
            is_transient: true,
            score,
            start_date: String::new(),
            end_date: String::new(),
            weather: String::new(),
            duration: String::new(),
        }
    }

    /// A plan document as `weekend-plan --md-out` writes it — the one writer of
    /// that markdown, called rather than retyped.
    ///
    /// These two fixtures used to be hand-typed documents NO writer produces: a
    /// five-column header (`Price`, not `Estimated Price (CAD)`), headings with no
    /// `(Ranked by Fit Score)`, no `Dates` column, and — in the stale case — no
    /// transient section at all, where the writer always writes one. Nothing
    /// failed: the parser is header-keyed, so a status page could read a plan
    /// that never existed and report it `ok`.
    fn saved_plan(
        dates: &str,
        transient: &[WeekendEvent],
        fixed: &[WeekendEvent],
        health: &PlanHealth,
    ) -> String {
        format_weekend_plan(
            transient,
            fixed,
            "Vaughan",
            "6-12",
            dates,
            "Fri 28.2°C (clear), Sat 32.0°C (precipraction)",
            health,
        )
    }

    /// The column set the status page must be reading. Seven columns, `Dates`
    /// third: the stale spelling had five and put the ages where the dates go.
    ///
    /// This is what fails if [`saved_plan`] is ever swapped back for a literal —
    /// not the row counts, which a header-keyed parser reads out of any spelling.
    ///
    /// `has_transient_rows` is not decoration. With no transient rows the writer
    /// emits the heading and the warning and NO table at all, so the transient
    /// header is genuinely absent from that document — a hollow plan is a plan
    /// with one table, and a fixture that "fixed" this by pasting the header in
    /// would describe a document the writer cannot produce.
    fn assert_current_plan_document(plan: &str, has_transient_rows: bool) {
        for (part, expected) in [
            ("fixed heading", FIXED_SECTION_HEADING),
            ("transient heading", TRANSIENT_SECTION_HEADING),
            ("fixed header", FIXED_TABLE_HEADER),
            ("separator", TABLE_SEPARATOR),
        ] {
            assert!(plan.contains(expected), "the plan has no {part}:\n{plan}");
        }
        assert_eq!(
            plan.contains(TRANSIENT_TABLE_HEADER),
            has_transient_rows,
            "the transient table is present iff the plan has transient rows:\n{plan}"
        );
    }

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
        // The tab's pointer copy is newest of all and must NOT win: it has
        // no window in its name.
        std::fs::write(td.path().join("weekend_plan_latest.md"), "y").unwrap();
        set_mtime(&td.path().join("weekend_plan_latest.md"), 1);
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
        // The directory must be NAMED and ABSENT, and that is two facts, so it is
        // two steps: `fixture` points the managed variable at a fresh subdirectory
        // of the sandbox (never the old fixed `/tmp` name two concurrent runs
        // shared), and the `remove_dir` makes the name point at nothing.
        //
        // `unset` is NOT the alternative. An absent `WEEKEND_OUTPUT_DIR` falls
        // back to `~/Documents`, which is the operator's real one -- so "absent"
        // here has to mean absent at a path, not absent as a variable.
        let (_env, named) = TestEnv::new().fixture("WEEKEND_OUTPUT_DIR", "absent");
        std::fs::remove_dir(&named).unwrap();
        let status = build_status(d(2026, 8, 15));
        assert_eq!(status["state"], "unknown");
        assert!(
            status["summary"]
                .as_str()
                .unwrap()
                .contains("no plan directory")
        );
    }

    #[test]
    #[serial_test::serial]
    fn status_flags_a_stale_plan_and_hollow_transient() {
        // The plan fixture and the variable that finds it live in the SAME sandbox:
        // `set_path` re-points the managed variable at a subdirectory of it, so the
        // fixture needs no separate temp dir and the restore is the guard's rather
        // than a hand-rolled capture/restore wrapped around the assertions.
        let (_env, plan_dir) = TestEnv::new().fixture("WEEKEND_OUTPUT_DIR", "plans");
        // A plan covering last weekend, with no transient rows — so the writer
        // emits the transient heading and the warning that says the weekend was
        // quiet, and `transient_rows` counts none of it.
        let text = saved_plan(
            "August 07 to August 09, 2026",
            &[],
            &[event(
                "Air Riderz",
                "Vaughan, ON",
                "$18",
                "6-13",
                "",
                "Good",
                3.0,
            )],
            &PlanHealth::nominal(),
        );
        assert_current_plan_document(&text, false);
        // The empty branch is not an absent section: the writer always writes the
        // heading, then the warning that names the cause. A fixture with no
        // transient section at all is the document this test used to read.
        assert!(
            text.contains("Plan Degraded"),
            "an empty transient section must say WHY it is empty:\n{text}"
        );
        std::fs::write(
            plan_dir.join("weekend_plan_August_07_to_August_09_2026.md"),
            text,
        )
        .unwrap();

        let status = build_status(d(2026, 8, 15));
        assert_eq!(status["state"], "attention");
        let items = status["items"].as_array().unwrap();
        assert!(items.iter().any(|i| i["name"] == "transient events"));
        assert!(items.iter().any(|i| {
            i["name"]
                .as_str()
                .unwrap_or_default()
                .starts_with("plan for 2026-08-14")
        }));
        assert_eq!(status["details"]["fixed_rows"], 1);
        assert_eq!(status["details"]["transient_rows"], 0);
        assert_eq!(status["details"]["covers_upcoming"], false);
        assert!(
            status["summary"]
                .as_str()
                .unwrap()
                .contains("(upcoming weekend 2026-08-14 not planned)")
        );
    }

    #[test]
    #[serial_test::serial]
    fn status_is_ok_for_a_current_full_plan() {
        // Same sandbox, same reason as the stale-plan test above.
        let (_env, plan_dir) = TestEnv::new().fixture("WEEKEND_OUTPUT_DIR", "plans");
        // The provenance the status page reports is the plan's OWN ledger, so it
        // is set through the writer's `PlanHealth` rather than typed into a
        // `_Provenance:` line that nothing produces.
        let mut health = PlanHealth::nominal();
        health.provenance.extracted = 9;
        health.provenance.unsourced = 5;
        health.provenance.outside_window = 2;
        health.provenance.excluded = 1;
        let text = saved_plan(
            "August 14 to August 16, 2026",
            &[event(
                "Maple Syrup Festival",
                "Vaughan",
                "By donation",
                "6-13",
                "Saturday 10am-4pm",
                "Fresh maple",
                4.0,
            )],
            &[event(
                "Air Riderz",
                "Vaughan, ON",
                "$18",
                "6-13",
                "",
                "Good",
                3.0,
            )],
            &health,
        );
        assert_current_plan_document(&text, true);
        std::fs::write(
            plan_dir.join("weekend_plan_August_14_to_August_16_2026.md"),
            text,
        )
        .unwrap();

        let status = build_status(d(2026, 8, 13));
        assert_eq!(status["state"], "ok");
        let items = status["items"].as_array().expect("items is an array");
        assert_empty!(items);
        assert_eq!(status["details"]["fixed_rows"], 1);
        assert_eq!(status["details"]["transient_rows"], 1);
        assert_eq!(status["details"]["covers_upcoming"], true);
        // The plan's own ledger reaches the status page, so a week of
        // dashboards carries the invented-row rate without reopening plans.
        assert_eq!(status["details"]["provenance"]["extracted"], 9);
        assert_eq!(status["details"]["provenance"]["unsourced"], 5);
        assert_eq!(status["details"]["provenance"]["outside_window"], 2);
        assert_eq!(status["details"]["provenance"]["excluded"], 1);
    }
}
