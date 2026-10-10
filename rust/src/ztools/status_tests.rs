//! Tests for `status.rs`, split out for the 500-line cap.

use super::*;
use crate::test_env::TestEnv;
use crate::ztools::weekend::holidays::Province::Ontario as ON;
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
fn upcoming_weekend_includes_a_holiday_monday() {
    // Thanksgiving 2026: a plan for Oct 9-11 does not cover the upcoming
    // weekend, which runs to Monday Oct 12.
    assert_eq!(
        upcoming_weekend(d(2026, 10, 8), ON),
        (d(2026, 10, 9), d(2026, 10, 12))
    );
}

#[test]
fn upcoming_weekend_is_this_weekend_during_it() {
    // Wednesday 2026-08-12 -> upcoming Fri-Sun.
    assert_eq!(
        upcoming_weekend(d(2026, 8, 12), ON),
        (d(2026, 8, 14), d(2026, 8, 16))
    );
    // Saturday 2026-08-15 -> this weekend, not next.
    assert_eq!(
        upcoming_weekend(d(2026, 8, 15), ON),
        (d(2026, 8, 14), d(2026, 8, 16))
    );
    // Monday 2026-08-17 -> the following Friday.
    assert_eq!(
        upcoming_weekend(d(2026, 8, 17), ON),
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
    let status = build_status(d(2026, 8, 15), ON);
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

    let status = build_status(d(2026, 8, 15), ON);
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

    let status = build_status(d(2026, 8, 13), ON);
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
