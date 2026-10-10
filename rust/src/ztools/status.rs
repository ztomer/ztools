//! Emit this project's status for the `routines` harness.
//!
//! Port of `references/routines_status.py`, honouring the `routines` harness's
//! `docs/STATUS_CONTRACT.md`. ADDITIVE AND READ-ONLY: it adds a
//! machine-readable view and changes nothing about how ztools runs standalone.
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
fn upcoming_weekend(
    today: chrono::NaiveDate,
    province: crate::ztools::weekend::holidays::Province,
) -> (chrono::NaiveDate, chrono::NaiveDate) {
    crate::ztools::weekend::plan_window(today, province)
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
fn build_status(
    today: chrono::NaiveDate,
    province: crate::ztools::weekend::holidays::Province,
) -> serde_json::Value {
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

    let (wanted_start, wanted_end) = upcoming_weekend(today, province);
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
pub fn run(config: &crate::config::ZtoolsConfig) -> anyhow::Result<()> {
    // The window depends on the province's holidays; without them "upcoming"
    // is a guess, and a guess is an `unknown`, never a short weekend.
    let status = match crate::ztools::weekend::holidays::load_province(&config.weekend_region_paths)
    {
        Ok(province) => build_status(Local::now().date_naive(), province),
        Err(reason) => unknown(&reason),
    };
    println!("{}", serde_json::to_string_pretty(&status)?);
    Ok(())
}

#[cfg(test)]
#[path = "status_tests.rs"]
mod tests;
