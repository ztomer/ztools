//! How an eval run PRINTS: the per-task rows, and the persistence + trend
//! tables that follow a sweep.
//!
//! Split out of `cli_ztools.rs` for the house 500-line cap, at the seam the
//! module already had: `cli_ztools_capabilities.rs` reports what a model IS
//! without running a task, and this file reports what a run PRODUCED once it
//! has run them. The dispatch that decides whether a run happened at all stayed
//! behind, because a renderer that lives next to the verdict it renders is how a
//! table keeps printing scores for a run that measured nothing.
//!
//! The rule both renderers below obey is in `ztools::model_eval`: a row that was
//! never measured carries [`NO_SCORE`] and says `NOT MEASURED` in its status,
//! and no summary averages those rows in.

use anyhow::Result;

/// One model's outcomes: a JSON array under `--json-output` (stdout carries
/// nothing else), otherwise the rendered table with substitutions noted.
pub fn print_outcomes(
    outcomes: &[crate::ztools::eval::TaskOutcome],
    json_output: bool,
) -> Result<()> {
    if json_output {
        use serde::Serialize;
        #[derive(Serialize)]
        struct OutcomeRow<'a> {
            task: &'a str,
            score: u8,
            status: &'a str,
            time_secs: f64,
            error: Option<&'a String>,
            failure_category: &'a str,
            #[serde(skip_serializing_if = "Option::is_none")]
            substituted_to: Option<&'a String>,
            #[serde(skip_serializing_if = "Option::is_none")]
            substitution_reason: Option<&'a String>,
        }
        let rows: Vec<OutcomeRow> = outcomes
            .iter()
            .map(|o| OutcomeRow {
                task: &o.task,
                score: o.score,
                status: o.status.as_str(),
                time_secs: o.time_secs,
                error: o.error.as_ref(),
                failure_category: o.failure_category.as_str(),
                substituted_to: o.substituted_to.as_ref(),
                substitution_reason: o.substitution_reason.as_ref(),
            })
            .collect();
        println!("{}", serde_json::to_string_pretty(&rows)?);
    } else {
        for note in outcomes
            .iter()
            .filter_map(|o| o.substitution_reason.as_deref())
        {
            eprintln!("⚠ {note}");
        }
        print!(
            "{}",
            crate::ztools::model_eval::render_task_outcomes(outcomes)
        );
    }
    Ok(())
}

/// Persistence + reporting, matching the Python evaluator's exports: the
/// per-(model, task) CSV sheet, the historical trends table, the delta from
/// the last run and the verbosity table. Nothing to do for an empty sweep.
pub fn report_suite(runs: &[crate::ztools::eval::ModelRun], started_at: f64) {
    if runs.is_empty() {
        return;
    }
    let csv_path = crate::ztools::eval::default_eval_dir().join("eval_results.csv");
    match crate::ztools::eval::export_csv(runs, &csv_path) {
        Ok(()) => println!("→ Exported to {}", csv_path.display()),
        Err(e) => eprintln!("⚠ CSV export failed: {e}"),
    }
    for line in crate::ztools::eval::render_historical_trends(None) {
        println!("{line}");
    }
    for line in crate::ztools::eval::render_diff_from_last_run(runs, None, started_at) {
        println!("{line}");
    }
    for line in crate::ztools::eval::render_verbosity(&crate::ztools::eval::compute_verbosity(runs))
    {
        println!("{line}");
    }
}

/// Say, on the run's own output, that a model was NOT measured.
///
/// Two channels on purpose and never one. `--json-output` owns stdout, so the
/// line goes to stderr there; on a human run it goes to stdout, next to the
/// table it is explaining, because a refusal printed only to stderr is invisible
/// to whoever reads the saved log. `✗` and the word NOT MEASURED carry it as
/// text, not as colour.
pub fn note_unmeasured(model: &str, reason: &str, json_output: bool) {
    let line = format!(
        "✗ {} {model}: {reason}",
        crate::ztools::model_eval::NOT_MEASURED
    );
    if json_output {
        eprintln!("{line}");
    } else {
        println!("{line}");
    }
}
