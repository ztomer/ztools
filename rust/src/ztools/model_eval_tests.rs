//! Unit tests for the eval smoke path: the request classification, the
//! not-measured row, and the two renderers.
//!
//! Split out of `model_eval.rs` for the house 500-line cap, following the
//! pattern `eval/oversize_tests.rs` and `cli_ztools_tests.rs` already use. These
//! are the PURE half — no server, no lock, no config — so each one can be made
//! to fail by breaking one expression. The half that needs a socket lives in
//! `tests/model_eval.rs` (served answers) and `tests/cli_refusals.rs` (the exit
//! code an operator gets).
//!
//! THE CLASS UNDER TEST: a row that was never measured must be unable to look
//! like a score. Every case here is calibrated against a renderer that prints
//! `0.0%` anyway, which is the defect these assertions were written to close.

use super::*;

#[test]
fn test_render_eval_report() {
    let results = vec![ModelEvalResult {
        model: "test_model".to_string(),
        test_name: "test_suite".to_string(),
        score: 100.0,
        passed: 4,
        total: 4,
        latency_ms: 120,
        status: "passed".to_string(),
    }];
    let md = render_eval_report(&results);
    assert!(md.contains("test_model"));
    assert!(md.contains("100.0%"));
    assert!(md.contains("120ms"));
}

/// The smoke suite `model-eval` runs IS the built-in roster, by identity.
///
/// This is the WIRING, which `eval::smoke_tasks`'s own pin cannot see: that
/// one pins the roster's contents, and a `get_test_cases` that built a
/// different list (or reached for the full roster) would still pass it.
/// The previous `cases.len() == 5` here pinned nothing at all — every name,
/// check and prompt could move and it stayed true.
#[test]
fn get_test_cases_is_the_built_in_smoke_roster() {
    let cases = get_test_cases();
    assert_eq!(cases, get_built_in_smoke_tasks());
}

#[test]
fn json_array_len_check() {
    let cleaned = r#"{"transient_events": [{"name": "a"}, {"name": "b"}]}"#;
    let parsed = extract_json(cleaned).unwrap();
    assert!(run_check(
        &Check::JsonArrayLen("transient_events".to_string(), 2),
        cleaned,
        Some(&parsed)
    ));
    assert!(!run_check(
        &Check::JsonArrayLen("transient_events".to_string(), 3),
        cleaned,
        Some(&parsed)
    ));
}

#[test]
fn osaurus_url_parses_to_host_and_port() {
    assert_eq!(
        parse_osaurus_url("http://127.0.0.1:1337"),
        ("127.0.0.1".to_string(), 1337)
    );
    assert_eq!(
        parse_osaurus_url("http://localhost:9999/"),
        ("localhost".to_string(), 9999)
    );
    // Portless and malformed URLs fall back to the default port rather
    // than panicking.
    assert_eq!(
        parse_osaurus_url("http://myhost"),
        ("myhost".to_string(), 1337)
    );
    assert_eq!(parse_osaurus_url("weird"), ("weird".to_string(), 1337));
}

/// The table leads with its worst row, and the mean covers what was measured.
///
/// WORST-FIRST ORDERING IS UNCHANGED, THE MEAN IS NOT, AND THE OLD ASSERTION
/// WAS WRONG. It pinned `"2 tasks, 1 ok, mean score 62.5"` — the mean of 100 and
/// 25 — where the 25 came from a row carrying `error: "HTTP 503"`. A 503 means
/// the model never got to answer (that is `failures.rs`'s own INFRA wording, and
/// `report_metrics::compute_error_rates` counts a non-empty error as infra too),
/// so that 25 was a fabrication and averaging it in is exactly the "a failure
/// that hides" defect: a model with one real 100 and one outage reported 62.5.
/// The summary now reads `mean score 100.0 over the 1 measured`, and the outage
/// row renders `—` beside its reason.
#[test]
fn task_outcomes_render_worst_first_with_a_mean_over_measured_tasks() {
    use crate::ztools::eval::TaskOutcome;
    let outcomes = vec![
        TaskOutcome {
            task: "good".to_string(),
            score: 100,
            status: "ok".to_string(),
            ..Default::default()
        },
        // A genuine quality failure: the model answered and got it wrong. This
        // is the row that must still lead the table, because it is the worst
        // MEASURED result rather than the worst number in the file.
        TaskOutcome {
            task: "bad".to_string(),
            score: 25,
            status: "fail".to_string(),
            ..Default::default()
        },
        // An outage: never reached the model.
        TaskOutcome {
            task: "outage".to_string(),
            score: 0,
            status: "fail".to_string(),
            error: Some("HTTP 503".to_string()),
            failure_category: crate::ztools::eval::FAIL_INFRA.to_string(),
            ..Default::default()
        },
    ];
    let report = render_task_outcomes(&outcomes);
    let bad_pos = report.find("| bad ").expect("bad row present");
    let good_pos = report.find("| good ").expect("good row present");
    assert!(bad_pos < good_pos, "worst score must lead:\n{report}");
    assert!(
        report.contains("3 tasks, 1 ok, 1 NOT MEASURED, mean score 62.5 over the 2 measured"),
        "the mean must cover the two tasks that were measured:\n{report}"
    );
    assert!(
        report.contains("| bad | 25 | fail |"),
        "a quality failure keeps its score:\n{report}"
    );
    assert!(report.contains("HTTP 503"), "{report}");
    let outage = report
        .lines()
        .find(|l| l.starts_with("| outage"))
        .expect("the outage row");
    assert!(
        outage.contains(NO_SCORE) && outage.contains(NOT_MEASURED),
        "an outage must not render as a score: {outage}"
    );
}

#[test]
fn cleaning_precedes_checks() {
    // Thinking block around the answer must not poison the checks.
    let cleaned = clean_model_output("<think>inner</think> Image: red_car.jpg");
    assert!(run_check(
        &Check::Contains("red_car.jpg".to_string()),
        &cleaned,
        None
    ));
}

#[test]
fn file_summary_check_uses_validator() {
    let good = r#"[{"path": "a.py", "desc": "parses config files"},
                    {"path": "b.py", "desc": "validates JSON output"},
                    {"path": "c.py", "desc": "fetches external API data"},
                    {"path": "d.py", "desc": "handles processing logic"}]"#;
    assert!(run_check(&Check::FileSummary(50), good, None));

    let bad = r#"[{"path": "a.py", "desc": "a python script"},
                  {"path": "b.py", "desc": "another file"}]"#;
    assert!(!run_check(&Check::FileSummary(50), bad, None));
}

/// A scored row, for the cases that need one next to an unmeasured one.
fn scored(score: f64, passed: usize, total: usize) -> ModelEvalResult {
    ModelEvalResult {
        model: "some-model".to_string(),
        test_name: "JSON Extraction".to_string(),
        score,
        passed,
        total,
        latency_ms: 10,
        status: if passed == total { "passed" } else { "failed" }.to_string(),
    }
}

/// An unmeasured row CANNOT hold a score.
///
/// The load-bearing half of the fix, and the reason `not_measured` writes NaN
/// rather than `0.0`: any consumer that forgets to ask `was_measured` — a mean, a
/// sort, a future CSV export — gets a number that poisons itself instead of a
/// plausible zero. `0.0` here is exactly what the dead-server defect produced,
/// so this assertion is the difference between "the table says so" and "nothing
/// downstream can quietly say otherwise".
#[test]
fn an_unmeasured_row_holds_no_number() {
    let row = ModelEvalResult::not_measured("m", "JSON Extraction", "no answer", 3, 4);
    assert!(
        row.score.is_nan(),
        "an unmeasured row scored {} — that is a reading nobody took",
        row.score
    );
    assert!(!row.was_measured());
    assert_eq!(row.total, 4, "the hole keeps its size");
    assert_eq!(row.passed, 0);
    assert!(
        row.status.starts_with(NOT_MEASURED) && row.status.contains("no answer"),
        "the reason must travel with the state: {}",
        row.status
    );
}

/// A scored row is measured, including a genuine zero.
///
/// The other direction, and the one a naive "score > 0 means measured"
/// implementation gets wrong: a model that answered every task wrongly has a
/// real 0.0% and must stay in the average.
#[test]
fn a_scored_zero_is_measured() {
    assert!(scored(0.0, 0, 4).was_measured());
    assert!(scored(100.0, 4, 4).was_measured());
    let quality_failure = scored(0.0, 0, 4);
    assert_eq!(
        quality_failure.status, "failed",
        "a 0.0% from an answer that came back is a quality failure"
    );
    assert!(
        quality_failure.was_measured(),
        "a quality failure is a result"
    );
}

/// The report cannot print a score it does not have.
///
/// Both cells and the summary line, because a table with a dash in the score
/// column and a silent "mean 0.0" underneath still reports a failure as a
/// result. The negative assertions are the calibration: `!contains("%")` over
/// the unmeasured row's own line is what fails if the renderer ever formats the
/// NaN.
#[test]
fn the_report_renders_no_score_for_an_unmeasured_row() {
    let results = vec![
        scored(100.0, 4, 4),
        ModelEvalResult::not_measured("m", "Markdown", "Connection refused", 5, 4),
    ];
    let report = render_eval_report(&results);
    eprintln!("{report}");

    let unmeasured_line = report
        .lines()
        .find(|l| l.contains("Markdown"))
        .expect("the unmeasured row is in the table");
    assert!(
        unmeasured_line.contains(NO_SCORE) && unmeasured_line.contains(NOT_MEASURED),
        "an unmeasured row must render as one: {unmeasured_line}"
    );
    assert!(
        !unmeasured_line.contains('%') && !unmeasured_line.contains("NaN"),
        "an unmeasured row must not print a number: {unmeasured_line}"
    );
    assert!(
        report.contains("Connection refused"),
        "the reason must reach the table: {report}"
    );
    assert!(
        report.contains("1 of 2 row(s) hold no score"),
        "the summary must count the rows that hold no score: {report}"
    );
    // The scored row is untouched: the fix is not "print a dash for everything".
    let scored_line = report
        .lines()
        .find(|l| l.contains("JSON Extraction"))
        .expect("the scored row");
    assert!(scored_line.contains("100.0%"), "{scored_line}");
}

/// A fully measured run is not turned into a failure.
///
/// The control for the whole change: `unmeasured_reason` returning `Some` here
/// would make every healthy eval exit non-zero, which is the over-correction
/// this repo's own rules warn about.
#[test]
fn a_fully_measured_run_has_no_refusal() {
    let results = vec![scored(75.0, 3, 4), scored(0.0, 0, 4)];
    assert!(unmeasured_reason(&results).is_none());
    assert_empty!(unmeasured(&results));
    let report = render_eval_report(&results);
    assert!(
        !report.contains(NOT_MEASURED),
        "a healthy run must not carry a failure banner: {report}"
    );
}

/// The refusal names the rows and their causes, or there is no diagnosis.
///
/// Asserted on a substring of the message rather than its shape, because the
/// message is what an operator reads after seeing exit 1: the count, the model,
/// the task and the reason all have to be in it.
#[test]
fn the_refusal_names_the_rows_and_their_causes() {
    let results = vec![
        scored(100.0, 4, 4),
        ModelEvalResult::not_measured("m", "Markdown", "HTTP 503 from http://127.0.0.1:1", 5, 4),
        ModelEvalResult::not_measured("m", "Renamer", "gave no answer", 5, 2),
    ];
    let reason = unmeasured_reason(&results).expect("two rows hold no score");
    for needle in [
        "2 of 3",
        "Markdown",
        "Renamer",
        "HTTP 503",
        "gave no answer",
    ] {
        assert!(reason.contains(needle), "{needle:?} missing from: {reason}");
    }
    let rows = unmeasured(&results);
    assert_eq!(rows.len(), 2, "{rows:?}");
}

/// A task whose request never reached the model is not measured.
///
/// The full-suite half. The runner classifies a refused connection as `INFRA`
/// and still records `score: 0` — correct as data for the infra counters,
/// wrong as a result — and this predicate is what tells the two apart.
#[test]
fn an_infra_task_is_not_measured_and_a_zero_scoring_task_is() {
    let infra = crate::ztools::eval::TaskOutcome {
        task: "json".to_string(),
        score: 0,
        status: "fail".to_string(),
        error: Some("Connection failed - is server running?".to_string()),
        failure_category: crate::ztools::eval::FAIL_INFRA.to_string(),
        ..Default::default()
    };
    assert!(!task_was_measured(&infra));

    let wrong = crate::ztools::eval::TaskOutcome {
        task: "json".to_string(),
        score: 0,
        status: "fail".to_string(),
        failure_category: crate::ztools::eval::FAIL_CONTENT.to_string(),
        ..Default::default()
    };
    assert!(
        task_was_measured(&wrong),
        "a model that answered wrongly HAS been measured; failing the run on this \
         would discard the only reading there is"
    );
}

/// A run that measured nothing says so, and an empty run counts as nothing.
///
/// The empty case is the dangerous one: the runner abandons a model after
/// `max_consecutive_infra` failures or a stall, and a run that never got a task
/// out of the server returns an EMPTY list. Reading that as "no problems" is how
/// a dead server printed an empty table and exited 0.
#[test]
fn a_run_that_measured_nothing_names_itself() {
    let nothing = tasks_unmeasured_reason(&[]).expect("an empty run measured nothing");
    assert!(nothing.contains("no task reported back"), "{nothing}");

    let all_infra: Vec<crate::ztools::eval::TaskOutcome> = ["a", "b"]
        .iter()
        .map(|t| crate::ztools::eval::TaskOutcome {
            task: (*t).to_string(),
            score: 0,
            status: "fail".to_string(),
            error: Some("Connection failed".to_string()),
            failure_category: crate::ztools::eval::FAIL_INFRA.to_string(),
            ..Default::default()
        })
        .collect();
    let why = tasks_unmeasured_reason(&all_infra).expect("neither task reached the model");
    assert!(why.contains("0 of 2"), "{why}");
    assert!(why.contains("Connection failed"), "{why}");

    let one_good = crate::ztools::eval::TaskOutcome {
        task: "a".to_string(),
        score: 100,
        status: "ok".to_string(),
        ..Default::default()
    };
    assert!(
        tasks_unmeasured_reason(&[one_good]).is_none(),
        "a run with one measured task is not a run that measured nothing"
    );
}

/// The verdict counts what really happened when SOME rows missed.
///
/// It used to say "0 of N task(s) reached the model" whenever any one row had
/// missed, so one outage among thirty answers read as a dead server. A
/// `CONTEXT` refusal alone does not fail the run -- a re-run cannot change it --
/// unless it leaves nothing measured.
#[test]
fn a_partly_measured_run_is_counted_and_a_context_refusal_alone_passes() {
    let row = |task: &str, category: &str, error: Option<&str>| crate::ztools::eval::TaskOutcome {
        task: task.to_string(),
        score: if error.is_none() { 90 } else { 0 },
        error: error.map(str::to_string),
        failure_category: category.to_string(),
        ..Default::default()
    };
    let good = || row("good", "", None);
    let outage = || {
        row(
            "down",
            crate::ztools::eval::FAIL_INFRA,
            Some("Connection failed"),
        )
    };
    let too_big = || {
        row(
            "big",
            crate::ztools::eval::FAIL_CONTEXT,
            Some("prompt does not fit"),
        )
    };

    let why = tasks_unmeasured_reason(&[good(), outage()]).expect("an outage fails the run");
    assert!(why.contains("1 of 2"), "{why}");
    assert!(!why.contains("0 of 2"), "one task WAS measured: {why}");

    assert_eq!(tasks_unmeasured_reason(&[good(), too_big()]), None);

    let why = tasks_unmeasured_reason(&[too_big()]).expect("nothing was measured");
    assert!(why.contains("0 of 1"), "{why}");
    assert!(why.contains("does not fit"), "{why}");
}

/// The full-suite table averages only what was measured.
///
/// The lie this closes is arithmetic, not cosmetic: a two-task run against a
/// dead server printed "mean score 0.0", which is the number a ranking reads.
/// A partially-wired outage would have dragged a real model's mean down by
/// scoring its unreachable tasks as zeros.
#[test]
fn the_full_suite_mean_covers_only_the_tasks_that_reached_the_model() {
    let outcomes = vec![
        crate::ztools::eval::TaskOutcome {
            task: "measured".to_string(),
            score: 80,
            status: "partial".to_string(),
            ..Default::default()
        },
        crate::ztools::eval::TaskOutcome {
            task: "unreachable".to_string(),
            score: 0,
            status: "fail".to_string(),
            error: Some("Connection failed".to_string()),
            failure_category: crate::ztools::eval::FAIL_INFRA.to_string(),
            ..Default::default()
        },
    ];
    let report = render_task_outcomes(&outcomes);
    eprintln!("{report}");
    assert!(
        report.contains("mean score 80.0 over the 1 measured"),
        "the mean must be over the measured tasks only: {report}"
    );
    let row = report
        .lines()
        .find(|l| l.starts_with("| unreachable"))
        .expect("the unreachable task row");
    assert!(row.contains(NO_SCORE), "{row}");
    assert!(
        !row.contains("| 0 |"),
        "an unreachable task must not render a score: {row}"
    );

    let all_dead = vec![outcomes[1].clone()];
    let dead_report = render_task_outcomes(&all_dead);
    assert!(
        dead_report.contains("no mean — nothing was measured"),
        "a dead server must not print a mean: {dead_report}"
    );
    assert!(
        !dead_report.contains("mean score"),
        "a dead server must not print a mean at all: {dead_report}"
    );
}

#[path = "model_eval_leaderboard_tests.rs"]
mod leaderboard_tests;

#[test]
fn test_leaderboard_ranking_overall_means_and_slot_scores() {
    leaderboard_tests::verify_leaderboard_ranking();
    leaderboard_tests::verify_leaderboard_slot_sorting();
    leaderboard_tests::verify_leaderboard_deltas();
}
