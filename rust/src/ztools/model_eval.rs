use crate::units::count;
use crate::ztools::eval::scoring_math::ratio;
use anyhow::Result;
use reqwest::blocking::Client;
use serde::{Deserialize, Serialize};
use std::fmt::Write as _;
use std::time::{Duration, Instant};

pub use crate::ztools::eval::{
    ChatMessage, Check, DEFAULT_MAX_IDLE_SECS, EvalTask, GpuLockGuard, clean_model_output,
    extract_content_from_code_blocks, extract_json, get_built_in_smoke_tasks, load_all_eval_tasks,
    load_taxes_tasks_from_dir, run_check, validate_file_summary,
};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ModelEvalResult {
    pub model: String,
    pub test_name: String,
    pub score: f64,
    pub passed: usize,
    pub total: usize,
    pub latency_ms: u64,
    pub status: String,
}

/// The marker in a row's `status` when the model was NEVER ASKED.
///
/// A prefix rather than a whole status because the reason travels with it: a
/// reader who sees only `NOT MEASURED` still has to be told WHICH failure, and an
/// operator deciding whether to re-run needs the URL and the error.
pub const NOT_MEASURED: &str = "NOT MEASURED";

/// What prints in a score cell that holds no score.
///
/// An em dash, never `0.0`: `0.0` is a reading, it is what a model that answered
/// every task wrongly scores, and it is exactly what a dead server used to
/// produce for every row. `tools/sweep_models.sh` counts a row as a scored task
/// by matching `| task | <digits> | `, so a dash here also stops the sweep filing
/// an outage as a model that ran every task.
pub const NO_SCORE: &str = "—";

impl ModelEvalResult {
    /// A row for something the model was never asked.
    ///
    /// `score` is [`f64::NAN`] rather than `0.0`, and that is the load-bearing
    /// choice: a struct field that must hold "no score" cannot hold a number
    /// that a mean, a sort or a CSV export will happily average into a
    /// leaderboard. NaN propagates, so a consumer that forgets to ask
    /// [`ModelEvalResult::was_measured`] produces a visibly broken number
    /// instead of a plausible wrong one.
    ///
    /// `total` keeps the checks that were never run, because that is the size of
    /// the hole; `passed` stays 0 because nothing was passed.
    #[must_use]
    pub fn not_measured(
        model: &str,
        test_name: &str,
        reason: &str,
        latency_ms: u64,
        total: usize,
    ) -> Self {
        Self {
            model: model.to_string(),
            test_name: test_name.to_string(),
            score: f64::NAN,
            passed: 0,
            total,
            latency_ms,
            status: format!("{NOT_MEASURED}: {reason}"),
        }
    }

    /// Does this row hold a measurement?
    ///
    /// The one predicate every consumer asks, so the table, the exit code and
    /// any future export cannot disagree about which rows are results.
    #[must_use]
    pub fn was_measured(&self) -> bool {
        !self.status.starts_with(NOT_MEASURED)
    }
}

/// Score one case's answer, or record that there was no answer.
///
/// Split out of [`eval_model`] so the two branches cannot blur: the branch that
/// scores and the branch that refuses are the whole defect, and a single
/// function holding both is a function where the refusal can be written as
/// "leave `checks_hit` at zero".
fn row_for(model_name: &str, case: &EvalTask, answer: Answer, elapsed: u64) -> ModelEvalResult {
    let total = case.checks.len();
    match answer {
        Answer::Content(output_text) => {
            let mut checks_hit = 0;
            // Parity with the Python eval: clean the output (thinking blocks,
            // stats, markdown fences) BEFORE judging it.
            let cleaned = clean_model_output(&output_text);
            let parsed = extract_json(&cleaned);
            for check in &case.checks {
                if run_check(check, &cleaned, parsed.as_ref()) {
                    checks_hit += 1;
                }
            }

            if checks_hit != total {
                println!(
                    "Test '{}' failed ({}/{}). Model output:\n---\n{}\n---",
                    case.name, checks_hit, total, output_text
                );
            }

            ModelEvalResult {
                model: model_name.to_string(),
                test_name: case.name.clone(),
                score: (ratio(checks_hit, total)) * 100.0,
                passed: checks_hit,
                total,
                latency_ms: elapsed,
                status: if checks_hit == total {
                    "passed"
                } else {
                    "failed"
                }
                .to_string(),
            }
        }
        Answer::Refused { reason, detail } => {
            // The full chain to stderr, the one-line verdict to the row: the
            // reason a sweep's log needs is in the row, and the reason a
            // debugging session needs is on stderr.
            eprintln!("✗ {NOT_MEASURED} {model_name} / {}: {detail}", case.name);
            ModelEvalResult::not_measured(model_name, &case.name, &reason, elapsed, total)
        }
    }
}

/// Measure `model_name` against the smoke roster.
///
/// # Errors
///
/// Only when the HTTP client cannot be built — a process-wide condition that
/// would fail identically for every model.
///
/// Everything else is a ROW. A transport failure, a non-2xx status and an
/// undecodable body each produce a [`ModelEvalResult::not_measured`] row
/// carrying the reason, rather than an `Err` and rather than a zero: the run
/// keeps whatever it really measured, the table cannot be misread, and
/// [`unmeasured_reason`] is what turns the rows into a non-zero exit.
pub fn eval_model(
    base_url: &str,
    model_name: &str,
    config: &crate::config::ZtoolsConfig,
) -> Result<Vec<ModelEvalResult>> {
    let defects = crate::ztools::model_health::probe_model_defects(model_name, None);
    if !defects.is_empty() {
        println!(
            "⚠ Skipping broken model '{}': {}",
            model_name,
            defects.join("; ")
        );
        // NOT MEASURED, and visibly so: this row used to carry `score: 0.0`
        // and `status: "refused: …"`, which a table rendered as a model that
        // scored nothing on one test. Nothing was asked of the model, so there
        // is no score, and the reason is the whole status.
        return Ok(vec![ModelEvalResult::not_measured(
            model_name,
            "packaging_health",
            &format!("refused: {}", defects.join("; ")),
            0,
            1,
        )]);
    }

    let lock_label = format!("eval {model_name}");
    let guard = GpuLockGuard::acquire(
        &lock_label,
        Duration::from_secs(config.llm_quick_timeout_secs),
        Duration::from_secs(DEFAULT_MAX_IDLE_SECS),
    );

    let timeout_secs = crate::ztools::eval::model_timeout(model_name)
        .map_or(config.llm_timeout_secs, |t| t.max(config.llm_timeout_secs));
    let client = Client::builder()
        .timeout(Duration::from_secs(timeout_secs))
        .build()?;
    let url = format!("{}/v1/chat/completions", base_url.trim_end_matches('/'));

    let mut results = Vec::new();

    // The loop does not stop at the first refusal. A server that dies halfway
    // through has still been asked about the tasks before it, and throwing those
    // away would trade one kind of missing data for another; the verdict is
    // carried by the rows, not by an early return.
    for case in get_test_cases() {
        if let Ok(ref g) = guard {
            g.heartbeat();
        }
        let start = Instant::now();
        let payload = serde_json::json!({
            "model": model_name,
            "messages": case.messages,
            "temperature": 0.0
        });
        let answer = ask(&client, &url, &payload);
        let elapsed = crate::units::millis(start.elapsed());
        results.push(row_for(model_name, &case, answer, elapsed));
    }

    Ok(results)
}

/// Every model the server lists, measured against the smoke roster.
///
/// # Errors
///
/// When the model list cannot be fetched, or the HTTP client cannot be built.
///
/// An individual model that cannot be REACHED no longer disappears from the
/// results: `eval_model` returns its reason as a [`ModelEvalResult::not_measured`]
/// row, so a sweep that lost two of twenty still reports the eighteen it
/// measured AND the two it did not, and [`unmeasured_reason`] fails the run. The
/// old `if let Ok(res)` dropped those two entirely — the same defect as the
/// swallowed transport error, one level up.
pub fn eval_all_models(
    base_url: &str,
    config: &crate::config::ZtoolsConfig,
) -> Result<Vec<ModelEvalResult>> {
    let models = get_available_models(base_url, config)?;
    let mut all_results = Vec::new();

    for model in models {
        all_results.extend(eval_model(base_url, &model, config)?);
    }
    Ok(all_results)
}

/// The rows in `results` that hold no measurement.
#[must_use]
pub fn unmeasured(results: &[ModelEvalResult]) -> Vec<&ModelEvalResult> {
    results.iter().filter(|r| !r.was_measured()).collect()
}

/// The sentence a run with unmeasured rows exits non-zero over.
///
/// `None` when everything was measured, so the caller has one condition to test
/// and the wording lives with the rule. It names the count AND the reasons,
/// because "the run failed" without them is a refusal with no diagnosis — the
/// same defect one level up. The causes are capped at [`CAUSES_LISTED`]: a
/// twenty-model sweep whose server is down has twenty identical reasons, and the
/// first three plus a count is the diagnosis. An empty roster is the other edge
/// this has to survive — no rows means nothing to be wrong about, which is not
/// the same as "measured nothing".
const CAUSES_LISTED: usize = 3;

#[must_use]
pub fn unmeasured_reason(results: &[ModelEvalResult]) -> Option<String> {
    let rows = unmeasured(results);
    if rows.is_empty() {
        return None;
    }
    let mut causes: Vec<String> = rows
        .iter()
        .map(|r| format!("{} / {}: {}", r.model, r.test_name, r.status))
        .collect();
    let rest = causes.len().saturating_sub(CAUSES_LISTED);
    causes.truncate(CAUSES_LISTED);
    if rest > 0 {
        causes.push(format!("…and {rest} more"));
    }
    Some(format!(
        "{NOT_MEASURED}: {} of {} row(s) hold no score, so this run measured \
         nothing about them: {}",
        rows.len(),
        results.len(),
        causes.join("; ")
    ))
}

/// Did this full-suite task's request reach the model at all?
///
/// The runner classifies a refused connection or a 5xx as `INFRA` and still
/// records a `score: 0` — right as data (the infra counters read it) and wrong as
/// a RESULT. This separates the two readings, and asks the same question
/// [`ModelEvalResult::was_measured`] asks for the smoke path; the answer is
/// [`crate::ztools::eval::TaskOutcome::was_measured`].
#[must_use]
pub fn task_was_measured(outcome: &crate::ztools::eval::TaskOutcome) -> bool {
    outcome.was_measured()
}

pub use crate::ztools::eval::completeness::tasks_unmeasured_reason;

/// Split an osaurus base URL ("<http://127.0.0.1:1337>") into host and port.
#[must_use]
pub fn parse_osaurus_url(url: &str) -> (String, u16) {
    let stripped = url
        .strip_prefix("http://")
        .or_else(|| url.strip_prefix("https://"))
        .unwrap_or(url);
    let trimmed = stripped.trim_end_matches('/');
    let whole = || (trimmed.to_string(), 1337);
    trimmed.rsplit_once(':').map_or_else(whole, |(host, port)| {
        port.parse()
            .map_or_else(|_| whole(), |p| (host.to_string(), p))
    })
}

/// Render full-suite [`TaskOutcome`]s as a markdown table, worst first: the
/// failures are what a reader scans for, so they lead.
///
/// A task whose request never reached the model renders NO SCORE. The runner
/// records those as `score: 0` with an `INFRA` category — right as data, wrong
/// as a result — and this table is where a reader decides whether a model is
/// any good, so a 0 in the score column has to mean "the model got it wrong".
#[must_use]
pub fn render_task_outcomes(outcomes: &[crate::ztools::eval::TaskOutcome]) -> String {
    use std::fmt::Write;
    let mut report = String::from("# Full Suite Eval\n\n");
    let _ = writeln!(report, "| task | score | status | time | error |");
    let _ = writeln!(report, "|------|-------|--------|------|-------|");
    let mut rows: Vec<_> = outcomes.iter().collect();
    rows.sort_by_key(|o| o.score);
    let mut not_measured = 0usize;
    for o in &rows {
        let (score, status) = if task_was_measured(o) {
            (o.score.to_string(), o.status.clone())
        } else {
            not_measured += 1;
            (
                NO_SCORE.to_string(),
                format!(
                    "{NOT_MEASURED}: {}",
                    o.error.as_deref().unwrap_or(o.failure_category.as_str())
                ),
            )
        };
        let _ = writeln!(
            report,
            "| {} | {score} | {status} | {:.1}s | {} |",
            o.task,
            o.time_secs,
            o.error.as_deref().unwrap_or("-")
        );
    }
    if !rows.is_empty() {
        // The mean is over the tasks that were MEASURED. Averaging in rows the
        // model never saw is how a dead server printed "mean score 0.0" for a
        // model that had not been measured at all — and how a 4-task outage
        // would have dragged a real score down if the rest had answered.
        let measured: Vec<u8> = rows
            .iter()
            .filter(|o| task_was_measured(o))
            .map(|o| o.score)
            .collect();
        let ok = rows.iter().filter(|o| o.status == "ok").count();
        let _ = writeln!(report);
        if measured.is_empty() {
            let _ = writeln!(
                report,
                "{} tasks, {ok} ok, {not_measured} {NOT_MEASURED}; no mean — nothing was measured",
                rows.len()
            );
        } else {
            let mean: f64 =
                measured.iter().map(|s| f64::from(*s)).sum::<f64>() / count(measured.len());
            let _ = writeln!(
                report,
                "{} tasks, {ok} ok, {not_measured} {NOT_MEASURED}, mean score \
                 {mean:.1} over the {} measured",
                rows.len(),
                measured.len()
            );
        }
    }
    report
}

/// Render the smoke-path report.
///
/// The two rules the table exists to enforce: a row that was never measured
/// carries [`NO_SCORE`] and says so in its status, and the summary line counts
/// those rows rather than averaging them away. A reader who sees only the
/// percentages must not be able to mistake a dead server for a bad model.
#[must_use]
pub fn render_eval_report(results: &[ModelEvalResult]) -> String {
    let mut out = String::new();
    out.push_str("# Model Quality Evaluation Benchmark\n\n");
    out.push_str("| Model | Test | Score | Passed | Latency | Status |\n");
    out.push_str("| :--- | :--- | :--- | :--- | :--- | :--- |\n");

    for r in results {
        let (score, passed) = if r.was_measured() {
            (
                format!("{:.1}%", r.score),
                format!("{}/{}", r.passed, r.total),
            )
        } else {
            (NO_SCORE.to_string(), NO_SCORE.to_string())
        };
        let _ = writeln!(
            out,
            "| **{}** | {} | {score} | {passed} | {}ms | {} |",
            r.model, r.test_name, r.latency_ms, r.status
        );
    }

    if let Some(reason) = unmeasured_reason(results) {
        let _ = write!(out, "\n> [!WARNING]\n> {reason}\n");
    }

    out
}

#[must_use]
pub fn get_test_cases() -> Vec<EvalTask> {
    get_built_in_smoke_tasks()
}

/// # Errors
///
/// When the HTTP client cannot be built, and when the `/v1/models` endpoint
/// cannot be reached or does not answer JSON. A well-formed answer listing
/// no usable models is `Ok(vec![])`.
pub fn get_available_models(
    base_url: &str,
    config: &crate::config::ZtoolsConfig,
) -> Result<Vec<String>> {
    let client = Client::builder()
        .timeout(Duration::from_secs(config.llm_quick_timeout_secs))
        .build()?;
    let url = format!("{}/v1/models", base_url.trim_end_matches('/'));
    let resp: serde_json::Value = client.get(&url).send()?.json()?;

    let mut models = Vec::new();
    if let Some(data) = resp.get("data").and_then(|d| d.as_array()) {
        for m in data {
            if let Some(id) = m.get("id").and_then(|id| id.as_str())
                && !id.contains("foundation")
                && !id.contains("diffusion")
            {
                models.push(id.to_string());
            }
        }
    }
    Ok(models)
}

// The request classification lives beside the loop that consumes it: "did this
// answer?" and "what do we do with the answer?" are one change, and the bug
// this replaces was one of them quietly assuming the other.
#[path = "model_eval_transport.rs"]
mod transport;
use transport::{Answer, ask};

#[cfg(test)]
#[path = "model_eval_tests.rs"]
mod tests;
