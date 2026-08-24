//! Eval runner: the per-model evaluation loop.
//!
//! Ported from `references/eval/run.py::run_eval`, written against the real
//! Rust modules (`transport`, `task_loader`) and kept compiling from its first
//! commit -- the previous draft was an orphan that never compiled, which is how
//! 75 errors of drift accumulated unseen.
//!
//! The loop: for each task, request a completion through the transport's full
//! pipeline (quirks, reasoning-overrun guard, missing-model substitution),
//! score the cleaned output against the task's checks, retry on failure, keep
//! the best attempt. A model whose SERVER is clearly failing (consecutive
//! transport errors) is abandoned after [`RunnerConfig::max_consecutive_infra`]
//! attempts: grinding on only produces more zeros, and those zeros must never
//! be read as quality results.
//!
//! Scoring is the established Rust pattern (as in `model_eval.rs`): each check
//! is boolean via `run_check`, and the task score is the fraction passed.
//!
//! The learning behaviours -- prefill measurement, per-task learned timeouts,
//! signal recording, the stall watchdog -- are OFF by default so tests stay
//! hermetic; the CLI sweep enables them via [`run_eval_with_signals`], which is
//! the [`run_eval`] analog of Python's `measure_prefill=True` production path.

use std::time::Instant;

use serde::Serialize;

use crate::ztools::eval::prefill::{measure_prefill_rate, record_prefill_rate};
use crate::ztools::eval::signals::{effective_timeout, load_signals, record_signal, save_signals};
use crate::ztools::eval::failures::{
    classify_failure, reasoning_overrun_was_guard_aborted, reasoning_retry_budget,
    FAIL_REASONING,
};
use crate::ztools::eval::model_resolve::is_generative_model;
use crate::ztools::eval::task_loader::{check_graded_score, run_check, EvalTask};
use crate::ztools::eval::transport::{self, RequestSpec};
use crate::ztools::eval::watchdog::{is_stalled, model_stall_duration};
use crate::ztools::eval::SignalStore;

/// Result of evaluating ONE task for ONE model. Errors are data.
#[derive(Debug, Clone, Default, PartialEq, Serialize)]
pub struct TaskOutcome {
    pub task: String,
    /// 0-100: fraction of the task's checks that passed on the best attempt.
    pub score: u8,
    /// "ok" (>=90), "partial" (>=50), "fail" (<50).
    pub status: String,
    pub time_secs: f64,
    pub error: Option<String>,
    pub finish_reason: String,
    pub aborted: bool,
    pub abort_reason: String,
    /// Why the best attempt failed: INFRA / TIMEOUT / PARSE / FORMAT /
    /// CONTENT / REASONING ("" when the task passed). Drives retry escalation
    /// and infra abandonment, matching eval/failures.py.
    #[serde(default)]
    pub failure_category: String,
    /// Set when the configured tag was dead and a stand-in answered instead.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub substituted_from: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub substituted_to: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub substitution_reason: Option<String>,
}

/// Knobs for one eval sweep over one model.
#[derive(Debug, Clone)]
pub struct RunnerConfig {
    pub host: String,
    pub port: u16,
    /// Greedy by default: a leaderboard has to be reproducible. Sampling turned
    /// a 100% run into a 0% run on identical input once; ranking models on
    /// sampled runs measures the sampler.
    pub temperature: f64,
    pub max_tokens: u32,
    /// Floor timeout used when learning is off (`record_signals = false`);
    /// otherwise the learned per-task value supersedes it.
    pub timeout_secs: u64,
    /// Retries AFTER the first attempt (0 = no retries).
    pub max_retries: u32,
    /// Consecutive transport failures before abandoning the model.
    pub max_consecutive_infra: u32,
    /// Retry once against a servable stand-in when the configured tag is gone.
    pub allow_model_substitution: bool,
    /// Production-path switches, off so unit/integration tests stay hermetic:
    /// measure this model's prefill rate up front, size every request from the
    /// learned per-task timeout, record signals after each task, and enforce
    /// the stall watchdog between tasks. See [`run_eval_with_signals`].
    pub record_signals: bool,
}

impl Default for RunnerConfig {
    fn default() -> Self {
        Self {
            host: "127.0.0.1".to_string(),
            port: 1337,
            temperature: 0.0,
            max_tokens: 2048,
            timeout_secs: 600,
            max_retries: 1,
            max_consecutive_infra: 4,
            allow_model_substitution: true,
            record_signals: false,
        }
    }
}

fn status_for(score: u8) -> &'static str {
    if score >= 90 {
        "ok"
    } else if score >= 50 {
        "partial"
    } else {
        "fail"
    }
}

/// Score one output against every check of a task, 0-100.
///
/// When EVERY check is inherently graded (the taxes grounded validators), the
/// task score is the mean of their numeric verdicts -- an 80/100 answer must
/// surface as 80/partial, not collapse to 0/fail behind a boolean threshold.
/// Mixed or purely-boolean tasks keep the passed-fraction semantics.
pub fn score_output(task: &EvalTask, cleaned: &str, parsed: Option<&serde_json::Value>) -> u8 {
    if task.checks.is_empty() {
        return 0;
    }
    let graded: Vec<i64> = task
        .checks
        .iter()
        .filter_map(|c| check_graded_score(c, cleaned, parsed))
        .collect();
    if graded.len() == task.checks.len() {
        let mean = graded.iter().sum::<i64>() / graded.len() as i64;
        return mean.clamp(0, 100) as u8;
    }
    let passed = task
        .checks
        .iter()
        .filter(|c| run_check(c, cleaned, parsed))
        .count();
    ((passed * 100 + task.checks.len() / 2) / task.checks.len()) as u8
}

/// Is this outcome a SERVER problem rather than a model-quality result?
/// Transport errors and timeouts are infra; a reasoning overrun is recorded as
/// what it is -- this model could not finish this task here.
fn is_infra_failure(outcome: &TaskOutcome) -> bool {
    outcome.error.is_some()
}

fn outcome_from(task: &EvalTask, r: &transport::TransportResult) -> TaskOutcome {
    TaskOutcome {
        task: task.name.clone(),
        time_secs: r.time_secs,
        error: r.error.clone(),
        finish_reason: r.finish_reason.clone(),
        aborted: r.aborted,
        abort_reason: r.abort_reason.clone(),
        substituted_from: r.substituted_from.clone(),
        substituted_to: r.substituted_to.clone(),
        substitution_reason: r.substitution_reason.clone(),
        ..Default::default()
    }
}

/// Evaluate `model` against `tasks` in order. Never panics on transport
/// problems; every failure lands in the returned outcomes as data. Learning
/// (signals, prefill, learned timeouts, watchdog) stays OFF -- see
/// [`run_eval_with_signals`] for the production path.
pub fn run_eval(model: &str, tasks: &[EvalTask], cfg: &RunnerConfig) -> Vec<TaskOutcome> {
    run_eval_inner(model, tasks, cfg, &mut SignalStore::new())
}

/// The production path (`ztools model-eval --suite full`): measures and records
/// this model's prefill/cold-start/decode capabilities up front, sizes each
/// request from the learned per-task timeout instead of the static floor,
/// records p95/retry signals after each task, and stops when the watchdog sees
/// no task completion within the stall ceiling. Signals are loaded from and
/// saved back to `eval_signals.json` by this function.
pub fn run_eval_with_signals(model: &str, tasks: &[EvalTask], cfg: &RunnerConfig) -> Vec<TaskOutcome> {
    let mut signals = load_signals();

    // Measure this model's ingestion rate before timing anything else. It is
    // what every tool's context budget is sized from, and the alternative was a
    // hand-picked constant that turned out to be 35-90x too low. One extra
    // request per model per run -- skipped for embedding models, which have
    // no ingestion rate to measure (judged by their config, not their name).
    if is_generative_model(model) {
        let rate = measure_prefill_rate(&mut signals, model, &cfg.host, cfg.port);
        record_prefill_rate(&mut signals, model, rate);
    }

    let outcomes = run_eval_inner(model, tasks, cfg, &mut signals);
    save_signals(&signals);
    outcomes
}

fn run_eval_inner(
    model: &str,
    tasks: &[EvalTask],
    cfg: &RunnerConfig,
    signals: &mut SignalStore,
) -> Vec<TaskOutcome> {
    let mut outcomes = Vec::new();
    let mut consecutive_infra: u32 = 0;
    let mut last_completion = Instant::now();
    let stall_limit = model_stall_duration();
    // Whether THIS model has already proven it cannot use a bigger budget. Two
    // model shapes produce an identical first attempt -- one wants more room and
    // answers once it has it (nemotron), the other expands its reasoning to fill
    // whatever it is handed and never answers (ornith-9b: 72,005 chars at 32,000
    // tokens, 144,441 at 64,000, guard-aborted at both). One attempt cannot tell
    // them apart, so the escalation is bought ONCE per model and the outcome
    // remembered; the shape is a property of the model, not of the task.
    let mut reasoning_escalation_futile = false;

    for task in tasks {
        // Progress, not duration: a healthy multi-hour sweep keeps completing
        // tasks; a wedged server does not, and must be cut before it eats the
        // GPU reservation.
        if cfg.record_signals && is_stalled(last_completion, stall_limit) {
            eprintln!("⚠ Abandoning {model}: no task completed within the stall ceiling");
            break;
        }

        let mut best: Option<TaskOutcome> = None;
        let prompt_chars: usize = task.messages.iter().map(|m| m.content.len()).sum();
        // Production path resolves the output budget per task/model from
        // config exactly like the Python eval (`get_max_tokens_for_task`);
        // the hermetic path keeps the configured constant.
        let max_tokens = if cfg.record_signals {
            crate::ztools::eval::budgets::max_tokens_for_task(&task.name, model)
        } else {
            cfg.max_tokens
        };
        let timeout_secs = if cfg.record_signals {
            effective_timeout(model, &task.name, prompt_chars, max_tokens)
        } else {
            cfg.timeout_secs
        };
        let mut attempts_used: u32 = 0;
        let mut best_diagnosis = crate::ztools::eval::failures::Diagnosis {
            category: "",
            reason: String::new(),
            evidence: String::new(),
        };

        for attempt in 0..=cfg.max_retries {
            attempts_used += 1;
            // A retry that repeats the identical call cannot fix a reasoning
            // overrun -- the model will think itself past the budget again.
            // Retry with MORE room, bounded: reasoning scales with the TASK.
            if attempt > 0
                && best_diagnosis.category == FAIL_REASONING
                && reasoning_escalation_futile
            {
                // Proven on an earlier task: this model fills whatever budget it
                // gets and the guard cuts it every time. Escalating again buys a
                // longer failure, and repeating the base call cannot help either --
                // attempt 1 already hit the guard at exactly that budget. Take the
                // zero now instead of paying twice more for it.
                eprintln!(
                    "  · skipping the retry for {}: {model} already reasoned past an \
                     escalated budget, so more room cannot help",
                    task.name
                );
                break;
            }
            let attempt_tokens = if attempt > 0 && best_diagnosis.category == FAIL_REASONING {
                let escalated = reasoning_retry_budget(max_tokens);
                eprintln!(
                    "  · previous attempt reasoned past {max_tokens}; retrying with {escalated}"
                );
                escalated
            } else {
                max_tokens
            };
            let spec = RequestSpec {
                model,
                messages: &task.messages,
                host: &cfg.host,
                port: cfg.port,
                temperature: cfg.temperature,
                max_tokens: attempt_tokens,
                timeout_secs,
                allow_substitution: cfg.allow_model_substitution,
                stream_guard: true,
            };
            let r = transport::call(&spec, false);
            let candidate = outcome_from(task, &r);
            let is_best = match &best {
                // Errors rank below any scored attempt.
                Some(b) => b.error.is_some() && candidate.error.is_none(),
                None => true,
            };
            if is_best {
                best = Some(candidate);
            }
            let score = score_output(task, &r.content, None);
            best_diagnosis = classify_failure(
                r.error.as_deref(),
                &r.content,
                &r.reasoning_content,
                &r.finish_reason,
                None,
                score,
                false,
            );
            // An escalated attempt the guard cut as well is the evidence that this
            // model expands to fill: strictly more room, strictly more of it spent
            // thinking, still no answer. Recorded next to the attempt that proves
            // it, so no later task re-buys the proof.
            if attempt_tokens > max_tokens
                && reasoning_overrun_was_guard_aborted(&r.finish_reason)
            {
                reasoning_escalation_futile = true;
            }
            // Archive what the model actually said BEFORE anything decides
            // what this score means. A scorer question asked after the fact is
            // unanswerable without the output, and re-running costs hours on a
            // one-model-at-a-time machine.
            if cfg.record_signals {
                crate::ztools::eval::outputs::save_output(
                    &crate::ztools::eval::outputs::OutputRecord {
                        model,
                        task: &task.name,
                        content: &r.content,
                        reasoning: &r.reasoning_content,
                        error: r.error.as_deref(),
                        score,
                        failure_reason: &best_diagnosis.reason,
                    },
                    None,
                );
            }
            // A scored attempt always outranks the error/empty placeholder in
            // `best`, even at 0 -- otherwise the placeholder's blank status
            // leaks into the result. Ties take the later attempt.
            if let Some(b) = best.as_mut() {
                if b.error.is_some() || score >= b.score {
                    b.score = score;
                    b.status = status_for(score).to_string();
                }
            }
            if best.as_ref().is_some_and(|b| b.score >= 90) {
                break;
            }
        }

        let mut outcome = best.unwrap_or_else(|| TaskOutcome {
            task: task.name.clone(),
            status: "fail".to_string(),
            ..Default::default()
        });
        // What the BEST attempt died of -- "" when nothing failed. Drives
        // parse-failure counting below and names the failure in reports.
        outcome.failure_category = best_diagnosis.category.to_string();

        if cfg.record_signals {
            let is_parse_failure = outcome.failure_category == "PARSE";
            record_signal(
                signals,
                model,
                &task.name,
                outcome.time_secs,
                attempts_used > 1,
                is_parse_failure,
            );
            last_completion = Instant::now();
        }

        if is_infra_failure(&outcome) {
            consecutive_infra += 1;
        } else {
            consecutive_infra = 0;
        }
        let abandoning = consecutive_infra >= cfg.max_consecutive_infra;
        outcomes.push(outcome);
        if abandoning {
            break;
        }
    }

    outcomes
}
