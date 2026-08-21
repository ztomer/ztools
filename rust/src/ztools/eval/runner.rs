//! Eval runner: the per-model evaluation loop.
//!
//! Ported from `references/eval/run.py::run_eval`, written against the real
//! Rust modules (`transport`, `task_loader`) and kept compiling from its first
//! commit -- the previous draft was an orphan that never compiled, which is how
//! 75 errors of drift accumulated unseen.
//!
//! The loop: for each task, stream a completion through the reasoning-overrun
//! guard, score the cleaned output against the task's checks, retry on failure,
//! keep the best attempt. A model whose SERVER is clearly failing (consecutive
//! transport errors) is abandoned after [`RunnerConfig::max_consecutive_infra`]
//! attempts: grinding on only produces more zeros, and those zeros must never be
//! read as quality results.
//!
//! Scoring is the established Rust pattern (as in `model_eval.rs`): each check
//! is boolean via `run_check`, and the task score is the fraction passed.

use crate::ztools::eval::task_loader::{check_graded_score, run_check, EvalTask};
use crate::ztools::eval::transport::{self, stream_with_overrun_guard, RequestSpec};

/// Result of evaluating ONE task for ONE model. Errors are data.
#[derive(Debug, Clone, Default, PartialEq)]
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
    pub timeout_secs: u64,
    /// Retries AFTER the first attempt (0 = no retries).
    pub max_retries: u32,
    /// Consecutive transport failures before abandoning the model.
    pub max_consecutive_infra: u32,
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

/// Only the happy path is served from the stream, as in the Python original:
/// any transport error -- or a stream that produced nothing at all (a server
/// without SSE support answers plain JSON, which the SSE parser skips) -- falls
/// through to the blocking call.
fn call_with_guard(spec: &RequestSpec) -> transport::TransportResult {
    let streamed = stream_with_overrun_guard(spec);
    let produced_nothing = streamed.content.is_empty()
        && streamed.reasoning_content.is_empty()
        && streamed.finish_reason.is_empty();
    if streamed.error.is_none() && !produced_nothing {
        return streamed;
    }
    let blocking = transport::call(spec, false);
    if blocking.error.is_none() || streamed.error.is_some() {
        blocking
    } else {
        streamed
    }
}

/// Evaluate `model` against `tasks` in order. Never panics on transport
/// problems; every failure lands in the returned outcomes as data.
pub fn run_eval(model: &str, tasks: &[EvalTask], cfg: &RunnerConfig) -> Vec<TaskOutcome> {
    let mut outcomes = Vec::new();
    let mut consecutive_infra: u32 = 0;

    for task in tasks {
        let mut best: Option<TaskOutcome> = None;

        for _attempt in 0..=cfg.max_retries {
            let spec = RequestSpec {
                model,
                messages: &task.messages,
                host: &cfg.host,
                port: cfg.port,
                temperature: cfg.temperature,
                max_tokens: cfg.max_tokens,
                timeout_secs: cfg.timeout_secs,
            };
            let r = call_with_guard(&spec);
            let outcome = TaskOutcome {
                task: task.name.clone(),
                time_secs: r.time_secs,
                error: r.error.clone(),
                finish_reason: r.finish_reason.clone(),
                aborted: r.aborted,
                abort_reason: r.abort_reason.clone(),
                ..Default::default()
            };
            let is_best = match &best {
                // Errors rank below any scored attempt.
                Some(b) => b.error.is_some() && outcome.error.is_none(),
                None => true,
            };
            if is_best {
                best = Some(outcome);
            }
            if r.error.is_some() {
                continue;
            }
            let score = score_output(task, &r.content, None);
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

        let outcome = best.unwrap_or_else(|| TaskOutcome {
            task: task.name.clone(),
            status: "fail".to_string(),
            ..Default::default()
        });

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
