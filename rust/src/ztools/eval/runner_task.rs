//! One task of the eval loop: its budget, its retries, and a single attempt.
//!
//! Split from `runner.rs` (the per-model loop) at the file-length cap; the
//! seam is the one the loop already had -- `run_task` is what the loop calls
//! per task, and everything below it is how one task is answered.

use super::{outcome_from, score_output, status_for, RunnerConfig, TaskOutcome};
use crate::ztools::eval::failures::{
    classify_failure, reasoning_overrun_was_guard_aborted, reasoning_retry_budget, FAIL_REASONING,
};
use crate::ztools::eval::signals::effective_timeout;
use crate::ztools::eval::task_loader::EvalTask;
use crate::ztools::eval::transport::{self, RequestSpec};

/// One task's output budget and deadline.
pub(super) struct TaskBudget {
    max_tokens: u32,
    timeout_secs: u64,
}

/// The production path resolves the output budget per task/model from config
/// exactly like the Python eval (`get_max_tokens_for_task`) and the deadline
/// from the learned signals; the hermetic path keeps the configured constants.
pub(super) fn task_budget(model: &str, task: &EvalTask, cfg: &RunnerConfig) -> TaskBudget {
    if !cfg.record_signals {
        return TaskBudget {
            max_tokens: cfg.max_tokens,
            timeout_secs: cfg.timeout_secs,
        };
    }
    let prompt_chars: usize = task.messages.iter().map(|m| m.content.len()).sum();
    let max_tokens = crate::ztools::eval::budgets::max_tokens_for_task(&task.name, model);
    TaskBudget {
        max_tokens,
        timeout_secs: effective_timeout(model, &task.name, prompt_chars, max_tokens),
    }
}

/// One task, with its retries: the best attempt and how many were used.
///
/// A retry that repeats the identical call cannot fix a reasoning overrun --
/// the model will think itself past the budget again -- so that retry gets
/// MORE room, bounded, and only once per model (`escalation_futile`).
pub(super) fn run_task(
    model: &str,
    task: &EvalTask,
    cfg: &RunnerConfig,
    budget: &TaskBudget,
    escalation_futile: &mut bool,
) -> (TaskOutcome, u32) {
    let mut best: Option<TaskOutcome> = None;
    let mut attempts_used: u32 = 0;
    let mut best_diagnosis = crate::ztools::eval::failures::Diagnosis {
        category: "",
        reason: String::new(),
        evidence: String::new(),
    };

    for attempt in 0..=cfg.max_retries {
        attempts_used += 1;
        let overran = attempt > 0 && best_diagnosis.category == FAIL_REASONING;
        if overran && *escalation_futile {
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
        let attempt_tokens = if overran {
            let escalated = reasoning_retry_budget(budget.max_tokens);
            eprintln!(
                "  · previous attempt reasoned past {}; retrying with {escalated}",
                budget.max_tokens
            );
            escalated
        } else {
            budget.max_tokens
        };
        let attempt = attempt_task(model, task, cfg, attempt_tokens, budget.timeout_secs);
        let is_best = match &best {
            // Errors rank below any scored attempt.
            Some(b) => b.error.is_some() && attempt.candidate.error.is_none(),
            None => true,
        };
        if is_best {
            best = Some(attempt.candidate);
        }
        best_diagnosis = attempt.diagnosis;
        // An escalated attempt the guard cut as well is the evidence that this
        // model expands to fill: strictly more room, strictly more of it spent
        // thinking, still no answer. Recorded next to the attempt that proves
        // it, so no later task re-buys the proof.
        if attempt_tokens > budget.max_tokens && attempt.guard_aborted {
            *escalation_futile = true;
        }
        // A scored attempt always outranks the error/empty placeholder in
        // `best`, even at 0 -- otherwise the placeholder's blank status
        // leaks into the result. Ties take the later attempt.
        if let Some(b) = best.as_mut() {
            if b.error.is_some() || attempt.score >= b.score {
                b.score = attempt.score;
                b.status = status_for(attempt.score).to_string();
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
    // parse-failure counting and names the failure in reports.
    outcome.failure_category = best_diagnosis.category.to_string();
    (outcome, attempts_used)
}

/// What one attempt at a task produced.
struct Attempt {
    candidate: TaskOutcome,
    score: u8,
    diagnosis: crate::ztools::eval::failures::Diagnosis,
    /// The stream guard cut the model off mid-reasoning.
    guard_aborted: bool,
}

/// One request: call, score, diagnose, and archive what the model said.
fn attempt_task(
    model: &str,
    task: &EvalTask,
    cfg: &RunnerConfig,
    max_tokens: u32,
    timeout_secs: u64,
) -> Attempt {
    let spec = RequestSpec {
        model,
        messages: &task.messages,
        host: &cfg.host,
        port: cfg.port,
        temperature: cfg.temperature,
        max_tokens,
        timeout_secs,
        allow_substitution: cfg.allow_model_substitution,
        thinking: cfg.thinking,
        stream_guard: true,
    };
    let r = transport::call(&spec, task.parse_json);
    let candidate = outcome_from(task, &r);
    // A JSON task's answer is parsed once here and handed to every
    // check; an unparseable answer scores as the raw text would.
    let parsed = if task.parse_json {
        serde_json::from_str::<serde_json::Value>(&r.content).ok()
    } else {
        None
    };
    let score = score_output(task, &r.content, parsed.as_ref());
    let diagnosis = classify_failure(
        r.error.as_deref(),
        &r.content,
        &r.reasoning_content,
        &r.finish_reason,
        parsed.as_ref(),
        score,
        task.parse_json,
    );
    // Archive what the model actually said BEFORE anything decides
    // what this score means. A scorer question asked after the fact is
    // unanswerable without the output, and re-running costs hours on a
    // one-model-at-a-time machine.
    if cfg.record_signals {
        // Informational path; a lost output must not stop the run.
        let _ = crate::ztools::eval::outputs::save_output(
            &crate::ztools::eval::outputs::OutputRecord {
                model,
                task: &task.name,
                content: &r.content,
                reasoning: &r.reasoning_content,
                error: r.error.as_deref(),
                score,
                failure_reason: &diagnosis.reason,
            },
            None,
        );
    }
    Attempt {
        candidate,
        score,
        guard_aborted: reasoning_overrun_was_guard_aborted(&r.finish_reason),
        diagnosis,
    }
}
