#!/usr/bin/env python3
"""The per-task attempt loop: retry policy and reasoning-overrun escalation.

Split out of eval/run.py for the 500-line limit. One task, up to MAX_RETRIES + 1
attempts, and the decision of when another attempt is worth paying for; the
model loop in eval/run_loop.py owns everything ACROSS tasks (stall detection,
infrastructure abandonment, reporting).

Owns `MAX_RETRIES`, `_validate_result` and `save_output`: they are read here, so
this is the module to patch them on.
"""

import os
from typing import NamedTuple

from lib.config_getters import get_max_tokens_for_task
from lib.logging_config import osaurus_logger as eval_logger

from eval.failures import (
    FAIL_CONTENT,
    FAIL_INFRA,
    FAIL_NONE,
    FAIL_REASONING,
    reasoning_overrun_was_guard_aborted,
    reasoning_retry_budget,
)
from eval.outputs import save_output
from eval.run_transport import _call_model
from eval.run_validate import _validate_result

MAX_RETRIES = int(os.environ.get("EVAL_MAX_RETRIES", "1"))


class TaskAttempts(NamedTuple):
    """The best of a task's attempts, plus whether it took more than one."""

    score: int
    result: dict | None
    failure: str
    diagnosis: dict
    first_attempt_failed: bool
    #: Carried forward across tasks by the caller -- see reasoning_escalation_futile
    #: below. Once true for a model, stays true for the rest of that model's run.
    reasoning_escalation_futile: bool


def run_task_attempts(
    model: str,
    task_cfg: dict,
    task_name: str,
    host: str,
    port: int,
    backend: str,
    task_timeout: int,
    reasoning_escalation_futile: bool = False,
) -> TaskAttempts:
    """Run one task until it passes, stops being worth retrying, or runs out of attempts.

    `reasoning_escalation_futile` is per-MODEL state the caller carries across
    tasks (see eval/run_loop.py): once an earlier task proves this model's
    reasoning expands to fill whatever budget it is given rather than answering
    once the budget covers it, every later task skips both the escalation and
    the retry it can't win. Returned back out so the caller can carry it
    forward, since it can also newly become true partway through THIS task.
    """
    best_score = -1
    best_result = None
    best_failure = ""
    best_diagnosis = {"category": FAIL_NONE, "reason": "", "evidence": ""}
    first_attempt_failed = False

    for attempt in range(MAX_RETRIES + 1):
        if attempt > 0:
            eval_logger.warning(
                f"Retrying task '{task_name}' with model {model} "
                f"(Attempt {attempt + 1}/{MAX_RETRIES + 1})..."
            )
            first_attempt_failed = True

        # A retry that repeats the identical call cannot fix a reasoning overrun
        # -- the model will think itself past the budget again. Retry with MORE
        # room, because "reasoned past its budget" is the model telling us the
        # budget was too small, and a reasoning model's chain of thought scales
        # with the task rather than fitting whatever ceiling we picked.
        retry_tokens = None
        if attempt > 0 and best_diagnosis.get("category") == FAIL_REASONING:
            if reasoning_escalation_futile:
                # Proven earlier: more room buys a longer failure, and a repeat
                # of the base call already hit the guard at that same budget.
                eval_logger.warning(
                    f"Skipping the retry for '{task_name}': {model} reasoned "
                    f"past an escalated budget, so more room cannot help"
                )
                break
            base_budget = get_max_tokens_for_task(task_name, model)
            retry_tokens = reasoning_retry_budget(base_budget)
            eval_logger.warning(
                f"Previous attempt reasoned past its budget ({base_budget}); "
                f"retrying with max_tokens={retry_tokens}"
            )

        try:
            result = _call_model(
                model, task_cfg, task_name, host, port, backend,
                timeout=task_timeout, max_tokens=retry_tokens,
            )
        except Exception as e:
            eval_logger.error(f"Model call failed with exception: {e}")
            result = {"content": None, "error": str(e), "time": None, "model": model}

        try:
            score, failure_reason, diagnosis = _validate_result(
                result, task_cfg, task_name, debug=True
            )
        except Exception as e:
            eval_logger.error(f"Validation failed with exception: {e}")
            score, failure_reason, diagnosis = (
                0,
                f"Validation error: {e}",
                {"category": FAIL_INFRA, "reason": str(e), "evidence": ""},
            )

        eval_logger.info(f"Quality score: {score}/100")

        # Before anything decides what this score MEANS. A scorer question
        # asked after the fact is unanswerable without the output, and
        # re-running costs hours on a one-model-at-a-time machine.
        save_output(model, task_name, result, score, failure_reason)

        if score < 90:
            category = diagnosis.get("category", "")
            evidence = diagnosis.get("evidence", "")
            eval_logger.warning(
                f"[DEBUG_OUTPUT] model={model} task={task_name} score={score} "
                f"category={category} failure={failure_reason} "
                f"evidence={evidence}"
            )

        # More room, cut by the guard anyway: this model expands to fill.
        if retry_tokens and reasoning_overrun_was_guard_aborted(result):
            reasoning_escalation_futile = True

        if score > best_score:
            best_score = score
            best_result = result
            best_failure = failure_reason
            best_diagnosis = diagnosis

        if score >= 90:
            break
        if diagnosis.get("category") == FAIL_CONTENT:
            break

    return TaskAttempts(
        score=best_score,
        result=best_result,
        failure=best_failure,
        diagnosis=best_diagnosis,
        first_attempt_failed=first_attempt_failed,
        reasoning_escalation_futile=reasoning_escalation_futile,
    )


__all__ = ["MAX_RETRIES", "TaskAttempts", "run_task_attempts"]
