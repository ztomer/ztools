#!/usr/bin/env python3
"""The model loop: everything that spans tasks rather than attempts.

Split out of eval/run.py for the 500-line limit; `eval.run` re-exports
`run_eval` and `run_eval_quick`, so existing imports are unaffected.

One task's retry policy lives in eval/run_attempt.py. What lives HERE is the
policy that only makes sense with the whole run in view: the prefill probe, the
GPU-lock heartbeat, the per-task stall check, and abandoning a model once the
server -- not the model -- is clearly the problem.

Owns `measure_prefill_rate`, `contended_server_warning`, `gpu_lock` and
`console`, so those are the names to patch on this module.
"""

import os
import time

from lib import gpu_lock
from lib.config_getters import get_max_tokens_for_task
from lib.model_caps import is_generative_model
from lib.tui import FAIL, STEP, WARN, console

from eval import run_attempt, run_transport
from eval.failures import FAIL_INFRA, FAIL_TIMEOUT
from eval.prefill import measure_prefill_rate, record_prefill_rate
from eval.run_attempt import run_task_attempts
from eval.run_summary import print_signal_noise_summary, print_source_matching_summary
from eval.signals import (
    DEFAULT_EVAL_TIMEOUT,
    _effective_timeout,
    _record_signal,
    contended_server_warning,
)
from eval.tasks_core import TASKS
from eval.validate import safe_content
from eval.watchdog import check_stall

# Consecutive INFRA/TIMEOUT failures before abandoning a model. Chosen from
# observation, not taste: a model that can serve at all recovers within a task
# or two, while qwen3.6-35b returned 46 straight infrastructure failures without
# a single success.
MAX_CONSECUTIVE_INFRA_FAILURES = int(os.environ.get("EVAL_MAX_INFRA_FAILURES", "4"))
MEMORY_WARNING_THRESHOLD = 80


def run_eval_quick(
    model: str,
    tasks: dict = None,
    host: str = "localhost",
    port: int = 1337,
    backend: str = "osaurus",
    verbose: bool = False,
    timeout: int = DEFAULT_EVAL_TIMEOUT,
    on_complete: callable = None,
    measure_prefill: bool = True,
) -> dict:
    """Run evaluation with no retries (quick mode)."""
    orig_retries = run_attempt.MAX_RETRIES
    run_attempt.MAX_RETRIES = 0
    try:
        return run_eval(
            model,
            tasks=tasks,
            host=host,
            port=port,
            backend=backend,
            verbose=verbose,
            timeout=timeout,
            on_complete=on_complete,
            measure_prefill=measure_prefill,
        )
    finally:
        run_attempt.MAX_RETRIES = orig_retries


def run_eval(
    model: str,
    tasks: dict = None,
    host: str = "localhost",
    port: int = 1337,
    backend: str = "osaurus",
    verbose: bool = False,
    timeout: int = DEFAULT_EVAL_TIMEOUT,
    on_complete: callable = None,
    measure_prefill: bool = True,
) -> dict:
    """Run evaluation on model using real-world tasks.

    This function owns all validation and retry logic.
    The library call() functions are pure transport/parsing layers.
    """
    tasks = tasks or TASKS
    results = []

    console.print(f"{STEP} Testing {model} ({backend})")

    # Measure this model's ingestion rate before timing anything else. It is
    # what every tool's context budget is sized from, and the alternative was a
    # hand-picked constant that turned out to be 35-90x too low. One extra
    # request per model per run.
    #
    # `run_transport.call` by attribute, not a local alias: the probe and the
    # task calls have to be the SAME mock seam. See eval/run_transport.py.
    if measure_prefill and backend == "osaurus" and is_generative_model(model):
        rate = measure_prefill_rate(model, host, port, transport=run_transport.call)
        record_prefill_rate(model, rate)
        if rate:
            console.print(f"{STEP} {model} prefill: {rate:,.0f} chars/sec")

    consecutive_infra = 0
    last_completion = time.monotonic()
    # Set once this model proves it cannot use a bigger budget (see failures.py).
    reasoning_escalation_futile = False

    for task_name, task_cfg in tasks.items():
        if "messages" not in task_cfg:
            console.print(f"{WARN} Skipping '{task_name}' (no messages key)")
            continue
        # Progress, not duration: the lock's wedge ceiling runs from the last beat,
        # so a healthy multi-hour run never loses the GPU to a peer while a hung one
        # still does. A no-op when this process holds no lock. See lib/gpu_lock.py.
        gpu_lock.heartbeat()
        # The single-server invariant is not established by having been true at
        # startup. osaurus_one.sh runs once before a model and then hours pass.
        contended = contended_server_warning(model, task_name)
        if contended:
            console.print(f"{WARN} {contended}")
        prompt_chars = sum(
            len(m.get("content") or "") for m in task_cfg.get("messages", [])
        )
        if check_stall(model, last_completion):
            break

        task_timeout = _effective_timeout(
            model, task_name, prompt_chars, get_max_tokens_for_task(task_name)
        )

        attempts = run_task_attempts(
            model, task_cfg, task_name, host, port, backend, task_timeout,
            reasoning_escalation_futile=reasoning_escalation_futile,
        )
        best_score = attempts.score
        best_result = attempts.result
        best_failure = attempts.failure
        best_diagnosis = attempts.diagnosis
        first_attempt_failed = attempts.first_attempt_failed
        reasoning_escalation_futile = attempts.reasoning_escalation_futile

        status = "ok" if best_score >= 90 else ("partial" if best_score >= 50 else "fail")
        category = best_diagnosis.get("category")

        results.append(
            {
                "task": task_name,
                "status": status,
                "quality_score": best_score,
                "time": best_result.get("time") if best_result else None,
                "error": best_result.get("error") if best_result else None,
                "failure_reason": best_failure,
                "failure_category": category,
                "failure_evidence": best_diagnosis.get("evidence", ""),
                "result": best_result,
                "first_attempt_failed": first_attempt_failed,
            }
        )

        _record_signal(
            model,
            task_name,
            time_taken=(best_result.get("time") or 0) if best_result else 0,
            had_retries=first_attempt_failed,
            is_parse_failure=(category == "PARSE"),
        )

        # Stop once the SERVER, not the model, is clearly the problem. A model
        # too large for the host answers every request with HTTP 503 "at
        # inference capacity" or nothing at all, and grinding through the rest
        # of the suite only produces more zeros: qwen3.6-35b took 3h09m to
        # return 23 of them, 34 x 503 and 12 timeouts, on a host where the 27b
        # sibling ran fine. Aborting says "cannot run here" in minutes and
        # leaves the remaining GPU time for models that can.
        if category in (FAIL_INFRA, FAIL_TIMEOUT):
            consecutive_infra += 1
            if consecutive_infra >= MAX_CONSECUTIVE_INFRA_FAILURES:
                console.print(
                    f"{FAIL} Abandoning {model}: {consecutive_infra} consecutive "
                    f"infrastructure failures ({best_failure[:60]}). The server "
                    f"cannot serve this model on this host -- this is not a "
                    f"quality result and must not be read as one."
                )
                break
        else:
            consecutive_infra = 0
            last_completion = time.monotonic()

        status_symbol = STEP if status == "ok" else (WARN if status == "partial" else FAIL)
        category_tag = f" [{category}]" if category else ""
        fail_info = f" - {best_failure}" if best_failure else ""
        evidence_info = (
            f"\n    - {best_diagnosis['evidence']}" if best_diagnosis.get("evidence") else ""
        )
        time_taken = best_result.get("time") if best_result else None
        time_taken_str = f"{time_taken}s" if time_taken is not None else "N/A"
        console.print(
            f"  {status_symbol} {task_name}: {best_score}% "
            f"({time_taken_str}){category_tag}{fail_info}{evidence_info}"
        )

        if verbose and best_result:
            content = safe_content(best_result)[:500]
            if content:
                console.print(f"  Raw output: {content}")

    if on_complete:
        on_complete()

    print_source_matching_summary(tasks, results)
    print_signal_noise_summary(tasks, results)

    return results


__all__ = [
    "MAX_CONSECUTIVE_INFRA_FAILURES",
    "MEMORY_WARNING_THRESHOLD",
    "run_eval",
    "run_eval_quick",
]
