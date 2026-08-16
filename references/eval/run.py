#!/usr/bin/env python3
"""
Evaluation runner module.
Contains the main eval loop, model calling, and validation orchestration.
"""

import json
import os
import re

from lib.config_getters import get_max_tokens_for_task
from lib.logging_config import osaurus_logger as eval_logger
from lib.mlx_lib import call as mlx_call
from lib.model_caps import is_generative_model
from lib.osaurus_lib import call
from lib.tui import FAIL, STEP, WARN, console
from lib.validators_lib import get_source_matching_details

from eval.failures import (
    FAIL_CONTENT,
    FAIL_INFRA,
    FAIL_NONE,
    FAIL_REASONING,
    FAIL_TIMEOUT,
    _classify_failure,
    reasoning_retry_budget,
)
from eval.outputs import save_output
from eval.prefill import measure_prefill_rate, record_prefill_rate
from eval.signals import (
    DEFAULT_EVAL_TIMEOUT,
    _effective_timeout,
    _record_signal,
)
from eval.tasks_core import TASKS, _extract_items_from_text
from eval.validate import safe_content

MAX_RETRIES = int(os.environ.get("EVAL_MAX_RETRIES", "1"))
# Consecutive INFRA/TIMEOUT failures before abandoning a model. Chosen from
# observation, not taste: a model that can serve at all recovers within a task
# or two, while qwen3.6-35b returned 46 straight infrastructure failures without
# a single success.
MAX_CONSECUTIVE_INFRA_FAILURES = int(os.environ.get("EVAL_MAX_INFRA_FAILURES", "4"))
# Greedy decoding, because a leaderboard has to be reproducible.
#
# The eval inherited DEFAULT_TEMPERATURE (0.1) and never pinned it, so every run
# sampled. Running ornith twice on an unchanged image_rename scored 100% (190s)
# and then 0% (523s) -- a 100-point swing on identical input. Ranking models on
# single sampled runs measures the sampler, and any best_models derived that way
# would be noise wearing a number.
#
# Production still runs at 0.1: this measures the model's best behaviour rather
# than its average, which is the right basis for comparing models and for
# telling whether a prompt change helped. It does not eliminate GPU batching
# non-determinism, only the sampling that dominated it.
EVAL_TEMPERATURE = float(os.environ.get("EVAL_TEMPERATURE", "0"))
MEMORY_WARNING_THRESHOLD = 80



def _validate_result(
    result: dict, task_cfg: dict, task_name: str, debug: bool = False
) -> tuple[int, str, dict]:
    """Run validation on a library result. Returns (score, failure_reason, diagnosis)."""
    validator = task_cfg["validator"]

    if result.get("error"):
        diagnosis = _classify_failure(result, task_cfg, 0, result["error"])
        return 0, result["error"], diagnosis

    is_parse_json = task_cfg.get("parse_json", False)
    parsed = result.get("parsed")
    content = safe_content(result)
    source = task_cfg.get("source", "")

    if is_parse_json and parsed:
        validated = validator(parsed, source_text=source, **task_cfg.get("validator_kwargs", {}))

        if isinstance(validated, tuple):
            score, failure_reason = validated
        else:
            score, failure_reason = validated, ""

        diagnosis = _classify_failure(result, task_cfg, score, failure_reason)
        return score, failure_reason, diagnosis

    if is_parse_json and content:
        json_match = re.search(r"\[[\s\S]*\]", content) or re.search(r"\{[\s\S]*\}", content)
        extracted = None
        if json_match:
            try:
                extracted = json.loads(json_match.group())
                if isinstance(extracted, dict):
                    extracted = [extracted]
            except Exception:
                pass

        if not extracted:
            extracted = _extract_items_from_text(content)

        if extracted:
            validated = validator(
                extracted, source_text=source, **task_cfg.get("validator_kwargs", {})
            )
            items_for_debug = extracted
        else:
            # A JSON task that produced no JSON has failed the task, whatever the
            # prose says. Scoring the leftover text with validate_summary — a
            # different task's validator — awarded structure and synthesis points
            # to refusals ("I could not find any events..."), recorded a failure
            # reason about headers and user mentions instead of "no JSON", and
            # kept the run out of the parse-failure signal.
            failure = "Empty content" if len(content) <= 50 else "No JSON in output"
            diagnosis = _classify_failure(result, task_cfg, 0, failure)
            return 0, failure, diagnosis

        if debug and source and "weekend" in task_name and items_for_debug:
            details = get_source_matching_details(items_for_debug, source)
            console.print(f"  Source matching for {task_name}:")
            console.print(
                f"    Matched: {len(details['matched'])}/"
                f"{len(details['matched']) + len(details['unmatched'])} "
                f"({details['ratio'] * 100:.0f}%)"
            )
            if details["unmatched"]:
                console.print("    Unmatched items:")
                for item in details["unmatched"][:3]:
                    if isinstance(item, dict):
                        console.print(
                            f"      - {item['name']} (terms: {item.get('terms', [])[:3]})"
                        )
                    else:
                        console.print(f"      - {item}")

        if isinstance(validated, tuple):
            score, failure_reason = validated
        else:
            score, failure_reason = validated, ""

        diagnosis = _classify_failure(result, task_cfg, score, failure_reason)
        return score, failure_reason, diagnosis

    if not content:
        failure = "Empty content"
        diagnosis = _classify_failure(result, task_cfg, 0, failure)
        return 0, failure, diagnosis
    validated = validator(content, source_text=source, **task_cfg.get("validator_kwargs", {}))

    if isinstance(validated, tuple):
        score, failure_reason = validated
    else:
        score, failure_reason = validated, ""

    diagnosis = _classify_failure(result, task_cfg, score, failure_reason)
    return score, failure_reason, diagnosis


def _call_model(
    model: str,
    task_cfg: dict,
    task_name: str,
    host: str,
    port: int,
    backend: str,
    timeout: int = None,
    max_tokens: int = None,
) -> dict:
    """Call model via the appropriate backend (pure transport, no validation)."""
    effective_timeout = timeout or DEFAULT_EVAL_TIMEOUT
    if backend == "mlx":
        return mlx_call(
            model,
            messages=task_cfg["messages"],
            host=host,
            port=port,
            temperature=EVAL_TEMPERATURE,
            timeout=effective_timeout,
        )
    else:
        return call(
            model=model,
            messages=task_cfg["messages"],
            host=host,
            port=port,
            task=task_name,
            parse_json=task_cfg["parse_json"],
            temperature=EVAL_TEMPERATURE,
            timeout=effective_timeout,
            max_tokens=max_tokens,
            # Watch the stream and cut a run that has reasoned past the point where
            # the remaining budget could hold an answer. Without it, a reasoning
            # overrun is only detectable AFTER paying for the whole budget, and a
            # sweep pays that once per affected task per model.
            stream_guard=True,
        )


def _quality_results_to_eval_format(scorecards: list, model: str) -> list[dict]:
    """Convert quality.py ScoreCards to model_eval's result format."""
    results = []
    for sc in scorecards:
        failures = [f for d in sc.dimensions for f in d.failures]
        composite = sc.composite
        status = "ok" if composite >= 90 else ("partial" if composite >= 50 else "fail")
        results.append(
            {
                "task": sc.task,
                "case_id": sc.case_id,
                "status": status,
                "quality_score": round(composite, 1),
                "time": round(sc.elapsed, 1),
                "error": None,
                "failure_reason": "; ".join(failures) if failures else "",
                "failure_category": None,
                "failure_evidence": "",
                "result": {"model": model, "time": sc.elapsed, "content": sc.output},
            }
        )
    return results


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
    global MAX_RETRIES
    orig_retries = MAX_RETRIES
    MAX_RETRIES = 0
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
        MAX_RETRIES = orig_retries


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
    if measure_prefill and backend == "osaurus" and is_generative_model(model):
        rate = measure_prefill_rate(model, host, port, transport=call)
        record_prefill_rate(model, rate)
        if rate:
            console.print(f"{STEP} {model} prefill: {rate:,.0f} chars/sec")

    consecutive_infra = 0

    for task_name, task_cfg in tasks.items():
        if "messages" not in task_cfg:
            console.print(f"{WARN} Skipping '{task_name}' (no messages key)")
            continue
        prompt_chars = sum(
            len(m.get("content") or "") for m in task_cfg.get("messages", [])
        )
        task_timeout = _effective_timeout(
            model, task_name, prompt_chars, get_max_tokens_for_task(task_name)
        )
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

            if score > best_score:
                best_score = score
                best_result = result
                best_failure = failure_reason
                best_diagnosis = diagnosis

            if score >= 90:
                break
            if diagnosis.get("category") == FAIL_CONTENT:
                break

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

    weekend_tasks = [k for k in tasks.keys() if "weekend" in k]
    mixed_tasks = [k for k in tasks.keys() if k.endswith("_mixed")]
    if weekend_tasks:
        console.print("")
        console.print("Quality Check Summary:")
        for r in results:
            task_name = r["task"]
            if task_name not in weekend_tasks:
                continue
            task_cfg = tasks[task_name]
            source = task_cfg.get("source", "")
            if not source:
                continue
            parsed = r.get("result", {}).get("parsed", [])
            if not parsed:
                continue
            details = get_source_matching_details(parsed, source)
            matched = len(details["matched"])
            total = matched + len(details["unmatched"])
            ratio = details["ratio"] * 100
            console.print(f"  {task_name}: {matched}/{total} items from source ({ratio:.0f}%)")
            if details["unmatched"]:
                names = [
                    u if isinstance(u, str) else u.get("name", "unnamed")
                    for u in details["unmatched"][:2]
                ]
                console.print(f"    {WARN} Not from source: {names}")

    if mixed_tasks:
        console.print("")
        console.print("Signal/Noise Filtering:")
        for r in results:
            task_name = r["task"]
            if task_name not in mixed_tasks:
                continue
            reason = r.get("failure_reason", "")
            noise_part = ""
            if "noise" in reason:
                noise_part = reason
            elif "missed" in reason or "coverage" in reason:
                noise_part = reason
            symbol = WARN if ("included" in reason and "noise" in reason) else STEP
            console.print(
                f"  {symbol} {task_name}: {r['quality_score']}% — {noise_part or 'filtered clean'}"
            )

    return results
