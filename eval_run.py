#!/usr/bin/env python3
"""
Evaluation runner module.
Contains the main eval loop, model calling, and validation orchestration.
"""

import json
import re
from rich.console import Console
from lib.osaurus_lib import call
from eval_tasks_core import TASKS, _extract_items_from_text
from eval_failures import FAIL_INFRA, FAIL_CONTENT, FAIL_NONE, _classify_failure
from eval_validate import safe_content
from lib.tui import STEP, WARN, FAIL


MAX_RETRIES = 1
EVAL_TIMEOUT = 300
MEMORY_WARNING_THRESHOLD = 80

console = Console()


def _validate_result(result: dict, task_cfg: dict, task_name: str, debug: bool = False) -> tuple[int, str, dict]:
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
        validated = validator(parsed, source_text=source)

        if isinstance(validated, tuple):
            score, failure_reason = validated
        else:
            score, failure_reason = validated, ""

        diagnosis = _classify_failure(result, task_cfg, score, failure_reason)
        return score, failure_reason, diagnosis

    if is_parse_json and content:
        json_match = re.search(r'\[[\s\S]*\]', content) or re.search(r'\{[\s\S]*\}', content)
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
            validated = validator(extracted, source_text=source)
            items_for_debug = extracted
        elif len(content) > 50:
            from lib.validators_lib import validate_summary
            validated = validate_summary(content)
            items_for_debug = None
        else:
            failure = "Empty content"
            diagnosis = _classify_failure(result, task_cfg, 0, failure)
            return 0, failure, diagnosis

        if debug and source and "weekend" in task_name and items_for_debug:
            from lib.validators_lib import get_source_matching_details
            details = get_source_matching_details(items_for_debug, source)
            console.print(f"  Source matching for {task_name}:")
            console.print(f"    Matched: {len(details['matched'])}/{len(details['matched']) + len(details['unmatched'])} ({details['ratio']*100:.0f}%)")
            if details['unmatched']:
                console.print(f"    Unmatched items:")
                for item in details['unmatched'][:3]:
                    console.print(f"      - {item['name']} (terms: {item.get('terms', [])[:3]})")

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
    validated = validator(content)

    if isinstance(validated, tuple):
        score, failure_reason = validated
    else:
        score, failure_reason = validated, ""

    diagnosis = _classify_failure(result, task_cfg, score, failure_reason)
    return score, failure_reason, diagnosis


def _call_model(model: str, task_cfg: dict, task_name: str, host: str, port: int, backend: str) -> dict:
    """Call model via the appropriate backend (pure transport, no validation)."""
    if backend == "mlx":
        from lib.mlx_lib import call as mlx_call
        return mlx_call(
            model,
            messages=task_cfg["messages"],
            host=host,
            port=port,
            timeout=EVAL_TIMEOUT,
        )
    else:
        return call(
            model=model,
            messages=task_cfg["messages"],
            host=host,
            port=port,
            task=task_name,
            parse_json=task_cfg["parse_json"],
            timeout=EVAL_TIMEOUT,
        )


def _quality_results_to_eval_format(scorecards: list, model: str) -> list[dict]:
    """Convert quality.py ScoreCards to model_eval's result format."""
    results = []
    for sc in scorecards:
        failures = [f for d in sc.dimensions for f in d.failures]
        composite = sc.composite
        status = "ok" if composite >= 90 else ("partial" if composite >= 50 else "fail")
        results.append({
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
        })
    return results


def run_eval(
    model: str, tasks: dict = None, host: str = "localhost", port: int = 1337, backend: str = "osaurus",
    verbose: bool = False
) -> dict:
    """Run evaluation on model using real-world tasks.

    This function owns all validation and retry logic.
    The library call() functions are pure transport/parsing layers.
    """
    from lib.logging_config import osaurus_logger as eval_logger

    tasks = tasks or TASKS
    results = []

    console.print(f"{STEP} Testing {model} ({backend})")

    for task_name, task_cfg in tasks.items():
        if "messages" not in task_cfg:
            console.print(f"{WARN} Skipping '{task_name}' (no messages key)")
            continue
        best_score = -1
        best_result = None
        best_failure = ""
        best_diagnosis = {"category": FAIL_NONE, "reason": "", "evidence": ""}
        first_attempt_failed = False

        for attempt in range(MAX_RETRIES + 1):
            if attempt > 0:
                eval_logger.warning(f"Retrying task '{task_name}' with model {model} (Attempt {attempt+1}/{MAX_RETRIES+1})...")
                first_attempt_failed = True

            try:
                result = _call_model(model, task_cfg, task_name, host, port, backend)
            except Exception as e:
                eval_logger.error(f"Model call failed with exception: {e}")
                result = {"content": None, "error": str(e), "time": None, "model": model}

            try:
                score, failure_reason, diagnosis = _validate_result(result, task_cfg, task_name, debug=True)
            except Exception as e:
                eval_logger.error(f"Validation failed with exception: {e}")
                score, failure_reason, diagnosis = 0, f"Validation error: {e}", {"category": FAIL_INFRA, "reason": str(e), "evidence": ""}

            eval_logger.info(f"Quality score: {score}/100")

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

        status_symbol = STEP if status == "ok" else (WARN if status == "partial" else FAIL)
        retry_tag = " (2nd try)" if first_attempt_failed else ""
        category_tag = f" [{category}]" if category else ""
        fail_info = f" - {best_failure}" if best_failure else ""
        evidence_info = f"\n    - {best_diagnosis['evidence']}" if best_diagnosis.get("evidence") else ""
        time_taken = best_result.get('time') if best_result else None
        time_taken_str = f"{time_taken}s" if time_taken is not None else "N/A"
        console.print(
            f"  {status_symbol} {task_name}: {best_score}% ({time_taken_str}){category_tag}{fail_info}{evidence_info}"
        )

        if verbose and best_result:
            content = safe_content(best_result)[:500]
            if content:
                console.print(f"  Raw output: {content}")

    weekend_tasks = [k for k in tasks.keys() if "weekend" in k]
    if weekend_tasks:
        from lib.validators_lib import get_source_matching_details
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
                names = [u["name"] for u in details["unmatched"][:2]]
                console.print(f"    {WARN} Not from source: {names}")

    return results
