#!/usr/bin/env python3
"""Scoring: turn one transport result into (score, failure_reason, diagnosis).

Split out of eval/run.py for the 500-line limit; `eval.run` re-exports
`_validate_result`, so existing imports are unaffected.

Owns `console` and `get_source_matching_details` for the debug source-matching
dump, which is why tests that want to read that output patch them HERE rather
than on the shim.
"""

import json
import re

from lib.tui import console
from lib.validators_lib import get_source_matching_details

from eval.failures import _classify_failure
from eval.tasks_core import _extract_items_from_text
from eval.validate import safe_content


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


__all__ = ["_validate_result"]
