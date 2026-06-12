#!/usr/bin/env python3
"""
Failure diagnosis module for model evaluation.
Classifies WHY a model result failed.
"""


from eval.validate import safe_content
from lib.validators_lib import has_item_details


FAIL_INFRA = "INFRA"
FAIL_TIMEOUT = "TIMEOUT"
FAIL_PARSE = "PARSE"
FAIL_FORMAT = "FORMAT"
FAIL_CONTENT = "CONTENT"
FAIL_NONE = None


def _classify_failure(result: dict, task_cfg: dict, score: int, failure_reason: str) -> dict:
    """Classify WHY a result failed. Returns a diagnosis dict.

    Returns:
        {
            "category": one of FAIL_* constants,
            "reason": human-readable failure reason,
            "evidence": specific evidence for the diagnosis,
        }
    """
    from eval.run import DEFAULT_EVAL_TIMEOUT

    error = result.get("error") or ""
    content = safe_content(result)
    parsed = result.get("parsed")
    raw_len = len(content)

    if score >= 90:
        return {"category": FAIL_NONE, "reason": "", "evidence": ""}

    if "Model not found" in error:
        return {
            "category": FAIL_INFRA,
            "reason": error,
            "evidence": "Model not loaded or wrong identifier",
        }
    if "Connection" in error:
        return {
            "category": FAIL_INFRA,
            "reason": error,
            "evidence": "Server unreachable",
        }

    if "Timeout" in error or "timed out" in error.lower():
        return {
            "category": FAIL_TIMEOUT,
            "reason": error,
            "evidence": f"Model did not respond within {DEFAULT_EVAL_TIMEOUT}s",
        }

    if task_cfg["parse_json"]:
        has_json_chars = "{" in content or "[" in content
        has_prose_before_json = False
        if has_json_chars:
            first_bracket = min(
                (content.find(c) for c in "[{" if content.find(c) >= 0),
                default=raw_len,
            )
            has_prose_before_json = first_bracket > 200

        if not parsed and not has_json_chars:
            return {
                "category": FAIL_FORMAT,
                "reason": failure_reason or "No JSON in output",
                "evidence": f"Output was {raw_len} chars of prose with no JSON brackets",
            }

        if not parsed and has_json_chars:
            return {
                "category": FAIL_PARSE,
                "reason": failure_reason or "JSON extraction failed",
                "evidence": f"Output had JSON-like content at char {first_bracket} of {raw_len} but parser couldn't extract valid JSON",
            }

        if parsed and has_prose_before_json:
            evidence = f"Model emitted {first_bracket} chars of reasoning before first JSON bracket (total {raw_len} chars)"
            if score < 50:
                return {
                    "category": FAIL_FORMAT,
                    "reason": failure_reason,
                    "evidence": evidence + "; reasoning may have consumed the context window",
                }
            return {
                "category": FAIL_CONTENT,
                "reason": failure_reason,
                "evidence": evidence,
            }

        return {
            "category": FAIL_CONTENT,
            "reason": failure_reason,
            "evidence": _describe_content_failure(parsed, failure_reason),
        }

    if not content:
        return {
            "category": FAIL_FORMAT,
            "reason": "Empty content",
            "evidence": "Model returned empty response",
        }

    reasoning_markers = ["Let me", "I'll", "Wait,", "Actually,", "Here's my", "Thinking"]
    has_reasoning = any(marker in content[:200] for marker in reasoning_markers)
    if has_reasoning and raw_len > 200:
        return {
            "category": FAIL_FORMAT,
            "reason": failure_reason,
            "evidence": f"Model output {raw_len} chars starting with reasoning instead of a direct answer",
        }

    return {
        "category": FAIL_CONTENT,
        "reason": failure_reason,
        "evidence": f"Output was {raw_len} chars but failed validation: {failure_reason}",
    }


def _describe_content_failure(parsed, failure_reason: str) -> str:
    """Generate human-readable evidence for content quality failures."""
    if isinstance(parsed, list):
        item_count = len(parsed)
        with_details = sum(
            1 for item in parsed
            if isinstance(item, dict) and has_item_details(item)
        )
        return f"Parsed {item_count} items, {with_details} with details. {failure_reason}"
    elif isinstance(parsed, dict):
        keys = list(parsed.keys())[:5]
        return f"Parsed dict with keys {keys}. {failure_reason}"
    return failure_reason
