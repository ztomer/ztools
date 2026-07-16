#!/usr/bin/env python3
"""
Validation module for model evaluation.
"""

import json
from typing import Any, Tuple

from lib.validators_lib import has_text_headers


def safe_content(result: dict) -> str:
    """Safely extract content from a result dict, handling None values."""
    content = result.get("content")
    if content is None:
        return ""
    if not isinstance(content, str):
        return str(content)
    return content


def validate_file_summary(data: Any, source_text: str = "") -> Tuple[int, str]:
    """Validate file summary quality - checks for ACTUAL content detail, not filename inference.

    STRICT checks:
    - No filename-only summaries (must describe what file does)
    - No generic patterns like "a python script"
    - Must have actionable content about file purpose/function
    """
    if not data:
        return 0, "empty response"

    if isinstance(data, list):
        failures = []
        items = data
        num_files = len(items)
        if num_files < 4:
            failures.append(f"only {num_files} files")

        detailed_count = 0
        content_verbs = [
            "parse",
            "validat",
            "evaluat",
            "extract",
            "load",
            "save",
            "read",
            "write",
            "fetch",
            "send",
            "process",
            "handle",
            "config",
            "setting",
            "option",
            "parameter",
            "api",
            "client",
            "model",
            "llm",
        ]

        for item in items:
            if not isinstance(item, dict):
                continue
            path = item.get("path", "")
            desc = item.get("desc", "") or item.get("summary", "")
            if not path or not desc:
                continue
            desc_lower = str(desc).lower()

            has_content = any(kw in desc_lower for kw in content_verbs)
            if has_content:
                detailed_count += 1

        if num_files == 0:
            return 0, "no items"
        if detailed_count >= num_files * 0.8:
            score = 100
        elif detailed_count >= num_files * 0.5:
            score = 85
        elif detailed_count >= 2:
            score = 70
        elif detailed_count >= 1:
            score = 50
        else:
            score = 25
            failures.append("no content details")

        return min(100, score), "; ".join(failures) if failures else ""

    if isinstance(data, dict):
        data = json.dumps(data)

    data_str = str(data).strip()
    failures = []
    score = 0

    parsed = None
    try:
        parsed = json.loads(data_str)
    except Exception:
        pass

    if not parsed:
        if has_text_headers(data_str):
            score += 20
        if len(data_str) >= 200:
            score += 20
        if score < 40:
            failures.append("no headers")
        return min(100, max(score, 20)), "; ".join(failures)

    items = list(parsed.items()) if isinstance(parsed, dict) else parsed
    num_files = len(items)

    detailed_count = 0
    content_verbs = [
        "parse",
        "validat",
        "evaluat",
        "extract",
        "load",
        "save",
        "read",
        "write",
        "fetch",
        "send",
        "process",
        "handle",
        "config",
        "setting",
        "option",
        "parameter",
        "api",
        "client",
        "model",
        "llm",
    ]

    for filepath, summary in items:
        if not filepath or not summary:
            continue
        summary_lower = str(summary).lower()

        has_content = any(kw in summary_lower for kw in content_verbs)
        if has_content:
            detailed_count += 1

    if detailed_count >= num_files * 0.8:
        score = 85
    elif detailed_count >= num_files * 0.5:
        score = 70
    elif detailed_count >= 2:
        score = 55
    elif detailed_count >= 1:
        score = 40
    else:
        score = 25

    if not detailed_count:
        failures.append("no content details")

    return min(100, score), "; ".join(failures) if failures else ""
