"""Scorers for the file_summary task.

Split out of quality_scorers.py for the repo's 500-line limit. Importing this
module registers its scorers; quality_scorers.py does that import.
"""

import json
import re

from lib.quality_models import Score, TestCase, _lower, _str
from lib.quality_scorers_core import register_scorer


@register_scorer("file_summary")
def _score_file_completeness(output: str, case: TestCase) -> Score:
    out = _str(output)
    if not out:
        return Score("Completeness", 0, 0.40, failures=["empty"])

    failures = []
    try:
        data = json.loads(out)
    except (json.JSONDecodeError, TypeError):
        return Score("Completeness", 0, 0.40, failures=["invalid JSON"])

    if not isinstance(data, list):
        return Score("Completeness", 0, 0.40, failures=["not a list"])

    ref = json.loads(case.reference)
    exp_paths = {item["path"] for item in ref}
    out_paths = {item.get("path", "") for item in data if isinstance(item, dict)}
    found = exp_paths & out_paths
    ratio = len(found) / len(exp_paths) if exp_paths else 0

    score = ratio * 100
    if ratio < 1.0:
        missing = exp_paths - out_paths
        failures.append(f"missing files: {', '.join(sorted(missing))}")

    return Score("Completeness", score, 0.40, failures)


@register_scorer("file_summary")
def _score_file_accuracy(output: str, case: TestCase) -> Score:
    out = _str(output)
    if not out:
        return Score("Accuracy", 0, 0.30, failures=["empty"])

    try:
        data = json.loads(out)
    except (json.JSONDecodeError, TypeError):
        return Score("Accuracy", 0, 0.30, failures=["invalid JSON"])

    if not isinstance(data, list):
        return Score("Accuracy", 0, 0.30, failures=["not a list"])

    ref = json.loads(case.reference)
    failures = []
    total_score = 0
    count = 0

    for ref_item in ref:
        ref_path = ref_item["path"]
        ref_desc = ref_item["desc"]
        match = next(
            (item for item in data if isinstance(item, dict) and item.get("path", "") == ref_path),
            None,
        )
        if match is None:
            failures.append(f"'{ref_path}' not found")
            continue

        out_desc = _str(match.get("desc", ""))
        if not out_desc:
            failures.append(f"'{ref_path}' has no description")
            continue

        ref_tokens = set(re.findall(r"[a-z]+", _lower(ref_desc)))
        out_tokens = set(re.findall(r"[a-z]+", _lower(out_desc)))
        if len(ref_tokens) == 0:
            continue

        overlap = set()
        for rt in ref_tokens:
            if rt in out_tokens:
                overlap.add(rt)
            else:
                for ot in out_tokens:
                    if rt in ot:
                        overlap.add(rt)
                        break
        ratio = len(overlap) / len(ref_tokens)
        item_score = min(100, ratio * 100)
        total_score += item_score
        count += 1

        if ratio < 0.3:
            failures.append(f"'{ref_path}' desc mismatch")

    if count == 0:
        return Score("Accuracy", 0, 0.30, failures=["no items scored"])

    return Score("Accuracy", total_score / count, 0.30, failures)


@register_scorer("file_summary")
def _score_file_format(output: str, case: TestCase) -> Score:
    out = _str(output)
    if not out:
        return Score("Format", 0, 0.30, failures=["empty"])

    failures = []
    try:
        data = json.loads(out)
    except (json.JSONDecodeError, TypeError):
        return Score("Format", 0, 0.30, failures=["invalid JSON"])

    if not isinstance(data, list):
        return Score("Format", 0, 0.30, failures=["not a list"])

    if len(data) == 0:
        return Score("Format", 30, 0.30, failures=["empty array"])

    valid = sum(1 for item in data if isinstance(item, dict) and "path" in item and "desc" in item)
    ratio = valid / len(data)
    score = ratio * 100

    if ratio < 1.0:
        failures.append(f"{valid}/{len(data)} items have valid schema")

    return Score("Format", score, 0.30, failures)
