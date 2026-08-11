"""Scorers for the filename task.

Split out of quality_scorers.py for the repo's 500-line limit. Importing this
module registers its scorers; quality_scorers.py does that import.
"""

import re

from lib.quality_models import GENERIC_FILENAMES, Score, TestCase, _lower
from lib.quality_scorers_core import register_scorer


@register_scorer("filename")
def _score_filename_relevance(output: str, case: TestCase) -> Score:
    out = _lower(output).strip()
    inp = _lower(case.input_text)
    ref = _lower(case.reference)

    if not out:
        return Score("Relevance", 0, 0.40, failures=["empty"])

    stopwords = {
        "the",
        "a",
        "an",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "to",
        "of",
        "in",
        "for",
        "on",
        "with",
        "at",
        "by",
        "from",
        "and",
        "or",
        "but",
        "not",
        "please",
        "try",
        "again",
        "showing",
    }
    inp_tokens = set(re.findall(r"[a-z0-9]+", inp)) - stopwords
    out_tokens = set(re.findall(r"[a-z0-9]+", out))

    matches = set()
    for it in inp_tokens:
        if it in out_tokens:
            matches.add(it)
        else:
            for ot in out_tokens:
                if it in ot:
                    matches.add(it)
                    break
    coverage = len(matches) / len(inp_tokens) if inp_tokens else 0

    ref_tokens = set(re.findall(r"[a-z0-9]+", ref))
    ref_matches = set()
    for rt in ref_tokens:
        if rt in out_tokens:
            ref_matches.add(rt)
        else:
            for ot in out_tokens:
                if rt in ot:
                    ref_matches.add(rt)
                    break

    failures = []
    score = 0
    if coverage >= 0.6:
        score = 100
    elif coverage >= 0.4:
        score = 75
    elif coverage >= 0.2:
        score = 50
    else:
        score = 25
        failures.append(f"only {len(matches)}/{len(inp_tokens)} input tokens covered")

    if ref_matches and len(ref_matches) < len(ref_tokens) * 0.4:
        score = min(score, 60)
        failures.append("missing key concepts from input")

    return Score("Relevance", score, 0.40, failures)


@register_scorer("filename")
def _score_filename_format(output: str, case: TestCase) -> Score:
    out = output.strip()
    if not out:
        return Score("Format", 0, 0.35, failures=["empty"])

    failures = []
    deduction = 0

    if _lower(out) in GENERIC_FILENAMES:
        return Score("Format", 0, 0.35, failures=["generic filename"])

    if "?" in out or "please" in _lower(out):
        deduction += 50
        failures.append("has question/instruction text")

    valid_part = re.sub(r"[a-zA-Z0-9_.-]", "", out)
    if valid_part:
        space_count = valid_part.count(" ")
        non_space = valid_part.replace(" ", "")
        if space_count > 0:
            deduction += 40 + (space_count * 5)
            failures.append(f"has {space_count} space(s)")
        if non_space:
            deduction += 20
            failures.append(f"invalid chars: {non_space[:10]}")

    if len(out) > 60:
        deduction += 20
        failures.append(f"too long ({len(out)} chars)")

    if out != _lower(out) and any(c.isupper() for c in out):
        deduction += 10
        failures.append("has uppercase")

    if "_" not in out and "-" not in out and "." not in out:
        deduction += 10
        failures.append("no separators")

    score = max(0, 100 - deduction)
    return Score("Format", score, 0.35, failures)


@register_scorer("filename")
def _score_filename_conciseness(output: str, case: TestCase) -> Score:
    out = output.strip()
    if not out:
        return Score("Conciseness", 0, 0.25, failures=["empty"])

    failures = []
    score = 100

    if "?" in out or "please" in _lower(out):
        return Score("Conciseness", 0, 0.25, failures=["not a filename (question)"])
    if " " in out:
        return Score("Conciseness", 10, 0.25, failures=["has spaces — not a filename"])

    length = len(out)
    if length < 5:
        score = 50
        failures.append(f"too short ({length} chars)")
    elif length < 10:
        score = 75
    elif 10 <= length <= 45:
        score = 100
    elif length <= 60:
        score = 80
        failures.append(f"slightly long ({length} chars)")
    else:
        score = 40
        failures.append(f"too long ({length} chars)")

    filler = ["the", "and", "of", "for", "with", "from", "this", "that"]
    if any(f in _lower(re.sub(r"[_-]", " ", out)).split() for f in filler):
        score = max(score - 15, 0)
        failures.append("has filler words")

    return Score("Conciseness", score, 0.25, failures)
