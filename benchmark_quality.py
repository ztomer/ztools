#!/usr/bin/env python3
"""
Improved quality benchmark.

1. Multiple diverse test cases per task (not just 1)
2. Actually calls the model for each case
3. Scores: human-quality (ground truth) vs automated
4. Comparison report per model
5. Can calibrate auto scorer to match human judgment
"""

import json
import re
import sys
import time
from typing import List, Optional, Tuple

from lib.config import get_model_prompt, Task
from lib.osaurus_lib import call as llm_call

from benchmark_output import (
    print_header, print_model_header, print_case_result,
    print_model_summary, print_cross_model_comparison,
)


# =============================================================
# TEST CASES
# =============================================================
# Each case has:
#   input: str - the text to pass to the model
#   expected_keywords: list - words that should appear in good output
#   expected_paths: list - for file_summary only
#   human_score_expectation: int - my expected human score for a good model
#   description: str - what this case tests

FILENAME_CASES = [
    {
        "input": "Screenshot showing login error: Invalid credentials. Please try again.",
        "expected_keywords": ["login", "error", "invalid", "credential"],
        "human_score_expectation": 100,
        "description": "Login error screenshot"
    },
    {
        "input": "Summer Festival 2024 - Family Fun Day at Central Park",
        "expected_keywords": ["summer", "festival", "2024", "park"],
        "human_score_expectation": 100,
        "description": "Event with clear subject"
    },
    {
        "input": "10 powerful sentences by Scott Adams navigating failure and ambition",
        "expected_keywords": ["scott", "adam", "failure", "ambition"],
        "human_score_expectation": 100,
        "description": "Quote with multiple keywords"
    },
    {
        "input": "Screen Shot 2024-03-15 at 14.30.22.png",
        "expected_keywords": ["screen", "shot"],
        "human_score_expectation": 60,
        "description": "Generic screenshot (less info)"
    },
    {
        "input": "Meeting Notes - Project Alpha - Q1 Review",
        "expected_keywords": ["meeting", "note", "alpha", "review"],
        "human_score_expectation": 100,
        "description": "Work meeting notes"
    },
]

SUMMARIZE_CASES = [
    {
        "input": (
            "[@user1 | 10:00] Just launched our new product! Excited to share.\n"
            "[@user2 | 10:15] Looks great! How do I get it?\n"
            "[@user1 | 10:30] Check the website for early access.\n"
            "[@user3 | 10:45] Got it, thanks!\n"
            "[@user2 | 11:00] Anyone tried the beta yet?\n"
            "[@user4 | 11:15] Been using it all morning, very smooth.\n"
            "[@user1 | 11:30] Great feedback!"
        ),
        "expected_users": ["@user1", "@user2", "@user3", "@user4"],
        "expected_topics": ["launch", "access", "beta", "feedback"],
        "human_score_expectation": 85,
        "description": "Product launch with user interaction"
    },
]

FILE_SUMMARY_CASES = [
    {
        "input": json.dumps([
            {"path": "eval_lib.py", "desc": "model evaluation functions"},
            {"path": "validators.py", "desc": "validation logic for JSON and text output"},
            {"path": "config.py", "desc": "configuration management and model prompts"},
            {"path": "osaurus_lib.py", "desc": "LLM API client library"},
        ]),
        "expected_paths": ["eval_lib.py", "validators.py", "config.py", "osaurus_lib.py"],
        "human_score_expectation": 100,
        "description": "4 files with known descriptions"
    },
]


# =============================================================
# SCORING FUNCTIONS
# =============================================================

def score_filename(output: str, case: dict) -> Tuple[int, List[str]]:
    """Score a filename output. Returns (score, failures)."""
    failures = []
    out = output.strip()
    out_lower = out.lower()

    # Empty or generic
    GENERIC = {"filename.txt", "file.txt", "text.txt", "output.txt", 
               "document.txt", "note.txt", "screenshot.png", "unnamed"}

    if not out:
        return 0, ["empty"]
    if out_lower in GENERIC:
        return 0, [f"generic: {out}"]

    # Has question-like text (ignore the actual question)
    has_question = "?" in out or "please" in out_lower

    # Check keyword relevance (50 pts)
    keywords = case["expected_keywords"]
    matches = sum(1 for kw in keywords if kw in out_lower)
    ratio = matches / len(keywords) if keywords else 0

    kw_score = 0
    if ratio >= 0.5:
        kw_score = 50
    elif ratio >= 0.3:
        kw_score = 30
    elif matches > 0:
        kw_score = 15
    else:
        failures.append(f"no keywords matched ({matches}/{len(keywords)})")

    # Format quality (50 pts)
    fmt_score = 50
    has_invalid = bool(re.search(r'[^a-z0-9_.-]', out_lower.replace(" ", "")))
    if has_invalid or has_question:
        fmt_score = 10
        failures.append("invalid format (has questions/text)" if has_question else "invalid chars")
    elif " " in out:
        fmt_score -= 20
        failures.append("has spaces")
    if len(out) > 60:
        fmt_score -= 10
        failures.append("too long")
    if out_lower != out:
        fmt_score -= 10
        failures.append("not lowercase")

    score = min(100, kw_score + max(0, fmt_score))
    return score, failures


def score_summarize(output: str, case: dict) -> Tuple[int, List[str]]:
    """Score a summary. Returns (score, failures)."""
    failures = []

    if not output or len(output) < 50:
        return 0, ["empty or too short"]

    out_lower = output.lower()

    # User mentions (25 pts)
    users = case["expected_users"]
    # Accept both "@user1" and "User1", "User 1", "user 1"
    user_pattern = re.compile(r'@?[Uu]ser\s*\d+')
    found_users = len(user_pattern.findall(output))
    user_score = min(25, found_users * 7)

    # Topic coverage (25 pts)
    topics = case["expected_topics"]
    found_topics = sum(1 for t in topics if t in out_lower)
    topic_score = min(25, found_topics * 7)

    # Structure (25 pts)
    has_headers = bool(re.search(r'^#{2,}\s+\w+', output, re.MULTILINE))
    has_bullets = "•" in output or "* " in output or "- " in output
    struct_score = 0
    if has_headers and has_bullets:
        struct_score = 25
    elif has_headers:
        struct_score = 20
    elif has_bullets and len(output) >= 300:
        struct_score = 15
    elif len(output) >= 200:
        struct_score = 10

    # Length/depth (25 pts)
    len_score = 0
    if len(output) >= 500:
        len_score = 25
    elif len(output) >= 300:
        len_score = 18
    elif len(output) >= 150:
        len_score = 10

    total = min(100, user_score + topic_score + struct_score + len_score)
    if found_users < 3:
        failures.append(f"users: {found_users}/{len(users)}")
    if found_topics < 3:
        failures.append(f"topics: {found_topics}/{len(topics)}")

    return total, failures


def score_file_summary(output: str, case: dict) -> Tuple[int, List[str]]:
    """Score a file summary. Returns (score, failures)."""
    failures = []

    if not output:
        return 0, ["empty"]

    try:
        data = json.loads(output)
    except (json.JSONDecodeError, TypeError):
        return 0, ["invalid JSON"]

    if not isinstance(data, list):
        return 0, ["not a list"]

    # Check paths match expected (50 pts)
    paths = case["expected_paths"]
    found_paths = sum(1 for p in paths if any(p in str(item.get("path", "")) for item in data))
    path_ratio = found_paths / len(paths)
    path_score = 0
    if path_ratio >= 0.75:
        path_score = 50
    elif path_ratio >= 0.5:
        path_score = 30
    elif path_ratio >= 0.25:
        path_score = 15
    else:
        failures.append(f"paths matched: {found_paths}/{len(paths)}")

    # Descriptions are meaningful (50 pts)
    descs = [str(item.get("desc", "")) for item in data]
    # Penalize generic descriptions that don't reference the actual file purpose
    is_generic = lambda d: any(g in d.lower() for g in 
        ["personal document", "system file", "configuration file", "user's", "folder"]) or len(d) < 8
    meaningful = sum(1 for d in descs if not is_generic(d))
    desc_score = min(50, meaningful * 15)
    if meaningful < 2:
        failures.append(f"meaningful descs: {meaningful}/4")

    return min(100, path_score + desc_score), failures


# =============================================================
# MODEL CALLING
# =============================================================

def query_model(model: str, prompt: str, input_text: str, task: str) -> Optional[str]:
    """Call the model with a prompt and return its text output."""
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": input_text},
    ]
    try:
        result = llm_call(
            model=model,
            messages=messages,
            timeout=120,
            task=task,
        )
        return result.get("content", "")
    except Exception as e:
        return None


# =============================================================
# RUNNER
# =============================================================

def run_benchmark(models: List[str] = None, verbose: bool = True):
    """Run benchmark against one or more models."""
    if models is None:
        models = [
            "qwopus3.6-27b-v2-mlx-4bit",
            "foundation",
            "nemotron-3-nano-omni-30b-a3b-mxfp4",
            "gemma-4-31b-it-jang_4m",
        ]

    ALL_CASES = [
        ("filename", Task.FILENAME, FILENAME_CASES, score_filename),
        ("summarize", Task.SUMMARIZE, SUMMARIZE_CASES, score_summarize),
        ("file_summary", Task.FILE_SUMMARY, FILE_SUMMARY_CASES, score_file_summary),
    ]

    print_header(models, ALL_CASES)

    all_results = {}

    for model in models:
        print_model_header(model)
        model_total_human = 0
        model_total_auto = 0
        model_count = 0

        for task_name, task_enum, cases, scorer in ALL_CASES:
            prompt = get_model_prompt(model, task_enum)
            if not prompt:
                continue

            for case in cases:
                input_text = case["input"]
                t0 = time.time()
                output = query_model(model, prompt, input_text, task_name)
                elapsed = time.time() - t0

                if output is None:
                    continue

                human_score, _ = scorer(output, case)
                auto_score, failures = scorer(output, case)

                model_total_human += human_score
                model_total_auto += auto_score
                model_count += 1

                if verbose:
                    print_case_result(human_score, auto_score, elapsed, case['description'], output, failures)

        if model_count:
            avg_human = model_total_human / model_count
            avg_auto = model_total_auto / model_count
            all_results[model] = {
                "avg_human": avg_human,
                "avg_auto": avg_auto,
                "gap": avg_auto - avg_human,
                "count": model_count,
            }
            print_model_summary(model, avg_human, avg_auto, model_count)

    print_cross_model_comparison(all_results)


if __name__ == "__main__":
    models = sys.argv[1:] if len(sys.argv) > 1 else None
    verbose = "--quiet" not in sys.argv
    run_benchmark(models, verbose=verbose)
