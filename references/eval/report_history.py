#!/usr/bin/env python3
"""Historical result persistence and trend reporting.

Split out of report.py for the repo's 500-line limit; report.py re-exports
everything here, so `from eval.report import ...` is unchanged.
"""

import json
import statistics
import time

from lib.tui import STEP, WARN

# Resolved through the module object, not by name: that keeps ONE patch
# point (eval.report_core.console / ._get_eval_dir) for the whole report
# surface, which is the seam the tests use.
from eval import report_core
from eval.completeness import is_complete
from eval.report_core import _make_table, default_eval_dir

# Only UNAMBIGUOUS test doubles. `mock-model` sat at mean 100 and topped the
# historical trend table. Short names like "m1" are left alone: they are used as
# fixtures but could plausibly be a real local model, and wrongly hiding a user's
# real results is worse than one stray row.
_TEST_MODEL_PREFIXES = ("mock", "test-", "fake")


def is_test_model(model: str) -> bool:
    """Whether a model name is a test double rather than a real served model."""
    return (model or "").strip().lower().startswith(_TEST_MODEL_PREFIXES)


def save_historical_results(
    all_results: list, stats: dict, categories: dict, eval_dir=None
) -> None:
    """Save per-model scores that persist even when models change."""
    eval_dir = eval_dir or default_eval_dir()
    eval_dir.mkdir(parents=True, exist_ok=True)
    history_file = eval_dir / "eval_history.json"
    history = {}

    if history_file.exists():
        try:
            with open(history_file) as f:
                history = json.load(f)
        except Exception:
            pass

    quarantined = []
    for r in all_results:
        model = r["model"]
        # Test doubles must not enter the production leaderboard: `mock-model`
        # sat at mean 100 and topped the historical trend table.
        if is_test_model(model):
            continue
        if model not in history:
            history[model] = []

        # MARKED, not dropped. A truncated run's individual task scores are real
        # -- the task that completed completed -- and deleting them would throw
        # away evidence. What is NOT real is any aggregate over them, because the
        # subset that finished is the subset the model found easy. So the entries
        # are written with the verdict attached and `load_historical_stats`
        # refuses to average them. Writing them to a separate quarantine FILE was
        # the first design and was wrong: a second store is a second thing
        # consumers forget to read, which is the parallel-pipeline drift class.
        complete = is_complete(r)
        if not complete:
            quarantined.append((model, (r.get("completeness") or {}).get("reason", "")))

        for res in r.get("results", []):
            entry = {
                "date": time.strftime("%Y-%m-%d"),
                "timestamp": time.time(),
                "task": res.get("task"),
                "score": res.get("quality_score", 0),
                "time": res.get("time"),
            }
            if not complete:
                entry["complete"] = False
            history[model].append(entry)

    # Never silent: something that refuses to count data says so at the moment it
    # refuses, or the next session reads a smaller mean and cannot tell why.
    for model, reason in quarantined:
        report_core.console.print(
            f"{WARN} {model}: entries recorded but EXCLUDED from historical "
            f"averages -- {reason}"
        )

    for model in history:
        history[model] = history[model][-100:]

    with open(history_file, "w") as f:
        json.dump(history, f, indent=2)


def load_historical_stats(eval_dir=None) -> dict:
    """Load per-model historical scores."""
    eval_dir = eval_dir or default_eval_dir()
    history_file = eval_dir / "eval_history.json"
    if not history_file.exists():
        return {}

    try:
        with open(history_file) as f:
            history = json.load(f)
    except Exception:
        return {}

    if not history:
        return {}

    stats = {}
    for model, entries in history.items():
        if not entries:
            continue

        # `if e.get("score")` is falsy for 0, so every total failure was
        # dropped from mean/median/min — a model that scored 0 on half its runs
        # looked identical to one that never failed.
        #
        # Entries from a truncated run are excluded here rather than at write
        # time. `ornith-1.0-9b-mxfp8` has 55 entries and its 11-of-30 run is
        # among them; averaging them reports the model's easy tasks as its
        # score. `complete` is absent on every entry written before this existed,
        # and absent means complete -- see eval/completeness.is_complete.
        countable = [e for e in entries if e.get("complete", True)]
        excluded = len(entries) - len(countable)
        scores = [e["score"] for e in countable if e.get("score") is not None]
        if scores:
            stats[model] = {
                "mean": statistics.mean(scores),
                "median": statistics.median(scores),
                "stdev": statistics.stdev(scores) if len(scores) > 1 else 0,
                "min": min(scores),
                "max": max(scores),
                "runs": len(countable),
                # Surfaced, not hidden: a model whose history is mostly truncated
                # runs has a `runs` count that no longer matches its entry count,
                # and that discrepancy is itself the finding.
                "excluded": excluded,
            }

    return stats


def check_model_history(model: str, eval_dir=None) -> dict:
    """Check if model has historical data."""
    eval_dir = eval_dir or default_eval_dir()
    history_file = eval_dir / "eval_history.json"
    if not history_file.exists():
        return {}

    try:
        with open(history_file) as f:
            history = json.load(f)
    except Exception:
        return {}

    return history.get(model, [])


def print_historical_trends(out=None) -> None:
    """Print historical score trends per model using Rich Table."""
    out = out or report_core.console
    stats = load_historical_stats()
    if not stats:
        return

    columns = [
        {"name": "Model", "justify": "left"},
        {"name": "Runs", "justify": "right", "style": "dim"},
        {"name": "Mean", "justify": "right", "style": "cyan"},
        {"name": "Stdev", "justify": "right", "style": "yellow"},
        {"name": "Trend", "justify": "left"},
    ]

    rows = []
    for model, s in sorted(stats.items(), key=lambda x: x[1]["mean"], reverse=True):
        runs = s.get("runs", 0)
        mean = s.get("mean", 0)
        stdev = s.get("stdev", 0)

        if runs >= 3:
            if stdev < 5:
                trend = "stable"
            elif stdev < 15:
                trend = "variable"
            else:
                trend = "unstable"
        else:
            trend = "new"

        rows.append([model, str(runs), f"{mean:.0f}", f"{stdev:.1f}", trend])

    table = _make_table(columns, rows, title=f"{STEP} Historical Trends")
    out.print(table)
