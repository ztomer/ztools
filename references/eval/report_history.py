#!/usr/bin/env python3
"""Historical result persistence and trend reporting.

Split out of report.py for the repo's 500-line limit; report.py re-exports
everything here, so `from eval.report import ...` is unchanged.
"""

import json
import statistics
import time

from lib.tui import STEP

# Resolved through the module object, not by name: that keeps ONE patch
# point (eval.report_core.console / ._get_eval_dir) for the whole report
# surface, which is the seam the tests use.
from eval import report_core
from eval.report_core import _make_table, default_eval_dir


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

    for r in all_results:
        model = r["model"]
        if model not in history:
            history[model] = []

        for res in r.get("results", []):
            entry = {
                "date": time.strftime("%Y-%m-%d"),
                "timestamp": time.time(),
                "task": res.get("task"),
                "score": res.get("quality_score", 0),
                "time": res.get("time"),
            }
            history[model].append(entry)

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

        scores = [e["score"] for e in entries if e.get("score")]
        if scores:
            stats[model] = {
                "mean": statistics.mean(scores),
                "median": statistics.median(scores),
                "stdev": statistics.stdev(scores) if len(scores) > 1 else 0,
                "min": min(scores),
                "max": max(scores),
                "runs": len(entries),
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
