#!/usr/bin/env python3
"""Derived metrics: tokens, verbosity, error rates, winners, diffs, CSV export.

Split out of report.py for the repo's 500-line limit; report.py re-exports
everything here, so `from eval.report import ...` is unchanged.
"""

import csv
import json
from pathlib import Path

from lib.tui import STEP

# Resolved through the module object, not by name: that keeps ONE patch
# point (eval.report_core.console / ._get_eval_dir) for the whole report
# surface, which is the seam the tests use.
from eval import report_core
from eval.report_core import _make_table, default_eval_dir


def compute_token_estimates(results: list) -> dict:
    """Rough token estimation from response length (~4 chars/token)."""
    input_tokens = 0
    output_tokens = 0

    for r in results:
        content = r.get("content", "")
        if content:
            output_tokens += len(content) // 4

        messages = r.get("messages", [])
        for msg in messages:
            content = msg.get("content", "")
            if content:
                input_tokens += len(content) // 4

    return {"input": input_tokens, "output": output_tokens, "total": input_tokens + output_tokens}


def compute_verbosity(all_results: list) -> dict:
    """Compute avg response length per task per model."""
    verbosity = {}

    for r in all_results:
        model = r["model"]
        verbosity[model] = {}

        for res in r.get("results", []):
            task = res.get("task")
            content = res.get("result", {}).get("content", "")
            length = len(content) if content else 0
            verbosity[model][task] = length

    return verbosity


def print_verbosity(verbosity: dict, out=None) -> None:
    """Print response length per task using Rich Table."""
    out = out or report_core.console
    if not verbosity:
        return

    models = list(verbosity.keys())
    first_model = models[0]
    tasks = list(verbosity[first_model].keys())

    columns = [{"name": "Task", "justify": "left"}]
    for m in models:
        columns.append({"name": m[:12], "justify": "right", "style": "dim"})

    rows = []
    for task in tasks:
        row = [task]
        for m in models:
            length = verbosity[m].get(task, 0)
            row.append(f"{length:,}")
        rows.append(row)

    table = _make_table(columns, rows, title=f"{STEP} Response Length per Task")
    out.print(table)


def compute_error_rates(all_results: list) -> dict:
    """Compute error rates: infra errors vs quality failures."""
    rates = {}

    for r in all_results:
        model = r["model"]
        infra = 0
        quality = 0
        success = 0

        for res in r.get("results", []):
            category = res.get("failure_category", "OK")
            error = res.get("error")

            if error or category == "INFRA":
                infra += 1
            elif res.get("quality_score", 0) < 50:
                quality += 1
            else:
                success += 1

        total = infra + quality + success
        rates[model] = {
            "infra": infra,
            "quality": quality,
            "success": success,
            "infra_rate": infra / total if total else 0,
            "quality_rate": quality / total if total else 0,
            "success_rate": success / total if total else 0,
        }

    return rates


def print_error_rates(rates: dict, out=None) -> None:
    """Print error rate breakdown using Rich Table."""
    out = out or report_core.console
    if not rates:
        return

    columns = [
        {"name": "Model", "justify": "left"},
        {"name": "Infra", "justify": "right", "style": "red"},
        {"name": "Quality", "justify": "right", "style": "yellow"},
        {"name": "Success", "justify": "right", "style": "green"},
        {"name": "Rate", "justify": "right", "style": "cyan"},
    ]

    rows = []
    for model, r in sorted(rates.items(), key=lambda x: x[1]["success_rate"], reverse=True):
        rate = r["success_rate"] * 100
        rows.append([model, str(r["infra"]), str(r["quality"]), str(r["success"]), f"{rate:.0f}%"])

    table = _make_table(columns, rows, title=f"{STEP} Error Rates")
    out.print(table)


def compute_task_winners(all_results: list) -> dict:
    """Find which model wins each task."""
    winners = {}

    for r in all_results:
        for res in r.get("results", []):
            task = res.get("task")
            score = res.get("quality_score", 0)

            if task not in winners or score > winners[task][1]:
                winners[task] = (r["model"], score)

    return winners


def diff_from_last_run(all_results: list, eval_dir=None) -> dict:
    """Compare current scores to last run for each model."""
    eval_dir = eval_dir or default_eval_dir()
    prev_file = eval_dir / "eval_results.json"

    if not prev_file.exists():
        return {}

    try:
        with open(prev_file) as f:
            prev_data = json.load(f)
    except Exception:
        return {}

    prev_results = prev_data.get("models", [])
    if not prev_results:
        return {}

    diffs = {}
    for r in all_results:
        model = r["model"]
        prev_model_data = next((p for p in prev_results if p.get("model") == model), None)

        if not prev_model_data:
            continue

        diffs[model] = {}

        for res in r.get("results", []):
            task = res.get("task")
            score = res.get("quality_score", 0)

            prev_score = 0
            for p in prev_model_data.get("results", []):
                if p.get("task") == task:
                    prev_score = p.get("quality_score", 0)
                    break

            diff = score - prev_score
            if diff != 0:
                diffs[model][task] = {"current": score, "prev": prev_score, "diff": diff}

    return diffs


def print_diff(diffs: dict, out=None) -> None:
    """Print score changes from last run using Rich Table."""
    out = out or report_core.console
    if not diffs:
        return

    has_changes = False
    for model, changes in diffs.items():
        for task, d in changes.items():
            if d.get("diff", 0) != 0:
                has_changes = True

    if not has_changes:
        return

    columns = [
        {"name": "Model", "justify": "left", "min_width": 18},
        {"name": "Task", "justify": "left", "min_width": 18},
        {"name": "Prev", "justify": "right", "style": "dim"},
        {"name": "Now", "justify": "right", "style": "cyan"},
        {"name": "Diff", "justify": "right"},
    ]

    rows = []
    for model, changes in diffs.items():
        for task, d in changes.items():
            diff = d.get("diff", 0)
            if diff != 0:
                arrow = "↑" if diff > 0 else "↓"
                rows.append(
                    [
                        model[:18],
                        task[:18],
                        str(d["prev"]),
                        str(d["current"]),
                        f"{arrow}{abs(diff)}",
                    ]
                )

    if rows:
        table = _make_table(columns, rows)
        out.print(table)


def export_to_csv(all_results: list, output_file: str = None, out=None, eval_dir=None) -> None:
    """Export results to CSV for reporting."""
    eval_dir = eval_dir or default_eval_dir()
    out = out or report_core.console
    if output_file is None:
        output_file = eval_dir / "eval_results.csv"
    else:
        output_file = Path(output_file)

    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["Model", "Task", "Score", "Status", "Time(s)", "Failure", "Failure_Category"]
        )

        for r in all_results:
            model = r["model"]
            for res in r.get("results", []):
                score = res.get("quality_score", 0)
                status = "PASS" if score >= 90 else ("WARN" if score >= 50 else "FAIL")
                time_s = res.get("time", "")
                failure = res.get("failure_reason", "")
                category = res.get("failure_category", "")

                writer.writerow([model, res.get("task"), score, status, time_s, failure, category])

    out.print(f"{STEP} Exported to {output_file}")
