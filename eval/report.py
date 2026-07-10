#!/usr/bin/env python3
"""
Reporting and analysis module for model evaluation.
Contains all display, comparison, and export functions.
"""

import csv
import json
import os
import statistics
import time
from pathlib import Path
from rich.console import Console
from rich.table import Table

from lib.tui import STEP

console = Console()


def _make_table(columns: list, rows: list, title: str = None, show_header: bool = True) -> Table:
    """Create a Rich table with consistent styling."""
    table = Table(show_header=show_header, header_style="bold", show_lines=False, title=title)
    for col in columns:
        table.add_column(col["name"], style=col.get("style", ""), justify=col.get("justify", "left"), 
                         no_wrap=col.get("nowrap", False), min_width=col.get("min_width", 0))
    for row in rows:
        table.add_row(*[str(c) for c in row])
    return table


def print_cross_model_comparison(all_results: list) -> None:
    """Print comparison table across all models using Rich Table."""
    if not all_results:
        return

    models = [r["model"] for r in all_results]
    if not models:
        return

    first_results = all_results[0].get("results", [])
    if not first_results:
        # Print empty table with header
        columns = [{"name": "Task", "justify": "left"}]
        for m in models:
            columns.append({"name": m[:15], "justify": "right", "style": "dim"})
        table = Table(show_header=True, header_style="bold", show_lines=False, 
                      title=f"{STEP} Cross-Model Comparison")
        for col in columns:
            table.add_column(col["name"], style=col.get("style", ""), 
                            justify=col.get("justify", "left"))
        console.print(table)
        return

    tasks = [res["task"] for res in first_results]
    if not tasks:
        return

    columns = [{"name": "Task", "justify": "left"}]
    for m in models:
        columns.append({"name": m[:15], "justify": "right", "style": "dim"})

    rows = []
    for task in tasks:
        row = [task]
        task_scores = {}
        for r in all_results:
            score = 0
            for res in r.get("results", []):
                if res.get("task") == task:
                    score = res.get("quality_score", 0)
                    break
            task_scores[r["model"]] = score
            row.append(str(score))
        best_model = max(task_scores, key=task_scores.get)
        best_score = task_scores[best_model]
        row.append(f"{best_score}*")
        rows.append(row)

    table = _make_table(columns, rows, title=f"{STEP} Cross-Model Comparison")
    console.print(table)


def compute_score_stats(all_results: list) -> dict:
    """Compute aggregate statistics for each model."""
    stats = {}

    for r in all_results:
        model = r["model"]
        scores = [res.get("quality_score", 0) for res in r.get("results", [])]

        if not scores:
            continue

        stats[model] = {
            "mean": statistics.mean(scores),
            "median": statistics.median(scores),
            "stdev": statistics.stdev(scores) if len(scores) > 1 else 0,
            "min": min(scores),
            "max": max(scores),
            "count": len(scores),
        }

    return stats


def print_score_stats(stats: dict) -> None:
    """Print score statistics table using Rich Table."""
    if not stats:
        return

    columns = [
        {"name": "Model", "justify": "left"},
        {"name": "Mean", "justify": "right", "style": "cyan"},
        {"name": "Med", "justify": "right", "style": "cyan"},
        {"name": "Stdev", "justify": "right", "style": "yellow"},
        {"name": "Min", "justify": "right", "style": "green"},
        {"name": "Max", "justify": "right", "style": "green"},
    ]

    rows = []
    for model, s in sorted(stats.items(), key=lambda x: x[1]["mean"], reverse=True):
        rows.append([model, f"{s['mean']:.1f}", f"{s['median']:.1f}", 
                     f"{s['stdev']:.1f}", str(s['min']), str(s['max'])])

    table = _make_table(columns, rows, title=f"{STEP} Score Statistics")
    console.print(table)


def categorize_failures(all_results: list) -> dict:
    """Group failures by category across all models."""
    categories = {}

    for r in all_results:
        for res in r.get("results", []):
            if res.get("quality_score", 0) >= 90:
                continue

            cat = res.get("failure_category", "UNKNOWN")
            if cat not in categories:
                categories[cat] = {"count": 0, "models": set(), "tasks": set()}

            categories[cat]["count"] += 1
            categories[cat]["models"].add(r["model"])
            categories[cat]["tasks"].add(res.get("task"))

    for cat in categories:
        categories[cat]["models"] = list(categories[cat]["models"])
        categories[cat]["tasks"] = list(categories[cat]["tasks"])

    return categories


def print_failure_summary(categories: dict) -> None:
    """Print failure categorization summary."""
    if not categories:
        return

    console.print("")
    for cat, info in sorted(categories.items(), key=lambda x: x[1]["count"], reverse=True):
        count = info["count"]
        models = ", ".join(info["models"][:3])
        tasks = ", ".join(info["tasks"][:2])
        console.print(f"  [{count}] {cat}: {models} ({tasks})")


def save_historical_results(all_results: list, stats: dict, categories: dict) -> None:
    """Save per-model scores that persist even when models change."""
    eval_dir = Path(os.path.expanduser("~/.config/ztools"))
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


def load_historical_stats() -> dict:
    """Load per-model historical scores."""
    eval_dir = Path(os.path.expanduser("~/.config/ztools"))
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


def check_model_history(model: str) -> dict:
    """Check if model has historical data."""
    eval_dir = Path(os.path.expanduser("~/.config/ztools"))
    history_file = eval_dir / "eval_history.json"
    if not history_file.exists():
        return {}

    try:
        with open(history_file) as f:
            history = json.load(f)
    except Exception:
        return {}

    return history.get(model, [])


def print_historical_trends() -> None:
    """Print historical score trends per model using Rich Table."""
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
    console.print(table)


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


def print_verbosity(verbosity: dict) -> None:
    """Print response length per task using Rich Table."""
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
    console.print(table)


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


def print_error_rates(rates: dict) -> None:
    """Print error rate breakdown using Rich Table."""
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
        rows.append([model, str(r['infra']), str(r['quality']), str(r['success']), f"{rate:.0f}%"])

    table = _make_table(columns, rows, title=f"{STEP} Error Rates")
    console.print(table)


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


def diff_from_last_run(all_results: list) -> dict:
    """Compare current scores to last run for each model."""
    eval_dir = Path(os.path.expanduser("~/.config/ztools"))
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


def print_diff(diffs: dict) -> None:
    """Print score changes from last run using Rich Table."""
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
                rows.append([model[:18], task[:18], str(d['prev']), str(d['current']), f"{arrow}{abs(diff)}"])

    if rows:
        table = _make_table(columns, rows)
        console.print(table)


def export_to_csv(all_results: list, output_file: str = None) -> None:
    """Export results to CSV for reporting."""
    if output_file is None:
        eval_dir = Path(os.path.expanduser("~/.config/ztools"))
        output_file = eval_dir / "eval_results.csv"
    else:
        output_file = Path(output_file)

    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Model", "Task", "Score", "Status", "Time(s)", "Failure", "Failure_Category"])

        for r in all_results:
            model = r["model"]
            for res in r.get("results", []):
                score = res.get("quality_score", 0)
                status = "PASS" if score >= 90 else ("WARN" if score >= 50 else "FAIL")
                time_s = res.get("time", "")
                failure = res.get("failure_reason", "")
                category = res.get("failure_category", "")

                writer.writerow([model, res.get("task"), score, status, time_s, failure, category])

    console.print(f"{STEP} Exported to {output_file}")