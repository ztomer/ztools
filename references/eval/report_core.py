#!/usr/bin/env python3
"""Table helpers, cross-model comparison, score stats and failure summary.

Split out of report.py for the repo's 500-line limit; report.py re-exports
everything here, so `from eval.report import ...` is unchanged.
"""

import statistics
from pathlib import Path

from lib.tui import STEP, console
from rich.table import Table


def default_eval_dir() -> Path:
    """Where eval artefacts live when the caller does not say otherwise.

    Callers take `eval_dir` as a parameter and fall back to this, so tests hand
    in a tmp dir instead of patching a module attribute.
    """
    return Path.home() / ".config" / "ztools"


# Retained for callers that still import the old private name.
_get_eval_dir = default_eval_dir


def _make_table(columns: list, rows: list, title: str = None, show_header: bool = True) -> Table:
    """Create a Rich table with consistent styling."""
    table = Table(show_header=show_header, header_style="bold", show_lines=False, title=title)
    for col in columns:
        table.add_column(
            col["name"],
            style=col.get("style", ""),
            justify=col.get("justify", "left"),
            no_wrap=col.get("nowrap", False),
            min_width=col.get("min_width", 0),
        )
    for row in rows:
        table.add_row(*[str(c) for c in row])
    return table


def print_cross_model_comparison(all_results: list, out=None) -> None:
    """Print comparison table across all models using Rich Table."""
    out = out or console
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
        table = Table(
            show_header=True,
            header_style="bold",
            show_lines=False,
            title=f"{STEP} Cross-Model Comparison",
        )
        for col in columns:
            table.add_column(
                col["name"], style=col.get("style", ""), justify=col.get("justify", "left")
            )
        out.print(table)
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
    out.print(table)


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


def print_score_stats(stats: dict, out=None) -> None:
    """Print score statistics table using Rich Table."""
    out = out or console
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
        rows.append(
            [
                model,
                f"{s['mean']:.1f}",
                f"{s['median']:.1f}",
                f"{s['stdev']:.1f}",
                str(s["min"]),
                str(s["max"]),
            ]
        )

    table = _make_table(columns, rows, title=f"{STEP} Score Statistics")
    out.print(table)


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


def print_failure_summary(categories: dict, out=None) -> None:
    """Print failure categorization summary."""
    out = out or console
    if not categories:
        return

    out.print("")
    for cat, info in sorted(categories.items(), key=lambda x: x[1]["count"], reverse=True):
        count = info["count"]
        models = ", ".join(info["models"][:3])
        tasks = ", ".join(info["tasks"][:2])
        out.print(f"  [{count}] {cat}: {models} ({tasks})")
