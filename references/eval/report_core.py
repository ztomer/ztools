#!/usr/bin/env python3
"""Table helpers, cross-model comparison, score stats and failure summary.

Split out of report.py for the repo's 500-line limit; report.py re-exports
everything here, so `from eval.report import ...` is unchanged.
"""

import statistics
from pathlib import Path

from lib.tui import STEP, console
from rich.table import Table

from eval.completeness import is_complete
from eval.discrimination import is_gate, ranking_mean


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

    # The row set is the UNION across models, not the first model's results.
    # Taking it from all_results[0] made the table order-dependent twice over:
    # when the first model's run was truncated, every task it never reached was
    # dropped as a ROW, silently hiding those scores for every OTHER model too;
    # and when the first model returned nothing at all, the whole table printed
    # empty while later models had full results. Same class as the mean over a
    # short run -- one incomplete run quietly redefining "all the tasks".
    tasks = []
    for r in all_results:
        for res in r.get("results", []):
            name = res.get("task")
            if name is not None and name not in tasks:
                tasks.append(name)

    if not tasks:
        # Header-only table: no model reported a single task.
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
    """Compute aggregate statistics for each model.

    `count` was already recorded and nothing ever compared it to the number of
    tasks the run was asked for, so a mean over 11 tasks and a mean over 30
    printed identically. `complete` is that comparison, carried so the printer
    can say which is which -- see eval/completeness.py.
    """
    stats = {}

    for r in all_results:
        model = r["model"]
        scores = [res.get("quality_score", 0) for res in r.get("results", [])]

        if not scores:
            continue

        results = r.get("results", [])
        stats[model] = {
            "mean": statistics.mean(scores),
            "median": statistics.median(scores),
            "stdev": statistics.stdev(scores) if len(scores) > 1 else 0,
            "min": min(scores),
            "max": max(scores),
            "count": len(scores),
            "complete": is_complete(r),
            # Gates cannot order the models that pass them, so counting them
            # pulls every model toward the same number. `mean` is kept beside it
            # because a gate FAILURE is real information -- it just is not a
            # ranking. See eval/discrimination.py.
            "ranking_mean": ranking_mean(results),
            "gate_tasks": sum(1 for res in results if is_gate(res.get("task"))),
        }

    return stats


def print_score_stats(stats: dict, out=None) -> None:
    """Print score statistics table using Rich Table.

    `Mean` is the RANKING mean -- gate tasks excluded, because a task every
    model passes cannot order the models that pass it. `All` is the mean over
    every task, kept beside it because a gate FAILURE is real information.
    They are equal for any run containing no gate tasks.
    """
    out = out or console
    if not stats:
        return

    columns = [
        {"name": "Model", "justify": "left"},
        {"name": "Mean", "justify": "right", "style": "cyan"},
        {"name": "All", "justify": "right", "style": "dim"},
        {"name": "Med", "justify": "right", "style": "cyan"},
        {"name": "Stdev", "justify": "right", "style": "yellow"},
        {"name": "Min", "justify": "right", "style": "green"},
        {"name": "Max", "justify": "right", "style": "green"},
        {"name": "N", "justify": "right", "style": "dim"},
    ]

    def _ranking(s):
        # Absent on stats dicts built before this existed, and on hand-built
        # ones in tests. Falling back to the plain mean keeps those honest
        # rather than reporting a zero.
        return s.get("ranking_mean", s["mean"])

    rows = []
    for model, s in sorted(stats.items(), key=lambda x: _ranking(x[1]), reverse=True):
        # The mark rides on the MEAN, not on a separate column, because the mean
        # is the number that gets copied into a table and quoted six weeks later.
        # A truncated run reported bonsai at 62% against a real 79%, and the
        # thing that made it dangerous was that it looked like every other cell.
        incomplete = not s.get("complete", True)
        mean = f"{_ranking(s):.1f}" + (" (partial)" if incomplete else "")
        count = str(s.get("count", 0)) + ("?" if incomplete else "")
        rows.append(
            [
                model,
                mean,
                f"{s['mean']:.1f}",
                f"{s['median']:.1f}",
                f"{s['stdev']:.1f}",
                str(s["min"]),
                str(s["max"]),
                count,
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
