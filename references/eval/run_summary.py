#!/usr/bin/env python3
"""End-of-run summaries printed after the task loop finishes.

Split out of eval/run.py for the 500-line limit. Pure reporting over results the
loop already produced -- nothing here feeds back into scoring, which is why it
was safe to lift whole.
"""

from lib.tui import STEP, WARN, console
from lib.validators_lib import get_source_matching_details


def print_source_matching_summary(tasks: dict, results: list) -> None:
    """For weekend tasks: how much of the output actually came from the source."""
    weekend_tasks = [k for k in tasks.keys() if "weekend" in k]
    if not weekend_tasks:
        return

    console.print("")
    console.print("Quality Check Summary:")
    for r in results:
        task_name = r["task"]
        if task_name not in weekend_tasks:
            continue
        task_cfg = tasks[task_name]
        source = task_cfg.get("source", "")
        if not source:
            continue
        parsed = r.get("result", {}).get("parsed", [])
        if not parsed:
            continue
        details = get_source_matching_details(parsed, source)
        matched = len(details["matched"])
        total = matched + len(details["unmatched"])
        ratio = details["ratio"] * 100
        console.print(f"  {task_name}: {matched}/{total} items from source ({ratio:.0f}%)")
        if details["unmatched"]:
            names = [
                u if isinstance(u, str) else u.get("name", "unnamed")
                for u in details["unmatched"][:2]
            ]
            console.print(f"    {WARN} Not from source: {names}")


def print_signal_noise_summary(tasks: dict, results: list) -> None:
    """For *_mixed tasks: whether the model kept the signal and dropped the noise."""
    mixed_tasks = [k for k in tasks.keys() if k.endswith("_mixed")]
    if not mixed_tasks:
        return

    console.print("")
    console.print("Signal/Noise Filtering:")
    for r in results:
        task_name = r["task"]
        if task_name not in mixed_tasks:
            continue
        reason = r.get("failure_reason", "")
        noise_part = ""
        if "noise" in reason:
            noise_part = reason
        elif "missed" in reason or "coverage" in reason:
            noise_part = reason
        symbol = WARN if ("included" in reason and "noise" in reason) else STEP
        console.print(
            f"  {symbol} {task_name}: {r['quality_score']}% — {noise_part or 'filtered clean'}"
        )


__all__ = ["print_signal_noise_summary", "print_source_matching_summary"]
