#!/usr/bin/env python3
"""Emit this project's status for the `routines` harness.

ADDITIVE AND READ-ONLY. This file adds a machine-readable view; it changes
nothing about how ztools runs standalone, and no existing file was modified to
make it work.

IT DOES NOT PLAN. A `wk` run scrapes the web and drives a 35B model for
minutes; a status command that did that would make every daily report a
multi-minute job, and would burn a model run just to answer "how did the last
one go". It reads back the newest `weekend_plan_*.md` instead.

IT REUSES THE CHECKERS' OWN PARSERS. `eval.report_classes` is what the G3
quality checks parse plans with, so the numbers on the daily page and the
numbers the checks assert on come from one place. A private copy of the parsing
here is exactly how enforcement and its checker drift apart — this project has
already paid for that once (six checkers wrong, several keeping their own copy
of the logic they were checking).

WHAT IT WATCHES. Two failure modes, both of which look fine from outside:
  - a plan that is HOLLOW. An empty plan passes every content check — no
    fabricated constant, no stale date, no excluded venue, because there are no
    rows. This is the open defect (PENDING 5.1: event supply), so the page
    should say it out loud every day until it is fixed.
  - a plan that is STALE. Last week's plan sitting in the output directory
    looks identical to this week's unless someone compares its window to the
    calendar.

Contract: ~/Projects/routines/docs/STATUS_CONTRACT.md
"""

from __future__ import annotations

import json
import os
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from eval.report_classes import (
    fixed_rows,
    parse_window_from_wk_filename,
    transient_rows,
)
from weekend.cli import OUTPUT_DIR_PATH, OUTPUT_FILE_SUFFIX, PLAN_FILE_PREFIX

NAME = "ztools-weekend"


def unknown(summary: str) -> dict:
    """The honest answer when there is nothing to read.

    Never `ok`: "0 events" from a missing file is indistinguishable from a
    genuinely empty plan, and both are indistinguishable from a working tool
    that simply has not run yet.
    """
    return {"name": NAME, "state": "unknown", "summary": summary}


def upcoming_weekend(today: date) -> tuple[date, date]:
    """The Friday-to-Sunday a plan should currently cover.

    During a weekend the answer is *this* one, not the next: a plan for the
    days you are living through is current, not stale.
    """
    # Monday = 0 ... Friday = 4, Saturday = 5, Sunday = 6.
    if today.weekday() >= 4:
        friday = today - timedelta(days=today.weekday() - 4)
    else:
        friday = today + timedelta(days=4 - today.weekday())
    return friday, friday + timedelta(days=2)


def newest_plan(directory: Path) -> Path | None:
    plans = sorted(
        directory.glob(f"{PLAN_FILE_PREFIX}*{OUTPUT_FILE_SUFFIX}"),
        key=lambda p: p.stat().st_mtime,
    )
    return plans[-1] if plans else None


def build_status(today: date | None = None) -> dict:
    today = today or date.today()
    directory = Path(os.path.expanduser(OUTPUT_DIR_PATH))
    if not directory.is_dir():
        return unknown(f"no plan directory at {directory}")

    plan = newest_plan(directory)
    if plan is None:
        return unknown(f"no weekend plan has ever been written to {directory}")

    text = plan.read_text(encoding="utf-8")
    window = parse_window_from_wk_filename(plan)
    fixed = len(fixed_rows(text))
    transient = len(transient_rows(text))

    wanted_start, wanted_end = upcoming_weekend(today)
    covers_upcoming = window is not None and window[0] == wanted_start

    when = (
        f"{window[0].isoformat()}..{window[1].isoformat()}"
        if window
        else "an unreadable date range"
    )

    items: list[dict] = []
    state = "ok"
    if not covers_upcoming:
        state = "attention"
        items.append(
            {
                "name": f"plan for {wanted_start.isoformat()}",
                "action": "not generated yet",
            }
        )
    if transient == 0:
        # The known open defect, and the one an empty plan hides best: every
        # content check passes when there are no rows to be wrong.
        state = "attention"
        items.append({"name": "transient events", "action": "none in the latest plan"})

    summary = f"latest plan {when}: {fixed} fixed, {transient} transient"
    if not covers_upcoming:
        summary += f" (upcoming weekend {wanted_start.isoformat()} not planned)"

    return {
        "name": NAME,
        "state": state,
        "summary": summary,
        # When the plan was actually written, not when we asked.
        "ran_at": datetime.fromtimestamp(plan.stat().st_mtime)
        .astimezone()
        .isoformat(timespec="seconds"),
        "items": items,
        "details": {
            "plan": str(plan),
            "window": [w.isoformat() for w in window] if window else None,
            "fixed_rows": fixed,
            "transient_rows": transient,
            "upcoming_weekend": [wanted_start.isoformat(), wanted_end.isoformat()],
            "covers_upcoming": covers_upcoming,
        },
    }


def main() -> None:
    try:
        status = build_status()
    except Exception as exc:
        # The harness needs a stated reason, not a traceback.
        status = unknown(f"status unavailable: {type(exc).__name__}: {exc}")

    json.dump(status, sys.stdout, indent=2)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
