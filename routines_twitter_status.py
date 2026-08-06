#!/usr/bin/env python3
"""Emit Twitter status for the `routines` harness.

Reads back the newest twitter summary file from Documents/twitter_summaries
and reports status without making network calls or running model inference.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

NAME = "ztools-twitter"


def unknown(summary: str) -> dict:
    return {"name": NAME, "state": "unknown", "summary": summary}


def default_output_dir() -> Path:
    return Path.home() / "Documents" / "twitter_summaries"


def newest_summary(directory: Path) -> Path | None:
    summaries = sorted(
        directory.glob("tw_*.md"),
        key=lambda p: p.stat().st_mtime,
    )
    return summaries[-1] if summaries else None


def build_status() -> dict:
    env_path = os.environ.get("TWITTER_OUTPUT_DIR")
    directory = Path(os.path.expanduser(env_path)) if env_path else default_output_dir()
    if not directory.is_dir():
        return unknown(f"no summary directory at {directory}")

    summary_file = newest_summary(directory)
    if summary_file is None:
        return unknown(f"no twitter summary found in {directory}")

    mtime = summary_file.stat().st_mtime
    ran_at = datetime.fromtimestamp(mtime).astimezone().isoformat()

    text = summary_file.read_text(encoding="utf-8")
    lines = [line.strip() for line in text.splitlines() if line.strip()]

    header_summary = ""
    for line in lines:
        if line.startswith("**Tweets:**"):
            header_summary = line.replace("**Tweets:**", "").strip()
            break

    if header_summary:
        summary_text = f"latest summary {summary_file.stem}: {header_summary}"
    else:
        summary_text = f"latest summary {summary_file.name}"

    return {
        "name": NAME,
        "state": "ok",
        "summary": summary_text,
        "ran_at": ran_at,
    }


def main() -> None:
    status = build_status()
    print(json.dumps(status))


if __name__ == "__main__":
    main()
