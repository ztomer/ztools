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


def output_dir() -> Path:
    """The summary directory, honouring the $TWITTER_OUTPUT_DIR override.

    The single source the status page AND the dashboard tab both resolve, so
    a test seam (`TWITTER_OUTPUT_DIR`) redirects both at once and they can
    never disagree about the directory.
    """
    env_path = os.environ.get("TWITTER_OUTPUT_DIR")
    return Path(os.path.expanduser(env_path)) if env_path else default_output_dir()


def newest_summary(directory: Path) -> Path | None:
    """The most recent summary by mtime, whatever the naming generation.

    The summary writer has used two names: `{since}_to_{until}.md` (the python
    writer) and `{when}_HHMM_summary.md` (the current generation). The old
    discovery globbed `tw_*.md`, which surprisingly matched neither — so the
    status answered "no summary found" on a directory full of them. Discovery
    sorts every `*.md` by mtime; taking the newest is resilient to the next
    naming change instead of pinning today's.
    """
    summaries = sorted(
        directory.glob("*.md"),
        key=lambda p: p.stat().st_mtime,
    )
    return summaries[-1] if summaries else None


def build_status() -> dict:
    directory = output_dir()
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
