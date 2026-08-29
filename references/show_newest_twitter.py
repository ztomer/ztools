#!/usr/bin/env python3
"""Print the newest Twitter summary as markdown for the `routines` dashboard tab.

READ-ONLY AND CHEAP. Reads back one file; no network, no model, no writes.
The tab twin of `routines_twitter_status.py`: its discovery is imported here,
so the tab and the status line can never disagree about which file is current.

Output is the raw markdown; the dashboard's shared renderer styles it.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import routines_twitter_status  # noqa: E402


def main() -> None:
    directory = routines_twitter_status.output_dir()
    summary = (
        routines_twitter_status.newest_summary(directory) if directory.is_dir() else None
    )
    if summary is None:
        print(f"No twitter summary has been written to {directory}", file=sys.stderr)
        sys.exit(1)
    sys.stdout.write(summary.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()