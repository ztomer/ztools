#!/usr/bin/env python3
"""Print the newest weekend plan as markdown for the `routines` dashboard tab.

READ-ONLY AND CHEAP. Reads back one file and prints it; no model, no network,
no writes. The tab twin of `routines_status.py` — that script answers "how did
the last run go", this one shows the plan itself.

The discovery is IMPORTED from `routines_status`, not re-derived, so the tab
can never point at a newer file than the page claims is current: a naming or
directory drift breaks both together instead of one of them silently.

Output is the raw markdown; the dashboard's shared renderer styles it.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import routines_status  # noqa: E402


def main() -> None:
    directory = Path(os.path.expanduser(routines_status.OUTPUT_DIR_PATH))
    plan = routines_status.newest_plan(directory) if directory.is_dir() else None
    if plan is None:
        print(f"No weekend plan has been written to {directory}", file=sys.stderr)
        sys.exit(1)
    sys.stdout.write(plan.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()