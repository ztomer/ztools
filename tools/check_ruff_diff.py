#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["ruff"]
# ///

"""
check_ruff_diff.py — run ruff only on *added* lines of changed Python files.

This mirrors the config-debt gate philosophy: legacy style debt in untouched
code is not re-litigated, but every new line a commit introduces must be ruff
clean (zero warnings, zero errors). Used by the Quality Gate CI workflow.

Exit 1 if any added line has a ruff violation, 0 if clean.
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def get_changed_files(base: str, head: str = "HEAD") -> list[str]:
    # Diff base against the working tree + index (not just committed HEAD), so
    # uncommitted local edits are also gated in development.
    r = subprocess.run(
        ["git", "diff", "--name-only", base, "--", "*.py"],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    return [f for f in r.stdout.strip().split("\n") if f]


def get_added_lines(file_path: str, base: str, head: str = "HEAD") -> set[int]:
    r = subprocess.run(
        ["git", "diff", "-U0", base, "--", file_path],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    added: set[int] = set()
    for line in r.stdout.split("\n"):
        m = re.match(r"^\+(\d+)(?:,(\d+))? @@", line)
        if not m:
            continue
        start = int(m.group(1))
        count = int(m.group(2)) if m.group(2) else 1
        added.update(range(start, start + count))
    return added


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="HEAD~1")
    ap.add_argument("--head", default="HEAD")
    args = ap.parse_args()

    changed = get_changed_files(args.base, args.head)
    if not changed:
        print("No changed Python files — clean.")
        return 0

    errors: list[str] = []
    for f in changed:
        fp = ROOT / f
        if not fp.exists() or fp.suffix != ".py":
            continue
        added = get_added_lines(f, args.base, args.head)
        if not added:
            continue
        r = subprocess.run(
            ["ruff", "check", "--output-format=concise", str(fp)],
            capture_output=True,
            text=True,
            cwd=ROOT,
        )
        for line in r.stdout.strip().split("\n"):
            m = re.match(r"^(.+?):(\d+):\d+: ", line)
            if not m:
                continue
            lineno = int(m.group(2))
            if lineno in added:
                errors.append(f"{f}:{lineno}: {line.split(':', 2)[-1].strip()}")

    if errors:
        print("Ruff violations on added lines (gate fails):\n")
        print("\n".join(errors))
        return 1
    print("Ruff clean on added lines.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
