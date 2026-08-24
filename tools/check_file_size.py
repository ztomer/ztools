#!/usr/bin/env python3
"""File size gate: no file may exceed 500 lines. No exemptions.

One implementation for every language in the repo, called by .githooks/pre-commit
so the rule cannot be enforced in Python and quietly skipped in Rust. It was
Python-only until 2026-08-23 -- the hook filtered `f.endswith(".py")`, so `rust/`
inherited none of the cap while the suite still reported green, and
json_validator.rs reached 1126 lines beside a 485-line Python file the same gate
was about to block.

There was a test exemption (by directory in Python, by filename/inline
`#[cfg(test)]` in Rust). It is gone as of 2026-08-24: a test file over the limit
is exactly as unmaintainable to read and diff as a production file over the
limit, and an exemption is one more place the rule can silently narrow when a
third language or a new test layout doesn't match the carve-out's shape. Every
tracked `.py` and `.rs` file is counted by raw line count -- full stop.
"""

import sys
from pathlib import Path

MAX_LINES = 500


def line_count(text: str) -> int:
    return text.count("\n") + (0 if text.endswith("\n") or not text else 1)


def check(paths: list[str], root: Path) -> list[str]:
    errors = []
    for rel in paths:
        if not (rel.endswith(".py") or rel.endswith(".rs")):
            continue
        fp = root / rel
        if not fp.exists():
            continue
        count = line_count(fp.read_text(encoding="utf-8", errors="replace"))
        if count > MAX_LINES:
            errors.append(f"  File exceeds {MAX_LINES} lines: {rel} ({count} lines)")
    return errors


def main(argv: list[str]) -> int:
    root = Path.cwd()
    errors = check(argv[1:], root)
    for e in errors:
        print(e)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
