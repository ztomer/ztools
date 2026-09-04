#!/usr/bin/env python3
"""
Structural gate: prohibit `#[allow]` attributes in Rust source.
This gate ensures no `#[allow]` attributes remain in the Rust codebase,
enforcing the house rule that all unsafe/partial work must be explicitly
documented rather than silently suppressed.

Usage (pre-commit / pre-push):
    python3 tools/check_no_allow.py

Exits with status 0 if no `#[allow]` attributes found; 1 otherwise.
"""

import re
import sys
from pathlib import Path


def main() -> int:
    repo_root = Path(__file__).resolve().parent.parent
    rs_files = list(repo_root.rglob("*.rs"))

    allowed_paths = {repo_root / "target", repo_root / ".git", repo_root / "vendor"}
    # Exclude cargo cache, target, and vendor dirs from search
    rs_files = [f for f in rs_files if not any(f.is_relative_to(p) for p in allowed_paths)]

    pattern = re.compile(r"#\[allow\([^\]]*\)\]")
    found = False

    for path in rs_files:
        try:
            content = path.read_text(encoding="utf-8")
        except Exception:
            continue
        matches = pattern.findall(content)
        if matches:
            for m in matches:
                print(f"{path.relative_to(repo_root)}: {m}")
            found = True

    if found:
        print(
            "ERROR: Found `#[allow(...)]` attributes in Rust source. "
            "Structural gate failed: all #[allow] must be justified or removed."
        )
        return 1

    print("OK: No `#[allow]` attributes found in Rust source.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
