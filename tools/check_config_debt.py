#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# ///

"""
check_config_debt.py — scan ztools codebase for hardcoded values
that should come from config (CI gate / pre-commit hook).

Exit 1 if violations found, 0 if clean.
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Same-directory module (this file is always run as tools/check_config_debt.py,
# so tools/ is sys.path[0]).
from config_debt_checks import (
    _has_config_fallback,
    _is_constant_localhost_def,
    _is_env_fallback,
    _is_test_file,
    check_absolute_paths,
    check_hardcoded_models,
    check_hardcoded_years,
    check_home_paths,
    check_layout_paths,
    check_localhost_1337,
)

ROOT = Path(__file__).resolve().parent.parent

SKIP_DIRS = frozenset(
    {
        ".git",
        # Session worktrees: gitignored copies of the tree, often on an older
        # layout. Scanning them reports debt that exists in no tracked file.
        ".claude",
        "__pycache__",
        ".venv",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        "otools.egg-info",
        "benchmarks",
        "tools",
        "docs",
        "eval_outputs",
        "build",
    }
)

SKIP_FILES = frozenset(
    {
        "eval_results.json",
        "coverage.json",
    }
)

SOURCE_EXTS = frozenset({".py", ".json", ".sh", ".md"})

def _skip_path(path: Path) -> bool:
    try:
        rel = path.relative_to(ROOT)
    except ValueError:
        return True
    if not rel.parts:
        return True
    if path.name in SKIP_FILES:
        return True
    for part in rel.parts[:-1]:
        if part in SKIP_DIRS:
            return True
    return False


# ---------------------------------------------------------------------------
# Fix mode helpers
# ---------------------------------------------------------------------------


def _detect_config_pattern(lines: List[str]) -> str:
    """Detect which config pattern the file uses."""
    text = "\n".join(lines)
    has_os = "import os" in text or "from os import" in text
    if any("_RENAME_CFG" in line for line in lines):
        return "rename_cfg"
    if any("_TWITTER_CFG" in line for line in lines):
        return "twitter_cfg"
    if any("_CFG" in line for line in lines):
        return "generic_cfg"
    if has_os and ("os.environ" in text or "os.getenv" in text):
        return "environ"
    if has_os:
        return "environ"
    return "unknown"


def _make_fix_replacement(config_pattern: str, value: str) -> str:
    if config_pattern == "rename_cfg":
        return f'_RENAME_CFG.get("llm_url", {value})'
    elif config_pattern == "twitter_cfg":
        return f'_TWITTER_CFG.get("llm_url", {value})'
    elif config_pattern == "generic_cfg":
        return f'_CFG.get("llm_url", {value})'
    elif config_pattern == "environ":
        return f'os.environ.get("OLLAMA_BASE_URL", {value})'
    else:
        return value


def fix_file_localhost(path: Path, dry_run: bool = True) -> Tuple[bool, int]:
    """Fix bare 'http://localhost:1337' strings (entire string literals, not path-suffixed)."""
    if path.suffix != ".py" or _is_test_file(path) or _skip_path(path):
        return False, 0
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()

    config_pattern = _detect_config_pattern(lines)
    modified = False
    fix_count = 0
    new_lines: List[str] = []

    for line in lines:
        if '"http://localhost:1337"' not in line and "'http://localhost:1337'" not in line:
            new_lines.append(line)
            continue
        if _has_config_fallback(line):
            new_lines.append(line)
            continue
        if _is_constant_localhost_def(line):
            new_lines.append(line)
            continue
        if _is_env_fallback(line):
            new_lines.append(line)
            continue

        old = '"http://localhost:1337"'
        # Only fix if the bare URL is an entire string literal (no path suffix)
        if old not in line:
            new_lines.append(line)
            continue

        new = _make_fix_replacement(config_pattern, old)
        new_line = line.replace(old, new, 1)
        if new_line != line:
            modified = True
            fix_count += 1
        new_lines.append(new_line)

    if modified and not dry_run:
        path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")

    return modified, fix_count


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Scan ztools for hardcoded values that should come from config.",
    )
    parser.add_argument(
        "files",
        nargs="*",
        help="Specific files to scan (default: all source files)",
    )
    parser.add_argument(
        "--fix",
        action="store_true",
        help="Auto-fix simple cases (replace localhost:1337 with config expression)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show extra detail",
    )
    args = parser.parse_args()

    if args.files:
        # SKIP_DIRS is the definition of "not policed" and must hold however a
        # file arrives: the pre-commit hook passes explicit paths, which used to
        # bypass this filter entirely — so this checker's own pattern constants
        # (localhost URLs, model names, home paths) were reported as debt the
        # moment tools/ files were staged.
        scan_files = [
            f
            for f in (Path(x).resolve() for x in args.files)
            if f.suffix in SOURCE_EXTS and not _skip_path(f)
        ]
    else:
        all_files = sorted(ROOT.rglob("*"))
        scan_files = [
            f for f in all_files if f.is_file() and f.suffix in SOURCE_EXTS and not _skip_path(f)
        ]

    if args.verbose:
        print(f"Scanning {len(scan_files)} files...")

    check_funcs = [
        ("Absolute paths", check_absolute_paths),
        ("Hardcoded years", check_hardcoded_years),
        ("Hardcoded localhost:1337", check_localhost_1337),
        ("Hardcoded home paths", check_home_paths),
        ("Hardcoded model names", check_hardcoded_models),
        ("Layout-dependent resource path", check_layout_paths),
    ]

    all_violations: Dict[str, List[Tuple[str, int, str, str]]] = {}

    for label, func in check_funcs:
        for f in scan_files:
            try:
                violations = func(f)
            except Exception as e:
                if args.verbose:
                    print(f"  [SKIP] {f.relative_to(ROOT)}: {e}", file=sys.stderr)
                continue
            for line_num, msg, fix_suggestion in violations:
                rel = str(f.relative_to(ROOT))
                all_violations.setdefault(rel, [])
                all_violations[rel].append((label, line_num, msg, fix_suggestion))

    total = sum(len(v) for v in all_violations.values())
    if not all_violations:
        if args.verbose:
            print("✓ No config debt violations found.")
        sys.exit(0)

    print(f"Found {total} config debt violation(s) across {len(all_violations)} file(s):\n")

    for file_path in sorted(all_violations):
        print(f"  {file_path}")
        for label, line_num, msg, fix_suggestion in all_violations[file_path]:
            print(f"    L{line_num:>4}  [{label}]")
            print(f"            {msg}")
            if fix_suggestion:
                print(f"            → {fix_suggestion}")
        print()

    if args.fix:
        fixed_any = False
        total_fixed = 0
        for f in scan_files:
            _, count = fix_file_localhost(f, dry_run=False)
            if count > 0:
                print(f"  ✓ Fixed {f.relative_to(ROOT)} ({count} occurrence(s))")
                fixed_any = True
                total_fixed += count
        if fixed_any:
            print(f"\nFixed {total_fixed} hardcoded localhost:1337 occurrence(s).")
        else:
            print("No auto-fixable violations found.")
        sys.exit(1)

    sys.exit(1)


if __name__ == "__main__":
    main()
