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
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parent.parent

SKIP_DIRS = frozenset(
    {
        ".git",
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

CONFIG_FILE_NAMES = frozenset(
    {
        "config.py",
        "config_core.py",
        "config_getters.py",
        "config_tasks.py",
        "constants.py",
        "osaurus_models.py",
    }
)

Violation = Tuple[int, str, str]


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


def _is_test_file(path: Path) -> bool:
    return path.parent.name == "tests" or path.name.startswith("test_")


def _is_conf_file(path: Path) -> bool:
    return path.parent.name == "conf"


_CONFIG_FALLBACK_RE = re.compile(
    r'\.get\s*\(\s*["\'][^"\']*["\']\s*,\s*["\']http://localhost:1337["\']'
)


def _has_config_fallback(line: str) -> bool:
    """Detect '.get("key", "http://localhost:1337")' pattern (NOT requests.get)."""
    return bool(_CONFIG_FALLBACK_RE.search(line))


def _is_constant_localhost_def(line: str) -> bool:
    """Detect 'SOME_CONSTANT = "http://localhost:1337"' pattern (module-level constant)."""
    stripped = line.strip()
    if "=" not in stripped:
        return False
    if '"http://localhost:1337"' not in stripped and "'http://localhost:1337'" not in stripped:
        return False
    before_eq = stripped.split("=", 1)[0].strip()
    return bool(re.match(r"^_?[A-Z][A-Z0-9_]*$", before_eq))


def _is_env_fallback(line: str) -> bool:
    return "os.environ.get(" in line or "os.getenv(" in line


def _is_model_config_expr(line: str) -> bool:
    return any(
        kw in line
        for kw in (
            "os.environ.get(",
            "os.getenv(",
            "get_best_model",
            "select_best_model",
            "find_best_mlx_model",
            ".get(",
        )
    )


# ---------------------------------------------------------------------------
# Check 1: Machine-specific absolute paths
# ---------------------------------------------------------------------------


def check_absolute_paths(path: Path) -> List[Violation]:
    if path.suffix == ".md":
        return []
    text = path.read_text(encoding="utf-8", errors="replace")
    violations: List[Violation] = []
    for i, line in enumerate(text.splitlines(), 1):
        if "/Users/ztomer" in line:
            violations.append(
                (
                    i,
                    "Machine-specific absolute path `/Users/ztomer` — use a "
                    "config key or relative path",
                    "Replace with a path derived from project root (Path(__file__).parent / ...)",
                )
            )
    return violations


# ---------------------------------------------------------------------------
# Check 2: Hardcoded years in code
# ---------------------------------------------------------------------------

_YEAR_RE = re.compile(r"(?<!\w)(2024|2025|2026)(?!\w)")


def check_hardcoded_years(path: Path) -> List[Violation]:
    if path.suffix != ".py":
        return []
    if _is_test_file(path) or _is_conf_file(path):
        return []
    text = path.read_text(encoding="utf-8", errors="replace")
    violations: List[Violation] = []
    for i, line in enumerate(text.splitlines(), 1):
        line_s = line.strip()
        if not line_s or line_s.startswith("#"):
            continue
        if "# check-ok:" in line:
            continue
        if "%Y" in line or "strftime" in line:
            continue
        if re.search(r"datetime\s*\(\s*\d{4}", line):
            continue
        m = _YEAR_RE.search(line)
        if m:
            violations.append(
                (
                    i,
                    f"Hardcoded year `{m.group(1)}` — should come from "
                    f"config or computed dynamically",
                    "Use datetime.date.today().year or a config key",
                )
            )
    return violations


# ---------------------------------------------------------------------------
# Check 3: Hardcoded localhost:1337
# ---------------------------------------------------------------------------

_LOCALHOST_RE = re.compile(r"http://localhost:1337")


def check_localhost_1337(path: Path) -> List[Violation]:
    if path.suffix != ".py":
        return []
    if _is_test_file(path):
        return []
    text = path.read_text(encoding="utf-8", errors="replace")
    violations: List[Violation] = []
    for i, line in enumerate(text.splitlines(), 1):
        if not _LOCALHOST_RE.search(line):
            continue
        if _has_config_fallback(line):
            continue
        if _is_constant_localhost_def(line):
            continue
        if _is_env_fallback(line):
            continue
        violations.append(
            (
                i,
                "Hardcoded `http://localhost:1337` — should come from config or a constant",
                'Replace with config lookup: `_CFG.get("llm_url", "http://localhost:1337")`',
            )
        )
    return violations


# ---------------------------------------------------------------------------
# Check 4: Hardcoded user home paths
# ---------------------------------------------------------------------------

_HOME_PATH_PATTERNS = [
    ("~/Documents", "Hardcoded `~/Documents` path"),
    ("~/MLXModels", "Hardcoded `~/MLXModels` path"),
    ("~/Library", "Hardcoded `~/Library` path"),
]
_HOME_DOT_RE = re.compile(r"""['"]~/\.\S*['"]""")


def check_home_paths(path: Path) -> List[Violation]:
    if path.suffix != ".py":
        return []
    if _is_test_file(path):
        return []
    text = path.read_text(encoding="utf-8", errors="replace")
    violations: List[Violation] = []
    for i, line in enumerate(text.splitlines(), 1):
        line_s = line.strip()
        if line_s.startswith("#"):
            continue
        for pat, msg in _HOME_PATH_PATTERNS:
            if pat in line_s:
                violations.append(
                    (
                        i,
                        msg,
                        'Use Path.home() / "subdir" or a config key',
                    )
                )
        if _HOME_DOT_RE.search(line):
            violations.append(
                (
                    i,
                    'Hardcoded `~/.` path — should use Path.home() / ".file"',
                    'Use Path.home() / ".filename" or a config key',
                )
            )
    return violations


# ---------------------------------------------------------------------------
# Check 5: Hardcoded model names
# ---------------------------------------------------------------------------

CONFIG_MODEL_NAMES = frozenset(
    {
        "foundation",
        "qwopus",
        "qwen",
        "gemma",
        "nemotron",
        "laguna",
    }
)
_MODEL_VERSION_RE = re.compile(r"""['"]qwen3\.6-""")


def check_hardcoded_models(path: Path) -> List[Violation]:
    if path.suffix != ".py":
        return []
    if _is_test_file(path) or _is_conf_file(path):
        return []
    try:
        rel_parent = str(path.relative_to(ROOT).parent)
    except ValueError:
        rel_parent = ""
    if path.name in CONFIG_FILE_NAMES and rel_parent in ("lib", "lib/llm"):
        return []
    text = path.read_text(encoding="utf-8", errors="replace")
    violations: List[Violation] = []
    for i, line in enumerate(text.splitlines(), 1):
        line_s = line.strip()
        if not line_s or line_s.startswith(("#", "//")):
            continue
        if "# check-ok:" in line_s:
            continue

        if _is_model_config_expr(line_s):
            continue

        if " in model" in line_s or "model." in line_s or "model_lower" in line_s:
            continue

        if 'default="' in line_s or "default='" in line_s:
            continue
        if 'or ["' in line_s or "or ['" in line_s:
            continue
        if 'else "' in line_s or "else '" in line_s:
            continue

        for model_name in CONFIG_MODEL_NAMES:
            if f'"{model_name}"' not in line_s and f"'{model_name}'" not in line_s:
                continue
            if "==" in line_s or "!=" in line_s:
                continue
            if "_FALLBACK_MODEL" in line_s or "DEFAULT_MODEL" in line_s:
                continue
            violations.append(
                (
                    i,
                    f"Hardcoded model name `{model_name}` — should come from config (best_models)",
                    'Use get_best_model(task) or os.environ.get("OLLAMA_MODEL", ...)',
                )
            )

        if _MODEL_VERSION_RE.search(line_s):
            violations.append(
                (
                    i,
                    "Hardcoded model version `qwen3.6-...` — should come from config",
                    'Use get_best_model(task) or os.environ.get("OLLAMA_MODEL", ...)',
                )
            )

    return violations


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
        scan_files = [Path(f).resolve() for f in args.files if Path(f).suffix in SOURCE_EXTS]
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
