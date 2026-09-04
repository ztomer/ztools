#!/usr/bin/env python3
"""Individual config-debt checks and their line-level predicates.

Split out of check_config_debt.py so both files stay under the repo's 500-line
limit. Each check takes a path and returns (line, message, fix) violations;
check_config_debt.py owns file discovery, reporting and exit codes.
"""

import re
from pathlib import Path
from typing import List, Tuple

ROOT = Path(__file__).resolve().parent.parent

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


def _is_test_file(path: Path) -> bool:
    """True for a file that IS test code, by name or by living under a tests/ tree.

    Checks every ancestor, not just the immediate parent: `references/tests/
    conftest_fixtures/legacy.py` is test-support code one level below
    `references/tests/`, and a parent-only check exempted `conftest.py` while
    missing the fixtures split out of it under a subdirectory -- flagging
    verbatim moved fixture data (a hardcoded sample year, a mock model name) as
    new config debt the moment it was split into its own file.
    """
    return "tests" in path.parts[:-1] or path.name.startswith("test_")


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
                    "Replace with a lib.paths helper (conf_path/eval_tasks_path/repo_path)",
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
# Check 6: Shipped-resource paths derived from __file__
# ---------------------------------------------------------------------------

# `conf/`, `docs/` and `eval_tasks/` sit next to `lib/` in an installed wheel but
# one level above it in the checkout (Python lives under `references/`). Any
# path derived by walking up from __file__ is therefore right in exactly one
# layout and silently wrong in the other.
_LAYOUT_PATH_RE = re.compile(r'parent\s*(?:\.\w+\s*)*/\s*"(conf|docs|eval_tasks)"')


# A cwd-relative path into a package directory. `open("eval/benchmark_quality.py")`
# resolved from the repo root until the Python moved under references/, then
# silently stopped — four tests were red for months because of exactly this.
# Unlike the __file__ rule, this one applies to tests too: that is where it bit.
_CWD_RELATIVE_RE = re.compile(
    r"""(?:open|Path)\(\s*["'](lib|eval|weekend|twitter|rename|tui|tests)/"""
)


def check_layout_paths(path: Path) -> List[Violation]:
    if path.suffix != ".py":
        return []
    # lib/paths.py owns the layout knowledge, and test_paths.py is its test: it
    # must name the candidate paths to pin them. Nothing else may.
    if path.name in ("paths.py", "test_paths.py"):
        return []
    violations: List[Violation] = []
    # Tests are NOT exempt from either rule: this class reached four C8 tests via
    # `rc.ROOT / "conf" / "weekend.toml"` and four more via a cwd-relative open,
    # and all eight sat red for months. The patterns only match the three shipped
    # resource names, so ordinary `Path(__file__).parent / "fixtures"` is unaffected.
    for i, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
        m = _CWD_RELATIVE_RE.search(line)
        if m:
            violations.append(
                (
                    i,
                    f"cwd-relative path into `{m.group(1)}/` — resolves only when "
                    "run from the repo root, and not at all since the move",
                    "Locate it through the import system (inspect.getfile) or lib.paths",
                )
            )
        m = _LAYOUT_PATH_RE.search(line)
        if m:
            resource = m.group(1)
            violations.append(
                (
                    i,
                    f"`{resource}/` path derived from __file__ — correct in one "
                    "layout only (installed wheel vs source checkout)",
                    "Use lib.paths: conf_path(...) / eval_tasks_path(...) / repo_path(...)",
                )
            )
    return violations


