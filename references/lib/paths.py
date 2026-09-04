"""Filesystem layout resolution for shipped resources.

`conf/` and `eval_tasks/` are top-level packages that ship flat next to `lib/`
in an installed wheel (site-packages/lib, site-packages/conf), but in the source
checkout the Python lives under `references/` while those directories stay at
the repo root. Deriving them as `Path(__file__).parent.parent / "conf"` is only
correct for the installed layout and silently resolves to a nonexistent
`references/conf` in a checkout — where it either crashes (`init_config`) or,
worse, degrades to fallback defaults without the user noticing.

Every consumer resolves through the helpers here so the layout assumption lives
in exactly one place. Resolution order, most authoritative first:

  1. `ZTOOLS_CONF` (conf only) — an explicit instruction, so a bad value raises
     rather than degrading.
  2. The installed package's own location, via the import system. Correct in
     both layouts and immune to sibling directories that merely share a name.
  3. A content-validated candidate scan, as a fallback for exotic layouts.

Step 2 matters: a bare `is_dir()` check would accept a stray empty
`references/conf/` (which unfixed writers used to create) and resurrect the
original crash.
"""

import importlib.util
import os
from pathlib import Path
from typing import Optional

__all__ = [
    "conf_dir",
    "conf_path",
    "eval_tasks_dir",
    "eval_tasks_path",
    "repo_root",
    "repo_path",
]

_LIB_DIR = Path(__file__).resolve().parent

# A directory only counts as the config dir if it actually holds the config.
_CONF_MARKER = "config.toml"

# Ordered by specificity: installed (flat) layout first, then the source
# checkout where `lib/` sits one level deeper than `conf/`.
_CANDIDATES = (
    _LIB_DIR.parent / "conf",  # site-packages/lib -> site-packages/conf
    _LIB_DIR.parent.parent / "conf",  # references/lib -> <repo>/conf
)

# Marker files that identify the repo root when running from a checkout.
_REPO_MARKERS = ("pyproject.toml", ".git")


def _package_dir(name: str) -> Optional[Path]:
    """Locate an installed top-level package directory without importing it."""
    try:
        spec = importlib.util.find_spec(name)
    except (ImportError, ValueError, ModuleNotFoundError):
        return None
    if spec is None or not spec.submodule_search_locations:
        return None
    for location in spec.submodule_search_locations:
        path = Path(location)
        if path.is_dir():
            return path
    return None


def conf_dir() -> Path:
    """Return the directory holding the bundled TOML/JSON config.

    Raises FileNotFoundError if `ZTOOLS_CONF` names a directory that does not
    exist — an explicit override that silently falls back to defaults is the
    exact failure mode this module exists to prevent.
    """
    override = os.environ.get("ZTOOLS_CONF")
    if override:
        path = Path(override)
        if not path.is_dir():
            raise FileNotFoundError(f"ZTOOLS_CONF points to a nonexistent directory: {override}")
        return path

    package = _package_dir("conf")
    if package is not None and (package / _CONF_MARKER).is_file():
        return package

    for candidate in _CANDIDATES:
        if (candidate / _CONF_MARKER).is_file():
            return candidate

    # Nothing resolved: return the installed-layout path so callers can name a
    # plausible location in their error message instead of an empty string.
    return package if package is not None else _CANDIDATES[0]


def conf_path(*parts: str) -> Path:
    """Return a path inside the config directory, e.g. `conf_path("config.toml")`."""
    return conf_dir().joinpath(*parts)


def eval_tasks_dir() -> Path:
    """Return the `eval_tasks/` package directory (rubrics and task data)."""
    package = _package_dir("eval_tasks")
    if package is not None:
        return package
    for base in (_LIB_DIR.parent, _LIB_DIR.parent.parent):
        candidate = base / "eval_tasks"
        if candidate.is_dir():
            return candidate
    return _LIB_DIR.parent / "eval_tasks"


def eval_tasks_path(*parts: str) -> Path:
    """Return a path inside `eval_tasks/`, e.g. `eval_tasks_path("data", "taxes")`."""
    return eval_tasks_dir().joinpath(*parts)


def repo_root() -> Optional[Path]:
    """Return the source checkout root, or None when running from an install.

    Only a checkout has directories that are not packaged into the wheel (most
    notably `docs/`). Callers must handle None by degrading with a stated
    reason rather than writing to an invented path.
    """
    for parent in (_LIB_DIR, *_LIB_DIR.parents):
        if any((parent / marker).exists() for marker in _REPO_MARKERS):
            return parent
    return None


def repo_path(*parts: str) -> Optional[Path]:
    """Return a path inside the checkout root, or None when not in a checkout."""
    root = repo_root()
    return root.joinpath(*parts) if root is not None else None
