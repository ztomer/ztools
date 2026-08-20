"""Loader for `conf/prompts.toml` — the single home for shared prompt texts.

The Rust binary embeds a fallback copy of each prompt so the static binary works
with no checkout; the drift-gate test in `rust/src/config.rs` fails if that
fallback ever diverges from this file. Python reads this file directly, so a
prompt edit lands on both sides by construction — the parallel-copy failure this
module exists to prevent.

A missing file is a loud error, not a silent built-in fallback: prompts are
load-bearing, and a quiet fallback would resurrect the very copy-drift this
module removes.
"""

import tomllib
from typing import Any, Dict

from .paths import conf_path


def load_prompts_conf() -> Dict[str, Any]:
    path = conf_path("prompts.toml")
    try:
        with open(path, "rb") as fh:
            return tomllib.load(fh)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"shared prompts file missing at {path} — prompts are the single "
            "source for both Rust and Python and cannot be silently replaced"
        ) from exc


def load_prompt(section: str, key: str) -> str:
    """Fetch one prompt text, e.g. `load_prompt("twitter", "summarize")["instructions"]`."""
    value = load_prompts_conf()[section][key]
    if isinstance(value, dict):
        return value.get("instructions", "")
    return value
