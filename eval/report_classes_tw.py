"""Content-class checks for `tw` reports, and checks over the repo itself.

Split from eval/report_classes.py to stay under the repo's 500-line cap.
report_classes re-exports every name here, so callers import from one place.

Two groups live together because neither reads a `wk` report:
- `check_tw_*` read a saved twitter summary
- `check_model_prompts_render` / `check_declared_config_keys_are_read` read the
  SOURCE and CONFIG rather than any artifact
"""

from __future__ import annotations

import re
from datetime import date
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parent.parent

_MONTHS = (
    "january february march april may june july august september october "
    "november december"
).split()

# A bare wall-clock time with no day qualifier anywhere near it (C2a).
_BARE_TIME = re.compile(r"(?<![\d:])([01]?\d|2[0-3]):([0-5]\d)(?![\d:])")
_DAY_QUALIFIER = re.compile(
    r"(\b(" + "|".join(m[:3] for m in _MONTHS) + r")\w*\b"
    r"|\b(mon|tue|wed|thu|fri|sat|sun)(day|s|nes|rs|ur)?\w*\b"
    r"|\b\d{4}-\d{2}-\d{2}\b"
    r"|\byesterday\b|\btoday\b)",
    re.IGNORECASE,
)



# --------------------------------------------------------------------------
# C2a / C10 — twitter report checks
# --------------------------------------------------------------------------


def check_tw_timestamps_are_day_qualified(text: str) -> list[str]:
    """C2a. Over a multi-day window a bare `HH:MM` cannot be resolved to a day."""
    from eval.report_classes import parse_window_from_tw_report

    window = parse_window_from_tw_report(text)
    if window and window[0].date() == window[1].date():
        return []  # single-day report: a bare time is unambiguous
    failures = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.startswith(("-", "*")):
            continue
        if not _BARE_TIME.search(stripped):
            continue
        if _DAY_QUALIFIER.search(stripped):
            continue
        failures.append(f"bullet has an unqualified time: {stripped[:90]}")
    return failures


_ATTRIB_BRACKET = re.compile(r"\(@\w+\s*\|\s*\d{1,2}:\d{2}\)")
_ATTRIB_PROSE = re.compile(r"\bat \d{1,2}:\d{2}\b")


def attribution_styles(text: str) -> set[str]:
    styles = set()
    if _ATTRIB_BRACKET.search(text):
        styles.add("bracket")
    if _ATTRIB_PROSE.search(text):
        styles.add("prose")
    return styles


def check_tw_attribution_format_is_uniform(texts: Iterable[str]) -> list[str]:
    """C10. One attribution format, and the same one across reports."""
    seen: set[str] = set()
    failures = []
    for i, text in enumerate(texts):
        styles = attribution_styles(text)
        if len(styles) > 1:
            failures.append(f"report {i} mixes attribution styles: {sorted(styles)}")
        seen |= styles
    if len(seen) > 1:
        failures.append(f"attribution style is not stable across reports: {sorted(seen)}")
    return failures


def check_tw_names_its_backend(text: str) -> list[str]:
    """C9. A degraded backend must be distinguishable from the primary.

    Satisfied by either the quiet `**Model:**` line of a normal run or the
    `DEGRADED OUTPUT` block of a fallback run -- both name the backend. A report
    with neither cannot be attributed after the fact, which is the defect.
    """
    if re.search(r"(?im)^\*\*Model:\*\*\s*\S", text):
        return []
    if re.search(r"(?im)^>\s*.*DEGRADED OUTPUT", text) and re.search(
        r"(?im)^>\s*\*\*Backend:\*\*\s*\S", text
    ):
        return []
    return ["report does not record which model/backend produced it"]




# --------------------------------------------------------------------------
# C1 / C6 / C12 / C13 — source and config checks
# --------------------------------------------------------------------------

# The exact keyword set weekend/prompts.py passes at its two production call
# sites. Kept here so the check fails if production starts passing something else.
# Values are arbitrary; only the KEYS matter to the render check, and the year is
# derived so this file never pins a calendar year.
PRODUCTION_PROMPT_KWARGS = {
    "location": "Vaughan/Toronto",
    "age_range": "6-13",
    "date_range": f"July 31 to August 02, {date.today().year}",
    "exclusions": "Toronto Zoo, LEGOLAND",
}

WEEKEND_PROMPT_TASKS = ("weekend_fixed", "weekend_transient")


def check_model_prompts_render(models_dir: Path | None = None) -> list[str]:
    """C1. Every weekend prompt must render through PRODUCTION's renderer.

    Deliberately calls `weekend.prompts._render_model_prompt` -- the exact
    function the pipeline uses -- rather than re-implementing substitution here.
    A check that renders prompts its own way is the C12 mistake repeated.
    """
    import tomllib

    from lib.prompt_render import PromptRenderError, unrendered_placeholders
    from weekend.prompts import _render_model_prompt

    models_dir = models_dir or (ROOT / "conf" / "models")
    failures = []
    for path in sorted(models_dir.glob("*.toml")):
        cfg = tomllib.loads(path.read_text())
        for task in WEEKEND_PROMPT_TASKS:
            template = cfg.get("prompts", {}).get(task)
            if template is None:
                continue
            try:
                rendered = _render_model_prompt(
                    template,
                    f"{path.name}:{task}",
                    "SOURCE-CONTEXT",
                    **PRODUCTION_PROMPT_KWARGS,
                )
            except PromptRenderError as exc:
                failures.append(str(exc))
                continue
            left = unrendered_placeholders(rendered)
            if left:
                failures.append(f"{path.name}:{task} left placeholders unrendered: {left}")
    return failures


def check_declared_config_keys_are_read(
    config_path: Path | None = None, search_dirs: Iterable[Path] | None = None
) -> list[str]:
    """C13. A declared config key with no reader silently misdescribes the tool."""
    import tomllib

    config_path = config_path or (ROOT / "conf" / "twitter.toml")
    search_dirs = list(search_dirs or [ROOT / "twitter", ROOT / "lib", ROOT / "tui"])
    sources = "\n".join(
        p.read_text(errors="ignore")
        for d in search_dirs
        if d.is_dir()
        for p in d.rglob("*.py")
    )
    failures = []
    for key in tomllib.loads(config_path.read_text()):
        if f'"{key}"' not in sources and f"'{key}'" not in sources:
            failures.append(f"{config_path.name}: key {key!r} is declared but never read")
    return failures


