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


_ATTRIB_BRACKET = re.compile(r"\(@\w+\s*\|\s*(?:[A-Za-z]+\s+\d{1,2}\s+)?\d{1,2}:\d{2}\)")
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




def check_wk_no_column_echoes_config(text: str, configured: dict[str, str]) -> list[str]:
    """C4, config-echo form: a column constant across every row AND equal to a
    configured value is the config leaking into the output, not data.

    This is the SHAPE of the whole C4 class, and it is the assertion that was
    missing each time it recurred:
      - "2-3 hours" in every Duration cell    (a prompt-mandated constant)
      - "$18-35 per child or free" everywhere (a prompt-mandated constant)
      - the configured FAMILY age range in every Target Age(s) cell, a column
        that means "the ages this venue is for"

    Each was caught by eye, one at a time, after shipping. A column whose every
    value equals something the program already knew carries no information about
    the world, whatever the source of that value.

    `configured` maps a column-name substring to the value it must not equal,
    e.g. {"target age": "6-13"}.
    """
    from eval.report_classes import MISSING_SENTINEL, fixed_rows, transient_rows

    failures = []
    for label, rows in (("transient", transient_rows(text)), ("fixed", fixed_rows(text))):
        if len(rows) < 2:
            continue
        for col in rows[0]:
            low = col.lower()
            expected = next((v for k, v in configured.items() if k in low), None)
            if not expected:
                continue
            values = {r.get(col, "").strip() for r in rows}
            if values <= {"", MISSING_SENTINEL}:
                continue
            if len(values) == 1 and values.pop().strip() == str(expected).strip():
                failures.append(
                    f"{label}: column {col!r} is {expected!r} on all {len(rows)} rows, "
                    f"which is the CONFIGURED value -- the config is echoing into the "
                    f"output rather than the source being read"
                )
    return failures


def configured_echo_values() -> dict[str, str]:
    """The configured values a column must not simply repeat."""
    from weekend.config import AGE_RANGE

    return {"target age": AGE_RANGE, "age(s)": AGE_RANGE}


def check_wk_plan_is_not_hollow(text: str, min_transient: int = 1, min_fixed: int = 3) -> list[str]:
    """C18 HOLLOW-PLAN-PASSES-EVERY-CHECK.

    An empty table satisfies every content rule trivially: no fabricated
    constant, no stale date, no excluded venue, no mislabelled weather. A run
    that produced ZERO transient events was therefore reported as "every content
    check passes" -- the checks were green and the feature was useless.

    Traced with counts, the collapse was invisible without them:
      40 raw results -> 29 in-region -> 3 pages followed -> 20 candidates
      -> 10 draft -> 8 refined -> 7 structured -> 6 after exclusions -> 0 after
      the window filter
    The model had stamped one wrong date on every event, and the window filter
    correctly dropped all of them. Every stage "worked"; the output was empty.

    So emptiness itself has to be a failure. This check is the floor the other
    checks stand on: they say the rows are honest, this one says there are rows.
    """
    from eval.report_classes import fixed_rows, transient_rows

    failures = []
    n_transient, n_fixed = len(transient_rows(text)), len(fixed_rows(text))
    if n_transient < min_transient:
        failures.append(
            f"only {n_transient} transient event(s); a weekend plan that cannot name "
            f"anything happening is hollow, however clean its other columns are"
        )
    if n_fixed < min_fixed:
        failures.append(f"only {n_fixed} fixed activit(ies); expected at least {min_fixed}")
    return failures


# A name that is only a CATEGORY names no particular place. A real listing is
# "Sky Zone Trampoline Park" or "Playdium Vaughan"; a fabricated one is
# "Trampoline Park" at "Jump City", or "Zoo Adventure" at "Central Zoo".
_GENERIC_NAME_WORDS = frozenset(
    """indoor outdoor play centre center park zoo museum adventure day fun kids
    family sports complex zone city trip event events activity activities
    playground aquarium arcade splash summer winter central local main community
    town big little world land house place spot venue jump trampoline""".split()
)


def _is_generic(text: str) -> bool:
    """True when every significant word is a bare category word."""
    from weekend.enforce import _significant_tokens

    tokens = _significant_tokens(text)
    return bool(tokens) and tokens <= _GENERIC_NAME_WORDS


def check_wk_no_fabricated_rows(text: str) -> list[str]:
    """C19 FABRICATED-ROW. A row that names no particular place is invented.

    Written from a real artifact (tests/fixtures/reports/wk_fabricated_transient_sample.md)
    produced when the draft phase was starved of in-window candidates: rather
    than return fewer events the model invented five, with generic names and
    invented prices.

    An invented row satisfies every OTHER check trivially, because it corresponds
    to nothing: no excluded venue matches it, no stale date contradicts it, and
    "Central Park" is not even caught by the in-region check. Of ten checks, one
    caught it. Emptiness is honest; this is actively misleading -- a user could
    plan a Saturday around an event that does not exist.

    The signal is specificity. A real listing carries a proper noun; a fabricated
    one is a category ("Trampoline Park" at "Jump City"). Deliberately narrow: a
    row is only reported when BOTH its name and its location are generic, so a
    real venue with a plain name survives on the strength of its location.
    """
    import re

    from eval.report_classes import _cell, _row_name, fixed_rows, transient_rows

    failures = []
    for label, rows in (("transient", transient_rows(text)), ("fixed", fixed_rows(text))):
        for row in rows:
            cell = _row_name(row) or _cell(row, "location")
            if not cell:
                continue
            # The column is "Event & Location", rendered "**Name** (Location)".
            m = re.match(r"\s*\*\*(.+?)\*\*\s*(?:\((.*)\))?\s*$", cell)
            name = (m.group(1) if m else cell).strip()
            loc = ((m.group(2) if m and m.group(2) else "") or "").strip()
            if _is_generic(name) and (not loc or _is_generic(loc)):
                where = loc or "(no location)"
                failures.append(
                    f"{label} {name!r} at {where!r}: names no particular place -- "
                    f"a category, not a venue. Likely invented."
                )
    return failures
