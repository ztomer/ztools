"""Content-class checks for real, saved `tw` / `wk` reports.

Stage 0 of G3. Each `check_*` here encodes one weakness CLASS from
`docs/REPORT_WEAKNESS_CLASSES.md` as an executable predicate over a report that
was actually shipped, rather than over an eval fixture.

This module exists because of class C12: the `ev` evaluator scores synthetic
fixtures and never reads a produced artifact, so every class in the catalogue is
invisible to it. These checks read the artifact.

Every function returns `list[str]` — the failures found. An empty list means the
class is absent from that report. Nothing here calls an LLM or the network, so
the checks are deterministic and run in CI.
"""

from __future__ import annotations

import re
from datetime import date, datetime
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parent.parent

# Values the prompts mandate verbatim (class C4). A cell equal to one of these
# was authored by the prompt, not observed in the world.
MANDATED_LITERALS = frozenset(
    {
        "2-3 hours",
        "all day",
        "$20-30 per child or free",
        "$18-35 per child or free",
        "$20-30 per child",
    }
)

# Venue words that settle the indoor/outdoor question without a forecast (C5).
UNAMBIGUOUSLY_INDOOR = (
    "indoor",
    "trampoline park",
    "museum",
    "play centre",
    "play center",
    "playground",
    "library",
    "cinema",
    "aquarium",
)

_MONTHS = (
    "january february march april may june july august september october "
    "november december"
).split()

# A bare wall-clock time with no day qualifier anywhere near it (C2a).
_BARE_TIME = re.compile(r"(?<![\d:])([01]?\d|2[0-3]):([0-5]\d)(?![\d:])")
# Month names match on their 3-letter stem so both "July" and "Jul" qualify.
_DAY_QUALIFIER = re.compile(
    r"(\b(" + "|".join(m[:3] for m in _MONTHS) + r")\w*\b"
    r"|\b(mon|tue|wed|thu|fri|sat|sun)(day|s|nes|rs|ur)?\w*\b"
    r"|\b\d{4}-\d{2}-\d{2}\b"
    r"|\byesterday\b|\btoday\b)",
    re.IGNORECASE,
)


# --------------------------------------------------------------------------
# markdown table parsing
# --------------------------------------------------------------------------


def parse_tables(text: str) -> dict[str, list[dict[str, str]]]:
    """Split a `wk` report into {section heading: [row dicts]}.

    Keys are the raw column headers, so a caller can assert on the header text
    itself (class C6 cares that a column is *called* "Review Score").
    """
    tables: dict[str, list[dict[str, str]]] = {}
    heading = ""
    header: list[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if line.startswith("#"):
            heading = line.lstrip("# ").strip()
            header = []
            continue
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if set("".join(cells)) <= set(": -"):  # the |:---|:---| separator
            continue
        if not header:
            header = cells
            tables.setdefault(heading, [])
            continue
        tables.setdefault(heading, []).append(dict(zip(header, cells)))
    return tables


def transient_rows(text: str) -> list[dict[str, str]]:
    return _rows_matching(text, "transient")


def fixed_rows(text: str) -> list[dict[str, str]]:
    return _rows_matching(text, "fixed")


def _rows_matching(text: str, needle: str) -> list[dict[str, str]]:
    for heading, rows in parse_tables(text).items():
        if needle in heading.lower():
            return rows
    return []


def _cell(row: dict[str, str], *needles: str) -> str:
    for key, val in row.items():
        low = key.lower()
        if any(n in low for n in needles):
            return val
    return ""


def _row_name(row: dict[str, str]) -> str:
    return _cell(row, "activity", "event")


# --------------------------------------------------------------------------
# date helpers
# --------------------------------------------------------------------------


def parse_window_from_wk_filename(path: Path) -> tuple[date, date] | None:
    """`weekend_plan_<Month>_<D>_to_<Month>_<D>_<YYYY>.md` -> (start date, end date)."""
    m = re.search(
        r"([A-Z][a-z]+)_(\d{1,2})_to_([A-Z][a-z]+)_(\d{1,2})_(\d{4})", path.name
    )
    if not m:
        return None
    m1, d1, m2, d2, year = m.groups()
    try:
        start = datetime.strptime(f"{m1} {d1} {year}", "%B %d %Y").date()
        end = datetime.strptime(f"{m2} {d2} {year}", "%B %d %Y").date()
    except ValueError:
        return None
    return start, end


def parse_window_from_tw_report(text: str) -> tuple[datetime, datetime] | None:
    m = re.search(
        r"\*\*Period:\*\*\s*(\d{4}-\d{2}-\d{2} \d{2}:\d{2})\s*\S+\s*"
        r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2})",
        text,
    )
    if not m:
        return None
    fmt = "%Y-%m-%d %H:%M"
    return datetime.strptime(m.group(1), fmt), datetime.strptime(m.group(2), fmt)


def find_dates_in(value: str, year: int) -> list[date]:
    """Pull explicit calendar dates out of a cell. Durations are not dates."""
    found: list[date] = []
    for m in re.finditer(r"(\d{4})-(\d{2})-(\d{2})", value):
        try:
            found.append(date(int(m.group(1)), int(m.group(2)), int(m.group(3))))
        except ValueError:
            pass
    pattern = r"\b(" + "|".join(_MONTHS) + r")\w*\.?\s+(\d{1,2})\b"
    for m in re.finditer(pattern, value, re.IGNORECASE):
        month = _MONTHS.index(m.group(1).lower()) + 1
        try:
            found.append(date(year, month, int(m.group(2))))
        except ValueError:
            pass
    return found


# --------------------------------------------------------------------------
# C2a / C10 — twitter report checks
# --------------------------------------------------------------------------


def check_tw_timestamps_are_day_qualified(text: str) -> list[str]:
    """C2a. Over a multi-day window a bare `HH:MM` cannot be resolved to a day."""
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
    """C9. A degraded backend must be distinguishable from the primary."""
    if re.search(r"(?im)^\*\*(model|backend|generated by model):?\*\*", text):
        return []
    if re.search(r"(?i)generated by .*\bmodel\b", text):
        return []
    return ["report does not record which model/backend produced it"]


# --------------------------------------------------------------------------
# C2b / C3 / C4 / C5 / C7 / C8 — weekend report checks
# --------------------------------------------------------------------------


def check_wk_transient_rows_carry_a_date(text: str, year: int) -> list[str]:
    """C2b. The `Duration / End Date` column must hold a date, not a duration."""
    failures = []
    for row in transient_rows(text):
        cell = _cell(row, "duration", "end date")
        if find_dates_in(cell, year):
            continue
        failures.append(f"{_row_name(row)!r}: no date in date column (got {cell!r})")
    return failures


def check_wk_no_row_outside_window(text: str, start: date, end: date) -> list[str]:
    """C3. Every explicit date in the report must fall inside the plan window."""
    failures = []
    for row in transient_rows(text):
        blob = " ".join(row.values())
        for found in find_dates_in(blob, start.year):
            if not (start <= found <= end):
                failures.append(
                    f"{_row_name(row)!r}: date {found.isoformat()} is outside "
                    f"{start.isoformat()}..{end.isoformat()}"
                )
    return failures


def check_wk_no_stale_event_name(text: str, start: date, end: date) -> list[str]:
    """C3, name-level. A holiday named in a row must not belong to another date.

    Catches the catalogue's headline instance: "Canada Day" (July 1) listed in a
    July 31 - August 02 plan.
    """
    holidays = {
        "canada day": (7, 1),
        "new year": (1, 1),
        "christmas": (12, 25),
        "halloween": (10, 31),
        "valentine": (2, 14),
        "thanksgiving": (10, 13),
        "victoria day": (5, 20),
        "boxing day": (12, 26),
    }
    failures = []
    for row in transient_rows(text):
        name = _row_name(row).lower()
        for holiday, (month, day) in holidays.items():
            if holiday not in name:
                continue
            when = date(start.year, month, day)
            if not (start <= when <= end):
                failures.append(
                    f"{_row_name(row)!r}: {holiday} falls on {when.isoformat()}, "
                    f"outside {start.isoformat()}..{end.isoformat()}"
                )
    return failures


def check_wk_no_mandated_placeholder(text: str) -> list[str]:
    """C4, literal form. A cell equal to a prompt-mandated constant is invented."""
    failures = []
    for label, rows in (("transient", transient_rows(text)), ("fixed", fixed_rows(text))):
        for row in rows:
            for col, val in row.items():
                if val.strip().lower() in MANDATED_LITERALS:
                    failures.append(
                        f"{label} {_row_name(row)!r}: column {col!r} is the "
                        f"prompt-mandated literal {val!r}"
                    )
    return failures


# Columns that are legitimately identical across rows are not evidence of C4.
_KEY_COLUMNS = ("activity", "event", "score", "location")


def check_wk_no_constant_column(text: str) -> list[str]:
    """C4, structural form. A column with one distinct value carries no data."""
    failures = []
    for label, rows in (("transient", transient_rows(text)), ("fixed", fixed_rows(text))):
        if len(rows) < 3:
            continue
        for col in rows[0]:
            if any(k in col.lower() for k in _KEY_COLUMNS):
                continue
            values = {r.get(col, "").strip() for r in rows}
            if len(values) == 1:
                failures.append(
                    f"{label}: column {col!r} is the constant {values.pop()!r} "
                    f"across all {len(rows)} rows"
                )
    return failures


def check_wk_weather_label_matches_venue(text: str) -> list[str]:
    """C5. An unambiguously indoor venue must not be labelled outdoor."""
    failures = []
    for rows in (transient_rows(text), fixed_rows(text)):
        for row in rows:
            name = _row_name(row).lower()
            weather = _cell(row, "weather").strip().lower()
            if weather != "outdoor":
                continue
            for marker in UNAMBIGUOUSLY_INDOOR:
                if marker in name:
                    failures.append(
                        f"{_row_name(row)!r} contains {marker!r} but is labelled 'outdoor'"
                    )
                    break
    return failures


def check_wk_transient_rows_are_time_bounded(text: str, year: int) -> list[str]:
    """C7. A row with no time bound is evergreen and belongs in the fixed table."""
    failures = []
    for row in transient_rows(text):
        cell = _cell(row, "duration", "end date")
        blob = " ".join(row.values())
        if find_dates_in(cell, year) or find_dates_in(blob, year):
            continue
        failures.append(
            f"{_row_name(row)!r}: no date bound — evergreen content in the "
            f"transient table"
        )
    return failures


def check_wk_no_excluded_place(text: str, excluded: Iterable[str]) -> list[str]:
    """C8. The user's `exclude_places` must be enforced, not merely suggested."""
    failures = []
    rows = transient_rows(text) + fixed_rows(text)
    for row in rows:
        haystack = f"{_row_name(row)} {_cell(row, 'location')}".lower()
        for place in excluded:
            token = place.strip().lower()
            if token and token in haystack:
                failures.append(f"{_row_name(row)!r} matches excluded place {place!r}")
                break
    return failures


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
