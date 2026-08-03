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
# What weekend/output.py renders for a value the source never stated.
MISSING_SENTINEL = "—"

MANDATED_LITERALS = frozenset(
    {
        "2-3 hours",
        "all day",
        "$20-30 per child or free",
        "$18-35 per child or free",
        "$20-30 per child",
    }
)

# C5 markers come from weekend.enforce, never a private copy. A checker with its
# own list drifts from the enforcement it measures and reports PASS on the rows
# the enforcement missed -- exactly how C8b hid.
def _indoor_markers():
    from weekend.enforce import INDOOR_MARKERS

    return INDOOR_MARKERS

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
# C2b / C3 / C4 / C5 / C7 / C8 — weekend report checks
# --------------------------------------------------------------------------


def check_wk_transient_rows_carry_a_date(text: str, year: int) -> list[str]:
    """C2b. The date column must hold a date -- never a duration standing in.

    A row whose date cell is the missing sentinel is NOT a C2b failure: the
    source genuinely gave no date and the report says so. That row is class C7
    (evergreen content in the transient table), reported by
    `check_wk_transient_rows_are_time_bounded`. Conflating the two made the fixed
    renderer look broken because it had started telling the truth.
    """
    failures = []
    for row in transient_rows(text):
        cell = _cell(row, "dates", "duration", "end date").strip()
        if find_dates_in(cell, year):
            continue
        if cell in ("", MISSING_SENTINEL):
            continue
        failures.append(f"{_row_name(row)!r}: date column holds a non-date {cell!r}")
    return failures


def check_wk_no_row_outside_window(text: str, start: date, end: date) -> list[str]:
    """C3. Every dated row's RANGE must overlap the plan window.

    Overlap, not endpoint containment. A long-running exhibition
    (e.g. late June to mid August) is legitimately in an early-August plan
    even though neither endpoint falls inside the window. This checker previously tested
    each date for containment and so reported a correct row as a failure, while
    the enforcement -- which used overlap -- kept it. Both now call
    `weekend.enforce.window_overlap`, so there is one decision in one place.
    """
    from weekend.enforce import window_overlap

    failures = []
    for row in transient_rows(text):
        cell = _cell(row, "dates", "duration", "end date")
        found = find_dates_in(cell, start.year)
        if not found:
            continue
        item = {"start_date": min(found).isoformat(), "end_date": max(found).isoformat()}
        if window_overlap(item, start, end) is False:
            failures.append(
                f"{_row_name(row)!r}: runs {min(found).isoformat()}..{max(found).isoformat()}, "
                f"which does not overlap {start.isoformat()}..{end.isoformat()}"
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
    from weekend.enforce import normalize_for_match

    failures = []
    for row in transient_rows(text):
        name = normalize_for_match(_row_name(row))
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


def check_wk_no_mandated_placeholder(text: str) -> list[str]:  # noqa: D401
    """C4, literal form. A cell equal to a prompt-mandated constant is invented."""
    from weekend.enforce import normalize_for_match

    _NORMALIZED_MANDATED = {normalize_for_match(v) for v in MANDATED_LITERALS}
    failures = []
    for label, rows in (("transient", transient_rows(text)), ("fixed", fixed_rows(text))):
        for row in rows:
            for col, val in row.items():
                if normalize_for_match(val) in _NORMALIZED_MANDATED:
                    failures.append(
                        f"{label} {_row_name(row)!r}: column {col!r} is the "
                        f"prompt-mandated literal {val!r}"
                    )
    return failures


# Columns that are legitimately identical across rows are not evidence of C4.
# "weather" is here because it is a three-value categorical (indoor/outdoor/both)
# that can honestly be uniform -- a real 2026-08-07 run picked seven indoor
# venues for a rainy weekend, which is correct behaviour, not a fabricated
# default. C4 is about a FREE-TEXT field (price, ages, duration) repeating a
# constant the prompt supplied.
_KEY_COLUMNS = ("activity", "event", "score", "location", "weather")


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
            # An empty/sentinel column is an HONEST blank -- the C4 fix makes
            # unknown values render as nothing at all. Only a repeated
            # NON-empty value is evidence of a fabricated default.
            if values <= {"", MISSING_SENTINEL}:
                continue
            if len(values) == 1:
                failures.append(
                    f"{label}: column {col!r} is the constant {values.pop()!r} "
                    f"across all {len(rows)} rows"
                )
    return failures


def check_wk_weather_label_matches_venue(text: str) -> list[str]:
    """C5. An unambiguously indoor venue must not be labelled outdoor."""
    failures = []
    from weekend.enforce import normalize_for_match

    for rows in (transient_rows(text), fixed_rows(text)):
        for row in rows:
            name = normalize_for_match(_row_name(row))
            weather = _cell(row, "weather").strip().lower()
            if weather != "outdoor":
                continue
            for marker in _indoor_markers():
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
        # Dates must come from the DATE column, not from anywhere in the row.
        # Scanning the whole row let a date range that the model had dumped into
        # the `Day` column satisfy this check, so five fabricated rows with an
        # empty Dates column were reported as properly time-bounded.
        cell = _cell(row, "dates", "duration", "end date")
        if find_dates_in(cell, year):
            continue
        failures.append(
            f"{_row_name(row)!r}: no date bound — evergreen content in the "
            f"transient table"
        )
    return failures


def check_wk_no_excluded_place(text: str, excluded: Iterable[str]) -> list[str]:
    """C8. The user's `exclude_places` must be enforced, not merely suggested."""
    failures = []
    from weekend.enforce import matches_exclusion

    rows = transient_rows(text) + fixed_rows(text)
    for row in rows:
        haystack = f"{_row_name(row)} {_cell(row, 'location')}"
        for place in excluded:
            if matches_exclusion(place, haystack):
                failures.append(f"{_row_name(row)!r} matches excluded place {place!r}")
                break
    return failures


# ---------------------------------------------------------------------------
# C14 -- AGGREGATOR-PAGE-AS-ACTIVITY
# ---------------------------------------------------------------------------

# Phrases that make a row a DIRECTORY of activities rather than an activity.
# This is a measurement instrument, not an enforcement filter: deciding "can a
# family go and DO this?" is a judgement call the model makes (see
# PHASE_EXTRACT_EVENTS). A checker may flag a suspicious row for review; code
# must not silently drop rows on a title pattern.
# Markers come from weekend.followup -- the SAME list the pipeline uses to
# decide which pages to follow. A checker with a private copy drifts from the
# code it measures and then agrees with the bug (see C8b, and the C5
# library/libraries miss).
def _aggregator_markers():
    from weekend.followup import AGGREGATOR_MARKERS

    return AGGREGATOR_MARKERS


def check_wk_no_aggregator_rows(text: str) -> list[str]:
    """C14. A row must be an activity, not a page that lists activities.

    The user's verdict on the row this was written for -- "Vaughan Events &
    Activities Guides for Kids & Families" -- was "this specifically means
    nothing". You cannot attend a guide.

    Distinct from C7: C7 is "transient because an events query returned it"
    (an evergreen venue in the wrong table). C14 is "this is not an activity at
    all", and it can appear in either table.
    """
    from weekend.enforce import normalize_for_match

    failures = []
    for label, rows in (("transient", transient_rows(text)), ("fixed", fixed_rows(text))):
        for row in rows:
            name = normalize_for_match(_row_name(row))
            hit = next((m for m in _aggregator_markers() if m in name), None)
            if hit:
                failures.append(
                    f"{label} {_row_name(row)!r}: reads as a directory page, not an "
                    f"activity (matched {hit!r})"
                )
    return failures


def check_wk_day_matches_dates(text: str, year: int) -> list[str]:
    """The purely checkable consistency rule: Day must fall inside the row's dates.

    A real run shipped a row whose Dates column covered Tuesday to Friday and
    whose Day column said Saturday. No model needed to be involved for code to
    catch that.
    """

    failures = []
    for row in transient_rows(text):
        stated = _cell(row, "day").strip()
        if not stated or stated == MISSING_SENTINEL:
            continue
        cell = _cell(row, "dates", "duration", "end date")
        found = find_dates_in(cell, year)
        if not found:
            continue
        first, last = min(found), max(found)
        covered = set()
        cursor = first
        while cursor <= last:
            covered.add(cursor.strftime("%A"))
            cursor = cursor.fromordinal(cursor.toordinal() + 1)
        if stated not in covered:
            failures.append(
                f"{_row_name(row)!r}: Day {stated!r} is not within its own dates "
                f"{first.isoformat()}..{last.isoformat()} (covers {sorted(covered)})"
            )
    return failures


# Re-export shim: the tw and repo-level checks live in report_classes_tw to keep
# this module under the 500-line cap. Callers import from one place.
from eval.report_classes_tw import (  # noqa: E402,F401
    PRODUCTION_PROMPT_KWARGS,
    WEEKEND_PROMPT_TASKS,
    attribution_styles,
    check_declared_config_keys_are_read,
    check_model_prompts_render,
    check_tw_attribution_format_is_uniform,
    check_tw_names_its_backend,
    check_tw_timestamps_are_day_qualified,
    check_wk_no_column_echoes_config,
    check_wk_plan_is_not_hollow,
    configured_echo_values,
)

# Places far outside the configured region that the scrape keeps returning.
# This is a MEASUREMENT instrument, not an enforcement filter: "could a family
# here drive to this?" is a judgement call for the model (see the WHERE rule in
# PHASE_EXTRACT_EVENTS). A checker may flag an obviously-foreign row so the
# quality gate fails; code must not drop rows on a place-name list, because the
# list can never be complete and would silently discard real local venues.
_OUT_OF_REGION_HINTS = (
    "san diego", "new york", "chicago", "boston", "london", "dublin", "oswego",
    "edmonton", "calgary", "vancouver", "montreal", "ottawa", "seattle",
    "los angeles", "california", "texas", "florida", "australia", "queensland",
)


def check_wk_rows_are_in_region(text: str) -> list[str]:
    """C16. Every row must be somewhere the user could actually go.

    A real run returned "San Diego Zoo Nighttime Zoo", "Urban Air Trampoline
    (Dublin)" and "Altitude Trampoline Park (Oswego)" for a Vaughan/Toronto
    plan. DDGS returns results globally and nothing checked the geography.
    """
    from weekend.enforce import normalize_for_match

    failures = []
    for label, rows in (("transient", transient_rows(text)), ("fixed", fixed_rows(text))):
        for row in rows:
            blob = normalize_for_match(f"{_row_name(row)} {_cell(row, 'location')}")
            hit = next((h for h in _OUT_OF_REGION_HINTS if h in blob), None)
            if hit:
                failures.append(
                    f"{label} {_row_name(row)!r}: names {hit!r}, which is not in the "
                    f"configured region"
                )
    return failures
