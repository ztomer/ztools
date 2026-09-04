"""Finding calendar dates in free text.

Shared on purpose. This scanner is used by three things that MUST agree: the
weekend planner's candidate prioritiser (`weekend.supply`), which decides what
looks like it happens during the plan window; the constraint enforcer
(`weekend.enforce`), which drops rows outside it; and the report checkers
(`eval.report_classes`), which later assert that shipped rows really do.

If those kept private copies, the pipeline could prioritise a candidate the
checker then rejects — enforcement and its checker disagreeing about the same
question, which is exactly the failure this project has already paid for. They
DID disagree: this module matched only full month names while `enforce.py` used
three-letter stems, so `find_dates_in("Aug 15")` returned nothing while the
enforcer read it fine. Search snippets overwhelmingly write "Aug 15" and
"Sun 09 Aug", so in-window prioritisation reported 0 candidates on corpora that
demonstrably held this-weekend events, and the plan came out empty.
"""

from __future__ import annotations

import re
from datetime import date

MONTHS = (
    "january february march april may june july august september october november december"
).split()

# Three-letter stems match both "Aug" and "August"; the trailing `[a-z]*\.?`
# absorbs the rest of the word and an optional abbreviating period.
_STEMS = "|".join(mo[:3] for mo in MONTHS)
_MONTH = rf"({_STEMS})[a-z]*\.?"

_ISO_RE = re.compile(r"(\d{4})-(\d{2})-(\d{2})")
# "Aug 15", "August 15, 2026", "Aug. 15 2026"
_MONTH_FIRST_RE = re.compile(rf"\b{_MONTH}\s+(\d{{1,2}})(?:\s*,?\s*(\d{{4}}))?\b", re.IGNORECASE)
# "15 Aug", "09 Aug 2026", "Sun 09 Aug"
_DAY_FIRST_RE = re.compile(rf"\b(\d{{1,2}})\s+{_MONTH}(?:\s*,?\s*(\d{{4}}))?\b", re.IGNORECASE)


def _month_number(stem: str) -> int:
    """1-12 for a month name or its three-letter stem."""
    stem = stem.lower()[:3]
    return next(i for i, mo in enumerate(MONTHS, 1) if mo.startswith(stem))


def _add(found: list[date], year: int, month: int, day: int) -> None:
    try:
        value = date(year, month, day)
    except ValueError:
        return
    if value not in found:
        found.append(value)


def find_dates_in(value: str, year: int) -> list[date]:
    """Pull explicit calendar dates out of a cell. Durations are not dates.

    `year` is the fallback for formats that omit it; an explicit four-digit year
    in the text always wins, so a snippet carrying a past year is not silently
    promoted into this year's plan window.
    """
    if not value:
        return []

    found: list[date] = []

    for m in _ISO_RE.finditer(value):
        _add(found, int(m.group(1)), int(m.group(2)), int(m.group(3)))

    for m in _MONTH_FIRST_RE.finditer(value):
        stem, day, explicit_year = m.group(1), int(m.group(2)), m.group(3)
        _add(found, int(explicit_year) if explicit_year else year, _month_number(stem), day)

    for m in _DAY_FIRST_RE.finditer(value):
        day, stem, explicit_year = int(m.group(1)), m.group(2), m.group(3)
        _add(found, int(explicit_year) if explicit_year else year, _month_number(stem), day)

    return found
