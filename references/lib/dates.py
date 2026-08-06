"""Finding calendar dates in free text.

Shared on purpose. This scanner is used by two things that MUST agree: the
weekend planner's candidate prioritiser (`weekend.supply`), which decides what
looks like it happens during the plan window, and the report checkers
(`eval.report_classes`), which later assert that shipped rows really do.

If those two kept private copies, the pipeline could prioritise a candidate the
checker then rejects — enforcement and its checker disagreeing about the same
question, which is exactly the failure this project has already paid for.
"""

from __future__ import annotations

import re
from datetime import date

MONTHS = (
    "january february march april may june july august september october november december"
).split()


def find_dates_in(value: str, year: int) -> list[date]:
    """Pull explicit calendar dates out of a cell. Durations are not dates."""
    found: list[date] = []
    for m in re.finditer(r"(\d{4})-(\d{2})-(\d{2})", value):
        try:
            found.append(date(int(m.group(1)), int(m.group(2)), int(m.group(3))))
        except ValueError:
            pass
    pattern = r"\b(" + "|".join(MONTHS) + r")\w*\.?\s+(\d{1,2})\b"
    for m in re.finditer(pattern, value, re.IGNORECASE):
        month = MONTHS.index(m.group(1).lower()) + 1
        try:
            found.append(date(year, month, int(m.group(2))))
        except ValueError:
            pass
    return found
