"""Post-parse enforcement of constraints the prompt can only ask for politely.

Classes C3, C5, C7 and C8 in `docs/REPORT_WEAKNESS_CLASSES.md` share one shape:
a rule was stated in a prompt and never checked in code, so whether it held
depended on the model's mood. A user constraint that is only ever a suggestion
to a model is not a feature.

Everything here is a pure function over the parsed item list -- no LLM, no
network -- so it is deterministic and cheap to test. Each returns
`(kept_items, notes)`; the notes are surfaced to the operator rather than
discarded, so a filtered run says what it dropped and why.
"""

from __future__ import annotations

import re
from datetime import date

__all__ = [
    "matches_exclusion",
    "normalize_for_match",
    "drop_excluded_places",
    "correct_weather_labels",
    "drop_events_outside_window",
    "reconcile_day_with_dates",
    "INDOOR_MARKERS",
    "parse_any_date",
    "window_overlap",
    "flag_constant_columns",
    "CONSTANT_COLUMN_FIELDS",
    "PROMPT_CONSTANTS",
]

# Venue words that settle indoor/outdoor without consulting a forecast. Kept
# deliberately small: only terms where an "outdoor" label is unambiguously wrong.
#
# These are STEMS, matched as substrings, so plurals are covered: "librar"
# catches both "library" and "libraries". A real run labelled "Vaughan Public
# Libraries" as outdoor and the singular-only marker missed it.
INDOOR_MARKERS = (
    "indoor",
    "trampoline park",
    "museum",
    "play centre",
    "play center",
    "playground",
    "librar",
    "cinema",
    "aquarium",
    "arcade",
    "bowling",
)


# Scraped venue names use typographic punctuation; a hand-written config uses
# ASCII. "Ripley's" in conf/weekend.toml did NOT match a scraped
# "Ripley's Aquarium of Canada" because of U+2019 vs U+0027 -- found by a real
# `wk` run, after the exclusion filter had already been declared working.
_PUNCT_FOLD = str.maketrans(
    {
        "’": "'", "‘": "'", "ʼ": "'",
        "“": '"', "”": '"',
        "–": "-", "—": "-", "−": "-",
        " ": " ",
    }
)


def normalize_for_match(text: str) -> str:
    """Fold typographic punctuation and whitespace for constraint matching.

    Shared with eval/report_classes.py on purpose: when the checker and the
    enforcement normalise differently, the checker reports PASS on exactly the
    rows the enforcement failed to drop. That is how this bug hid.
    """
    folded = (text or "").translate(_PUNCT_FOLD).lower()
    return " ".join(folded.split())


# Connector words carry no identifying signal, so requiring them would make a
# config entry miss a re-worded venue ("The Art of the Brick" vs "Art of Brick").
_CONNECTORS = frozenset({"of", "the", "and", "at", "in", "a", "an", "on", "for"})


def _significant_tokens(text: str) -> set[str]:
    """Identifying tokens of a venue name.

    Three things are deliberately discarded, each because it caused a real miss:
    - possessive `'s`, which otherwise tokenises to a stray "s" the scraped name
      will not have ("Canada's Wonderland" vs "Wonderland Canada")
    - single characters, which carry no signal
    - connector words (see `_CONNECTORS`)
    """
    cleaned = re.sub(r"'s\b", "", normalize_for_match(text))
    tokens = {t for t in re.findall(r"[a-z0-9]+", cleaned) if len(t) > 1}
    return tokens - _CONNECTORS


def _required_tokens(entry: str) -> set[str]:
    """Tokens an entry REQUIRES of a candidate name.

    A parenthetical is an annotation, not a requirement: the config's
    "Royal Ontario Museum (ROM)" must still match a venue called simply
    "Royal Ontario Museum". Dropping it keeps the match conservative -- the
    remaining tokens are all still required.
    """
    without_parens = re.sub(r"\([^)]*\)", " ", entry)
    return _significant_tokens(without_parens) or _significant_tokens(entry)


def matches_exclusion(entry: str, haystack: str) -> bool:
    """Does `entry` name the same venue as `haystack`?

    Class NAME-MATCHED-BY-CONTAINMENT. Containment matching assumes the config's
    wording is a contiguous substring of the scraped wording, which it usually is
    not: the scraper interleaves and reorders words. `"Sky Zone Toronto"` is not a
    substring of `"Sky Zone Trampoline Park (Vaughan/Toronto)"`, so an excluded
    venue shipped in a real run -- twice, because the first fix only
    addressed the U+2019 instance rather than the class.

    The rule is token-SUBSET: every significant token of the entry must appear in
    the haystack, in any order, with anything interleaved. That is still
    conservative -- ALL tokens are required, so "Toronto Zoo" does not match
    "Toronto Islands" -- but it survives word order, interpolated words and
    punctuation, which containment does not.

    Contiguous containment is kept as an additional accept so that nothing which
    matched before stops matching (e.g. an entry whose token set is not fully
    present but whose exact phrase is).
    """
    entry_n, hay_n = normalize_for_match(entry), normalize_for_match(haystack)
    if not entry_n:
        return False
    if entry_n in hay_n:
        return True
    entry_tokens = _required_tokens(entry)
    return bool(entry_tokens) and entry_tokens <= _significant_tokens(hay_n)


def _item_text(item: dict) -> str:
    name = item.get("name") or item.get("event") or item.get("title") or ""
    loc = item.get("location") or item.get("address") or ""
    return normalize_for_match(f"{name} {loc}")


SEASONAL_EVENT_MARKERS = (
    "festival",
    "fair",
    "lumina",
    "spooktacular",
    "lights",
    "exhibit",
    "market",
    "harvest",
    "show",
    "carnival",
    "spectacular",
    "concert",
    "expo",
    "pumpkin",
    "maple",
)


def is_seasonal_event_exception(item: dict, hit_exclusion: str) -> bool:
    """Check if an item at an excluded venue is a specific seasonal event exception.

    Generic visits to excluded places (e.g. 'Toronto Zoo') are dropped. But specific,
    time-limited seasonal events/exhibits (e.g. 'Terra Lumina at Toronto Zoo' or
    'Royal Agricultural Winter Fair at Exhibition Place') are allowed as exceptions.
    """
    name = (item.get("name") or item.get("title") or "").strip()
    name_norm = normalize_for_match(name)
    hit_norm = normalize_for_match(hit_exclusion)

    has_event_marker = any(m in name_norm for m in SEASONAL_EVENT_MARKERS)
    is_more_specific = len(name_norm.split()) > len(hit_norm.split())

    return has_event_marker and is_more_specific


def drop_excluded_places(items: list[dict], excluded: list[str]) -> tuple[list[dict], list[str]]:
    """C8. Remove rows matching the user's `exclude_places` unless it is a seasonal event."""
    kept, notes = [], []
    for item in items:
        haystack = _item_text(item)
        hit = next((p for p in excluded if matches_exclusion(p, haystack)), None)
        if hit:
            if is_seasonal_event_exception(item, hit):
                notes.append(f"kept seasonal event {item.get('name', '?')!r} at {hit!r}")
                kept.append(item)
            else:
                notes.append(f"dropped {item.get('name', '?')!r} — matches excluded place {hit!r}")
        else:
            kept.append(item)
    return kept, notes


OUTDOOR_MARKERS = (
    "high park",
    "conservation",
    "nature walk",
    "botanical garden",
    "provincial park",
    "national park",
    "hiking trail",
)


def correct_weather_labels(items: list[dict]) -> tuple[list[dict], list[str]]:
    """C5. Correct weather label inversions (indoor/outdoor).

    Local LLMs occasionally invert weather labels (e.g. High Park or Nature Walk
    labeled as 'indoor', or Trampoline Park labeled as 'outdoor'). Only clear-cut
    cases are corrected; ambiguous venues are left alone.
    """
    notes = []
    for item in items:
        weather = str(item.get("weather") or "").strip().lower()
        text = _item_text(item)
        if weather == "outdoor":
            marker = next((m for m in INDOOR_MARKERS if m in text), None)
            if marker:
                item["weather"] = "indoor"
                notes.append(
                    f"corrected {item.get('name', '?')!r} from 'outdoor' to 'indoor' "
                    f"(name contains {marker!r})"
                )
        elif weather == "indoor":
            marker = next((m for m in OUTDOOR_MARKERS if m in text), None)
            if marker:
                item["weather"] = "outdoor"
                notes.append(
                    f"corrected {item.get('name', '?')!r} from 'indoor' to 'outdoor' "
                    f"(name contains {marker!r})"
                )
    return items, notes


_MONTHS = (
    "january february march april may june july august september october "
    "november december"
).split()


def _parse_any_date(value: str, year: int) -> date | None:
    value = (value or "").strip()
    if not value:
        return None
    m = re.search(r"(\d{4})-(\d{2})-(\d{2})", value)
    if m:
        try:
            return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except ValueError:
            return None
    # Match on the three-letter stem so scraped abbreviations parse too:
    # "Aug 9", "Aug. 9" and "August 9" are all the same date, and event listings
    # use all three. Requiring the full name silently dropped the abbreviated
    # ones, which is a date lost at exactly the boundary class C2 is about.
    m = re.search(
        r"\b(" + "|".join(mo[:3] for mo in _MONTHS) + r")[a-z]*\.?\s+(\d{1,2})\b",
        value,
        re.IGNORECASE,
    )
    if m:
        try:
            month = next(
                i for i, mo in enumerate(_MONTHS, 1) if mo.startswith(m.group(1).lower())
            )
            return date(year, month, int(m.group(2)))
        except ValueError:
            return None
    return None


def drop_events_outside_window(
    items: list[dict], start: date, end: date
) -> tuple[list[dict], list[str]]:
    """C3. A dated event outside the plan's weekend is dropped.

    Only rows that actually carry a parseable date are judged. A row with no
    date is NOT dropped here -- undated rows are class C7's problem (evergreen
    content in the transient table), and silently discarding them would hide
    that rather than fix it.
    """
    kept, notes = [], []
    for item in items:
        overlaps = window_overlap(item, start, end)
        if overlaps is None:
            kept.append(item)
            continue
        starts = _parse_any_date(item.get("start_date", ""), start.year)
        ends = _parse_any_date(item.get("end_date", ""), start.year)
        first, last = starts or ends, ends or starts
        if not overlaps:
            notes.append(
                f"dropped {item.get('name', '?')!r} — runs "
                f"{first.isoformat()}..{last.isoformat()}, outside "
                f"{start.isoformat()}..{end.isoformat()}"
            )
        else:
            kept.append(item)
    return kept, notes


def parse_any_date(value: str, year: int):
    """Public alias -- the checker must parse dates exactly as enforcement does."""
    return _parse_any_date(value, year)


def reconcile_day_with_dates(
    items: list[dict], start: date, end: date
) -> tuple[list[dict], list[str]]:
    """Make `day` agree with the row's own dates, or blank it.

    A real run shipped a row whose date range covered Tuesday to Friday while
    its Day column said Saturday. Nothing in the pipeline ever compared the two
    columns, so a row could disagree with itself.

    This is the purely CHECKABLE half of correctness -- no judgement, no model.
    Where the row's dates overlap the plan window, `day` is derived from them.
    Where they do not overlap at all the row is left for
    `drop_events_outside_window`. Where there are no dates, `day` is left alone:
    it cannot be verified, and inventing one would be class C4 again.
    """
    notes: list[str] = []
    for item in items:
        first = _parse_any_date(item.get("start_date", ""), start.year)
        last = _parse_any_date(item.get("end_date", ""), start.year) or first
        if first is None:
            continue
        if last < first:
            first, last = last, first

        lo, hi = max(first, start), min(last, end)
        if lo > hi:
            continue  # entirely outside the window; not this function's job

        covered = []
        cursor = lo
        while cursor <= hi:
            covered.append(cursor.strftime("%A"))
            cursor = cursor.fromordinal(cursor.toordinal() + 1)

        stated = str(item.get("day") or "").strip()
        if stated and stated in covered:
            continue
        corrected = covered[0] if len(covered) == 1 else ""
        item["day"] = corrected
        if stated:
            notes.append(
                f"{item.get('name', '?')!r}: day {stated!r} is not within "
                f"{first.isoformat()}..{last.isoformat()} — "
                + (f"corrected to {corrected!r}" if corrected else "cleared")
            )
    return items, notes


def window_overlap(item: dict, start: date, end: date):
    """Does this row's date range overlap [start, end]?

    Returns True / False, or None when the row carries no parseable date.

    Shared with eval/report_classes.py deliberately. A long-running exhibition
    (e.g. late June to mid August) is IN the plan if it spans the weekend, even
    though neither of its endpoints falls inside it. A checker that tested each
    endpoint for containment instead of testing the range for overlap disagreed
    with this enforcement and reported a correct row as a failure -- the fifth
    time in this project that a checker written from a second mental model was
    wrong. There is one decision, in one place.
    """
    first = _parse_any_date(item.get("start_date", ""), start.year)
    last = _parse_any_date(item.get("end_date", ""), start.year) or first
    if first is None:
        last = _parse_any_date(item.get("end_date", ""), start.year)
        if last is None:
            return None
        first = last
    if last < first:
        first, last = last, first
    return not (last < start or first > end)


# Fields whose value the prompt asks the model to always supply, paired with the
# aliases the parsed item may carry them under. These are the columns a model
# fills mechanically when it has nothing real to say.
CONSTANT_COLUMN_FIELDS: dict[str, tuple[str, ...]] = {
    "Target Age(s)": ("target_ages", "age_group"),
    "Estimated Price": ("price", "cost"),
    "Duration": ("duration",),
}

# The values that actually shipped, filled from the instructions rather than
# from any event. Kept as data so a fourth instance is one line, not a new
# check -- and so the historical evidence for this class stays readable.
PROMPT_CONSTANTS: dict[str, list[str]] = {
    "Estimated Price": ["$18-35", "18-35"],
    "Duration": ["2-3 hours"],
}


# A column of one row is trivially constant. Flagging it would be noise, and a
# check that cries wolf on every single-row table is a check the reader learns
# to skip.
_MIN_ROWS_FOR_CONSTANT = 2


def _column_value(item: dict, aliases: tuple[str, ...]) -> str:
    for key in aliases:
        value = item.get(key)
        if value:
            return str(value).strip()
    return ""


def flag_constant_columns(
    items: list[dict],
    suspect_values: dict[str, list[str]],
) -> list[str]:
    """Report columns that are identical on every row **and** equal to a value
    the configuration or the prompt supplied.

    **The assertion this class always wanted.** Three separate bugs shipped in
    the same shape: `Duration` reading `2-3 hours` on every row, `Estimated
    Price` reading `$18-35` on every row, and `Target Age(s)` reading the
    configured family range `6-13` on every row. Each was a field the model was
    told never to leave empty, filled from the instructions rather than from the
    event -- and each was caught by a human noticing, which is exactly why all
    three reached a report.

    The conjunction is what makes it precise. *Constant* alone fires on a
    genuinely uniform column (every event in one city). *Equal to a configured
    value* alone fires on the one row that legitimately matches. Together they
    say: this column was answered from the question.

    Returns notes and **changes nothing**. A mechanically-filled column is a
    signal that the extraction is wrong, not a reason to delete the rows --
    dropping would trade a wrong answer for an empty one, which is how this
    report went honest and hollow once already.
    """
    if len(items) < _MIN_ROWS_FOR_CONSTANT:
        return []

    notes: list[str] = []
    for label, aliases in CONSTANT_COLUMN_FIELDS.items():
        values = [_column_value(item, aliases) for item in items]
        first = values[0]
        if any(v.lower() != first.lower() for v in values[1:]):
            continue
        # `if s` is load-bearing, not defensive tidying: it is what keeps an
        # empty column from ever matching. A blank cell is an honest "unknown",
        # and the failure being hunted here is a column uniformly *populated*
        # from the prompt -- flagging uniformly blank would invert the lesson
        # that empty beats fabricated.
        suspects = [s.strip().lower() for s in suspect_values.get(label, []) if s]
        if first.lower() not in suspects:
            continue
        notes.append(
            f"{label}: every one of {len(items)} rows reads {first!r}, which is "
            f"the configured value -- the column was answered from the prompt, "
            f"not from the events. Rows kept; treat the column as unverified."
        )
    return notes
