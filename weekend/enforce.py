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


def drop_excluded_places(items: list[dict], excluded: list[str]) -> tuple[list[dict], list[str]]:
    """C8. Remove rows matching the user's `exclude_places`.

    The exclusion list previously reached nothing: it was nested under
    [[children]] in the TOML so it parsed as empty, and no prompt template ever
    interpolated `{exclusions}` anyway. Enforcing it here means it holds
    regardless of what any model does with the instruction.

    Matching is `matches_exclusion` (token-subset). An earlier version used
    contiguous substring matching and this docstring told the reader to "add the
    variant to exclude_places if one slips through" -- an escape hatch that
    turned a matcher defect into unbounded manual config maintenance and let an
    excluded venue ship from two consecutive real runs. Fix the matcher, not the
    config.
    """
    kept, notes = [], []
    for item in items:
        haystack = _item_text(item)
        hit = next((p for p in excluded if matches_exclusion(p, haystack)), None)
        if hit:
            notes.append(f"dropped {item.get('name', '?')!r} — matches excluded place {hit!r}")
        else:
            kept.append(item)
    return kept, notes


def correct_weather_labels(items: list[dict]) -> tuple[list[dict], list[str]]:
    """C5. An unambiguously indoor venue must not be labelled 'outdoor'.

    The label is free LLM choice and was never rechecked, which is how a
    trampoline park shipped as 'outdoor'. Only clear-cut cases are corrected;
    genuinely ambiguous venues are left alone rather than guessed at.
    """
    notes = []
    for item in items:
        weather = str(item.get("weather") or "").strip().lower()
        if weather != "outdoor":
            continue
        text = _item_text(item)
        marker = next((m for m in INDOOR_MARKERS if m in text), None)
        if marker:
            item["weather"] = "indoor"
            notes.append(
                f"corrected {item.get('name', '?')!r} from 'outdoor' to 'indoor' "
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
    m = re.search(r"\b(" + "|".join(_MONTHS) + r")\w*\.?\s+(\d{1,2})\b", value, re.IGNORECASE)
    if m:
        try:
            return date(year, _MONTHS.index(m.group(1).lower()) + 1, int(m.group(2)))
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
