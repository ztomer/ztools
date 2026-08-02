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
]

# Venue words that settle indoor/outdoor without consulting a forecast. Kept
# deliberately small: only terms where an "outdoor" label is unambiguously wrong.
_INDOOR_MARKERS = (
    "indoor",
    "trampoline park",
    "museum",
    "play centre",
    "play center",
    "playground",
    "library",
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
        marker = next((m for m in _INDOOR_MARKERS if m in text), None)
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
        starts = _parse_any_date(item.get("start_date", ""), start.year)
        ends = _parse_any_date(item.get("end_date", ""), start.year)
        if starts is None and ends is None:
            kept.append(item)
            continue
        first, last = starts or ends, ends or starts
        if last < start or first > end:
            notes.append(
                f"dropped {item.get('name', '?')!r} — runs "
                f"{first.isoformat()}..{last.isoformat()}, outside "
                f"{start.isoformat()}..{end.isoformat()}"
            )
        else:
            kept.append(item)
    return kept, notes
