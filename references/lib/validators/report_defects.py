"""Defects a well-formed report can still have, which the JSON scorer misses.

validate_detailed_json is a structure checklist: items present, enough of them,
fields populated, grounded in the source. Everything it asks is about shape, so
output with real venue names and useless content passes it perfectly. Measured
against a hand-built set of defective reports, three distinct failures scored
100 -- identical to correct output:

  every location the generic word the prompt explicitly forbids
  one column the same value in every row
  six rows that are the same venue under four spellings

Only wholesale hallucination was detected. These three are what a reader
notices first, and each is checkable without a model.
"""

from __future__ import annotations

import re

__all__ = [
    "GENERIC_LOCATION_RE",
    "constant_column_ratio",
    "generic_location_ratio",
    "near_duplicate_ratio",
]

# The weekend prompts say, verbatim: "NEVER output generic words like 'Indoor
# venue' or 'Outdoor venue'". A location field is meant to carry a street
# address or at minimum a place name; these are the model restating the column
# header instead of answering it.
GENERIC_LOCATION_RE = re.compile(
    r"^\s*(?:"
    r"(?:in|out)door\s+(?:venue|location|activity|space)|"
    r"(?:various|multiple|several)\s+(?:locations?|venues?|places?)|"
    r"n/?a|tbd|tba|unknown|unspecified|not\s+specified|none|"
    r"local\s+(?:area|venue)|nearby|online|virtual|"
    r"venue|location|place|address|city"
    r")\s*\.?\s*$",
    re.IGNORECASE,
)

# Below this many rows, a repeated value is a coincidence rather than a pattern.
MIN_ROWS_FOR_CONSTANT = 3
# Fields where one value legitimately repeats across a whole report.
_CONSTANT_EXEMPT = frozenset({"day", "weather"})

_DUP_NOISE_RE = re.compile(r"\b(?:the|a|an|of|at|in|on|and|museum|centre|center|park)\b")


def _rows(items) -> list[dict]:
    return [i for i in (items or []) if isinstance(i, dict)]


def generic_location_ratio(items) -> float:
    """Share of rows whose location restates the column instead of answering it."""
    rows = _rows(items)
    if not rows:
        return 0.0
    generic = sum(
        1 for r in rows if GENERIC_LOCATION_RE.match(str(r.get("location", "") or ""))
    )
    return generic / len(rows)


def constant_column_ratio(items) -> tuple[float, list[str]]:
    """(share of scored columns that never vary, their names).

    A column identical in every row is a value the model was told to emit rather
    than one it observed -- six different venues do not all cost exactly $20.
    `day` and `weather` are exempt: a one-day report legitimately says Saturday
    throughout, and an all-indoor list is a real answer.
    """
    rows = _rows(items)
    if len(rows) < MIN_ROWS_FOR_CONSTANT:
        return 0.0, []
    names = [k for k in rows[0] if k not in _CONSTANT_EXEMPT and k != "name"]
    if not names:
        return 0.0, []
    constant = [
        k
        for k in names
        if len({str(r.get(k, "")).strip().lower() for r in rows}) == 1
        and str(rows[0].get(k, "")).strip()
    ]
    return len(constant) / len(names), constant


def _words(name: str) -> list[str]:
    return re.sub(r"[^a-z0-9\s]", " ", str(name or "").lower()).split()


def _name_tokens(name: str) -> frozenset[str]:
    """The words that make a venue name distinct, minus generic noise."""
    stripped = _DUP_NOISE_RE.sub(" ", " ".join(_words(name)))
    # Keep digits whatever their length: "Place 1" and "Place 2" differ only by
    # the numeral, and dropping it collapsed ten distinct venues into one.
    return frozenset(w for w in stripped.split() if len(w) > 2 or w.isdigit())


def _acronym(name: str) -> str:
    """Initials of a multi-word name: "Royal Ontario Museum" -> "rom".

    Token containment cannot see that "The ROM" is the same venue as "Royal
    Ontario Museum" -- they share no words at all. Abbreviating is the most
    natural way to restate a venue, so a padded report reaches for it first.
    Built from the FULL word list, before noise words are dropped, since the
    dropped ones ("Museum", "Centre", "Park") supply the final initial.
    """
    words = [w for w in _words(name) if w not in {"the", "a", "an", "of", "and", "at"}]
    return "".join(w[0] for w in words) if len(words) > 1 else ""


def near_duplicate_ratio(items) -> float:
    """Share of rows that repeat an earlier row under a different spelling.

    The existing check compares names exactly, so "Royal Ontario Museum", "The
    ROM" and "ROM Museum" counted as three venues and a padded report scored
    full marks on item count.

    Containment, not equality: one name being a subset of another is what
    distinguishes a restatement ("The ROM" inside "Royal Ontario Museum (ROM)")
    from two genuinely different places, which share at most a city name.
    """
    rows = _rows(items)
    names = [str(r.get("name", "") or "") for r in rows]
    if not names:
        return 0.0
    kept: list[tuple[frozenset[str], str]] = []
    dupes = 0
    for name in names:
        tokens = _name_tokens(name)
        if not tokens:
            continue
        acronym = _acronym(name)
        single = next(iter(tokens)) if len(tokens) == 1 else ""
        duplicate = False
        for seen_tokens, seen_acronym in kept:
            if tokens <= seen_tokens or seen_tokens <= tokens:
                duplicate = True
                break
            # "The ROM" against "Royal Ontario Museum": a one-word name that IS
            # the other's initials, in either direction.
            seen_single = next(iter(seen_tokens)) if len(seen_tokens) == 1 else ""
            if (single and single == seen_acronym) or (seen_single and seen_single == acronym):
                duplicate = True
                break
        if duplicate:
            dupes += 1
        else:
            kept.append((tokens, acronym))
    return dupes / len(names)
