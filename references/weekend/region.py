"""Positive in-region evidence -- a whitelist, never a blocklist.

Class C16 OUT-OF-REGION-ROW. A real run produced a San Diego zoo, a Dublin
trampoline park and an Oswego one for a Vaughan/Toronto plan.

A blocklist of everywhere-else is unbounded and can never be complete; trying to
maintain one is the C8 escape-hatch trap in a new costume. The INVERSE is
tractable: the home region is small and knowable, and most of it is already in
conf/weekend.toml. So the rule is positive evidence -- a result must name
somewhere in the region to be kept -- rather than "is not on a list of foreign
places".

Probed on the real scraper: DDGS answers a query with no good local matches by
returning whatever matched the YEAR. A query of the form "<city> community
centres kids <month> <year>" came back with a mass-spectrometry conference, an
Astana film festival, a Bulgarian concert and a Bengali government scheme. There
was no relevance floor at all, and the pipeline treated that as signal.

Note this module is used to filter RAW SCRAPE RESULTS, where dropping noise is
safe. A curated row that lacks region evidence is a judgement call and belongs to
the model -- see `weekend/prompts.py` -- not to the floor.
"""

from __future__ import annotations

import re

__all__ = ["region_tokens", "has_region_evidence"]

# Municipalities and neighbourhoods of the Greater Toronto Area and its commuter
# ring. The region is the GTA, not the city limits: a plan must be able to keep
# "Jurassic Quest (International Centre, Mississauga, ON)".
_GTA = (
    "toronto", "vaughan", "markham", "richmond hill", "mississauga", "brampton",
    "scarborough", "north york", "etobicoke", "york", "woodbridge", "concord",
    "thornhill", "maple", "kleinburg", "aurora", "newmarket", "oakville",
    "burlington", "pickering", "ajax", "whitby", "oshawa", "milton", "caledon",
    "king city", "stouffville", "bolton", "georgetown", "hamilton", "brantford",
    "guelph", "barrie", "niagara", "gta", "greater toronto", "ontario",
    "durham region", "peel region", "halton", "yorkdale", "downsview",
)

# The province abbreviation is matched ONLY in its address form, on the RAW
# text before case-folding. Matching a lowercased whole word "on" made the
# filter a near-no-op: "tickets on sale", "posted on Tuesday" and "deals on the
# beach" all counted as Ontario evidence, so Boston tourism pages and iRacing
# calendars entered a Vaughan plan as candidates.
_PROVINCE_ADDRESS_RE = re.compile(
    r",\s*ON\b"  # "Mississauga, ON"
    r"|\bON\s+[A-Z]\d[A-Z]"  # "ON L4K 5W4" (postal code follows)
    r"|\bOnt\.",  # "Ont." abbreviation with its period
)


def region_tokens() -> tuple[str, ...]:
    """In-region place names: the configured city/region plus the GTA."""
    from weekend.config import CITY, REGION

    configured = tuple(t.strip().lower() for t in (CITY, REGION) if t and t.strip())
    return tuple(dict.fromkeys(configured + _GTA))


def has_region_evidence(text: str) -> bool:
    """Does `text` name somewhere in the region?

    STRICT: a result must positively name somewhere in the region. Text that
    names no place at all returns False.

    That is a deliberate trade, and it is only acceptable because of WHERE this
    is applied. Raw scrape results are redundant -- five queries times ten
    results -- so losing an unlabelled local listing costs little, while the
    noise it removes ("Empires & Puzzles Calendar of Events", "Скачать СберKids")
    would otherwise reach the model as candidate activities. If it ever starts
    thinning real results, the low-item-count warning surfaces it.

    The strictness must NOT be carried to curated rows. A finished row that
    lacks region evidence is a judgement call for the model, never a code drop --
    dropping is not free, and an honest-but-hollow plan is its own failure.
    """
    from weekend.enforce import normalize_for_match

    normalized = normalize_for_match(text)
    if not normalized:
        return True
    if any(token in normalized for token in region_tokens()):
        return True
    # Checked against the ORIGINAL text: the case and the comma are the whole
    # signal that distinguishes the province from the preposition.
    return bool(_PROVINCE_ADDRESS_RE.search(text or ""))
