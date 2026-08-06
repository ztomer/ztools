"""C16 OUT-OF-REGION-ROW: positive in-region evidence, scoped at source.

Split from tests/test_report_class_fixes.py to stay under the repo's 500-line
cap. See docs/REPORT_WEAKNESS_CLASSES.md C16.
"""

import pytest

# ---------------------------------------------------------------------------
# C16 -- positive in-region evidence (whitelist, never a blocklist)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "Toronto Zoo | Events",
        "Playcious Indoor Playground in Vaughan",
        "Jurassic Quest (International Centre, Mississauga, ON)",
        "Kite Festival, Scarborough",
        "Drop-in art, Woodbridge",
        "Museum programme, Hamilton Ontario",
    ],
)
def test_C16_in_region_places_are_kept(text):
    """The region is the GTA, not the city limits -- Mississauga must survive."""
    from weekend.region import has_region_evidence

    assert has_region_evidence(text)


@pytest.mark.parametrize(
    "text",
    [
        "San Diego Zoo Nighttime Zoo",
        "Urban Air Trampoline and Adventure Park (Dublin)",
        "Altitude Trampoline Park (Oswego)",
        "PETROMASS 2026 - International Mass Spectrometry Conference",
        "AAIFF - Astana AI Film Festival",
        "Boston's Official Calendar of Events",
        "Family Fun Edmonton",
    ],
)
def test_C16_out_of_region_noise_has_no_evidence(text):
    """Every one of these actually reached the model as a candidate activity."""
    from weekend.region import has_region_evidence

    assert not has_region_evidence(text)


def test_C16_evidence_is_required_and_the_trade_is_explicit():
    """Strict by design, and only at the SOURCE layer where results are
    redundant. Empty text is kept (nothing to judge); text naming no place is
    dropped, which is what removes the year-matched global noise."""
    from weekend.region import has_region_evidence

    assert has_region_evidence("")  # nothing to judge
    assert not has_region_evidence("Summer Splash Festival")
    assert not has_region_evidence("August 2026 Calendar of Events | Empires & Puzzles")


def test_C16_scrape_filter_drops_noise_but_keeps_local():
    """Scoped at SOURCE: the noise never reaches the model."""
    from weekend.data import _clean_search_results

    out = _clean_search_results(
        [
            {"title": "Toronto Zoo | Events", "body": "Special events this August"},
            {"title": "PETROMASS - Mass Spectrometry Conference", "body": "Held abroad"},
            {"title": "Jurassic Quest", "body": "International Centre, Mississauga, ON"},
            {"title": "San Diego Zoo Nighttime Zoo", "body": "San Diego, California"},
        ],
        "Event",
    )
    assert "Toronto Zoo" in out and "Jurassic Quest" in out
    assert "PETROMASS" not in out and "San Diego" not in out


def test_C16_is_a_whitelist_not_a_blocklist():
    """The design constraint, asserted. A blocklist of everywhere-else is
    unbounded and is the C8 escape-hatch trap in a new costume."""
    from weekend.region import region_tokens

    tokens = {t.lower() for t in region_tokens()}
    for foreign in ("san diego", "dublin", "oswego", "boston", "edmonton", "california"):
        assert foreign not in tokens, (
            f"{foreign!r} is in the region token list -- those are places to KEEP, "
            f"so a foreign name there means the list has become a blocklist"
        )
    assert {"toronto", "vaughan", "mississauga"} <= tokens
