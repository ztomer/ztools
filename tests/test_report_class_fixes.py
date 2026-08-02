"""Post-fix cases: proof that a weakness class is actually CLOSED.

Split from tests/test_report_class_cases.py to stay under the repo's 500-line
cap. That module holds the cases asserted against the FROZEN historical
artifacts -- evidence a class existed, and proof each checker can still fail.
This module renders through the CURRENT pipeline and asserts the checkers pass,
which is the only thing that can show a fix landed.

See docs/REPORT_WEAKNESS_CLASSES.md.
"""

import tomllib
from datetime import date
from pathlib import Path

import pytest  # noqa: F401  (used by cases added here over time)

from eval import report_classes as rc

FIXTURES = Path(__file__).parent / "fixtures" / "reports"
WINDOW_START = date(2026, 7, 31)
WINDOW_END = date(2026, 8, 2)


def _declared_exclusions() -> list[str]:
    raw = tomllib.loads((rc.ROOT / "conf" / "weekend.toml").read_text())
    if "exclude_places" in raw:
        return raw["exclude_places"]
    for child in raw.get("children", []):
        if "exclude_places" in child:
            return child["exclude_places"]
    return []


# ---------------------------------------------------------------------------
# Post-fix pipeline cases.
#
# The cases above assert against the FROZEN historical fixture, which is the
# evidence that a class existed and the proof that a checker can still fail. But
# a frozen artifact can never become compliant, so it cannot show that a fix
# landed. These render representative items through the CURRENT renderer and
# assert the checkers pass -- that is what proves the class is closed.
# ---------------------------------------------------------------------------


def _render_current(transient, fixed=None):
    """Render items through the real pipeline renderer."""
    from weekend.output import build_markdown_tables

    return build_markdown_tables(
        "August 07 to August 09, 2026",
        "Fri 24C Clear",
        {"transient_events": list(transient)},
        list(fixed or []),
    )


def _post_fix_items():
    """What the fixed pipeline yields: real dates, honest blanks for unknowns."""
    return [
        {"name": "Harbour Kite Festival", "location": "Pier 4", "price": "$12",
         "start_date": "2026-08-07", "end_date": "2026-08-07",
         "target_ages": "5-12", "weather": "outdoor", "day": "Friday"},
        {"name": "Clay Studio Drop-In", "location": "Maple", "price": "",
         "start_date": "2026-08-08", "end_date": "", "target_ages": "",
         "weather": "indoor", "day": "Saturday"},
        {"name": "Night Market", "location": "Concord", "price": "free",
         "start_date": "2026-08-09", "end_date": "2026-08-09",
         "target_ages": "6-13", "weather": "both", "day": "Sunday"},
    ]


def test_C2b_current_renderer_puts_real_dates_in_the_date_column():
    md = _render_current(_post_fix_items())
    assert rc.check_wk_transient_rows_carry_a_date(md, 2026) == []
    assert "Duration / End Date" not in md
    assert "2-3 hours" not in md


def test_C4_current_renderer_emits_no_mandated_literal_and_no_constant_column():
    md = _render_current(_post_fix_items())
    assert rc.check_wk_no_mandated_placeholder(md) == []
    assert rc.check_wk_no_constant_column(md) == []


def test_C4_unknown_value_renders_as_an_honest_blank_not_a_plausible_one():
    """The point of C4: absence must look like absence."""
    md = _render_current(_post_fix_items())
    assert "—" in md, "a row with no price/ages must render the missing sentinel"


def test_C5_current_pipeline_corrects_an_impossible_weather_label():
    from weekend.enforce import correct_weather_labels

    items = [{"name": "Sky Zone Trampoline Park", "location": "Toronto",
              "weather": "outdoor", "start_date": "2026-08-08"}]
    fixed, notes = correct_weather_labels(items)
    assert fixed[0]["weather"] == "indoor"
    assert notes and "trampoline park" in notes[0]
    assert rc.check_wk_weather_label_matches_venue(_render_current(fixed)) == []


def test_C7_current_renderer_flags_an_undated_transient_row():
    """C7 is NOT closed: an evergreen row still reaches the transient table. The
    date column now exposes it as blank instead of showing a fake duration."""
    evergreen = [{"name": "Discover family fun in Vaughan", "location": "Various",
                  "weather": "both", "day": "Saturday"}]
    md = _render_current(evergreen)
    assert rc.check_wk_transient_rows_are_time_bounded(md, 2026) != [], (
        "an undated row must still be reported -- classifying it is C7's job"
    )


def test_C8_current_pipeline_drops_excluded_places():
    from weekend.enforce import drop_excluded_places

    items = _post_fix_items() + [
        {"name": "Canada Day at Your Toronto Zoo", "location": "Toronto Zoo",
         "weather": "outdoor", "start_date": "2026-08-08"}
    ]
    kept, notes = drop_excluded_places(items, _declared_exclusions())
    assert notes and "Toronto Zoo" in notes[0]
    assert rc.check_wk_no_excluded_place(_render_current(kept), _declared_exclusions()) == []


def test_C3_current_pipeline_drops_a_dated_event_outside_the_window():
    from datetime import date as _date

    from weekend.enforce import drop_events_outside_window

    items = _post_fix_items() + [
        {"name": "Canada Day Fireworks", "location": "Downsview",
         "start_date": "2026-07-01", "end_date": "2026-07-01", "weather": "outdoor"}
    ]
    kept, notes = drop_events_outside_window(items, _date(2026, 8, 7), _date(2026, 8, 9))
    assert notes and "2026-07-01" in notes[0]
    assert "Canada Day" not in _render_current(kept)


def test_C8_typographic_apostrophe_does_not_defeat_the_exclusion():
    """Regression: found by a REAL wk run on 2026-08-02, after C8 had already
    been declared fixed. conf/weekend.toml says "Ripley's" (U+0027); the scraper
    returned "Ripley’s Aquarium of Canada" (U+2019), so the substring match
    failed and an excluded venue shipped.

    The checker missed it too, because it normalised the same (wrong) way -- an
    instrument that shares the bug reports PASS. Both now call one shared
    normalizer, so they cannot drift apart again.
    """
    from weekend.enforce import drop_excluded_places, normalize_for_match

    items = [
        {"name": "Ripley’s Aquarium of Canada", "location": "Toronto, Ontario"},
        {"name": "Union Summer", "location": "Union Station Plaza"},
    ]
    kept, notes = drop_excluded_places(items, ["Ripley's"])
    assert len(notes) == 1 and "Ripley" in notes[0]
    assert [i["name"] for i in kept] == ["Union Summer"]

    # The checker must agree with the enforcement on the same input.
    rendered = _render_current([], items)
    assert rc.check_wk_no_excluded_place(rendered, ["Ripley's"]) != []
    assert normalize_for_match("Ripley’s") == normalize_for_match("Ripley's")


def test_C4_an_honest_blank_column_is_not_reported_as_fabricated():
    """The C4 fix makes unknown values render as the missing sentinel, so a
    column of blanks is CORRECT. Flagging it would punish the pipeline for
    telling the truth -- a real run tripped exactly that false positive."""
    blank = (
        "### Transient / Limited-Time Events\n"
        "| Event & Location | Est. Price | Dates |\n| :--- | :--- | :--- |\n"
        "| **A** (X) | — | 2026-08-07 |\n| **B** (Y) | — | 2026-08-08 |\n"
        "| **C** (Z) | — | 2026-08-09 |\n"
    )
    assert rc.check_wk_no_constant_column(blank) == []

    fabricated = blank.replace("—", "$20-30 per child or free")
    assert rc.check_wk_no_constant_column(fabricated) != [], (
        "a repeated NON-empty value is still the C4 defect"
    )

# ---------------------------------------------------------------------------
# C8 class-level regression: NAME-MATCHED-BY-CONTAINMENT
# ---------------------------------------------------------------------------


def test_C8_zero_excluded_venues_in_the_output_not_merely_one_drop():
    """The assertion that would have caught the real miss.

    C8 was twice declared fixed on the evidence that SOME exclusion fired. One
    venue dropping does not license "no excluded venue shipped" -- and a real
    2026-08-07 run shipped "Sky Zone Trampoline Park (Vaughan/Toronto)" while
    "LEGOLAND Discovery Centre Toronto" was being dropped in the same run. The
    bar is ZERO excluded venues in the rendered output.
    """
    from weekend.enforce import drop_excluded_places

    excluded = _declared_exclusions()
    scraped = [
        {"name": "Sky Zone Trampoline Park", "location": "Vaughan/Toronto"},
        {"name": "LEGOLAND Discovery Centre Toronto", "location": "Vaughan Mills"},
        {"name": "Ripley’s Aquarium of Canada", "location": "Toronto, Ontario"},
        {"name": "Canada Day at Your Toronto Zoo", "location": "Toronto Zoo"},
        {"name": "Harbour Kite Festival", "location": "Pier 4"},
    ]
    kept, notes = drop_excluded_places(scraped, excluded)

    assert [i["name"] for i in kept] == ["Harbour Kite Festival"]
    assert len(notes) == 4
    assert rc.check_wk_no_excluded_place(_render_current([], kept), excluded) == []


def test_C8_matcher_survives_reordering_interpolation_and_punctuation():
    """The CLASS: the config's wording is not a contiguous substring of the
    scraped wording. Each row below is a variant containment silently missed."""
    from weekend.enforce import matches_exclusion

    should_match = [
        ("Sky Zone Toronto", "Sky Zone Trampoline Park (Vaughan/Toronto)"),  # interpolated
        ("Canada's Wonderland", "Wonderland Canada thrill rides"),  # reordered
        ("Ripley's", "Ripley’s Aquarium of Canada"),  # U+2019
        ("Museum of Illusions", "Illusions Museum Toronto"),  # reordered
        ("Royal Ontario Museum (ROM)", "Royal Ontario Museum"),  # parenthetical
    ]
    for entry, scraped in should_match:
        assert matches_exclusion(entry, scraped), f"{entry!r} should match {scraped!r}"


def test_C8_matcher_stays_conservative_and_does_not_over_drop():
    """All tokens are required, so a shared word is not enough to drop a row."""
    from weekend.enforce import matches_exclusion

    should_not_match = [
        ("Toronto Zoo", "Toronto Islands ferry"),
        ("Toronto Islands", "Toronto Zoo"),
        ("CN Tower", "Tower of London exhibit"),
        ("Little Canada", "Canada Day at the Zoo"),
    ]
    for entry, scraped in should_not_match:
        assert not matches_exclusion(entry, scraped), f"{entry!r} must NOT match {scraped!r}"


def test_C8_checker_and_enforcement_agree_on_the_same_input():
    """A checker that cannot catch what the enforcement misses is worse than
    none -- it manufactures confidence. They now share one matcher."""
    from weekend.enforce import drop_excluded_places

    excluded = ["Sky Zone Toronto"]
    row = [{"name": "Sky Zone Trampoline Park", "location": "Vaughan/Toronto"}]

    kept, notes = drop_excluded_places(list(row), excluded)
    assert notes and not kept
    # and the checker flags the same row when it is NOT dropped
    assert rc.check_wk_no_excluded_place(_render_current([], row), excluded) != []
