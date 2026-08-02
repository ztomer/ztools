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

# ---------------------------------------------------------------------------
# C14 AGGREGATOR-PAGE-AS-ACTIVITY + the Day/dates consistency rule
#
# All four cases below are built from ONE row that really shipped, which the
# user rejected with "this specifically means nothing":
#
#   Vaughan Events & Activities Guides for Kids & Families
#   (Vaughan Public Libraries, Vaughan, ON, Canada)
#   | 6-13 | — | 2026-08-04 → 2026-08-07 | Saturday | outdoor |
#
# It carries three independent defects, and the row is kept intact here so a
# regression in any one of them is caught by the row that produced it.
# ---------------------------------------------------------------------------

SHIPPED_AGGREGATOR_ROW = (
    "### Transient / Limited-Time Events\n"
    "| Score | Event & Location | Target Age(s) | Est. Price | Dates | Day | Weather Appr. |\n"
    "| :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n"
    "| * 2.0/5 | **Vaughan Events & Activities Guides for Kids & Families** "
    "(Vaughan Public Libraries, Vaughan, ON, Canada) | 6-13 | — | "
    "2026-08-04 → 2026-08-07 | Saturday | outdoor |\n"
)


def test_C14_a_directory_page_is_not_an_activity():
    """You cannot attend a guide."""
    failures = rc.check_wk_no_aggregator_rows(SHIPPED_AGGREGATOR_ROW)
    assert failures and "directory page" in failures[0]


@pytest.mark.parametrize(
    "name",
    [
        "Things to do in Vaughan with kids",
        "What's On in Toronto This Weekend",
        "Your Guide to Summer Fun",
        "August 2026 Archives",
        "Toronto Event Calendar",
        "Top 10 Family Attractions",
    ],
)
def test_C14_catches_the_common_aggregator_shapes(name):
    row = SHIPPED_AGGREGATOR_ROW.replace(
        "Vaughan Events & Activities Guides for Kids & Families", name
    )
    assert rc.check_wk_no_aggregator_rows(row), f"{name!r} should read as a directory page"


def test_C14_does_not_flag_a_real_activity():
    """Conservative: a genuine event must not be mistaken for a directory."""
    for name in ("Harbour Kite Festival", "Jurassic Quest", "TD Community Sunday at MOCA"):
        row = SHIPPED_AGGREGATOR_ROW.replace(
            "Vaughan Events & Activities Guides for Kids & Families", name
        )
        assert rc.check_wk_no_aggregator_rows(row) == [], f"{name!r} is a real activity"


def test_day_must_fall_within_the_rows_own_dates():
    """Pure internal consistency -- no model needed, code should never ship this."""
    failures = rc.check_wk_day_matches_dates(SHIPPED_AGGREGATOR_ROW, 2026)
    assert failures and "Saturday" in failures[0]

    from datetime import date as _date

    from weekend.enforce import reconcile_day_with_dates

    items = [
        {
            "name": "Vaughan Events Guide",
            "start_date": "2026-08-04",
            "end_date": "2026-08-07",
            "day": "Saturday",
        }
    ]
    fixed, notes = reconcile_day_with_dates(items, _date(2026, 8, 7), _date(2026, 8, 9))
    assert notes and "not within" in notes[0]
    assert fixed[0]["day"] != "Saturday"


def test_day_check_passes_when_the_row_agrees_with_itself():
    """Prove the check can go green."""
    good = SHIPPED_AGGREGATOR_ROW.replace("2026-08-04 → 2026-08-07", "2026-08-08").replace(
        "| Saturday |", "| Saturday |"
    )
    assert rc.check_wk_day_matches_dates(good, 2026) == []


def test_C5_marker_list_is_shared_with_enforcement_not_copied():
    """The C8b lesson applied to C5: one list, so checker and enforcement agree.

    'Vaughan Public Libraries' was labelled outdoor and the singular-only marker
    'library' missed it; the checker had its own copy of the list, so it agreed
    with the bug instead of catching it.
    """
    from weekend.enforce import INDOOR_MARKERS

    assert rc._indoor_markers() is INDOOR_MARKERS
    assert rc.check_wk_weather_label_matches_venue(SHIPPED_AGGREGATOR_ROW), (
        "a public library labelled 'outdoor' must be caught"
    )


def test_C3_a_long_running_exhibition_that_spans_the_weekend_is_in_the_plan():
    """Overlap, not endpoint containment.

    A real run produced "Monet: The Immersive Experience | 2026-06-29 →
    2026-08-16" for a 2026-08-07..09 plan. That is CORRECT -- it is on all
    weekend. The checker tested each endpoint for containment, so it reported a
    correct row as a failure while the enforcement (which used overlap) kept it.
    Fifth checker in this project to be wrong from a second mental model; both
    now call weekend.enforce.window_overlap.
    """
    from datetime import date as _date

    from weekend.enforce import drop_events_outside_window, window_overlap

    spans = {"name": "Monet", "start_date": "2026-06-29", "end_date": "2026-08-16"}
    outside = {"name": "Canada Day", "start_date": "2026-07-01", "end_date": "2026-07-01"}
    start, end = _date(2026, 8, 7), _date(2026, 8, 9)

    assert window_overlap(spans, start, end) is True
    assert window_overlap(outside, start, end) is False
    assert window_overlap({"name": "no dates"}, start, end) is None

    kept, notes = drop_events_outside_window([spans, outside], start, end)
    assert [i["name"] for i in kept] == ["Monet"]
    assert len(notes) == 1

    rendered = _render_current(kept)
    assert rc.check_wk_no_row_outside_window(rendered, start, end) == []


def test_C3_checker_and_enforcement_agree_on_the_same_rows():
    """They disagreed once; assert they cannot again."""
    from datetime import date as _date

    from weekend.enforce import drop_events_outside_window

    start, end = _date(2026, 8, 7), _date(2026, 8, 9)
    rows = [
        {"name": "Spans", "start_date": "2026-06-29", "end_date": "2026-08-16"},
        {"name": "Inside", "start_date": "2026-08-08", "end_date": "2026-08-08"},
        {"name": "Before", "start_date": "2026-07-01", "end_date": "2026-07-02"},
        {"name": "After", "start_date": "2026-09-01", "end_date": "2026-09-02"},
    ]
    kept, _ = drop_events_outside_window(list(rows), start, end)
    assert rc.check_wk_no_row_outside_window(_render_current(kept), start, end) == []
    # and everything the enforcement dropped IS reported by the checker
    dropped = [r for r in rows if r["name"] not in {k["name"] for k in kept}]
    assert rc.check_wk_no_row_outside_window(_render_current(dropped), start, end)
