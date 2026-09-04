"""C4 family: honest values -- absence must look like absence.

Split from tests/test_report_class_fixes.py to stay under the repo's 500-line
cap. These cases all guard one invariant: a cell must carry a value the source
actually stated, or the missing sentinel -- never a fabricated constant, never
the literal word "unknown", and never a value the program already knew.

See docs/REPORT_WEAKNESS_CLASSES.md C4 / C17.
"""

from pathlib import Path

import pytest
from eval import report_classes as rc


def _render_current(transient, fixed=None):
    from weekend.output import build_markdown_tables

    return build_markdown_tables(
        "August 07 to August 09, 2026",
        "Fri 24C Clear",
        {"transient_events": list(transient)},
        list(fixed or []),
    )


@pytest.mark.parametrize("absent", ["unknown", "Unknown", "n/a", "N/A", "none", "TBD", ""])
def test_C4_the_word_unknown_never_reaches_the_table(absent):
    """The prompts now ASK the model for "unknown" instead of a fabricated
    constant, so the renderer must turn that back into the missing sentinel.
    A real run shipped `| — | unknown | — |`: the honest answer had leaked into
    the table as a word, which reads like data.
    """
    from weekend.output import _fmt_missing

    assert _fmt_missing(absent) == "—"

    rows = [
        {"name": "A", "location": "X", "price": absent, "target_ages": absent,
         "start_date": "2026-08-08", "weather": "indoor", "score": 3.0},
    ]
    rendered = _render_current(rows, list(rows))
    assert "unknown" not in rendered.lower()
    assert "n/a" not in rendered.lower()


def test_C4_a_real_value_is_never_mistaken_for_absent():
    """Prove the normaliser is not over-eager."""
    from weekend.output import _fmt_missing

    for real in ("Free", "$12", "5-12", "all ages", "indoor"):
        assert _fmt_missing(real) == real


def test_C4_every_table_uses_the_same_missing_sentinel():
    """All four renderers previously had their own fallback; the fixed markdown
    table used "" and so showed an empty cell where the transient table showed
    the sentinel. One helper now, so the tables cannot disagree."""
    import inspect

    from weekend import output

    source = inspect.getsource(output)
    # no renderer may fall back to a bare literal for these columns
    assert 'item.get("target_ages") or item.get("age_group") or' not in source
    assert 'item.get("price") or item.get("cost") or' not in source


def test_C4_location_column_also_normalises_absent_words():
    """The location was the one column the absent-word fix had not reached, so a
    real run shipped "**Union Summer** (unknown)". Fix the class, not three of
    the four columns."""
    from weekend.output import _build_transient_table

    rendered = _build_transient_table(
        [{"name": "Union Summer", "location": "unknown", "start_date": "2026-08-08", "score": 1.4}]
    )
    assert "unknown" not in rendered.lower()
    assert "()" not in rendered, "an absent location must not leave empty parentheses"
    assert "**Union Summer**" in rendered


def test_C4_a_real_location_is_still_rendered():
    from weekend.output import _build_transient_table

    rendered = _build_transient_table(
        [{"name": "TD Sunday", "location": "MOCA", "start_date": "2026-08-09", "score": 2.2}]
    )
    assert "**TD Sunday** (MOCA)" in rendered


def test_C4_a_column_that_merely_echoes_config_is_caught():
    """The SHAPE of the whole C4 class, asserted at last.

    Each recurrence was caught by eye, after shipping, one at a time:
      "2-3 hours" in every Duration cell, "$18-35 per child or free" in every
      Price cell, and finally "6-13" -- the configured FAMILY age range -- in
      every Target Age(s) cell, a column that means "the ages this venue is for".

    A column whose every value equals something the program already knew carries
    no information about the world, whatever the source of that value.
    """
    from weekend.config import AGE_RANGE

    echoing = (
        "### Fixed / Year-Round Activities\n"
        "| Activity & Location | Target Age(s) | Weather Appropriateness |\n"
        "| :--- | :--- | :--- |\n"
        f"| **A** (Vaughan) | {AGE_RANGE} | indoor |\n"
        f"| **B** (Toronto) | {AGE_RANGE} | indoor |\n"
        f"| **C** (Vaughan) | {AGE_RANGE} | indoor |\n"
    )
    assert rc.check_wk_no_column_echoes_config(echoing, rc.configured_echo_values())

    varied = echoing.replace(f"| {AGE_RANGE} | indoor |\n| **B**", "| 1-12 | indoor |\n| **B**")
    assert rc.check_wk_no_column_echoes_config(varied, rc.configured_echo_values()) == []


def test_C4_config_echo_check_ignores_an_honestly_blank_column():
    """A column of sentinels is the CORRECT output when the source is silent --
    it must not be reported as a config echo."""
    blank = (
        "### Fixed / Year-Round Activities\n"
        "| Activity & Location | Target Age(s) | Weather Appropriateness |\n"
        "| :--- | :--- | :--- |\n"
        "| **A** (Vaughan) | — | indoor |\n"
        "| **B** (Toronto) | — | indoor |\n"
    )
    assert rc.check_wk_no_column_echoes_config(blank, rc.configured_echo_values()) == []


# --------------------------------------------------------------------------
# C18 -- a hollow plan passes every other check
# --------------------------------------------------------------------------


def test_C18_an_empty_transient_table_is_a_failure():
    """The floor the other checks stand on.

    An empty table satisfies every content rule trivially -- no fabricated
    constant, no stale date, no excluded venue, no mislabelled weather -- so a
    run producing ZERO events was reported as "every content check passes". The
    checks were green and the feature was useless.
    """
    hollow = (
        "# Weekend Plan: August 07 to August 09, 2026\n\n"
        "### Fixed / Year-Round Activities\n"
        "| Activity & Location | Weather Appropriateness |\n| :--- | :--- |\n"
        "| **A** (Vaughan) | indoor |\n| **B** (Toronto) | indoor |\n"
        "| **C** (Vaughan) | indoor |\n\n"
        "### Transient / Limited-Time Events\n"
        "| Event & Location | Dates |\n| :--- | :--- |\n"
    )
    failures = rc.check_wk_plan_is_not_hollow(hollow)
    assert failures and "hollow" in failures[0]

    # ...and every other content check is perfectly happy with it
    assert rc.check_wk_no_mandated_placeholder(hollow) == []
    assert rc.check_wk_no_aggregator_rows(hollow) == []
    assert rc.check_wk_transient_rows_carry_a_date(hollow, 2026) == []


def test_C18_a_populated_plan_passes():
    """Prove the floor can be met, not just tripped."""
    populated = (
        "### Fixed / Year-Round Activities\n"
        "| Activity & Location | Weather Appropriateness |\n| :--- | :--- |\n"
        "| **A** (Vaughan) | indoor |\n| **B** (Toronto) | indoor |\n"
        "| **C** (Vaughan) | indoor |\n\n"
        "### Transient / Limited-Time Events\n"
        "| Event & Location | Dates |\n| :--- | :--- |\n"
        "| **Kite Festival** (Vaughan) | 2026-08-08 |\n"
    )
    assert rc.check_wk_plan_is_not_hollow(populated) == []


def test_C18_reports_the_fixed_table_being_thin_too():
    thin = (
        "### Fixed / Year-Round Activities\n"
        "| Activity & Location | Weather Appropriateness |\n| :--- | :--- |\n"
        "| **A** (Vaughan) | indoor |\n\n"
        "### Transient / Limited-Time Events\n"
        "| Event & Location | Dates |\n| :--- | :--- |\n"
        "| **Kite Festival** (Vaughan) | 2026-08-08 |\n"
    )
    failures = rc.check_wk_plan_is_not_hollow(thin)
    assert failures and "fixed" in failures[0]


# --------------------------------------------------------------------------
# C19 -- FABRICATED-ROW
# --------------------------------------------------------------------------

FABRICATED = Path(__file__).parent / "fixtures" / "reports" / "wk_fabricated_transient_sample.md"


def test_C19_every_fabricated_row_in_the_real_artifact_is_caught():
    """Built from a REAL shipped artifact, kept as the fixture.

    The draft phase was starved of in-window candidates and the model invented
    five events rather than return fewer. A constraint a component cannot satisfy
    honestly gets satisfied dishonestly.
    """
    failures = rc.check_wk_no_fabricated_rows(FABRICATED.read_text())
    assert len(failures) == 5, f"expected all five invented rows, got {len(failures)}"
    for fragment in ("Indoor Play Centre", "Trampoline Park", "Zoo Adventure",
                     "Outdoor Park Day", "Sports Day"):
        assert any(fragment in f for f in failures), f"{fragment} not caught"


def test_C19_is_why_it_matters_the_other_checks_pass_on_it():
    """An invented row corresponds to nothing, so it satisfies almost everything.

    Of the checks below, only the constant-column one caught anything -- and
    "Central Park" slips the in-region check precisely because the venue does not
    exist. This is the assertion that documents the gap C19 closes.
    """
    text = FABRICATED.read_text()
    assert rc.check_wk_no_mandated_placeholder(text) == []
    assert rc.check_wk_no_aggregator_rows(text) == []
    assert rc.check_wk_rows_are_in_region(text) == []
    assert rc.check_wk_plan_is_not_hollow(text) == []
    # the two that do fire
    assert rc.check_wk_no_constant_column(text) != []
    assert rc.check_wk_no_fabricated_rows(text) != []


def test_C19_does_not_flag_real_venues():
    """Conservatism, checked against real shipped artifacts. A row is only
    reported when BOTH its name and location are bare categories, so a real venue
    with a plain name survives on the strength of its location."""
    historical = (Path(__file__).parent / "fixtures" / "reports"
                  / "wk_2026-07-31_sample.md").read_text()
    assert rc.check_wk_no_fabricated_rows(historical) == []

    real_rows = (
        "### Transient / Limited-Time Events\n"
        "| Event & Location | Dates |\n| :--- | :--- |\n"
        "| **Grange Festival** (Toronto) | 2026-08-07 |\n"
        "| **TD Jerkfest Toronto** (Toronto) | 2026-08-07 |\n"
        "| **Sky Zone Trampoline Park** (Vaughan) | 2026-08-08 |\n"
        "| **The Bubble** (Toronto & Vaughan) | 2026-08-08 |\n"
    )
    assert rc.check_wk_no_fabricated_rows(real_rows) == []
