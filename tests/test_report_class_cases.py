"""One failing case per weakness class in docs/REPORT_WEAKNESS_CLASSES.md.

Stage 0 of G3 ships when the catalogue exists with a failing case per class.
Each test here asserts the CORRECT behaviour, so it fails against today's code
and passes once the class is fixed. They are marked `xfail(strict=True)`:

  → the suite stays green while the classes are open
  → when Stage 1 fixes a class the test XPASSes, and strict mode turns that into
    a FAILURE, forcing the marker to be removed

That is the structural gate. A class cannot be quietly fixed-and-forgotten, and
it cannot be quietly left open either.

Fixtures in `tests/fixtures/reports/` back these. The `wk` fixture is the real
2026-07-31 plan verbatim. The `tw` fixtures are synthetic reproductions of the
two real dated samples' shapes -- the real ones carry the user's private
timeline and are deliberately not vendored; their paths and quotes are recorded
in the catalogue instead. `test_real_samples_still_exhibit_every_class` runs the
same checks against the real files when they are present on this machine.
"""

import tomllib
from datetime import date
from pathlib import Path

import pytest

from eval import report_classes as rc

FIXTURES = Path(__file__).parent / "fixtures" / "reports"
WK_FIXTURE = FIXTURES / "wk_2026-07-31_sample.md"
TW_BRACKET = FIXTURES / "tw_2026-07-29_bracket_style.md"
TW_PROSE = FIXTURES / "tw_2026-07-27_prose_style.md"

WINDOW_START = date(2026, 7, 31)
WINDOW_END = date(2026, 8, 2)

REAL_WK = Path.home() / "Documents" / "weekend_plan_July_31_to_August_02_2026.md"
REAL_TW_DIR = Path.home() / "Documents" / "twitter_summaries"


def _wk() -> str:
    return WK_FIXTURE.read_text()


def _declared_exclusions() -> list[str]:
    """Read exclude_places from wherever TOML actually put it.

    Deliberately not via weekend.config: that path is broken (class C8) and this
    helper must report what the user WROTE, not what the program manages to see.
    """
    raw = tomllib.loads((rc.ROOT / "conf" / "weekend.toml").read_text())
    if "exclude_places" in raw:
        return raw["exclude_places"]
    for child in raw.get("children", []):
        if "exclude_places" in child:
            return child["exclude_places"]
    return []


# ---------------------------------------------------------------------------
# fixture sanity -- these must pass, or the failing cases below prove nothing
# ---------------------------------------------------------------------------


def test_fixtures_parse_into_rows():
    text = _wk()
    assert len(rc.transient_rows(text)) == 7
    assert len(rc.fixed_rows(text)) == 8
    assert "Canada Day" in text


def test_window_is_recoverable_from_the_wk_filename():
    parsed = rc.parse_window_from_wk_filename(
        Path("weekend_plan_July_31_to_August_02_2026.md")
    )
    assert parsed == (WINDOW_START, WINDOW_END)


def test_checkers_pass_on_a_clean_report():
    """Proves the checkers can go green -- they are not unconditional failures."""
    clean = (
        "### Transient / Limited-Time Events\n"
        "| Score | Event & Location | Est. Price | Duration / End Date | Weather Appr. |\n"
        "| :--- | :--- | :--- | :--- | :--- |\n"
        "| 4.8/5 | **Harbour Kite Festival** (Pier 4) | $12 | August 01, 2026 | outdoor |\n"
        "| 4.1/5 | **Clay Studio Drop-In** (Maple) | free | August 02, 2026 | indoor |\n"
        "| 3.9/5 | **Night Market** (Concord) | $5 | July 31, 2026 | outdoor |\n"
    )
    assert rc.check_wk_transient_rows_carry_a_date(clean, 2026) == []
    assert rc.check_wk_no_row_outside_window(clean, WINDOW_START, WINDOW_END) == []
    assert rc.check_wk_no_stale_event_name(clean, WINDOW_START, WINDOW_END) == []
    assert rc.check_wk_no_mandated_placeholder(clean) == []
    assert rc.check_wk_no_constant_column(clean) == []
    assert rc.check_wk_weather_label_matches_venue(clean) == []
    assert rc.check_wk_transient_rows_are_time_bounded(clean, 2026) == []
    assert rc.check_wk_no_excluded_place(clean, ["Toronto Zoo"]) == []


def test_tw_checkers_pass_on_a_clean_report():
    clean = (
        "# Twitter Timeline Summary\n\n"
        "**Period:** 2026-07-29 16:33 → 2026-07-31 12:15 UTC\n"
        "**Model:** qwen-agentworld-35b-a3b-mxfp8 (primary)\n\n"
        "## Technology\n"
        "- @alpha_dev released a build tool (Jul 30 23:21).\n"
        "- @beta_labs shipped a driver update (Jul 31 07:28).\n"
    )
    assert rc.check_tw_timestamps_are_day_qualified(clean) == []
    assert rc.check_tw_names_its_backend(clean) == []
    assert rc.check_tw_attribution_format_is_uniform([clean]) == []


# ---------------------------------------------------------------------------
# the failing cases -- one per class
# ---------------------------------------------------------------------------


def test_C1_model_prompts_render_under_production_kwargs():
    """FIXED 2026-08-02. conf/models/*.toml prompts embed unescaped JSON braces,
    so str.format() raised and weekend/prompts.py swallowed it into a replace that
    substituted nothing -- the model received the literal text `{date_range}`.
    Rendering no longer goes through str.format(), and a template that cannot be
    rendered now raises PromptRenderError instead of passing through."""
    assert rc.check_model_prompts_render() == []


def test_C1_unrenderable_template_raises_rather_than_passing_through():
    """The CLASS, not the instance: a template error must be loud."""
    from lib.prompt_render import PromptRenderError
    from weekend.prompts import _render_model_prompt

    with pytest.raises(PromptRenderError, match="unrendered placeholder"):
        _render_model_prompt("Find events for {date_range}", "t.toml:x", "ctx")


@pytest.mark.xfail(strict=True, reason="C2a DATE-DROPPED-AT-THE-LLM-BOUNDARY (tw)")
def test_C2a_tw_timestamps_are_day_qualified():
    assert rc.check_tw_timestamps_are_day_qualified(TW_BRACKET.read_text()) == []


@pytest.mark.xfail(strict=True, reason="C2b DATE-DROPPED-AT-THE-LLM-BOUNDARY (wk)")
def test_C2b_wk_transient_rows_carry_a_date():
    assert rc.check_wk_transient_rows_carry_a_date(_wk(), 2026) == []


@pytest.mark.xfail(strict=True, reason="C3 NO-RECENCY-FILTER")
def test_C3_no_row_names_an_event_outside_the_window():
    assert rc.check_wk_no_stale_event_name(_wk(), WINDOW_START, WINDOW_END) == []


@pytest.mark.xfail(strict=True, reason="C4 MANDATED-PLACEHOLDER (literal)")
def test_C4_no_cell_is_a_prompt_mandated_literal():
    assert rc.check_wk_no_mandated_placeholder(_wk()) == []


@pytest.mark.xfail(strict=True, reason="C4 MANDATED-PLACEHOLDER (constant column)")
def test_C4_no_column_is_constant_across_all_rows():
    assert rc.check_wk_no_constant_column(_wk()) == []


@pytest.mark.xfail(strict=True, reason="C5 UNVERIFIED-SEMANTIC-LABEL")
def test_C5_weather_label_matches_venue_kind():
    assert rc.check_wk_weather_label_matches_venue(_wk()) == []


def test_C6_score_column_is_not_labelled_as_reviews(tmp_path):
    """FIXED 2026-08-02. The tables were headed "Ranked by Review Score" and
    rendered a N/5 rating, but the number is weekend/scoring.py's internal
    heuristic; scrape_review_score() is not on the pipeline's call path. The
    heading now names what the number actually is.

    Asserted against RENDERED output, not by grepping source -- an earlier
    version of this test grepped for the function name and was satisfied by a
    code comment that merely mentioned it.
    """
    from weekend.output import _build_fixed_table, _build_transient_table

    fixed = _build_fixed_table(
        [{"name": "A", "location": "Toronto", "weather": "indoor", "score": 4.5}]
    )
    transient = _build_transient_table(
        [{"name": "B", "location": "Vaughan", "weather": "indoor", "score": 4.2}]
    )
    for rendered in (fixed, transient):
        assert "Review Score" not in rendered
        assert "Fit Score" in rendered
        assert "computed" in rendered


@pytest.mark.xfail(strict=True, reason="C7 CLASSIFICATION-BY-QUERY-PROVENANCE")
def test_C7_transient_rows_are_time_bounded():
    assert rc.check_wk_transient_rows_are_time_bounded(_wk(), 2026) == []


@pytest.mark.xfail(strict=True, reason="C8 UNENFORCED-USER-CONSTRAINT (report)")
def test_C8_no_excluded_place_appears_in_the_report():
    assert rc.check_wk_no_excluded_place(_wk(), _declared_exclusions()) == []


def test_C8_declared_exclusions_reach_production():
    """FIXED 2026-08-02 (C8 layer 1 of 3). `exclude_places` sat after the
    [[children]] array-of-tables, so TOML made it a key of the LAST child and
    weekend/config.py:54 `.get("exclude_places", [])` silently yielded []. Moved
    above the first table. Layers 2 (no {exclusions} placeholder in any prompt)
    and 3 (no post-parse filter) remain open -- see the report-level case below."""
    from weekend.config import EXCLUDE_PLACES

    declared = _declared_exclusions()
    assert declared, "fixture guard: conf/weekend.toml must declare exclusions"
    assert sorted(EXCLUDE_PLACES) == sorted(declared)


def test_C9_report_names_its_backend(tmp_path):
    """FIXED 2026-08-02. write_markdown now emits a provenance banner, so a
    fallback summary can never be byte-identical to a primary one."""
    from datetime import datetime, timezone

    from twitter.output import write_markdown
    from twitter.provenance import FALLBACK, PRIMARY, Provenance, SummaryText

    tweets = [{"screen_name": "a"}, {"screen_name": "b"}]
    since = datetime(2026, 7, 29, 16, 33, tzinfo=timezone.utc)
    until = datetime(2026, 7, 31, 12, 15, tzinfo=timezone.utc)
    body = "## Topic\n- a fact\n- another\n- a third"

    primary = SummaryText(body, Provenance("osaurus", "qwen-agentworld-35b", PRIMARY))
    degraded = SummaryText(
        body, Provenance("osaurus", "foundation", FALLBACK, ("primary did not answer",))
    )

    _, primary_md = write_markdown(tweets, primary, since, until, tmp_path)
    _, degraded_md = write_markdown(tweets, degraded, since, until, tmp_path / 'd')

    assert rc.check_tw_names_its_backend(primary_md) == []
    assert rc.check_tw_names_its_backend(degraded_md) == []
    assert primary_md != degraded_md, "a degraded run must not be byte-identical"
    assert "DEGRADED OUTPUT" in degraded_md
    assert "primary did not answer" in degraded_md
    assert "DEGRADED OUTPUT" not in primary_md


def test_C9_unrecorded_provenance_is_not_reported_as_primary():
    """A bare str summary means nobody recorded it -- that must read as degraded,
    not as a good run, or an unattended job silently regains the ambiguity."""
    from twitter.provenance import provenance_of

    prov = provenance_of("a plain string summary")
    assert prov.degraded
    assert "DEGRADED OUTPUT" in prov.banner()


def test_C9_checker_still_fails_on_a_report_without_provenance():
    """Prove the check can fail: the pre-fix fixture must still be caught."""
    assert rc.check_tw_names_its_backend(TW_BRACKET.read_text()) != []


@pytest.mark.xfail(strict=True, reason="C10 UNSPECIFIED-OUTPUT-CONTRACT")
def test_C10_attribution_format_is_stable_across_reports():
    texts = [TW_PROSE.read_text(), TW_BRACKET.read_text()]
    assert rc.check_tw_attribution_format_is_uniform(texts) == []


@pytest.mark.xfail(
    strict=False, reason="C11 COVERAGE-OVERSTATED -- latent, not reproduced; see catalogue"
)
def test_C11_stated_count_matches_processed_count():
    """Non-strict on purpose: the artifact does not carry the processed count, so
    this cannot be evaluated from a saved report. Do not fix C11 on the strength
    of the mechanism alone -- probe a real run first."""
    text = TW_BRACKET.read_text()
    assert "**Tweets:**" in text
    assert "processed" in text.lower(), (
        "report states only the fetched count; the count actually sent to the "
        "model (twitter/summarize.py:186) never reaches the artifact"
    )


def test_C12_eval_and_production_share_one_renderer():
    """FIXED 2026-08-02. `ev` rendered prompts with a private str.replace helper
    while production used str.format, so the eval was green on templates
    production shipped corrupted. Both now call lib.prompt_render.render_prompt,
    so the divergence is structurally impossible rather than a convention."""
    from lib import config_tasks
    from lib.prompt_render import render_prompt
    from weekend.prompts import _render_model_prompt

    # A template with the exact shape that used to diverge: literal JSON braces
    # (which broke production's str.format) plus named placeholders (which the
    # eval's private helper substituted anyway).
    template = '{"transient_events": [{"name": "str"}]} for {age_range} in {location}'
    expected = '{"transient_events": [{"name": "str"}]} for 6-13 in Vaughan'

    assert render_prompt(template, age_range="6-13", location="Vaughan") == expected
    assert _render_model_prompt(template, "t", None, age_range="6-13", location="Vaughan") == expected

    # The eval renders the same template through the same function. Its VALUES
    # differ (they come from the eval input), but no template can now render in
    # one path and fail in the other.
    eval_rendered = config_tasks._safe_format_prompt(
        template, '[{"location": "Vaughan", "target_ages": "6-13"}]'
    )
    assert eval_rendered == expected


@pytest.mark.xfail(strict=True, reason="C13 DECLARED-BUT-UNREAD-CONFIG")
def test_C13_declared_config_keys_are_read():
    assert rc.check_declared_config_keys_are_read() == []


# ---------------------------------------------------------------------------
# the real dated samples -- skipped in CI, run on the author's machine
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not REAL_WK.exists(), reason="real wk sample not on this machine")
def test_real_wk_sample_exhibits_the_catalogued_classes():
    """prove-before-claim: the fixture must not be the only place these appear."""
    text = REAL_WK.read_text()
    window = rc.parse_window_from_wk_filename(REAL_WK)
    assert window is not None
    start, end = window
    assert rc.check_wk_transient_rows_carry_a_date(text, start.year), "C2b vanished"
    assert rc.check_wk_no_stale_event_name(text, start, end), "C3 vanished"
    assert rc.check_wk_no_mandated_placeholder(text), "C4 vanished"
    assert rc.check_wk_weather_label_matches_venue(text), "C5 vanished"
    assert rc.check_wk_no_excluded_place(text, _declared_exclusions()), "C8 vanished"


@pytest.mark.skipif(not REAL_TW_DIR.is_dir(), reason="real tw samples not on this machine")
def test_real_tw_samples_exhibit_the_catalogued_classes():
    texts = [p.read_text() for p in sorted(REAL_TW_DIR.glob("*.md"))]
    if len(texts) < 2:
        pytest.skip("need two tw reports to compare attribution styles")
    assert all(rc.check_tw_timestamps_are_day_qualified(t) for t in texts), "C2a vanished"
    assert all(rc.check_tw_names_its_backend(t) for t in texts), "C9 vanished"
    assert rc.check_tw_attribution_format_is_uniform(texts), "C10 vanished"


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

