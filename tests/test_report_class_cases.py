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


@pytest.mark.xfail(strict=True, reason="C1 SILENT-TEMPLATE-SUBSTITUTION-FAILURE")
def test_C1_model_prompts_render_under_production_kwargs():
    assert rc.check_model_prompts_render() == []


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


@pytest.mark.xfail(strict=True, reason="C6 PROVENANCE-LABEL-NOT-BACKED-BY-DATA")
def test_C6_review_score_heading_requires_review_data():
    """If a table claims "Review Score", the scraper must be on the call path."""
    assert "Ranked by Review Score" in _wk()
    live = (rc.ROOT / "weekend" / "llm.py").read_text()
    live += (rc.ROOT / "weekend" / "output.py").read_text()
    assert "scrape_review_score" in live, (
        "report claims 'Review Score' but weekend/data.py:scrape_review_score is "
        "never called by the pipeline -- the number is _score_item's heuristic"
    )


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


@pytest.mark.xfail(strict=True, reason="C9 BACKEND-PROVENANCE-DISCARDED")
def test_C9_report_names_its_backend():
    assert rc.check_tw_names_its_backend(TW_BRACKET.read_text()) == []


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


@pytest.mark.xfail(strict=True, reason="C12 EVAL-DOES-NOT-EXERCISE-PRODUCTION")
def test_C12_eval_and_production_render_prompts_identically():
    """`ev` uses _safe_format_prompt (str.replace) while production uses
    str.format, so the eval succeeds on templates production cannot render."""
    from lib.config_tasks import _safe_format_prompt

    template = tomllib.loads((rc.ROOT / "conf" / "models" / "qwen.toml").read_text())[
        "prompts"
    ]["weekend_transient"]
    eval_rendered = _safe_format_prompt(template, "[]")
    try:
        prod_rendered = template.format(**rc.PRODUCTION_PROMPT_KWARGS)
    except (KeyError, IndexError, ValueError) as exc:
        pytest.fail(f"production renderer raises where the eval renderer does not: {exc}")
    assert eval_rendered == prod_rendered


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
