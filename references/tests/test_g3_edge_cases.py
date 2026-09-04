"""Edge-case coverage for the modules added during the G3 quality work.

These are not coverage padding: every case below exercises a branch that decides
whether a real run fails loudly, degrades honestly, or silently ships something
wrong. The coverage gate only runs on push, so this whole session's "gates green"
reports were pre-commit only -- these are the branches that gap left untested.
"""

from datetime import date
from unittest.mock import Mock, patch

import pytest

# --------------------------------------------------------------------------
# lib/prompt_render -- the three ways a template refuses to render
# --------------------------------------------------------------------------


def test_render_rejects_a_non_string_template():
    from lib.prompt_render import PromptRenderError, render_prompt

    with pytest.raises(PromptRenderError, match="must be str"):
        render_prompt(None, template_id="t.toml:x")


def test_render_requires_a_positional_value_when_the_template_has_a_slot():
    """Silently rendering "{}" as empty is how the predecessor shipped a prompt
    asking the model to work from a list that was not there."""
    from lib.prompt_render import PromptRenderError, render_prompt

    with pytest.raises(PromptRenderError, match="positional"):
        render_prompt("Use this list: {}", template_id="t.toml:x")


def test_render_rejects_a_positional_value_the_template_cannot_use():
    """A caller passing context to a named-placeholder template has misread the
    template; saying so beats dropping the context on the floor."""
    from lib.prompt_render import PromptRenderError, render_prompt

    with pytest.raises(PromptRenderError, match="no '\\{\\}' slot"):
        render_prompt("Events for {date_range}", positional="ctx", date_range="Aug")


# --------------------------------------------------------------------------
# weekend/followup -- the real HTML path
# --------------------------------------------------------------------------


def test_fetch_page_text_strips_chrome_and_keeps_content():
    """Nav/script/footer text is not event data; keeping it would feed the model
    a page of share buttons, which is what the first real run produced."""
    from weekend.followup import fetch_page_text

    html = """
      <html><head><style>.x{color:red}</style></head>
      <body>
        <nav>Home Toronto Experience</nav>
        <script>trackEverything()</script>
        <main><p>TAIWANfest at Harbourfront Centre, August 28-30.</p></main>
        <footer>Click to share on Facebook</footer>
      </body></html>
    """
    resp = Mock(text=html)
    resp.raise_for_status = Mock()
    with patch("requests.get", return_value=resp):
        text = fetch_page_text("https://example.test/events")

    assert "TAIWANfest" in text and "August 28-30" in text
    assert "trackEverything" not in text
    assert "Click to share" not in text
    assert "Home Toronto Experience" not in text


def test_fetch_page_text_truncates_to_the_budget():
    from weekend.followup import fetch_page_text

    resp = Mock(text="<html><body>" + ("event " * 5000) + "</body></html>")
    resp.raise_for_status = Mock()
    with patch("requests.get", return_value=resp):
        assert len(fetch_page_text("https://x", max_chars=200)) == 200


def test_fetch_page_text_returns_empty_on_an_http_error():
    """A scheduled run degrades to fewer events; it never aborts on one site."""
    from weekend.followup import fetch_page_text

    resp = Mock()
    resp.raise_for_status = Mock(side_effect=RuntimeError("500"))
    with patch("requests.get", return_value=resp):
        assert fetch_page_text("https://x") == ""


# --------------------------------------------------------------------------
# weekend/enforce -- date parsing and the reconciliation edges
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value,expected",
    [
        ("2026-08-09", date(2026, 8, 9)),
        ("August 9", date(2026, 8, 9)),
        ("Aug 9", date(2026, 8, 9)),      # scraped listings abbreviate
        ("Aug. 9", date(2026, 8, 9)),
        ("Sept 12", date(2026, 9, 12)),
        ("2026-02-30", None),      # well-formed but not a real date
        ("February 30", None),     # ditto, via the month-name branch
        ("next weekend", None),
        ("", None),
    ],
)
def test_parse_any_date_handles_real_and_impossible_dates(value, expected):
    from weekend.enforce import parse_any_date

    assert parse_any_date(value, 2026) == expected


def test_matches_exclusion_ignores_an_empty_entry():
    """A blank line in exclude_places must not match every row."""
    from weekend.enforce import matches_exclusion

    assert not matches_exclusion("", "Toronto Zoo")
    assert not matches_exclusion("   ", "Toronto Zoo")


def test_reconcile_day_swaps_a_reversed_range():
    """A model that emits end before start must not silently produce an empty
    day set -- the row still has to be judged."""
    from weekend.enforce import reconcile_day_with_dates

    items = [{"name": "Backwards", "start_date": "2026-08-09", "end_date": "2026-08-08",
              "day": "Tuesday"}]
    fixed, notes = reconcile_day_with_dates(items, date(2026, 8, 7), date(2026, 8, 9))
    assert notes and fixed[0]["day"] in ("Saturday", "Sunday", "")


def test_reconcile_day_leaves_a_row_outside_the_window_alone():
    """That row belongs to drop_events_outside_window; correcting its day here
    would hide the fact that it should not be in the plan at all."""
    from weekend.enforce import reconcile_day_with_dates

    items = [{"name": "Old", "start_date": "2026-07-01", "end_date": "2026-07-01",
              "day": "Wednesday"}]
    fixed, notes = reconcile_day_with_dates(items, date(2026, 8, 7), date(2026, 8, 9))
    assert notes == []
    assert fixed[0]["day"] == "Wednesday"


def test_reconcile_day_clears_rather_than_guessing_across_a_multi_day_range():
    from weekend.enforce import reconcile_day_with_dates

    items = [{"name": "Long", "start_date": "2026-08-07", "end_date": "2026-08-09",
              "day": "Monday"}]
    fixed, notes = reconcile_day_with_dates(items, date(2026, 8, 7), date(2026, 8, 9))
    assert fixed[0]["day"] == ""
    assert notes and "cleared" in notes[0]


def test_window_overlap_handles_an_end_date_with_no_start():
    from weekend.enforce import window_overlap

    start, end = date(2026, 8, 7), date(2026, 8, 9)
    assert window_overlap({"end_date": "2026-08-08"}, start, end) is True
    assert window_overlap({"end_date": "2026-07-01"}, start, end) is False


def test_window_overlap_swaps_a_reversed_range():
    from weekend.enforce import window_overlap

    reversed_row = {"start_date": "2026-08-16", "end_date": "2026-06-29"}
    assert window_overlap(reversed_row, date(2026, 8, 7), date(2026, 8, 9)) is True


# --------------------------------------------------------------------------
# weekend/phases -- the extract batching and signal telemetry
# --------------------------------------------------------------------------


def test_extract_signals_survive_a_corrupt_cache(tmp_path):
    """Telemetry must never take a run down: a truncated signals file means
    'no learned batch size', not a crash on a scheduled morning."""
    from weekend import llm, phases

    bad = tmp_path / "extract_signals.json"
    bad.write_text("{ this is not json")
    with patch.object(llm, "EXTRACT_SIGNALS_PATH", bad):
        assert phases._load_extract_signals() == {}


def test_extract_signals_are_written_where_they_can_be_reloaded(tmp_path):
    from weekend import llm, phases

    path = tmp_path / "nested" / "extract_signals.json"
    with patch.object(llm, "EXTRACT_SIGNALS_PATH", path):
        phases._save_extract_signals({"model": {"events": {"batch_size": 4}}})
        assert phases._load_extract_signals() == {"model": {"events": {"batch_size": 4}}}


def test_extract_returns_the_raw_text_when_there_is_nothing_to_extract():
    """An empty corpus must pass through, not become the string 'None'."""
    from weekend import phases

    assert phases.extract_sources("", "events") == ""


def test_extract_grows_its_batch_after_a_streak_of_successes(tmp_path):
    """The adaptive batch size is persisted telemetry; if it never grows the
    extract phase stays needlessly slow on every future run."""
    from weekend import llm, phases

    signals = {}
    path = tmp_path / "sig.json"
    lines = "\n".join(f"- Event {i}: in Toronto" for i in range(24))
    with (
        patch.object(llm, "EXTRACT_SIGNALS_PATH", path),
        patch.object(phases, "_load_extract_signals", return_value=signals),
        patch.object(phases, "_save_extract_signals") as save,
        patch.object(llm, "_call_llm", return_value="- Event: Toronto"),
    ):
        phases.extract_sources(lines, "events", model_name="m")
    assert save.called, "a successful streak must persist the larger batch size"


# --------------------------------------------------------------------------
# eval/report_classes -- parsing edges the checkers depend on
# --------------------------------------------------------------------------


def test_cell_lookup_returns_empty_for_an_absent_column():
    from eval import report_classes as rc

    assert rc._cell({"Score": "1"}, "weather") == ""


@pytest.mark.parametrize(
    "name",
    ["weekend_plan.md", "weekend_plan_Notamonth_31_to_August_02_2026.md",
     "weekend_plan_February_30_to_March_01_2026.md"],
)
def test_unparseable_wk_filenames_return_none(name):
    from pathlib import Path

    from eval import report_classes as rc

    assert rc.parse_window_from_wk_filename(Path(name)) is None


def test_tw_period_header_is_optional():
    from eval import report_classes as rc

    assert rc.parse_window_from_tw_report("# Summary\nno period line") is None


def test_find_dates_in_skips_impossible_dates():
    from eval import report_classes as rc

    assert rc.find_dates_in("2026-13-45 and February 30", 2026) == []
    assert rc.find_dates_in("2026-08-09", 2026) == [date(2026, 8, 9)]


def test_checkers_ignore_rows_without_the_column_they_judge():
    """A table missing a column is not a violation of that column's rule."""
    from eval import report_classes as rc

    no_weather = (
        "### Transient / Limited-Time Events\n"
        "| Event & Location | Dates |\n| :--- | :--- |\n"
        "| **Indoor Playground** (Vaughan) | 2026-08-09 |\n"
    )
    assert rc.check_wk_weather_label_matches_venue(no_weather) == []
    assert rc.check_wk_day_matches_dates(no_weather, 2026) == []


# --------------------------------------------------------------------------
# eval/report_classes_tw
# --------------------------------------------------------------------------


def test_single_day_tw_report_allows_bare_times():
    """C2a only bites across a multi-day window; a bare time is unambiguous when
    the report covers one day."""
    from eval import report_classes as rc

    one_day = (
        "# Twitter Timeline Summary\n\n"
        "**Period:** 2026-07-31 08:00 → 2026-07-31 18:00 UTC\n\n"
        "- @a said something (@a | 09:15).\n"
    )
    assert rc.check_tw_timestamps_are_day_qualified(one_day) == []


def test_a_single_report_mixing_both_attribution_styles_is_reported():
    from eval import report_classes as rc

    mixed = "- @a did a thing (@a | 09:15).\n- @b did another at 10:20.\n"
    failures = rc.check_tw_attribution_format_is_uniform([mixed])
    assert failures and "mixes attribution styles" in failures[0]


def test_prompt_render_check_reports_an_unrenderable_template(tmp_path):
    """The C1 gate must name the file to edit, not just fail."""
    from eval import report_classes as rc

    (tmp_path / "broken.toml").write_text(
        '[prompts]\nweekend_transient = "Find events for {no_such_field}"\n'
    )
    failures = rc.check_model_prompts_render(models_dir=tmp_path)
    assert failures and "broken.toml" in failures[0]


def test_prompt_render_check_skips_a_model_with_no_weekend_prompts(tmp_path):
    from eval import report_classes as rc

    (tmp_path / "other.toml").write_text('[prompts]\nfilename = "Name this: {}"\n')
    assert rc.check_model_prompts_render(models_dir=tmp_path) == []


# --------------------------------------------------------------------------
# weekend/scoring -- the weather-fit branches
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "item_weather,forecast,expect_bonus",
    [
        ("rainy", "cloudy with rain", True),    # cloudy activity, cloudy forecast
        ("sunny", "clear and warm", True),      # sunny activity, sunny forecast
        ("sunny", "cloudy with rain", False),   # mismatch -> partial only
    ],
)
def test_score_rewards_weather_that_matches_the_forecast(item_weather, forecast, expect_bonus):
    from weekend.scoring import _score_item

    base = {"name": "A venue name here", "location": "Vaughan, Ontario"}
    matched = _score_item({**base, "weather": item_weather}, weather_str=forecast)
    mismatched = _score_item({**base, "weather": item_weather}, weather_str="")
    assert (matched > mismatched) is expect_bonus


def test_score_rewards_a_specific_weather_value_over_the_canonical_trio():
    """"indoor"/"outdoor"/"both" carry no information beyond the label; anything
    more specific is real detail and scores higher."""
    from weekend.scoring import _score_item

    base = {"name": "A venue name here", "location": "Vaughan, Ontario"}
    specific = _score_item({**base, "weather": "covered patio, rain-friendly"})
    canonical = _score_item({**base, "weather": "indoor"})
    assert specific > canonical


# --------------------------------------------------------------------------
# The last branches: "nothing to judge" must be distinct from "a violation".
# A checker that reports a row it cannot evaluate is the mirror image of one
# that passes a row it should have caught -- both mislead.
# --------------------------------------------------------------------------


def test_a_blank_date_cell_is_not_a_C2b_violation():
    """It is class C7 (an undated row), reported separately. Conflating them made
    the fixed renderer look broken for telling the truth."""
    from eval import report_classes as rc

    blank = (
        "### Transient / Limited-Time Events\n"
        "| Event & Location | Dates | Day |\n| :--- | :--- | :--- |\n"
        "| **Rib Fest** (Scarborough) | — | — |\n"
    )
    assert rc.check_wk_transient_rows_carry_a_date(blank, 2026) == []
    assert rc.check_wk_no_row_outside_window(blank, date(2026, 8, 7), date(2026, 8, 9)) == []
    assert rc.check_wk_day_matches_dates(blank, 2026) == []
    # ...but C7 still reports it, which is the point
    assert rc.check_wk_transient_rows_are_time_bounded(blank, 2026) != []


def test_reconcile_leaves_a_day_that_already_agrees():
    from weekend.enforce import reconcile_day_with_dates

    items = [{"name": "TD Sunday", "start_date": "2026-08-09", "end_date": "2026-08-09",
              "day": "Sunday"}]
    fixed, notes = reconcile_day_with_dates(items, date(2026, 8, 7), date(2026, 8, 9))
    assert notes == [], "a correct row must not be 'corrected'"
    assert fixed[0]["day"] == "Sunday"


def test_extract_falls_back_to_the_raw_text_when_every_batch_fails():
    """Losing the whole corpus because the extract phase failed would be silent
    data loss; passing the raw text through keeps the run degraded, not empty."""
    from weekend import llm, phases

    raw = "- Kite Festival: Woodbridge, Vaughan"
    with (
        patch.object(phases, "_load_extract_signals", return_value={}),
        patch.object(phases, "_save_extract_signals"),
        patch.object(llm, "_call_llm", return_value=None),
    ):
        assert phases.extract_sources(raw, "events", model_name="m") == raw


def test_venue_extract_prompt_renders_without_a_location_placeholder():
    """The venues template has no {location}; formatting it as if it did would
    raise on every fixed-venue fetch."""
    from weekend.prompts import build_source_extract_prompt

    prompt = build_source_extract_prompt("- Playdium: Vaughan", "venues")
    assert "Playdium" in prompt
    assert "{" not in prompt.replace("{raw_text}", "")


def test_ensure_server_with_no_retries_reports_the_observed_state():
    """max_retries=0 means "do not restart anything, just tell me". The answer
    must come from probing the server, not from the retry bookkeeping."""
    from lib import osaurus_server as srv

    with (
        patch.object(srv, "is_server_running", lambda: True),
        patch.object(srv, "restart_server") as restart,
    ):
        assert srv.ensure_server(max_retries=0) is True
    restart.assert_not_called()

    with patch.object(srv, "is_server_running", lambda: False):
        assert srv.ensure_server(max_retries=0) is False
