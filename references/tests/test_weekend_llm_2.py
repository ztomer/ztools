"""Tests for weekend_llm: get_llm_json, normalize_llm_items, _score_item, fetch_scores_for_items, phase pipeline."""

from unittest.mock import patch


class TestPhaseFunctions:
    """Test the multiphase pipeline phase functions."""

    def test_condense_weather_returns_api_content(self, mock_llm):
        import weekend.llm as wl

        with patch.object(wl, "_call_llm", return_value="Fri 25°C, Sat 28°C, Sun 19°C"):
            result = wl.condense_weather("Forecast: ...")
        assert result == "Fri 25°C, Sat 28°C, Sun 19°C"

    def test_condense_weather_fallback_on_none(self, mock_llm):
        import weekend.llm as wl

        with patch.object(wl, "_call_llm", return_value=None):
            result = wl.condense_weather("Daily Forecast:\nFri: 25°C")
        assert "25°C" in result

    def test_extract_sources_returns_api_content(self, mock_llm):
        import weekend.llm as wl
        import weekend.phases as wp

        with (
            patch.object(wl, "_call_llm", return_value="- Event A: details"),
            patch.object(wp, "_load_extract_signals", return_value={}),
            patch.object(wp, "_save_extract_signals"),
        ):
            result = wl.extract_sources("- raw result", "events")
        assert "Event A" in result

    def test_extract_sources_fallback_on_none(self, mock_llm):
        import weekend.llm as wl
        import weekend.phases as wp

        raw = "- raw result 1\n- raw result 2"
        with (
            patch.object(wl, "_call_llm", return_value=None),
            patch.object(wp, "_load_extract_signals", return_value={}),
            patch.object(wp, "_save_extract_signals"),
        ):
            result = wl.extract_sources(raw, "events")
        # With batch_size=1 fallback, results are raw lines
        assert "- raw result" in result

    def test_extract_sources_returns_raw_on_no_lines(self, mock_llm):
        import weekend.llm as wl

        result = wl.extract_sources("not a dash line", "events")
        assert result == "not a dash line"

    def test_extract_sources_reduces_batch_on_timeout(self, mock_llm):
        import weekend.llm as wl
        import weekend.phases as wp

        with (
            patch.object(wl, "_call_llm", side_effect=[None, "- Event B: details"]),
            patch.object(wp, "_load_extract_signals", return_value={}),
            patch.object(wp, "_save_extract_signals"),
        ):
            result = wl.extract_sources("- r1\n- r2", "events")
        # First call with batch=5 fails → batch halves to 2 → retries both items → succeeds
        assert "Event B" in result

    def test_draft_activities_returns_content(self, mock_llm):
        import weekend.llm as wl

        with patch.object(wl, "_call_llm", return_value="1. Go to park\n2. Visit museum"):
            result = wl.draft_activities(
                "Sunny", "- Park", "transient", "Toronto", "5-10", "June 5-7"
            )
        assert result == "1. Go to park\n2. Visit museum"

    def test_draft_activities_returns_none_on_failure(self, mock_llm):
        import weekend.llm as wl

        with patch.object(wl, "_call_llm", return_value=None):
            result = wl.draft_activities(
                "Sunny", "- Park", "transient", "Toronto", "5-10", "June 5-7"
            )
        assert result is None

    def test_refine_draft_returns_api_content(self, mock_llm):
        import weekend.llm as wl

        with patch.object(wl, "_call_llm", return_value="1. Park\n2. Museum"):
            result = wl.refine_draft("1. Go to big park\n2. Visit the museum")
        assert result == "1. Park\n2. Museum"

    def test_refine_draft_fallback_on_none(self, mock_llm):
        import weekend.llm as wl

        draft = "1. Go to park\n2. Visit museum"
        with patch.object(wl, "_call_llm", return_value=None):
            result = wl.refine_draft(draft)
        assert result == draft

    def test_structure_to_json_returns_parsed(self, mock_llm):
        import weekend.llm as wl

        mock_json = {"transient_events": [{"name": "Park", "location": "Toronto"}]}
        with patch.object(wl, "_call_llm", return_value=mock_json):
            result = wl.structure_to_json("1. Park in Toronto", "transient", "5-10", "Sunny")
        assert result == mock_json

    def test_structure_to_json_returns_none_on_failure(self, mock_llm):
        import weekend.llm as wl

        with patch.object(wl, "_call_llm", return_value=None):
            result = wl.structure_to_json("1. Park in Toronto", "transient", "5-10", "Sunny")
        assert result is None

    def test_generate_weekend_plan_threads_each_phase_into_the_next(self, mock_llm):
        """The orchestrator's real contract: output of phase N feeds phase N+1.

        Asserting only that the injected return values come back out passes for
        any wiring — including phases run in the wrong order, or a phase fed the
        raw input instead of its predecessor's output. These assertions check
        the chain itself.
        """
        import weekend.llm as wl

        mock_transient = {"transient_events": [{"name": "E1"}]}
        mock_fixed = {"fixed_activities": [{"name": "F1"}]}
        with (
            patch.object(wl, "condense_weather", return_value="Sunny") as weather,
            patch.object(
                wl, "extract_sources", side_effect=["cleaned events", "cleaned venues"]
            ) as extract,
            patch.object(
                wl, "draft_activities", side_effect=["draft text", "draft fixed"]
            ) as draft,
            patch.object(
                wl, "refine_draft", side_effect=["refined text", "refined fixed"]
            ) as refine,
            patch.object(
                wl, "structure_to_json", side_effect=[mock_transient, mock_fixed]
            ) as structure,
        ):
            t, f = wl.generate_weekend_plan(
                "model", "weather", "events", "venues", "June 5-7", "Toronto", "5-10", "June 5-7"
            )

        assert (t, f) == (mock_transient, mock_fixed)

        # The raw weather goes to condense_weather, and only the condensed form
        # travels onward.
        assert "weather" in weather.call_args.args

        # Each source corpus is cleaned before drafting, transient first.
        assert [c.args[0] for c in extract.call_args_list] == ["events", "venues"]

        # Drafts are built from the CLEANED sources, not the raw ones, and carry
        # the condensed weather (draft_activities(weather_condensed, cleaned, ...)).
        assert [c.args[0] for c in draft.call_args_list] == ["Sunny", "Sunny"]
        assert [c.args[1] for c in draft.call_args_list] == ["cleaned events", "cleaned venues"]

        # Refinement consumes the drafts; structuring consumes the refinements.
        assert [c.args[0] for c in refine.call_args_list] == ["draft text", "draft fixed"]
        assert [c.args[0] for c in structure.call_args_list] == ["refined text", "refined fixed"]

    def test_generate_weekend_plan_transient_draft_fails(self, mock_llm):
        import weekend.llm as wl

        mock_fixed = {"fixed_activities": [{"name": "F1"}]}
        with (
            patch.object(wl, "condense_weather", return_value="Sunny"),
            patch.object(wl, "extract_sources", side_effect=["cleaned events", "cleaned venues"]),
            patch.object(wl, "draft_activities", side_effect=[None, "draft fixed"]),
            patch.object(wl, "refine_draft", return_value="refined"),
            patch.object(wl, "structure_to_json", return_value=mock_fixed),
            patch.object(wl, "get_llm_json", return_value=None),
        ):
            t, f = wl.generate_weekend_plan(
                "model", "weather", "events", "venues", "June 5-7", "Toronto", "5-10", "June 5-7"
            )
        assert t == {}
        assert f == mock_fixed

    def test_generate_weekend_plan_fixed_draft_fails(self, mock_llm):
        import weekend.llm as wl

        mock_transient = {"transient_events": [{"name": "E1"}]}
        with (
            patch.object(wl, "condense_weather", return_value="Sunny"),
            patch.object(wl, "extract_sources", side_effect=["cleaned events", "cleaned venues"]),
            patch.object(wl, "draft_activities", side_effect=["draft text", None]),
            patch.object(wl, "refine_draft", return_value="refined"),
            patch.object(wl, "structure_to_json", return_value=mock_transient),
            patch.object(wl, "get_llm_json", return_value=None),
        ):
            t, f = wl.generate_weekend_plan(
                "model", "weather", "events", "venues", "June 5-7", "Toronto", "5-10", "June 5-7"
            )
        assert t == mock_transient
        assert f == {}


class TestPhaseTimeoutLearning:
    """The learned phase timeout must reflect real latency.

    `current_timeout` was only ever assigned `base_timeout`, so the persisted
    signal restated the configured default on every run: the file grew, nothing
    was learned, and the comment claimed adaptivity the code had lost.
    """

    def _run(self, tmp_path, elapsed, seed=None):
        import json as _json

        import weekend.llm as wl

        signals_path = tmp_path / "phase_signals.json"
        if seed is not None:
            signals_path.write_text(_json.dumps(seed))

        clock = iter([0.0, elapsed])

        with (
            patch.object(wl, "PHASE_SIGNALS_PATH", signals_path),
            patch.object(wl, "get_best_model", return_value="m1"),
            patch.object(wl.time, "monotonic", lambda: next(clock)),
            patch.object(wl, "call_with_fallback", return_value="ok"),
        ):
            wl._call_llm("sys", "usr", timeout=900, phase_key="draft", parse_json=False)

        return _json.loads(signals_path.read_text())

    def test_slow_phase_widens_the_timeout(self, tmp_path):
        saved = self._run(tmp_path, elapsed=1200.0)["m1"]["draft"]
        assert saved["p95_latency"] == 1200.0
        assert saved["timeout"] == 1800  # 1200 * 1.5, above the 900 default
        assert saved["samples"] == 1

    def test_fast_phase_never_drops_below_the_configured_default(self, tmp_path):
        saved = self._run(tmp_path, elapsed=12.0)["m1"]["draft"]
        assert saved["p95_latency"] == 12.0
        assert saved["timeout"] == 900

    def test_samples_accumulate_across_runs(self, tmp_path):
        seed = {"m1": {"draft": {"p95_latency": 1000.0, "samples": 3, "timeout": 1500}}}
        saved = self._run(tmp_path, elapsed=20.0, seed=seed)["m1"]["draft"]
        assert saved["samples"] == 4
        # A single fast run does not erase the slow history.
        assert saved["p95_latency"] > 900
