"""Integration tests for weekend.cli with mocked LLM and data fetchers."""

import types
from unittest.mock import patch


FAKE_WEATHER = "Daily Forecast:\nFriday: 18°C, Clear (0mm)\nSaturday: 22°C, Sunny (0mm)\nSunday: 20°C, Cloudy (2mm)"
FAKE_EVENTS = "- Spring Festival at City Hall\n- Coding Workshop at Library\n- Outdoor Movie Night in Park"
FAKE_VENUES = "- Indoor Play Centre on Main St\n- Trampoline Park on Oak Ave\n- Museum of Nature"
FAKE_DATES = "June 5 to June 7, 2026"

FAKE_TRANSIENT = {
    "transient_events": [
        {"name": "Spring Festival", "location": "Toronto", "target_ages": "All",
         "price": "Free", "weather": "outdoor", "day": "Saturday", "duration": "All day"},
        {"name": "Indoor Coding Workshop", "location": "Vaughan", "target_ages": "8-14",
         "price": "$25", "weather": "indoor", "day": "Sunday", "duration": "3 hours"},
        {"name": "Outdoor Movie Night", "location": "Markham", "target_ages": "6-12",
         "price": "$10", "weather": "outdoor", "day": "Friday", "duration": "2 hours"},
        {"name": "Farmers Market", "location": "Richmond Hill", "target_ages": "All",
         "price": "Free", "weather": "outdoor", "day": "Saturday", "duration": "Morning"},
        {"name": "Art Workshop", "location": "Toronto", "target_ages": "10-16",
         "price": "$15", "weather": "indoor", "day": "Sunday", "duration": "2 hours"},
    ]
}

FAKE_FIXED = {
    "fixed_activities": [
        {"name": "Vaughan Sports Arena", "location": "Vaughan", "target_ages": "6-13",
         "price": "$20", "weather": "indoor"},
        {"name": "High Park", "location": "Toronto", "target_ages": "All",
         "price": "Free", "weather": "outdoor"},
        {"name": "LEGOLAND", "location": "Vaughan", "target_ages": "3-12",
         "price": "$35", "weather": "indoor"},
        {"name": "Royal Ontario Museum", "location": "Toronto", "target_ages": "4-18",
         "price": "$23", "weather": "indoor"},
        {"name": "Ontario Science Centre", "location": "Toronto", "target_ages": "6-16",
         "price": "$22", "weather": "indoor"},
    ]
}


def _make_args(**overrides):
    defaults = dict(use_cache=True, model="mock-model", skip_web=True, debug=False)
    defaults.update(overrides)
    return types.SimpleNamespace(**defaults)


class TestWeekendMainFlow:
    """Test weekend.cli.main() full flow with mocked dependencies."""

    def test_main_happy_path(self, mock_llm, capsys, tmp_path):
        import weekend.cli as wp
        with patch("os.path.expanduser", return_value=str(tmp_path)), \
             patch.object(wp, "_fetch_data", return_value=(FAKE_WEATHER, FAKE_EVENTS, FAKE_VENUES, FAKE_DATES)), \
             patch.object(wp, "get_llm_json", side_effect=[FAKE_TRANSIENT, FAKE_FIXED]):
            wp.main(_make_args())
        captured = capsys.readouterr()
        assert "Spring" in captured.out or "Festival" in captured.out
        assert "Weekend plan" in captured.out or "weekend" in captured.out.lower()

    def test_main_creates_file(self, mock_llm, capsys, tmp_path):
        import weekend.cli as wp
        with patch("os.path.expanduser", return_value=str(tmp_path)), \
             patch.object(wp, "_fetch_data", return_value=(FAKE_WEATHER, FAKE_EVENTS, FAKE_VENUES, FAKE_DATES)), \
             patch.object(wp, "get_llm_json", side_effect=[FAKE_TRANSIENT, FAKE_FIXED]):
            wp.main(_make_args())
        out_files = list(tmp_path.iterdir())
        # At least one weekend_plan file written
        plan_files = [f for f in out_files if "weekend_plan" in f.name]
        assert len(plan_files) >= 1

    def test_main_resolves_model_arg(self, mock_llm, capsys, tmp_path):
        import weekend.cli as wp
        import os
        original = os.environ.pop("OLLAMA_MODEL", None)
        try:
            with patch("os.path.expanduser", return_value=str(tmp_path)), \
                 patch.object(wp, "_fetch_data", return_value=(FAKE_WEATHER, FAKE_EVENTS, FAKE_VENUES, FAKE_DATES)), \
                 patch.object(wp, "get_llm_json", side_effect=[FAKE_TRANSIENT, FAKE_FIXED]) as mock_get:
                wp.main(_make_args(model="custom-model"))
            captured = capsys.readouterr()
            # Model name appears in output (e.g., "Using model: custom-model")
            assert "custom-model" in captured.out
            # Verify the LLM was called twice (transient + fixed)
            assert mock_get.call_count == 2
        finally:
            if original is not None:
                os.environ["OLLAMA_MODEL"] = original

    def test_main_debug_mode(self, mock_llm, capsys, tmp_path):
        import weekend.cli as wp
        with patch("os.path.expanduser", return_value=str(tmp_path)), \
             patch.object(wp, "_fetch_data", return_value=(FAKE_WEATHER, FAKE_EVENTS, FAKE_VENUES, FAKE_DATES)), \
             patch.object(wp, "get_llm_json", side_effect=[FAKE_TRANSIENT, FAKE_FIXED]):
            wp.main(_make_args(debug=True))
        # Debug mode should print debug indicators
        captured = capsys.readouterr()
        assert len(captured.out) > 0

    def test_main_low_items_triggers_warning(self, mock_llm, capsys, tmp_path):
        import weekend.cli as wp
        few_items = {"transient_events": [{"name": "Only One", "location": "Toronto"}]}
        with patch("os.path.expanduser", return_value=str(tmp_path)), \
             patch.object(wp, "_fetch_data", return_value=(FAKE_WEATHER, FAKE_EVENTS, FAKE_VENUES, FAKE_DATES)), \
             patch.object(wp, "get_llm_json", side_effect=[few_items, {"fixed_activities": []}]):
            wp.main(_make_args())
        captured = capsys.readouterr()
        assert "Low item count" in captured.out

    def test_main_empty_llm_response_still_works(self, mock_llm, capsys, tmp_path):
        import weekend.cli as wp
        with patch("os.path.expanduser", return_value=str(tmp_path)), \
             patch.object(wp, "_fetch_data", return_value=(FAKE_WEATHER, FAKE_EVENTS, FAKE_VENUES, FAKE_DATES)), \
             patch.object(wp, "get_llm_json", side_effect=[None, None]):
            wp.main(_make_args())
        captured = capsys.readouterr()
        # Should not crash, should print something
        assert "FAIL" in captured.out or "error" in captured.out.lower() or len(captured.out) > 0

    def test_main_passes_use_cache_to_fetch_data(self, mock_llm, capsys, tmp_path):
        import weekend.cli as wp
        with patch("os.path.expanduser", return_value=str(tmp_path)), \
             patch.object(wp, "_fetch_data", return_value=(FAKE_WEATHER, FAKE_EVENTS, FAKE_VENUES, FAKE_DATES)) as mock_fetch, \
             patch.object(wp, "get_llm_json", side_effect=[FAKE_TRANSIENT, FAKE_FIXED]):
            wp.main(_make_args(use_cache=True))
        mock_fetch.assert_called_once()
        assert mock_fetch.call_args[0][4] is True



class TestWeekendFetchData:
    """Test _fetch_data with mocked internals."""

    def test_fetch_data_basic(self, mock_llm):
        import weekend.cli as wp
        from datetime import date
        fri = date(2026, 6, 5)
        sun = date(2026, 6, 7)
        with patch.object(wp, "ensure_server"), \
             patch.object(wp, "fetch_weather", return_value=FAKE_WEATHER), \
             patch.object(wp, "fetch_transient_events", return_value=FAKE_EVENTS), \
             patch.object(wp, "fetch_fixed_venues", return_value=FAKE_VENUES), \
             patch.object(wp, "save_events_cache"), \
             patch.object(wp, "save_venues_cache"):
            weather, events, venues, dates = wp._fetch_data(fri, sun, "2026", "June", use_cache=False)
        assert weather == FAKE_WEATHER
        assert events == FAKE_EVENTS
        assert venues == FAKE_VENUES
        assert "June" in dates and "2026" in dates

    def test_fetch_data_uses_cache_hit(self, mock_llm):
        import weekend.cli as wp
        from datetime import date
        fri = date(2026, 6, 5)
        sun = date(2026, 6, 7)
        with patch.object(wp, "ensure_server"), \
             patch.object(wp, "fetch_weather", return_value=FAKE_WEATHER), \
             patch.object(wp, "load_events_cache", return_value=FAKE_EVENTS), \
             patch.object(wp, "load_venues_cache", return_value=FAKE_VENUES), \
             patch.object(wp, "fetch_transient_events") as mock_transient, \
             patch.object(wp, "fetch_fixed_venues") as mock_fixed:
            weather, events, venues, dates = wp._fetch_data(fri, sun, "2026", "June", use_cache=True)
        assert events == FAKE_EVENTS
        assert venues == FAKE_VENUES
        mock_transient.assert_not_called()
        mock_fixed.assert_not_called()

    def test_fetch_data_cache_miss_loads_fresh(self, mock_llm):
        import weekend.cli as wp
        from datetime import date
        fri = date(2026, 6, 5)
        sun = date(2026, 6, 7)
        with patch.object(wp, "ensure_server"), \
             patch.object(wp, "fetch_weather", return_value=FAKE_WEATHER), \
             patch.object(wp, "load_events_cache", return_value=None), \
             patch.object(wp, "load_venues_cache", return_value=None), \
             patch.object(wp, "fetch_transient_events", return_value=FAKE_EVENTS), \
             patch.object(wp, "fetch_fixed_venues", return_value=FAKE_VENUES), \
             patch.object(wp, "save_events_cache"), \
             patch.object(wp, "save_venues_cache"):
            weather, events, venues, dates = wp._fetch_data(fri, sun, "2026", "June", use_cache=True)
        assert events == FAKE_EVENTS
        assert venues == FAKE_VENUES


class TestWeekendParseFunctions:
    """Test _parse_fixed and _parse_transient directly."""

    def test_parse_fixed_list(self):
        import weekend.cli as wp
        result = wp._parse_fixed(
            [{"name": "Place 1"}, {"name": "Place 2"}],
            "mock-model", {}
        )
        assert len(result) == 2
        assert result[0]["name"] == "Place 1"

    def test_parse_fixed_dict_with_key(self):
        import weekend.cli as wp
        result = wp._parse_fixed(
            {"fixed_activities": [{"name": "Place 1"}]},
            "mock-model", {}
        )
        assert len(result) == 1
        assert result[0]["name"] == "Place 1"

    def test_parse_fixed_empty(self):
        import weekend.cli as wp
        result = wp._parse_fixed({}, "mock-model", {})
        assert result == []

    def test_parse_fixed_single_object(self):
        import weekend.cli as wp
        result = wp._parse_fixed(
            {"name": "Single Place", "location": "Toronto"},
            "mock-model", {}
        )
        assert len(result) == 1
        assert result[0]["name"] == "Single Place"

    def test_parse_fixed_fallback_key(self):
        import weekend.cli as wp
        result = wp._parse_fixed(
            {"odd_key": [{"name": "Fallback Item"}]},
            "mock-model", {}
        )
        assert len(result) == 1
        assert result[0]["name"] == "Fallback Item"

    def test_parse_transient_list(self):
        import weekend.cli as wp
        result = wp._parse_transient(
            [{"name": "Event 1"}, {"name": "Event 2"}],
            "mock-model", {}
        )
        assert len(result) == 2

    def test_parse_transient_dict_with_key(self):
        import weekend.cli as wp
        result = wp._parse_transient(
            {"transient_events": [{"name": "Event 1"}]},
            "mock-model", {}
        )
        assert len(result) == 1

    def test_parse_transient_empty(self):
        import weekend.cli as wp
        result = wp._parse_transient({}, "mock-model", {})
        assert result == []

    def test_parse_transient_with_alt_name_keys_in_list(self):
        import weekend.cli as wp
        result = wp._parse_transient(
            [{"description": "Desc Event", "location": "Toronto"},
             {"description": "Another Event", "location": "Vaughan"}],
            "mock-model", {}
        )
        assert len(result) == 2
        assert result[0]["name"] == "Desc Event"

    def test_parse_transient_filters_weather_objects(self):
        import weekend.cli as wp
        result = wp._parse_transient(
            [{"temperature": "20C", "condition": "Sunny", "precipitation": "0"},
             {"name": "Real Event", "day": "Saturday"}],
            "mock-model", {}
        )
        assert len(result) == 1
        assert result[0]["name"] == "Real Event"


class TestWeekendPrompts:
    """Test prompt building functions."""

    def test_build_fixed_system_prompt_defaults(self):
        from weekend.prompts import build_fixed_system_prompt
        prompt = build_fixed_system_prompt()
        assert "Output JSON now" in prompt
        assert "fixed_activities" in prompt

    def test_build_fixed_system_prompt_with_args(self):
        from weekend.prompts import build_fixed_system_prompt
        prompt = build_fixed_system_prompt(model="mock-model", location="Vaughan/Toronto", age_range="4-12")
        assert "Output JSON now" in prompt
        assert "fixed_activities" in prompt

    def test_build_fixed_user_prompt(self):
        from weekend.prompts import build_fixed_user_prompt
        prompt = build_fixed_user_prompt("June 5-7", "Sunny", "- Venue A\n- Venue B")
        assert "June 5-7" in prompt
        assert "Venue A" in prompt

    def test_build_transient_system_prompt_defaults(self):
        from weekend.prompts import build_transient_system_prompt
        prompt = build_transient_system_prompt()
        assert "Output JSON now" in prompt
        assert "transient_events" in prompt

    def test_build_transient_user_prompt(self):
        from weekend.prompts import build_transient_user_prompt
        prompt = build_transient_user_prompt("June 5-7", "Rainy", "- Event A\n- Event B")
        assert "June 5-7" in prompt
        assert "Event A" in prompt
