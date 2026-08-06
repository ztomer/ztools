"""Tests for weekend_llm: get_llm_json, normalize_llm_items, _score_item, fetch_scores_for_items, phase pipeline."""

from unittest.mock import patch


class TestGetLlmJsonHappy:
    """Successful path: API returns content, JSON extracted."""

    def test_first_attempt_succeeds(self, mock_llm):
        import weekend.llm as wl

        with (
            patch.object(
                wl, "call_llm_api", return_value={"content": '{"items": [1, 2]}'}
            ) as mock_call,
            patch.object(wl, "strip_thinking", return_value='{"items": [1, 2]}'),
            patch.object(wl, "_extract_json_only", return_value='{"items": [1, 2]}'),
        ):
            result = wl.get_llm_json("sys", "user")
        assert result == {"items": [1, 2]}
        assert mock_call.call_count == 1

    def test_result_not_dict_debug(self, mock_llm):
        """When result is not a dict, debug_print still works."""
        import weekend.llm as wl

        with (
            patch.object(wl, "call_llm_api", return_value=[1, 2, 3]),
            patch.object(wl, "_extract_json_only", return_value=None),
        ):
            result = wl.get_llm_json("sys", "user", max_retries=1)
        assert result is None


class TestGetLlmJsonRetries:
    """Retry path: JSON parse fails, retries until success."""

    def test_retry_with_value_error(self, mock_llm):
        import weekend.llm as wl

        # First call returns content that fails JSON parse, second succeeds
        with (
            patch.object(
                wl,
                "call_llm_api",
                side_effect=[
                    {"content": "not json"},
                    {"content": '{"ok": true}'},
                ],
            ),
            patch.object(wl, "strip_thinking", side_effect=lambda x: x),
            patch.object(wl, "_extract_json_only", side_effect=[None, '{"ok": true}']),
            patch.object(wl, "ensure_server"),
        ):
            result = wl.get_llm_json("sys", "user", max_retries=2)
        assert result == {"ok": True}

    def test_max_retries_exhausted(self, mock_llm):
        """All retries fail, panic_dump is called."""
        import weekend.llm as wl

        with (
            patch.object(wl, "call_llm_api", return_value={"content": "bad"}),
            patch.object(wl, "strip_thinking", return_value="bad"),
            patch.object(wl, "_extract_json_only", return_value=None),
            patch.object(wl, "panic_dump") as mock_dump,
            patch.object(wl, "ensure_server"),
        ):
            result = wl.get_llm_json("sys", "user", max_retries=2)
        assert result is None
        assert mock_dump.call_count == 1

    def test_no_content_in_result(self, mock_llm):
        """Result has no 'content' key."""
        import weekend.llm as wl

        with (
            patch.object(wl, "call_llm_api", return_value={"status": "ok"}),
            patch.object(wl, "ensure_server"),
        ):
            result = wl.get_llm_json("sys", "user", max_retries=1)
        assert result is None


class TestNormalizeLlmItems:
    """Test normalize_llm_items edge cases."""

    def test_empty_items(self, mock_llm):
        from weekend.llm import normalize_llm_items

        assert normalize_llm_items([]) == []
        assert normalize_llm_items(None) is None

    def test_string_item(self, mock_llm):
        from weekend.llm import normalize_llm_items

        result = normalize_llm_items(["Just a name"])
        assert result == [{"name": "Just a name"}]

    def test_dict_with_field_mapping(self, mock_llm):
        from weekend.llm import normalize_llm_items

        items = [{"custom_name": "Foo", "location": "Bar"}]
        result = normalize_llm_items(items, field_mapping={"custom_name": "name"})
        assert result[0]["name"] == "Foo"
        assert result[0]["location"] == "Bar"

    def test_field_mapping_already_has_standard(self, mock_llm):
        """If standard_field already exists, don't overwrite."""
        from weekend.llm import normalize_llm_items

        items = [{"custom_name": "Foo", "name": "Bar"}]
        result = normalize_llm_items(items, field_mapping={"custom_name": "name"})
        assert result[0]["name"] == "Bar"

    def test_dict_with_alt_keys(self, mock_llm):
        from weekend.llm import normalize_llm_items

        items = [{"activity": "My Activity", "place": "My Place", "ages": "5-10"}]
        result = normalize_llm_items(items)
        assert result[0]["name"] == "My Activity"
        assert result[0]["location"] == "My Place"
        assert result[0]["target_ages"] == "5-10"

    def test_dict_with_price_alt(self, mock_llm):
        from weekend.llm import normalize_llm_items

        items = [{"name": "X", "cost": "Free"}]
        result = normalize_llm_items(items)
        assert result[0]["price"] == "Free"

    def test_dict_with_weather_alt(self, mock_llm):
        from weekend.llm import normalize_llm_items

        items = [{"name": "X", "indoor_outdoor": "outdoor"}]
        result = normalize_llm_items(items)
        assert result[0]["weather"] == "outdoor"

    def test_dict_with_day_alt(self, mock_llm):
        from weekend.llm import normalize_llm_items

        items = [{"name": "X", "event_date": "Saturday"}]
        result = normalize_llm_items(items)
        assert result[0]["day"] == "Saturday"

    def test_dict_with_dur_alt(self, mock_llm):
        from weekend.llm import normalize_llm_items

        items = [{"name": "X", "time": "2pm"}]
        result = normalize_llm_items(items)
        assert result[0]["duration"] == "2pm"

    def test_dict_with_no_matching_keys(self, mock_llm):
        from weekend.llm import normalize_llm_items

        items = [{"foo": "bar"}]
        result = normalize_llm_items(items)
        assert result == [{"foo": "bar"}]

    def test_mixed_types(self, mock_llm):
        from weekend.llm import normalize_llm_items

        items = ["Str Item", {"name": "Dict Item"}, 42, None]
        result = normalize_llm_items(items)
        assert result == [{"name": "Str Item"}, {"name": "Dict Item"}]


class TestScoreItem:
    """Test _score_item scoring logic."""

    def test_basic_score_no_fields(self, mock_llm):
        from weekend.llm import _score_item

        score = _score_item({})
        # No fields populated → populated/len*3.0 = 0; normalized round(0/2, 1) = 0
        assert score == 0.0

    def test_full_score_with_match(self, mock_llm):
        from weekend.llm import _score_item

        item = {
            "name": "X",
            "location": "Somewhere",
            "price": "$10",
            "target_ages": "5-10",
            "weather": "outdoor",
            "day": "Sat",
            "duration": "2h",
        }
        score = _score_item(item, weather_str="Sunny", age_range="5-10")
        # 7/7 fields populated → 3.0
        # age 5-10 vs 5-10 → overlap 6 ≥ 2 → +3.0
        # weather outdoor + Sunny → is_outdoor + forecast_sunny → +2.0
        # price $10 (not free) → +0.5
        # location len > 5 → +0.5
        # duration "2h" != "2-3 hours" → +0.3
        # raw = 9.3; round(9.3 / 2.0, 1) = 4.7
        assert score == 4.7

    def test_age_overlap(self, mock_llm):
        from weekend.llm import _score_item

        item = {"target_ages": "5-10"}
        score_overlap = _score_item(item, age_range="5-12")
        score_no_overlap = _score_item(item, age_range="20-30")
        # 1/7 fields populated → 0.4286
        # overlap: 5-10 vs 5-12 → 6 years → +3.0; vs 20-30 → 0 → +0
        # raw_overlap = 3.4286; raw_no_overlap = 0.4286
        # normalized: round(3.4286/2, 1) = 1.7; round(0.4286/2, 1) = 0.2
        assert score_overlap == 1.7
        assert score_no_overlap == 0.2
        # Key invariant: overlap scores higher than no overlap
        assert score_overlap > score_no_overlap

    def test_age_overlap_one_year(self, mock_llm):
        """Single year overlap — line 162 score += 1.5."""
        from weekend.llm import _score_item

        # age_range=5-10 → range(5, 11); target=10-15 → range(10, 16); overlap={10}, len=1
        item = {"target_ages": "10-15"}
        score_one = _score_item(item, age_range="5-10")
        score_none = _score_item(item, age_range="20-25")
        # 1/7 fields populated → 0.4286
        # raw_one = 0.4286 + 1.5 = 1.9286; round(1.9286/2, 1) = 1.0
        # raw_none = 0.4286; round(0.4286/2, 1) = 0.2
        assert score_one == 1.0
        assert score_none == 0.2
        # Key invariant: one-year overlap (1.5 bonus) scores higher than no overlap
        assert score_one > score_none

    def test_age_no_nums(self, mock_llm):
        from weekend.llm import _score_item

        item = {"target_ages": "all"}
        score = _score_item(item, age_range="5-10")
        # No digits → age range block skipped
        # 1/7 fields populated → 0.4286; round(0.4286/2, 1) = 0.2
        assert score == 0.2

    def test_weather_match_sunny(self, mock_llm):
        from weekend.llm import _score_item

        item = {"weather": "outdoor"}
        score_match = _score_item(item, weather_str="Sunny and clear")
        score_mismatch = _score_item(item, weather_str="Cloudy with rain")
        # 1/7 fields populated → 0.4286
        # "outdoor" + Sunny → is_outdoor + forecast_sunny → +2.0; raw_match = 2.4286 / 2 = 1.2
        # "outdoor" + Cloudy → is_outdoor + not forecast_sunny → 0 bonus; raw_mismatch = 0.4286 / 2 = 0.2
        assert score_match == 1.2
        assert score_mismatch == 0.2
        # Match is greater than mismatch
        assert score_match > score_mismatch

    def test_weather_match_cloudy(self, mock_llm):
        from weekend.llm import _score_item

        item = {"weather": "indoor"}
        score = _score_item(item, weather_str="Cloudy and wet")
        # 1/7 fields populated → 0.4286
        # "indoor" → is_indoor → +1.0 always (indoor is always appropriate)
        # raw = 1.4286 / 2 = 0.7
        assert score == 0.7

    def test_price_free(self, mock_llm):
        from weekend.llm import _score_item

        item = {"price": "Free"}
        score = _score_item(item)
        # 1/7 fields populated → 0.4286
        # "free" is in skip list → no price bonus
        # round(0.4286/2, 1) = 0.2
        assert score == 0.2

    def test_price_paid(self, mock_llm):
        from weekend.llm import _score_item

        item = {"price": "$20"}
        score_paid = _score_item(item)
        item2 = {"price": "Free"}
        score_free = _score_item(item2)
        assert score_paid > score_free

    def test_long_location_bonus(self, mock_llm):
        from weekend.llm import _score_item

        item1 = {"location": "A" * 10}
        item2 = {"location": "X"}
        assert _score_item(item1) > _score_item(item2)


class TestFetchScoresForItems:
    """Test fetch_scores_for_items mutates in place."""

    def test_assigns_scores(self, mock_llm):
        from weekend.llm import fetch_scores_for_items

        items = [
            {"name": "A", "target_ages": "5-10", "weather": "outdoor"},
            {"name": "B"},
        ]
        fetch_scores_for_items(items, weather_str="Sunny", age_range="5-10")
        assert "score" in items[0]
        assert "score" in items[1]
        # A: 3/7 fields → 1.2857, age overlap 6 → +3.0, outdoor+Sunny → +2.0; raw=6.2857/2=3.1
        # B: 1/7 fields → 0.4286/2=0.2
        assert items[0]["score"] == 3.1
        assert items[1]["score"] == 0.2


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

    def test_generate_weekend_plan_full_pipeline(self, mock_llm):
        import weekend.llm as wl

        mock_transient = {"transient_events": [{"name": "E1"}]}
        mock_fixed = {"fixed_activities": [{"name": "F1"}]}
        with (
            patch.object(wl, "condense_weather", return_value="Sunny"),
            patch.object(wl, "extract_sources", side_effect=["cleaned events", "cleaned venues"]),
            patch.object(wl, "draft_activities", side_effect=["draft text", "draft fixed"]),
            patch.object(wl, "refine_draft", side_effect=["refined text", "refined fixed"]),
            patch.object(wl, "structure_to_json", side_effect=[mock_transient, mock_fixed]),
        ):
            t, f = wl.generate_weekend_plan(
                "model", "weather", "events", "venues", "June 5-7", "Toronto", "5-10", "June 5-7"
            )
        assert t == mock_transient
        assert f == mock_fixed

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
