"""Tests for weekend_llm: get_llm_json, normalize_llm_items, _score_item, fetch_scores_for_items."""
import pytest
from unittest.mock import patch, MagicMock


class TestGetLlmJsonHappy:
    """Successful path: API returns content, JSON extracted."""

    def test_first_attempt_succeeds(self, mock_llm):
        import weekend_llm as wl
        with patch.object(wl, "call_llm_api", return_value={"content": '{"items": [1, 2]}'}) as mock_call, \
             patch.object(wl, "strip_thinking", return_value='{"items": [1, 2]}'), \
             patch.object(wl, "_extract_json_only", return_value='{"items": [1, 2]}'):
            result = wl.get_llm_json("sys", "user")
        assert result == {"items": [1, 2]}
        assert mock_call.call_count == 1

    def test_result_not_dict_debug(self, mock_llm):
        """When result is not a dict, debug_print still works."""
        import weekend_llm as wl
        with patch.object(wl, "call_llm_api", return_value=[1, 2, 3]), \
             patch.object(wl, "_extract_json_only", return_value=None):
            result = wl.get_llm_json("sys", "user", max_retries=1)
        assert result is None


class TestGetLlmJsonRetries:
    """Retry path: JSON parse fails, retries until success."""

    def test_retry_with_value_error(self, mock_llm):
        import weekend_llm as wl
        # First call returns content that fails JSON parse, second succeeds
        with patch.object(wl, "call_llm_api", side_effect=[
            {"content": "not json"},
            {"content": '{"ok": true}'},
        ]), \
             patch.object(wl, "strip_thinking", side_effect=lambda x: x), \
             patch.object(wl, "_extract_json_only", side_effect=[None, '{"ok": true}']), \
             patch.object(wl, "time", MagicMock()), \
             patch.object(wl, "ensure_server"):
            result = wl.get_llm_json("sys", "user", max_retries=2)
        assert result == {"ok": True}

    def test_max_retries_exhausted(self, mock_llm):
        """All retries fail, panic_dump is called and we fall back to MLX."""
        import weekend_llm as wl
        with patch.object(wl, "call_llm_api", return_value={"content": "bad"}), \
             patch.object(wl, "strip_thinking", return_value="bad"), \
             patch.object(wl, "_extract_json_only", return_value=None), \
             patch.object(wl, "panic_dump") as mock_dump, \
             patch.object(wl, "time", MagicMock()), \
             patch.object(wl, "ensure_server"), \
             patch.object(wl, "find_text_mlx_model", return_value=None):
            result = wl.get_llm_json("sys", "user", max_retries=2)
        assert result is None
        assert mock_dump.call_count == 1

    def test_no_content_in_result(self, mock_llm):
        """Result has no 'content' key."""
        import weekend_llm as wl
        with patch.object(wl, "call_llm_api", return_value={"status": "ok"}), \
             patch.object(wl, "find_text_mlx_model", return_value=None), \
             patch.object(wl, "time", MagicMock()), \
             patch.object(wl, "ensure_server"):
            result = wl.get_llm_json("sys", "user", max_retries=1)
        assert result is None


class TestGetLlmJsonMlxFallback:
    """MLX fallback path."""

    def test_mlx_fallback_succeeds(self, mock_llm):
        import weekend_llm as wl
        mlx_model = MagicMock()
        mlx_model.name = "qwen-test"
        with patch.object(wl, "call_llm_api", return_value={"content": "bad"}), \
             patch.object(wl, "strip_thinking", return_value="bad"), \
             patch.object(wl, "_extract_json_only", side_effect=[None, '{"a": 1}']), \
             patch.object(wl, "time", MagicMock()), \
             patch.object(wl, "ensure_server"), \
             patch.object(wl, "find_text_mlx_model", return_value=mlx_model), \
             patch.object(wl, "call_mlx", return_value='{"a": 1}'), \
             patch.object(wl, "process_mlx_content", return_value='{"a": 1}'):
            result = wl.get_llm_json("sys", "user", max_retries=1)
        assert result == {"a": 1}

    def test_mlx_fallback_no_json(self, mock_llm):
        """MLX returns content but no JSON."""
        import weekend_llm as wl
        mlx_model = MagicMock()
        mlx_model.name = "qwen-test"
        with patch.object(wl, "call_llm_api", return_value={"content": "bad"}), \
             patch.object(wl, "strip_thinking", return_value="bad"), \
             patch.object(wl, "_extract_json_only", return_value=None), \
             patch.object(wl, "time", MagicMock()), \
             patch.object(wl, "ensure_server"), \
             patch.object(wl, "find_text_mlx_model", return_value=mlx_model), \
             patch.object(wl, "call_mlx", return_value="nope"), \
             patch.object(wl, "process_mlx_content", return_value="nope"), \
             patch.object(wl, "_extract_json_only", side_effect=[None, None]):
            result = wl.get_llm_json("sys", "user", max_retries=1)
        assert result is None

    def test_mlx_fallback_exception(self, mock_llm):
        """MLX call raises exception."""
        import weekend_llm as wl
        mlx_model = MagicMock()
        mlx_model.name = "qwen-test"
        with patch.object(wl, "call_llm_api", return_value={"content": "bad"}), \
             patch.object(wl, "strip_thinking", return_value="bad"), \
             patch.object(wl, "_extract_json_only", return_value=None), \
             patch.object(wl, "time", MagicMock()), \
             patch.object(wl, "ensure_server"), \
             patch.object(wl, "find_text_mlx_model", return_value=mlx_model), \
             patch.object(wl, "call_mlx", side_effect=Exception("mlx error")):
            result = wl.get_llm_json("sys", "user", max_retries=1)
        assert result is None

    def test_mlx_fallback_empty_response(self, mock_llm):
        """MLX returns falsy (empty string)."""
        import weekend_llm as wl
        mlx_model = MagicMock()
        mlx_model.name = "qwen-test"
        with patch.object(wl, "call_llm_api", return_value={"content": "bad"}), \
             patch.object(wl, "strip_thinking", return_value="bad"), \
             patch.object(wl, "_extract_json_only", return_value=None), \
             patch.object(wl, "time", MagicMock()), \
             patch.object(wl, "ensure_server"), \
             patch.object(wl, "find_text_mlx_model", return_value=mlx_model), \
             patch.object(wl, "call_mlx", return_value=""):
            result = wl.get_llm_json("sys", "user", max_retries=1)
        assert result is None


class TestNormalizeLlmItems:
    """Test normalize_llm_items edge cases."""

    def test_empty_items(self, mock_llm):
        from weekend_llm import normalize_llm_items
        assert normalize_llm_items([]) == []
        assert normalize_llm_items(None) is None

    def test_string_item(self, mock_llm):
        from weekend_llm import normalize_llm_items
        result = normalize_llm_items(["Just a name"])
        assert result == [{"name": "Just a name"}]

    def test_dict_with_field_mapping(self, mock_llm):
        from weekend_llm import normalize_llm_items
        items = [{"custom_name": "Foo", "location": "Bar"}]
        result = normalize_llm_items(items, field_mapping={"custom_name": "name"})
        assert result[0]["name"] == "Foo"
        assert result[0]["location"] == "Bar"

    def test_field_mapping_already_has_standard(self, mock_llm):
        """If standard_field already exists, don't overwrite."""
        from weekend_llm import normalize_llm_items
        items = [{"custom_name": "Foo", "name": "Bar"}]
        result = normalize_llm_items(items, field_mapping={"custom_name": "name"})
        assert result[0]["name"] == "Bar"

    def test_dict_with_alt_keys(self, mock_llm):
        from weekend_llm import normalize_llm_items
        items = [{"activity": "My Activity", "place": "My Place", "ages": "5-10"}]
        result = normalize_llm_items(items)
        assert result[0]["name"] == "My Activity"
        assert result[0]["location"] == "My Place"
        assert result[0]["target_ages"] == "5-10"

    def test_dict_with_price_alt(self, mock_llm):
        from weekend_llm import normalize_llm_items
        items = [{"name": "X", "cost": "Free"}]
        result = normalize_llm_items(items)
        assert result[0]["price"] == "Free"

    def test_dict_with_weather_alt(self, mock_llm):
        from weekend_llm import normalize_llm_items
        items = [{"name": "X", "indoor_outdoor": "outdoor"}]
        result = normalize_llm_items(items)
        assert result[0]["weather"] == "outdoor"

    def test_dict_with_day_alt(self, mock_llm):
        from weekend_llm import normalize_llm_items
        items = [{"name": "X", "event_date": "Saturday"}]
        result = normalize_llm_items(items)
        assert result[0]["day"] == "Saturday"

    def test_dict_with_dur_alt(self, mock_llm):
        from weekend_llm import normalize_llm_items
        items = [{"name": "X", "time": "2pm"}]
        result = normalize_llm_items(items)
        assert result[0]["duration"] == "2pm"

    def test_dict_with_no_matching_keys(self, mock_llm):
        from weekend_llm import normalize_llm_items
        items = [{"foo": "bar"}]
        result = normalize_llm_items(items)
        assert result == [{"foo": "bar"}]

    def test_mixed_types(self, mock_llm):
        from weekend_llm import normalize_llm_items
        items = ["Str Item", {"name": "Dict Item"}, 42, None]
        result = normalize_llm_items(items)
        assert result == [{"name": "Str Item"}, {"name": "Dict Item"}]


class TestScoreItem:
    """Test _score_item scoring logic."""

    def test_basic_score_no_fields(self, mock_llm):
        from weekend_llm import _score_item
        score = _score_item({})
        assert 0 <= score <= 5

    def test_full_score_with_match(self, mock_llm):
        from weekend_llm import _score_item
        item = {
            "name": "X", "location": "Somewhere", "price": "$10",
            "target_ages": "5-10", "weather": "outdoor", "day": "Sat", "duration": "2h",
        }
        score = _score_item(item, weather_str="Sunny", age_range="5-10")
        assert score > 0

    def test_age_overlap(self, mock_llm):
        from weekend_llm import _score_item
        item = {"target_ages": "5-10"}
        score_overlap = _score_item(item, age_range="5-12")
        score_no_overlap = _score_item(item, age_range="20-30")
        assert score_overlap > score_no_overlap

    def test_age_overlap_one_year(self, mock_llm):
        """Single year overlap — line 162 score += 1.5."""
        from weekend_llm import _score_item
        # age_range=5-10 → range(5, 11); target=10-15 → range(10, 16); overlap={10}, len=1
        item = {"target_ages": "10-15"}
        score_one = _score_item(item, age_range="5-10")
        score_none = _score_item(item, age_range="20-25")
        assert score_one > score_none

    def test_age_no_nums(self, mock_llm):
        from weekend_llm import _score_item
        item = {"target_ages": "all"}
        score = _score_item(item, age_range="5-10")
        assert 0 <= score <= 5

    def test_weather_match_sunny(self, mock_llm):
        from weekend_llm import _score_item
        item = {"weather": "outdoor"}
        score_match = _score_item(item, weather_str="Sunny and clear")
        score_mismatch = _score_item(item, weather_str="Cloudy with rain")
        assert score_match > score_mismatch

    def test_weather_match_cloudy(self, mock_llm):
        from weekend_llm import _score_item
        item = {"weather": "indoor"}
        score_match = _score_item(item, weather_str="Cloudy and wet")
        score_mismatch = _score_item(item, weather_str="Sunny and clear")
        assert score_match > score_mismatch

    def test_price_free(self, mock_llm):
        from weekend_llm import _score_item
        item = {"price": "Free"}
        score = _score_item(item)
        assert 0 <= score <= 5

    def test_price_paid(self, mock_llm):
        from weekend_llm import _score_item
        item = {"price": "$20"}
        score_paid = _score_item(item)
        item2 = {"price": "Free"}
        score_free = _score_item(item2)
        assert score_paid > score_free

    def test_long_location_bonus(self, mock_llm):
        from weekend_llm import _score_item
        item1 = {"location": "A" * 10}
        item2 = {"location": "X"}
        assert _score_item(item1) > _score_item(item2)


class TestFetchScoresForItems:
    """Test fetch_scores_for_items mutates in place."""

    def test_assigns_scores(self, mock_llm):
        from weekend_llm import fetch_scores_for_items
        items = [
            {"name": "A", "target_ages": "5-10", "weather": "outdoor"},
            {"name": "B"},
        ]
        fetch_scores_for_items(items, weather_str="Sunny", age_range="5-10")
        assert "score" in items[0]
        assert "score" in items[1]
        assert 0 <= items[0]["score"] <= 5
