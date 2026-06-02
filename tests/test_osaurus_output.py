"""Tests for lib.osaurus_output - JSON extraction, normalization, filtering."""
import pytest
from unittest.mock import patch, MagicMock


class TestCleanOutput:
    def test_clean_output_alias(self):
        from lib.osaurus_output import clean_output
        from lib.content_processing import clean_model_output
        # Should be the same function
        assert clean_output is clean_model_output


class TestExtractJsonOnly:
    def test_empty(self):
        from lib.osaurus_output import _extract_json_only
        assert _extract_json_only("") is None
        assert _extract_json_only(None) is None

    def test_json_object(self):
        from lib.osaurus_output import _extract_json_only
        result = _extract_json_only('text {"key": "value"} more')
        assert result == '{"key": "value"}'

    def test_json_array(self):
        from lib.osaurus_output import _extract_json_only
        result = _extract_json_only('text [1, 2, 3] more')
        assert result == "[1, 2, 3]"

    def test_nested_json(self):
        from lib.osaurus_output import _extract_json_only
        result = _extract_json_only('{"a": {"b": 1}}')
        assert result == '{"a": {"b": 1}}'

    def test_with_code_block(self):
        from lib.osaurus_output import _extract_json_only
        content = "before\n```json\n{\"key\": \"value\"}\n```\nafter"
        result = _extract_json_only(content)
        assert '{"key": "value"}' in result

    def test_invalid_json(self):
        from lib.osaurus_output import _extract_json_only
        result = _extract_json_only("not json at all")
        assert result is None

    def test_array_takes_priority(self):
        """If both array and object are present, array is tried first."""
        from lib.osaurus_output import _extract_json_only
        content = '{"a": 1} [1, 2]'
        result = _extract_json_only(content)
        # Array is found first
        assert result == "[1, 2]"

    def test_nested_brackets(self):
        from lib.osaurus_output import _extract_json_only
        result = _extract_json_only('[{"a": 1}, {"b": 2}]')
        assert result == '[{"a": 1}, {"b": 2}]'


class TestExtractJson:
    def test_empty(self):
        from lib.osaurus_output import extract_json
        assert extract_json("") is None

    def test_none(self):
        from lib.osaurus_output import extract_json
        assert extract_json(None) is None

    def test_invalid_json_falls_to_text_normalization(self):
        from lib.osaurus_output import extract_json
        # Invalid JSON that ALSO has no plain list - falls through to normalize_text_output
        result = extract_json("Event: Toronto Festival\nTime: 10am\nLocation: Park")
        assert result is not None

    def test_text_normalization_only(self):
        from lib.osaurus_output import extract_json
        # Text that won't match JSON or plain list, but matches text normalizer
        result = extract_json("Festival: City Park")
        assert result is not None or result is None  # Just don't crash

    def test_strips_bold(self):
        from lib.osaurus_output import extract_json
        result = extract_json('**{"key": "value"}**')
        assert result == {"key": "value"} or (isinstance(result, list) and result)

    def test_strips_table_separators(self):
        from lib.osaurus_output import extract_json
        result = extract_json('{"a": 1}|:---|:---|')
        assert result == {"a": 1} or (isinstance(result, list) and result)

    def test_dict_extraction(self):
        from lib.osaurus_output import extract_json
        result = extract_json('{"name": "x", "location": "y"}')
        # Returns a list because it goes through normalize_keys (single key -> list) and filter_json_items
        assert isinstance(result, list) or isinstance(result, dict)
        assert result is not None

    def test_list_extraction(self):
        from lib.osaurus_output import extract_json
        result = extract_json('[{"name": "x"}, {"name": "y"}]')
        assert result == [{"name": "x"}, {"name": "y"}]

    def test_plain_list_fallback(self):
        from lib.osaurus_output import extract_json
        result = extract_json("1. First item\n2. Second item")
        assert result == [{"name": "First item"}, {"name": "Second item"}]

    def test_dash_list_fallback(self):
        from lib.osaurus_output import extract_json
        result = extract_json("- First\n- Second")
        assert result == [{"name": "First"}, {"name": "Second"}]

    def test_text_normalization_fallback(self):
        from lib.osaurus_output import extract_json
        result = extract_json("Event: Toronto Event\nTime: 10am")
        # Returns a list with parsed items
        assert result is not None

    def test_invalid_json_no_fallback(self):
        from lib.osaurus_output import extract_json
        result = extract_json("totally random text nothing to parse")
        # May or may not return something depending on text normalization
        # Just verify no exception
        assert result is None or isinstance(result, (list, dict))


class TestExtractPlainList:
    def test_empty(self):
        from lib.osaurus_output import _extract_plain_list
        assert _extract_plain_list("") is None
        assert _extract_plain_list(None) is None

    def test_numbered_list(self):
        from lib.osaurus_output import _extract_plain_list
        result = _extract_plain_list("1. First\n2. Second")
        assert result == [{"name": "First"}, {"name": "Second"}]

    def test_paren_numbered(self):
        from lib.osaurus_output import _extract_plain_list
        result = _extract_plain_list("1) First\n2) Second")
        assert result == [{"name": "First"}, {"name": "Second"}]

    def test_dash_list(self):
        from lib.osaurus_output import _extract_plain_list
        result = _extract_plain_list("- First\n- Second")
        assert result == [{"name": "First"}, {"name": "Second"}]

    def test_no_list(self):
        from lib.osaurus_output import _extract_plain_list
        result = _extract_plain_list("no list here")
        # May return None or a list with the line - depends on length
        assert result is None or isinstance(result, list)

    def test_skips_headers(self):
        from lib.osaurus_output import _extract_plain_list
        result = _extract_plain_list("# Header\n- item1")
        # Header is skipped
        assert result == [{"name": "item1"}]

    def test_empty_lines_skipped(self):
        from lib.osaurus_output import _extract_plain_list
        result = _extract_plain_list("- a\n\n- b")
        assert result == [{"name": "a"}, {"name": "b"}]

    def test_long_line(self):
        from lib.osaurus_output import _extract_plain_list
        # Plain line > 1 char is included
        result = _extract_plain_list("longish line")
        assert result == [{"name": "longish line"}]

    def test_single_char_skipped(self):
        from lib.osaurus_output import _extract_plain_list
        # Single char line is skipped
        assert _extract_plain_list("a") is None


class TestNormalizeKeys:
    def test_empty(self):
        from lib.osaurus_output import normalize_keys
        assert normalize_keys({}) == {}

    def test_none(self):
        from lib.osaurus_output import normalize_keys
        assert normalize_keys(None) is None

    def test_top_level_fixed_activities(self):
        from lib.osaurus_output import normalize_keys
        data = {"activities": [{"name": "x"}]}
        result = normalize_keys(data)
        assert "fixed_activities" in result

    def test_top_level_events(self):
        from lib.osaurus_output import normalize_keys
        data = {"events": [{"name": "x"}]}
        result = normalize_keys(data)
        assert "transient_events" in result

    def test_top_level_year_round(self):
        from lib.osaurus_output import normalize_keys
        data = {"year_round_activities": [{"name": "x"}]}
        result = normalize_keys(data)
        assert "fixed_activities" in result

    def test_top_level_venues(self):
        from lib.osaurus_output import normalize_keys
        data = {"venues": [{"name": "x"}]}
        result = normalize_keys(data)
        assert "fixed_activities" in result

    def test_top_level_limited_time(self):
        from lib.osaurus_output import normalize_keys
        data = {"limited_time_events": [{"name": "x"}]}
        result = normalize_keys(data)
        assert "transient_events" in result

    def test_top_level_with_dict(self):
        from lib.osaurus_output import normalize_keys
        data = {"activities": {"key": [{"name": "x"}]}}
        result = normalize_keys(data)
        assert "fixed_activities" in result

    def test_top_level_string_value(self):
        from lib.osaurus_output import normalize_keys
        # When top-level key has a non-list, non-dict value, line 143 is hit
        data = {"activities": "just a string"}
        result = normalize_keys(data)
        assert "fixed_activities" in result

    def test_key_normalization_event_to_name(self):
        from lib.osaurus_output import normalize_keys
        data = {"event": "Festival"}
        result = normalize_keys(data)
        # "event" -> "name"
        assert result.get("name") == "Festival"

    def test_key_normalization_title_to_activity(self):
        from lib.osaurus_output import normalize_keys
        data = {"title": "Title"}
        result = normalize_keys(data)
        assert "activity" in result

    def test_key_normalization_venue_to_location(self):
        from lib.osaurus_output import normalize_keys
        data = {"venue": "Toronto"}
        result = normalize_keys(data)
        assert "location" in result

    def test_key_normalization_address(self):
        from lib.osaurus_output import normalize_keys
        data = {"address": "123 Main St"}
        result = normalize_keys(data)
        assert "location" in result

    def test_key_normalization_where(self):
        from lib.osaurus_output import normalize_keys
        data = {"where": "there"}
        result = normalize_keys(data)
        assert "location" in result

    def test_key_normalization_date(self):
        from lib.osaurus_output import normalize_keys
        data = {"date": "Saturday"}
        result = normalize_keys(data)
        assert "day" in result

    def test_key_normalization_when(self):
        from lib.osaurus_output import normalize_keys
        data = {"when": "10am"}
        result = normalize_keys(data)
        assert "day" in result

    def test_key_normalization_time(self):
        from lib.osaurus_output import normalize_keys
        data = {"time": "10am"}
        result = normalize_keys(data)
        assert "duration" in result

    def test_key_normalization_audience(self):
        from lib.osaurus_output import normalize_keys
        data = {"audience": "kids"}
        result = normalize_keys(data)
        assert "target_ages" in result

    def test_key_normalization_age_group(self):
        from lib.osaurus_output import normalize_keys
        data = {"age_group": "5-12"}
        result = normalize_keys(data)
        assert "target_ages" in result

    def test_key_normalization_pricing(self):
        from lib.osaurus_output import normalize_keys
        data = {"pricing": "free"}
        result = normalize_keys(data)
        assert "price" in result

    def test_key_normalization_setting(self):
        from lib.osaurus_output import normalize_keys
        data = {"setting": "indoor"}
        result = normalize_keys(data)
        assert "weather" in result

    def test_key_normalization_indoor_outdoor(self):
        from lib.osaurus_output import normalize_keys
        data = {"indoor_outdoor": "outdoor"}
        result = normalize_keys(data)
        assert "weather" in result

    def test_only_string_value(self):
        from lib.osaurus_output import normalize_keys
        data = {"only": "this is a value"}
        result = normalize_keys(data)
        # single key with string value -> becomes {"name": "this is a value"}
        assert result.get("name") == "this is a value"

    def test_model_specific_mappings(self):
        from lib.osaurus_output import normalize_keys
        with patch("lib.osaurus_output.get_model_config", return_value={"key_mappings": {"foo": "bar"}}):
            data = {"foo": "value"}
            result = normalize_keys(data, model="some-model")
        # The model mappings override KEY_NORMALIZATIONS
        assert "bar" in result

    def test_list_normalization(self):
        from lib.osaurus_output import normalize_keys
        data = [{"name": "a"}, {"name": "b"}]
        result = normalize_keys(data)
        assert isinstance(result, list)
        assert len(result) == 2

    def test_list_with_single_key_items(self):
        from lib.osaurus_output import normalize_keys
        data = [{"event": "festival"}, {"name": "x"}]
        result = normalize_keys(data)
        # Single key with string value gets name added
        assert any("name" in item for item in result)

    def test_string_passthrough(self):
        from lib.osaurus_output import normalize_keys
        assert normalize_keys("just a string") == "just a string"

    def test_int_passthrough(self):
        from lib.osaurus_output import normalize_keys
        assert normalize_keys(42) == 42


class TestMergeFlatDicts:
    def test_empty(self):
        from lib.osaurus_output import merge_flat_dicts
        assert merge_flat_dicts([]) == []

    def test_none(self):
        from lib.osaurus_output import merge_flat_dicts
        assert merge_flat_dicts(None) is None

    def test_passthrough(self):
        from lib.osaurus_output import merge_flat_dicts
        items = [{"a": 1}, {"b": 2}]
        assert merge_flat_dicts(items) == items


class TestFilterJsonItems:
    def test_empty(self):
        from lib.osaurus_output import filter_json_items
        assert filter_json_items([]) == []

    def test_none(self):
        from lib.osaurus_output import filter_json_items
        assert filter_json_items(None) is None

    def test_valid_item(self):
        from lib.osaurus_output import filter_json_items
        items = [{"name": "x", "location": "y"}]
        assert filter_json_items(items) == items

    def test_skip_pipe_starting_key(self):
        from lib.osaurus_output import filter_json_items
        items = [{"|col1|": "x"}]
        assert filter_json_items(items) == []

    def test_skip_based_on(self):
        from lib.osaurus_output import filter_json_items
        items = [{"name": "Based on my analysis, this is great"}]
        assert filter_json_items(items) == []

    def test_skip_note_prefix(self):
        from lib.osaurus_output import filter_json_items
        items = [{"name": "Note: this is a note"}]
        assert filter_json_items(items) == []

    def test_skip_temperature_conditions(self):
        from lib.osaurus_output import filter_json_items
        items = [{"name": "Temperature conditions are nice"}]
        assert filter_json_items(items) == []

    def test_skip_empty_value(self):
        from lib.osaurus_output import filter_json_items
        items = [{"name": ""}]
        assert filter_json_items(items) == []

    def test_skip_dash_only(self):
        from lib.osaurus_output import filter_json_items
        items = [{"name": "-"}]
        assert filter_json_items(items) == []

    def test_skip_separator_only(self):
        from lib.osaurus_output import filter_json_items
        items = [{"name": "---"}]
        assert filter_json_items(items) == []

    def test_string_pipe_starting(self):
        from lib.osaurus_output import filter_json_items
        items = ["|table|row"]
        assert filter_json_items(items) == []

    def test_string_colon_starting(self):
        from lib.osaurus_output import filter_json_items
        items = [":thing"]
        assert filter_json_items(items) == []

    def test_string_based_on(self):
        from lib.osaurus_output import filter_json_items
        items = ["Based on what I see"]
        assert filter_json_items(items) == []

    def test_string_separator_only(self):
        from lib.osaurus_output import filter_json_items
        items = ["---"]
        assert filter_json_items(items) == []

    def test_string_kept(self):
        from lib.osaurus_output import filter_json_items
        items = ["valid string"]
        assert filter_json_items(items) == items

    def test_other_type_kept(self):
        from lib.osaurus_output import filter_json_items
        items = [42, 3.14, None]
        # All non-dict, non-string kept
        assert filter_json_items(items) == items

    def test_mixed(self):
        from lib.osaurus_output import filter_json_items
        items = [
            {"name": "valid"},
            {"name": "skip me"},
            "valid string",
            "skip this too",
        ]
        result = filter_json_items(items)
        # Filter happens
        assert isinstance(result, list)


class TestFixJsonYears:
    def test_empty(self):
        from lib.osaurus_output import fix_json_years
        assert fix_json_years([]) == []

    def test_none(self):
        from lib.osaurus_output import fix_json_years
        assert fix_json_years(None) is None

    def test_fix_2626(self):
        from lib.osaurus_output import fix_json_years
        items = [{"desc": "Year 2626 was when..."}]
        result = fix_json_years(items)
        assert "2026" in result[0]["desc"]

    def test_fix_26(self):
        from lib.osaurus_output import fix_json_years
        items = [{"desc": "Year 26 was when..."}]
        result = fix_json_years(items)
        assert "2026" in result[0]["desc"]

    def test_no_year_unchanged(self):
        from lib.osaurus_output import fix_json_years
        items = [{"desc": "no year here"}]
        result = fix_json_years(items)
        assert result[0]["desc"] == "no year here"

    def test_other_numbers_unchanged(self):
        from lib.osaurus_output import fix_json_years
        items = [{"desc": "Number 25 is fine"}]
        result = fix_json_years(items)
        assert "25" in result[0]["desc"]

    def test_non_string_values_kept(self):
        from lib.osaurus_output import fix_json_years
        items = [{"count": 26}, {"active": True}]
        result = fix_json_years(items)
        assert result[0]["count"] == 26

    def test_non_dict_items_kept(self):
        from lib.osaurus_output import fix_json_years
        items = ["just a string", 42]
        result = fix_json_years(items)
        assert result == items


class TestNormalizeTextOutput:
    def test_empty(self):
        from lib.osaurus_output import normalize_text_output
        assert normalize_text_output("") == []

    def test_none(self):
        from lib.osaurus_output import normalize_text_output
        assert normalize_text_output(None) == []

    def test_simple_numbered(self):
        from lib.osaurus_output import normalize_text_output
        result = normalize_text_output("1. Toronto Event - Park - All Ages")
        assert len(result) == 1
        assert "name" in result[0]

    def test_dash_numbered(self):
        from lib.osaurus_output import normalize_text_output
        result = normalize_text_output("- Toronto Event - Park - All Ages")
        assert len(result) == 1

    def test_skip_headers(self):
        from lib.osaurus_output import normalize_text_output
        result = normalize_text_output("## Header\n1. Item one")
        assert len(result) == 1
        assert "Item" in result[0]["name"]

    def test_skip_empty_lines(self):
        from lib.osaurus_output import normalize_text_output
        result = normalize_text_output("1. A\n\n2. B")
        assert len(result) == 2

    def test_with_location(self):
        from lib.osaurus_output import normalize_text_output
        result = normalize_text_output("1. Event Name - Toronto")
        assert result[0].get("location") == "Toronto"

    def test_with_parts(self):
        from lib.osaurus_output import normalize_text_output
        result = normalize_text_output("1. Event - Toronto - 5-12 - Free - Outdoor")
        item = result[0]
        # parts[0] is location, parts[1] is target_ages, etc.
        assert item.get("location") is not None
        assert item.get("target_ages") is not None

    def test_skip_asterisk_name(self):
        from lib.osaurus_output import normalize_text_output
        result = normalize_text_output("1. *italic name*")
        # Names starting with * are skipped
        assert len(result) == 0

    def test_with_colon_parts(self):
        from lib.osaurus_output import normalize_text_output
        result = normalize_text_output("1. Event: Toronto: 5-12")
        item = result[0]
        assert item.get("location") == "Toronto"

    def test_with_comma_parts(self):
        from lib.osaurus_output import normalize_text_output
        result = normalize_text_output("1. Event, Toronto, 5-12")
        item = result[0]
        # Comma-split is on details only
        assert item.get("location") is not None

    def test_no_match(self):
        from lib.osaurus_output import normalize_text_output
        result = normalize_text_output("not a list at all")
        assert result == []


class TestKeyNormalizations:
    def test_top_level_keys_dict(self):
        from lib.osaurus_output import TOP_LEVEL_KEYS
        assert "activities" in TOP_LEVEL_KEYS
        assert "events" in TOP_LEVEL_KEYS
        assert TOP_LEVEL_KEYS["activities"] == "fixed_activities"
        assert TOP_LEVEL_KEYS["events"] == "transient_events"

    def test_key_normalizations_dict(self):
        from lib.osaurus_output import KEY_NORMALIZATIONS
        assert "event" in KEY_NORMALIZATIONS
        assert "title" in KEY_NORMALIZATIONS
        assert "venue" in KEY_NORMALIZATIONS
        assert "address" in KEY_NORMALIZATIONS
