"""Tests for lib.validators.json_validator."""



class TestExtractListFromDict:
    def test_dict_with_known_key(self):
        from lib.validators.json_validator import extract_list_from_dict

        data = {"activities": [{"name": "a"}]}
        assert extract_list_from_dict(data) == [{"name": "a"}]

    def test_dict_with_fixed_activities(self):
        from lib.validators.json_validator import extract_list_from_dict

        data = {"fixed_activities": [{"name": "a"}]}
        assert extract_list_from_dict(data) == [{"name": "a"}]

    def test_dict_with_transient_events(self):
        from lib.validators.json_validator import extract_list_from_dict

        data = {"transient_events": [{"name": "a"}]}
        assert extract_list_from_dict(data) == [{"name": "a"}]

    def test_dict_with_events(self):
        from lib.validators.json_validator import extract_list_from_dict

        data = {"events": [{"name": "a"}]}
        assert extract_list_from_dict(data) == [{"name": "a"}]

    def test_dict_with_items(self):
        from lib.validators.json_validator import extract_list_from_dict

        data = {"items": [{"name": "a"}]}
        assert extract_list_from_dict(data) == [{"name": "a"}]

    def test_dict_with_results(self):
        from lib.validators.json_validator import extract_list_from_dict

        data = {"results": [{"name": "a"}]}
        assert extract_list_from_dict(data) == [{"name": "a"}]

    def test_dict_with_data(self):
        from lib.validators.json_validator import extract_list_from_dict

        data = {"data": [{"name": "a"}]}
        assert extract_list_from_dict(data) == [{"name": "a"}]

    def test_dict_with_places(self):
        from lib.validators.json_validator import extract_list_from_dict

        data = {"places": [{"name": "a"}]}
        assert extract_list_from_dict(data) == [{"name": "a"}]

    def test_dict_with_venues(self):
        from lib.validators.json_validator import extract_list_from_dict

        data = {"venues": [{"name": "a"}]}
        assert extract_list_from_dict(data) == [{"name": "a"}]

    def test_dict_with_recommendations(self):
        from lib.validators.json_validator import extract_list_from_dict

        data = {"recommendations": [{"name": "a"}]}
        assert extract_list_from_dict(data) == [{"name": "a"}]

    def test_dict_with_nested_dict(self):
        from lib.validators.json_validator import extract_list_from_dict

        data = {"nested": {"items": [{"name": "a"}]}}
        result = extract_list_from_dict(data)
        assert result == [{"name": "a"}]

    def test_dict_with_longest_list(self):
        from lib.validators.json_validator import extract_list_from_dict

        data = {"short": [1], "long": [1, 2, 3]}
        # Known keys are checked first, so if "short" is not in keys, falls to length
        result = extract_list_from_dict(data)
        assert len(result) == 3

    def test_dict_no_list(self):
        from lib.validators.json_validator import extract_list_from_dict

        data = {"name": "x"}
        assert extract_list_from_dict(data) == []

    def test_list_input(self):
        from lib.validators.json_validator import extract_list_from_dict

        data = [{"name": "a"}]
        assert extract_list_from_dict(data) == [{"name": "a"}]

    def test_non_list_non_dict(self):
        from lib.validators.json_validator import extract_list_from_dict

        assert extract_list_from_dict("string") == []
        assert extract_list_from_dict(42) == []

    def test_dict_with_non_list_value_at_known_key(self):
        from lib.validators.json_validator import extract_list_from_dict

        # activities key but value is a dict, not a list
        data = {"activities": {"nested": 1}, "items": [{"name": "a"}]}
        result = extract_list_from_dict(data)
        assert result == [{"name": "a"}]


class TestIsValidListItem:
    def test_valid_string(self):
        from lib.validators.json_validator import is_valid_list_item

        assert is_valid_list_item("hello") is True

    def test_empty_string(self):
        from lib.validators.json_validator import is_valid_list_item

        assert is_valid_list_item("") is False

    def test_whitespace_string(self):
        from lib.validators.json_validator import is_valid_list_item

        assert is_valid_list_item("   ") is False

    def test_dict_with_name(self):
        from lib.validators.json_validator import is_valid_list_item

        assert is_valid_list_item({"name": "x"}) is True

    def test_dict_with_activity(self):
        from lib.validators.json_validator import is_valid_list_item

        assert is_valid_list_item({"activity": "x"}) is True

    def test_dict_without_name(self):
        from lib.validators.json_validator import is_valid_list_item

        assert is_valid_list_item({"other": "x"}) is False

    def test_empty_dict(self):
        from lib.validators.json_validator import is_valid_list_item

        assert is_valid_list_item({}) is False

    def test_int(self):
        from lib.validators.json_validator import is_valid_list_item

        assert is_valid_list_item(42) is False

    def test_none(self):
        from lib.validators.json_validator import is_valid_list_item

        assert is_valid_list_item(None) is False


class TestHasItemDetails:
    def test_name_only(self):
        from lib.validators.json_validator import has_item_details

        assert has_item_details({"name": "x"}) is False

    def test_name_and_location(self):
        from lib.validators.json_validator import has_item_details

        assert has_item_details({"name": "x", "location": "y"}) is True

    def test_name_and_age(self):
        from lib.validators.json_validator import has_item_details

        assert has_item_details({"name": "x", "target_ages": "5-10"}) is True

    def test_no_name(self):
        from lib.validators.json_validator import has_item_details

        assert has_item_details({"location": "x"}) is False

    def test_non_dict(self):
        from lib.validators.json_validator import has_item_details

        assert has_item_details("string") is False
        assert has_item_details(None) is False

    def test_event_as_name(self):
        from lib.validators.json_validator import has_item_details

        assert has_item_details({"event": "x", "day": "Monday"}) is True

    def test_title_as_name(self):
        from lib.validators.json_validator import has_item_details

        assert has_item_details({"title": "x", "time": "10am"}) is True


class TestCheckSourceExtraction:
    def test_empty_items(self):
        from lib.validators.json_validator import check_source_extraction

        assert check_source_extraction([], "source text") == 0.0

    def test_empty_source(self):
        from lib.validators.json_validator import check_source_extraction

        assert check_source_extraction([{"a": 1}], "") == 0.0

    def test_no_source_terms(self):
        from lib.validators.json_validator import check_source_extraction

        # All words too short or stopwords
        assert check_source_extraction([{"a": "x"}], "the a") == 0.0

    def test_perfect_match(self):
        from lib.validators.json_validator import check_source_extraction

        items = [{"name": "toronto library"}]
        source = "toronto library is a great place"
        score = check_source_extraction(items, source)
        assert score == 1.0

    def test_partial_match(self):
        from lib.validators.json_validator import check_source_extraction

        # Need at least 2 common terms per item to count as match
        items = [{"name": "toronto library event"}, {"name": "vancouver park show"}]
        source = "toronto library event vancouver park show other things"
        score = check_source_extraction(items, source)
        assert score == 1.0

    def test_string_items(self):
        from lib.validators.json_validator import check_source_extraction

        items = ["toronto library"]
        source = "toronto library is a place"
        score = check_source_extraction(items, source)
        assert score == 1.0

    def test_no_common_terms(self):
        from lib.validators.json_validator import check_source_extraction

        items = [{"name": "completely different stuff"}]
        source = "toronto library is a place"
        score = check_source_extraction(items, source)
        assert score == 0.0

    def test_other_item_types(self):
        from lib.validators.json_validator import check_source_extraction

        # Items with various types but only string "toronto" has source terms
        items = [42, None, "toronto library event"]
        source = "toronto library event is a place"
        # String "toronto library event" has 3 terms matching >= 2
        # 1 match out of 3 items
        score = check_source_extraction(items, source)
        # 1 string item matches (3 terms in "toronto library event" matches), 2 non-string skipped
        # ratio = 1/3 = 0.33
        assert score == 1.0 / 3.0


class TestGetSourceMatchingDetails:
    def test_empty_items(self):
        from lib.validators.json_validator import get_source_matching_details

        result = get_source_matching_details([], "source")
        assert result["matched"] == []
        assert result["unmatched"] == []
        assert result["ratio"] == 0.0

    def test_empty_source(self):
        from lib.validators.json_validator import get_source_matching_details

        result = get_source_matching_details([{"a": 1}], "")
        assert result["ratio"] == 0.0
        assert result["source_preview"] == ""

    def test_no_source_terms(self):
        from lib.validators.json_validator import get_source_matching_details

        result = get_source_matching_details([{"a": 1}], "a the")
        assert result["source_preview"] == "a the"

    def test_all_matched(self):
        from lib.validators.json_validator import get_source_matching_details

        items = [{"name": "toronto library"}]
        source = "toronto library event"
        result = get_source_matching_details(items, source)
        assert len(result["matched"]) == 1
        assert result["ratio"] == 1.0

    def test_mixed(self):
        from lib.validators.json_validator import get_source_matching_details

        items = [{"name": "toronto library"}, {"name": "vancouver park"}]
        source = "toronto library is fun. vancouver park is great."
        result = get_source_matching_details(items, source)
        assert len(result["matched"]) == 2
        assert result["ratio"] == 1.0

    def test_unmatched(self):
        from lib.validators.json_validator import get_source_matching_details

        items = [{"name": "completely different stuff"}]
        source = "toronto library is a place"
        result = get_source_matching_details(items, source)
        assert len(result["unmatched"]) == 1
        assert result["ratio"] == 0.0

    def test_string_items(self):
        from lib.validators.json_validator import get_source_matching_details

        items = ["toronto library"]
        source = "toronto library event"
        result = get_source_matching_details(items, source)
        assert len(result["matched"]) == 1

    def test_other_item_types(self):
        from lib.validators.json_validator import get_source_matching_details

        items = [42, None, {"name": "toronto"}]
        source = "toronto library event"
        result = get_source_matching_details(items, source)
        # 42 and None converted to "42"/"None", dict to "{'name': 'toronto'}"
        # None of these strings appear in source → ratio 0
        assert result["ratio"] == 0.0
        assert result["matched"] == []

    def test_unnamed_item(self):
        from lib.validators.json_validator import get_source_matching_details

        items = ["toronto library"]
        source = "toronto library event"
        result = get_source_matching_details(items, source)
        assert result["matched"] == ["toronto library"]

    def test_source_preview_truncated(self):
        from lib.validators.json_validator import get_source_matching_details

        result = get_source_matching_details([{"a": 1}], "")
        assert result["source_preview"] == ""
