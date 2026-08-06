"""Tests for lib.validators.json_validator."""

from pathlib import Path


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


class TestValidateJson:
    def test_empty_data(self):
        from lib.validators.json_validator import validate_json

        score, msg = validate_json(None)
        assert score == 0
        assert "empty" in msg

    def test_empty_dict(self):
        from lib.validators.json_validator import validate_json

        score, msg = validate_json({})
        assert score == 0
        # Empty dict is falsy, returns empty response
        assert "empty" in msg

    def test_empty_list(self):
        from lib.validators.json_validator import validate_json

        score, msg = validate_json([])
        assert score == 0
        # Empty list is falsy, returns empty response
        assert "empty" in msg

    def test_dict_with_no_list_value(self):
        from lib.validators.json_validator import validate_json

        score, msg = validate_json({"key": "value"})
        assert score == 0
        assert "no items" in msg

    def test_non_list_non_dict(self):
        from lib.validators.json_validator import validate_json

        score, msg = validate_json(42)
        assert score == 0
        assert "no items" in msg

    def test_too_few_items(self):
        from lib.validators.json_validator import validate_json

        score, msg = validate_json([{"name": "x"}, {"name": "y"}])
        assert "only 2 items" in msg

    def test_good_count(self):
        from lib.validators.json_validator import validate_json

        items = [{"name": f"x{i}"} for i in range(10)]
        score, _ = validate_json(items)
        # JSON_STRUCTURE_WEIGHT + JSON_COUNT_GOOD + JSON_VALIDITY_WEIGHT = 20 + 25 + 30 = 75
        assert score == 75

    def test_ok_count(self):
        from lib.validators.json_validator import validate_json

        items = [{"name": f"x{i}"} for i in range(5)]
        score, _ = validate_json(items)
        # 20 + 15 + 30 = 65
        assert score == 65

    def test_all_invalid_items(self):
        from lib.validators.json_validator import validate_json

        items = [{}, {}]
        score, msg = validate_json(items)
        assert "0/2 items are valid" in msg

    def test_mostly_valid(self):
        from lib.validators.json_validator import validate_json

        items = [{"name": "x"}] * 8 + [{}] * 2  # 8/10 valid (80% >= 70% threshold)
        score, msg = validate_json(items)
        # Structure(20) + Count(25) + Validity partial(15) = 60
        assert score == 60
        assert "8/10 items are valid" in msg

    def test_source_match_high(self):
        from lib.validators.json_validator import validate_json

        # Items with 2-word names → "toronto park" appears in both source and items
        items = [{"name": f"toronto park {i}"} for i in range(10)]
        source = " ".join(f"toronto park event {i}" for i in range(10))
        score, msg = validate_json(items, source_text=source)
        # 10/10 items share 2+ terms with source → ratio 1.0
        # Structure(20) + Count(25) + Validity(30) + Source full(25) = 100
        assert score == 100
        assert msg == ""

    def test_source_match_low(self):
        from lib.validators.json_validator import validate_json

        items = [{"name": f"event {i}"} for i in range(10)]
        score, msg = validate_json(items, source_text="totally different text here")
        # 0 items share 2+ terms with source → ratio 0.0 → "hallucinated" failure
        assert "hallucinated" in msg
        assert score < 100

    def test_source_match_mid(self):
        from lib.validators.json_validator import validate_json

        items = [{"name": f"event {i}"} for i in range(10)]
        source = "event 0 event 1 event 2 event 3 event 4 event 5"
        score, _ = validate_json(items, source_text=source)
        # Mid match
        assert score > 20

    def test_dict_with_activities(self):
        from lib.validators.json_validator import validate_json

        data = {"activities": [{"name": f"x{i}"} for i in range(10)]}
        score, _ = validate_json(data)
        assert score > 50


class TestHasRequiredFields:
    def test_all_present(self):
        from lib.validators.json_validator import has_required_fields

        assert has_required_fields({"a": 1, "b": 2}, ["a", "b"]) is True

    def test_missing_field(self):
        from lib.validators.json_validator import has_required_fields

        assert has_required_fields({"a": 1}, ["a", "b"]) is False

    def test_empty_value(self):
        from lib.validators.json_validator import has_required_fields

        assert has_required_fields({"a": ""}, ["a"]) is False

    def test_none_value(self):
        from lib.validators.json_validator import has_required_fields

        assert has_required_fields({"a": None}, ["a"]) is False

    def test_no_required(self):
        from lib.validators.json_validator import has_required_fields

        assert has_required_fields({}, []) is True


class TestValidateDetailedJson:
    def test_empty(self):
        from lib.validators.json_validator import validate_detailed_json

        score, msg = validate_detailed_json(None)
        assert score == 0
        assert "empty" in msg

    def test_no_items(self):
        from lib.validators.json_validator import validate_detailed_json

        score, msg = validate_detailed_json({})
        assert score == 0

    def test_few_items(self):
        from lib.validators.json_validator import validate_detailed_json

        items = [{"name": f"x{i}", "location": "y"} for i in range(2)]
        score, msg = validate_detailed_json(items)
        assert "only 2 items" in msg

    def test_good_count_with_details(self):
        from lib.validators.json_validator import validate_detailed_json

        items = [{"name": f"x{i}", "location": "y", "age": "5"} for i in range(10)]
        score, _ = validate_detailed_json(items)
        # All have details
        assert score > 50

    def test_partial_details(self):
        from lib.validators.json_validator import validate_detailed_json

        items = [{"name": f"x{i}"} for i in range(10)]  # No details
        score, msg = validate_detailed_json(items)
        assert "no items with details" in msg

    def test_some_details(self):
        from lib.validators.json_validator import validate_detailed_json

        items = []
        for i in range(10):
            if i < 3:
                items.append({"name": f"x{i}", "location": "y"})
            else:
                items.append({"name": f"x{i}"})  # No details
        score, _ = validate_detailed_json(items)
        # Partial
        assert score > 0

    def test_all_unique_names(self):
        from lib.validators.json_validator import validate_detailed_json

        items = [{"name": f"x{i}", "location": "y"} for i in range(10)]
        score, _ = validate_detailed_json(items)
        # All unique names = bonus
        assert score > 50

    def test_duplicates(self):
        from lib.validators.json_validator import validate_detailed_json

        items = [{"name": "x", "location": "y"} for _ in range(10)]
        score, msg = validate_detailed_json(items)
        # 90% duplicates
        assert "duplicates" in msg

    def test_with_source_text(self):
        from lib.validators.json_validator import validate_detailed_json

        items = [{"name": f"event {i}", "location": "toronto"} for i in range(10)]
        source = "event 0 event 1 event 2 event 3 event 4 event 5 event 6 event 7 event 8 event 9 toronto"
        score, _ = validate_detailed_json(items, source_text=source)
        assert score > 30

    def test_source_hallucinated(self):
        from lib.validators.json_validator import validate_detailed_json

        items = [{"name": f"x{i}", "location": "y"} for i in range(10)]
        score, msg = validate_detailed_json(items, source_text="nothing related")
        assert "hallucinated" in msg

    def test_dict_input(self):
        from lib.validators.json_validator import validate_detailed_json

        data = {"items": [{"name": f"x{i}", "location": "y"} for i in range(10)]}
        score, _ = validate_detailed_json(data)
        assert score > 50

    def test_mostly_valid_details(self):
        from lib.validators.json_validator import validate_detailed_json

        items = []
        for i in range(10):
            if i < 9:
                items.append({"name": f"x{i}", "location": "y", "age": "5"})
            else:
                items.append({"name": f"x{i}", "location": "y"})
        score, _ = validate_detailed_json(items)
        # 9/10 have details
        assert score > 50

    def test_source_partial_match(self):
        from lib.validators.json_validator import validate_detailed_json

        items = [{"name": f"event {i}", "location": "toronto"} for i in range(10)]
        source = "event 0 event 1 event 2 event 3 event 4 event 5"
        score, _ = validate_detailed_json(items, source_text=source)
        assert score > 20

    def test_detailed_no_items_in_dict(self):
        from lib.validators.json_validator import validate_detailed_json

        score, msg = validate_detailed_json({"key": "value"})
        assert score == 0
        assert "no items" in msg

    def test_detailed_non_list_non_dict(self):
        from lib.validators.json_validator import validate_detailed_json

        score, msg = validate_detailed_json(42)
        assert score == 0
        assert "no items" in msg

    def test_source_extraction_empty_dict_item_text(self):
        """Item with all empty values produces empty item_text."""
        from lib.validators.json_validator import check_source_extraction

        items = [{"name": None}, {"name": ""}]  # All empty values
        source = "toronto library event"
        score = check_source_extraction(items, source)
        # No items contribute, ratio is 0
        assert score == 0.0

    def test_source_matching_details_empty_dict_item_text(self):
        from lib.validators.json_validator import get_source_matching_details

        items = [{"name": None}]
        result = get_source_matching_details(items, "toronto library event")
        # Empty item_text -> skipped
        assert result["matched"] == []
        assert result["unmatched"] == []

    def test_source_match_mid_score(self):
        from lib.validators.json_validator import validate_json

        # First 5 items share 2+ terms with source (matched), next 5 don't (unmatched)
        items = [{"name": f"toronto park {i}"} for i in range(5)] + [
            {"name": f"beach cafe {i}"} for i in range(5, 10)
        ]
        # 5/10 = 0.5 source ratio = mid tier
        source = " ".join("toronto park event" for _ in range(5))
        score, fail = validate_json(items, source_text=source)
        # Structure(20) + Count(25) + Validity(30) + Source mid(12) = 87
        assert score == 87
        assert fail == ""

    def test_source_match_minimal(self):
        from lib.validators.json_validator import validate_json

        items = [{"name": f"event {i}"} for i in range(10)]
        # Just 1/10 = 0.1 = minimal tier (but > 0). "event 0" is verbatim in the
        # source, so the substring fallback detects 1 grounded item.
        source = "event 0"
        score, _ = validate_json(items, source_text=source)
        # base (10 items, unique) 75 + minimal source tier (25//4 = 6) = 81
        assert score == 81

    def test_source_match_high_score(self):
        from lib.validators.json_validator import validate_json

        # All 10 items have unique names that match source
        items = [{"name": f"uniquename{i} with stuff"} for i in range(10)]
        source = " ".join(f"uniquename{i} with stuff" for i in range(10))
        score, _ = validate_json(items, source_text=source)
        # High tier
        assert score == 100

    def test_source_match_mid_score_six_of_ten(self):
        from lib.validators.json_validator import validate_json

        # 6/10 = 0.6 = mid tier
        items = [{"name": f"uniquename{i} with stuff"} for i in range(10)]
        source = " ".join(f"uniquename{i} with stuff" for i in range(6))
        score, _ = validate_json(items, source_text=source)
        # Mid tier (line 199): 10 items, 6 match source, 10 detailed
        assert score == 87

    def test_source_match_low_score(self):
        from lib.validators.json_validator import validate_json

        # 1/10 = 0.1 = low tier (between 0 and 0.5)
        items = [{"name": f"uniquename{i} with stuff"} for i in range(10)]
        source = "uniquename0 with stuff"
        score, _ = validate_json(items, source_text=source)
        # Low tier (line 201)
        assert score > 50  # still gets count + validity

    def test_detailed_count_ok(self):
        from lib.validators.json_validator import validate_detailed_json

        # 5-9 items = OK tier
        items = [{"name": f"x{i}", "location": "y"} for i in range(6)]
        score, _ = validate_detailed_json(items)
        # 15 (structure) + 10 (count ok) + ...
        assert score > 20

    def test_detailed_mostly_have_details(self):
        from lib.validators.json_validator import validate_detailed_json

        # 9/10 items have details (>= 80%), 1 has only name
        items = []
        for i in range(10):
            if i < 9:
                items.append({"name": f"x{i}", "location": "y", "age": "5"})
            else:
                # Only name - no details (name is not in DETAIL_FIELDS)
                items.append({"name": f"x{i}"})
        score, _ = validate_detailed_json(items)
        # DETAIL_PARTIAL_HIGH = 32 pts
        assert score > 30

    def test_detailed_source_high(self):
        from lib.validators.json_validator import validate_detailed_json

        items = [{"name": f"uniquename{i} toronto"} for i in range(10)]
        source = " ".join(f"uniquename{i} toronto" for i in range(10))
        score, _ = validate_detailed_json(items, source_text=source)
        # High source tier
        assert score > 30

    def test_detailed_source_mid(self):
        from lib.validators.json_validator import validate_detailed_json

        # 6/10 match
        items = [{"name": f"uniquename{i} toronto"} for i in range(10)]
        source = " ".join(f"uniquename{i} toronto" for i in range(6))
        score, _ = validate_detailed_json(items, source_text=source)
        # Mid source tier
        assert score > 20

    def test_detailed_source_low(self):
        from lib.validators.json_validator import validate_detailed_json

        # 1/10 match
        items = [{"name": f"uniquename{i} toronto"} for i in range(10)]
        source = "uniquename0 toronto"
        score, _ = validate_detailed_json(items, source_text=source)
        # Low source tier
        assert score > 10


class TestValidateMixedSignal:
    def test_perfect_signal_passes(self):
        from eval.tasks_core import WEEKEND_USR_TRANSIENT_MIXED
        from lib.validators.json_validator import parse_signal_noise, validate_mixed_signal

        sig, noise = parse_signal_noise(WEEKEND_USR_TRANSIENT_MIXED)
        # Output keeps every signal item, excludes all noise.
        items = [
            {
                "name": s,
                "location": "x",
                "target_ages": "6-13",
                "price": "Free",
                "weather": "indoor",
            }
            for s in sig
        ]
        score, reason = validate_mixed_signal(items, source_text=WEEKEND_USR_TRANSIENT_MIXED)
        assert score >= 90, reason
        assert "noise" not in reason

    def test_noise_included_fails(self):
        from eval.tasks_core import WEEKEND_USR_TRANSIENT_MIXED
        from lib.validators.json_validator import parse_signal_noise, validate_mixed_signal

        sig, noise = parse_signal_noise(WEEKEND_USR_TRANSIENT_MIXED)
        items = [
            {
                "name": s,
                "location": "x",
                "target_ages": "6-13",
                "price": "Free",
                "weather": "indoor",
            }
            for s in sig
        ]
        # Append several noise items that must be excluded.
        for n in noise[:4]:
            items.append(
                {
                    "name": n,
                    "location": "x",
                    "target_ages": "0-100",
                    "price": "Free",
                    "weather": "indoor",
                }
            )
        score, reason = validate_mixed_signal(items, source_text=WEEKEND_USR_TRANSIENT_MIXED)
        assert score < 90, reason
        assert "noise" in reason

    def test_score_capped_at_100(self):
        from eval.tasks_core import WEEKEND_USR_TRANSIENT_MIXED
        from lib.validators.json_validator import parse_signal_noise, validate_mixed_signal

        sig, noise = parse_signal_noise(WEEKEND_USR_TRANSIENT_MIXED)
        # Duplicate every signal item twice — recall must cap at 1.0, score at 100.
        items = []
        for s in sig:
            for _ in range(2):
                items.append(
                    {
                        "name": s,
                        "location": "x",
                        "target_ages": "6-13",
                        "price": "Free",
                        "weather": "indoor",
                    }
                )
        score, _ = validate_mixed_signal(items, source_text=WEEKEND_USR_TRANSIENT_MIXED)
        assert score <= 100, score


class TestValidateMixedSummary:
    def test_clean_summary_passes(self):
        from eval.tasks_core import TWITTER_PROMPT_MIXED
        from lib.validators.text_validator import validate_mixed_summary

        summary = (
            "@TechCrunch announced GPT-5. @Bloomberg: GDP grew. @LocalNews_TOR reopened CN Tower."
        )
        score, reason = validate_mixed_summary(summary, TWITTER_PROMPT_MIXED)
        assert score >= 90, reason

    def test_noise_summary_fails(self):
        from eval.tasks_core import TWITTER_PROMPT_MIXED
        from lib.validators.text_validator import validate_mixed_summary

        summary = (
            "@FakeNews reported aliens landed in Central Park. "
            "lorem ipsum dolor sit amet consectetur adipiscing. "
            "BUY NOW LIMITED TIME OFFER CLICK HERE. "
            "Cryptocurrency price prediction for next week. "
            "Also @LocalNews_TOR reopened CN Tower."
        )
        score, reason = validate_mixed_summary(summary, TWITTER_PROMPT_MIXED)
        assert score < 60, reason
        assert "noise" in reason


class TestValidateMixedFileSummary:
    def test_noise_file_fails(self):
        from eval.tasks_core import FILE_SUMMARY_PROMPT_MIXED
        from lib.validators.text_validator import validate_mixed_file_summary

        project_root = Path(__file__).parent.parent
        out = [
            {"path": str(project_root / "README.md"), "desc": "docs"},
            {"path": "/spam/buy_now/click_here.exe", "desc": "spam"},
        ]
        score, reason = validate_mixed_file_summary(out, FILE_SUMMARY_PROMPT_MIXED)
        assert score < 90, reason
        assert "noise" in reason

    def test_clean_file_summary_passes(self):
        from eval.tasks_core import FILE_SUMMARY_PROMPT_MIXED
        from lib.validators.text_validator import validate_mixed_file_summary

        project_root = Path(__file__).parent.parent
        out = [{"path": str(project_root / "README.md"), "desc": "docs"}]
        score, reason = validate_mixed_file_summary(out, FILE_SUMMARY_PROMPT_MIXED)
        assert "noise" not in reason


class TestValidateMixedFilename:
    def test_noise_filename_fails(self):
        from eval.tasks_core import RENAME_PROMPT_MIXED
        from lib.validators.text_validator import validate_mixed_filename

        out = ["manage_underperformers", "buy_now_click_here", "context_engineering"]
        score, reason = validate_mixed_filename(out, RENAME_PROMPT_MIXED)
        assert score < 90, reason
        assert "noise" in reason

    def test_clean_filename_passes(self):
        from eval.tasks_core import RENAME_PROMPT_MIXED
        from lib.validators.text_validator import validate_mixed_filename

        out = ["manage_underperformers", "scott_adams_essays", "context_engineering"]
        score, reason = validate_mixed_filename(out, RENAME_PROMPT_MIXED)
        assert "noise" not in reason
