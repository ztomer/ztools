"""Tests for lib.quality_weekend_scorers - all weekend scoring functions."""

import json

import pytest

from lib.quality_models import TestCase


def make_case(input_text="", reference="", description="test", task="weekend_fixed"):
    return TestCase(
        task=task,
        input_text=input_text,
        reference=reference,
        description=description,
    )


class TestExtractItems:
    def test_valid_json_list(self):
        from lib.quality_weekend_scorers import _extract_items

        items, fails = _extract_items('[{"name":"Park"}, {"name":"Museum"}]')
        assert len(items) == 2
        assert fails == []

    def test_valid_json_dict_with_items(self):
        from lib.quality_weekend_scorers import _extract_items

        items, fails = _extract_items('{"activities":[{"name":"A"},{"name":"B"}]}')
        assert len(items) == 2
        assert fails == []

    def test_dict_without_list_values(self):
        from lib.quality_weekend_scorers import _extract_items

        items, fails = _extract_items('{"name":"just a name"}')
        assert items == []
        assert "no dict items" in fails[0]

    def test_invalid_json(self):
        from lib.quality_weekend_scorers import _extract_items

        items, fails = _extract_items("{bad json}")
        assert items == []
        assert "invalid JSON" in fails[0]

    def test_empty_string(self):
        from lib.quality_weekend_scorers import _extract_items

        items, fails = _extract_items("")
        assert items == []
        assert fails == ["empty output"]

    def test_not_a_list_or_dict(self):
        from lib.quality_weekend_scorers import _extract_items

        items, fails = _extract_items('"just a string"')
        assert items == []
        assert "not a list" in fails[0]

    def test_list_with_non_dict_items(self):
        from lib.quality_weekend_scorers import _extract_items

        items, fails = _extract_items('[1, "two", null]')
        assert items == []
        assert "no dict items" in fails[0]


class TestParseReference:
    def test_valid_json(self):
        from lib.quality_weekend_scorers import _parse_reference

        case = make_case(reference=json.dumps({"age_range": [3, 8], "weather": {"sat": "rain"}}))
        ref = _parse_reference(case)
        assert ref["age_range"] == [3, 8]
        assert ref["weather"] == {"sat": "rain"}

    def test_invalid_json_uses_defaults(self):
        from lib.quality_weekend_scorers import _parse_reference

        case = make_case(reference="not json")
        ref = _parse_reference(case)
        assert ref["age_range"] == [6, 13]
        assert ref["weather"] == {}

    def test_empty_reference_uses_defaults(self):
        from lib.quality_weekend_scorers import _parse_reference

        case = make_case(reference="")
        ref = _parse_reference(case)
        assert ref["age_range"] == [6, 13]
        assert ref["exclude"] == []
        assert ref["expected_count"] == [5, 10]

    def test_partial_reference_merges_defaults(self):
        from lib.quality_weekend_scorers import _parse_reference

        case = make_case(reference=json.dumps({"age_range": [2, 5]}))
        ref = _parse_reference(case)
        assert ref["age_range"] == [2, 5]
        assert ref["weather"] == {}


class TestHasField:
    def test_field_present(self):
        from lib.quality_weekend_scorers import _has_field

        assert _has_field({"name": "Park"}, "name")

    def test_field_empty_string(self):
        from lib.quality_weekend_scorers import _has_field

        assert not _has_field({"name": ""}, "name")

    def test_field_missing(self):
        from lib.quality_weekend_scorers import _has_field

        assert not _has_field({"other": "val"}, "name")

    def test_field_none(self):
        from lib.quality_weekend_scorers import _has_field

        assert not _has_field({"name": None}, "name")


class TestAgeOverlap:
    def test_exact_match(self):
        from lib.quality_weekend_scorers import _age_overlap

        assert _age_overlap("6-13", 6, 13) == 1.0

    def test_partial_overlap(self):
        from lib.quality_weekend_scorers import _age_overlap

        overlap = _age_overlap("10-16", 6, 13)
        assert 0.3 < overlap < 0.6

    def test_no_overlap(self):
        from lib.quality_weekend_scorers import _age_overlap

        assert _age_overlap("20-30", 6, 13) == 0.0

    def test_nested_in_range(self):
        from lib.quality_weekend_scorers import _age_overlap

        # 8-10 ages fall entirely within ref range 6-13 with span 3 vs ref span 8
        assert _age_overlap("8-10", 6, 13) == 3 / max(3, 8)

    def test_empty_string(self):
        from lib.quality_weekend_scorers import _age_overlap

        assert _age_overlap("", 6, 13) == 0.0

    def test_single_age(self):
        from lib.quality_weekend_scorers import _age_overlap

        # single age 8 has span 1 vs ref span 8
        assert _age_overlap("8", 6, 13) == 1 / max(1, 8)

    def test_more_specific_outside(self):
        from lib.quality_weekend_scorers import _age_overlap

        overlap = _age_overlap("2-4", 6, 13)
        assert overlap == 0.0


class TestWeekendCompleteness:
    def test_all_fields_present(self):
        from lib.quality_weekend_scorers import _score_weekend_completeness as fn

        case = make_case()
        items = [{"name": "P", "location": "L", "target_ages": "6+", "price": "$", "weather": "in"}]
        s = fn(json.dumps(items), case)
        assert s.score == 100.0
        assert s.failures == []

    def test_missing_fields(self):
        from lib.quality_weekend_scorers import _score_weekend_completeness as fn

        case = make_case()
        items = [{"name": "P", "location": "L"}]
        s = fn(json.dumps(items), case)
        assert s.score == 0.0
        assert len(s.failures) > 0

    def test_empty_output(self):
        from lib.quality_weekend_scorers import _score_weekend_completeness as fn

        case = make_case()
        s = fn("", case)
        assert s.score == 0
        assert "empty output" in s.failures

    def test_partial_completeness(self):
        from lib.quality_weekend_scorers import _score_weekend_completeness as fn

        case = make_case()
        items = [
            {"name": "P", "location": "L", "target_ages": "6+", "price": "$", "weather": "in"},
            {"name": "Q", "location": "M"},
        ]
        s = fn(json.dumps(items), case)
        assert s.score == 50.0


class TestWeekendWeatherMatch:
    def test_indoor_for_rainy(self):
        from lib.quality_weekend_scorers import _score_weekend_weather_match as fn

        case = make_case(reference=json.dumps({"weather": {"sat": "rain"}}))
        items = [{"name": "P", "weather": "indoor", "day": "sat"}]
        s = fn(json.dumps(items), case)
        assert s.score == 100.0

    def test_outdoor_for_clear(self):
        from lib.quality_weekend_scorers import _score_weekend_weather_match as fn

        case = make_case(reference=json.dumps({"weather": {"sun": "sunny"}}))
        items = [{"name": "P", "weather": "outdoor", "day": "sun"}]
        s = fn(json.dumps(items), case)
        assert s.score == 100.0

    def test_mismatch_gets_partial(self):
        from lib.quality_weekend_scorers import _score_weekend_weather_match as fn

        case = make_case(reference=json.dumps({"weather": {"sat": "rain"}}))
        items = [{"name": "P", "weather": "outdoor", "day": "sat"}]
        s = fn(json.dumps(items), case)
        assert s.score < 100.0
        assert s.score >= 25

    def test_no_weather_reference_gets_full(self):
        from lib.quality_weekend_scorers import _score_weekend_weather_match as fn

        case = make_case(reference=json.dumps({"weather": {}}))
        items = [{"name": "P", "weather": "outdoor", "day": "sat"}]
        s = fn(json.dumps(items), case)
        assert s.score == 100.0

    def test_no_weather_on_item_skipped(self):
        from lib.quality_weekend_scorers import _score_weekend_weather_match as fn

        case = make_case(reference=json.dumps({"weather": {"sat": "rain"}}))
        items = [{"name": "P", "weather": "", "day": "sat"}]
        s = fn(json.dumps(items), case)
        assert s.score == 50  # no weather-labeled items → default 50

    def test_empty_output(self):
        from lib.quality_weekend_scorers import _score_weekend_weather_match as fn

        case = make_case()
        s = fn("", case)
        assert s.score == 0


class TestWeekendAgeMatch:
    def test_exact_age(self):
        from lib.quality_weekend_scorers import _score_weekend_age_match as fn

        case = make_case(reference=json.dumps({"age_range": [6, 13]}))
        items = [{"name": "P", "target_ages": "6-13"}]
        s = fn(json.dumps(items), case)
        assert s.score == 100.0

    def test_partial_age_overlap(self):
        from lib.quality_weekend_scorers import _score_weekend_age_match as fn

        case = make_case(reference=json.dumps({"age_range": [6, 13]}))
        items = [{"name": "P", "target_ages": "10-16"}]
        s = fn(json.dumps(items), case)
        assert 0 < s.score < 100

    def test_no_target_ages_gets_half(self):
        from lib.quality_weekend_scorers import _score_weekend_age_match as fn

        case = make_case(reference=json.dumps({"age_range": [6, 13]}))
        items = [{"name": "P", "target_ages": ""}]
        s = fn(json.dumps(items), case)
        assert s.score == 50.0

    def test_empty_output(self):
        from lib.quality_weekend_scorers import _score_weekend_age_match as fn

        case = make_case()
        s = fn("", case)
        assert s.score == 0


class TestWeekendSourceGrounding:
    def test_match_by_source_names(self):
        from lib.quality_weekend_scorers import _score_weekend_source_grounding as fn

        case = make_case(
            input_text="some source text",
            reference=json.dumps({"source_item_names": ["Central Park", "Museum"]}),
        )
        items = [{"name": "Central Park visit"}, {"name": "something else"}]
        s = fn(json.dumps(items), case)
        assert s.score == 50.0

    def test_match_by_token_overlap(self):
        from lib.quality_weekend_scorers import _score_weekend_source_grounding as fn

        case = make_case(
            input_text="Central Park and Museum of Art",
            reference=json.dumps({"source_item_names": []}),
        )
        items = [{"name": "Park visit"}, {"name": "Art show"}]
        s = fn(json.dumps(items), case)
        assert s.score > 0

    def test_no_match(self):
        from lib.quality_weekend_scorers import _score_weekend_source_grounding as fn

        case = make_case(
            input_text="completely different content",
            reference=json.dumps({"source_item_names": []}),
        )
        items = [{"name": "XYZ"}, {"name": "ABC"}]
        s = fn(json.dumps(items), case)
        assert s.score == 0.0

    def test_empty_output(self):
        from lib.quality_weekend_scorers import _score_weekend_source_grounding as fn

        case = make_case()
        s = fn("", case)
        assert s.score == 0


class TestWeekendExclusions:
    def test_no_exclusions(self):
        from lib.quality_weekend_scorers import _score_weekend_exclusions as fn

        case = make_case(reference=json.dumps({"exclude": []}))
        items = [{"name": "P"}, {"name": "Q"}]
        s = fn(json.dumps(items), case)
        assert s.score == 100.0

    def test_excluded_items_deducted(self):
        from lib.quality_weekend_scorers import _score_weekend_exclusions as fn

        case = make_case(reference=json.dumps({"exclude": ["McDonald"]}))
        items = [{"name": "McDonald visit"}, {"name": "Park"}]
        s = fn(json.dumps(items), case)
        assert s.score == 50.0

    def test_exclude_by_location(self):
        from lib.quality_weekend_scorers import _score_weekend_exclusions as fn

        case = make_case(reference=json.dumps({"exclude": ["mall"]}))
        items = [
            {"name": "Shop", "location": "Westfield mall"},
            {"name": "Park", "location": "Green"},
        ]
        s = fn(json.dumps(items), case)
        assert s.score == 50.0

    def test_all_excluded(self):
        from lib.quality_weekend_scorers import _score_weekend_exclusions as fn

        case = make_case(reference=json.dumps({"exclude": ["McDonald"]}))
        items = [{"name": "McDonald lunch"}]
        s = fn(json.dumps(items), case)
        assert s.score == 0.0

    def test_empty_output(self):
        from lib.quality_weekend_scorers import _score_weekend_exclusions as fn

        case = make_case()
        s = fn("", case)
        assert s.score == 0


@pytest.fixture
def mock_llm():
    from lib.testing import MockLLM

    mock = MockLLM()
    mock.patch_all()
    yield mock
    mock.unpatch()


class TestWeekendScorersIntegration:
    def test_tasks_register_correctly(self):
        from lib.quality_scorers import TASK_SCORERS, get_scorers

        trans = get_scorers("weekend_transient")
        fixed = get_scorers("weekend_fixed")
        assert len(trans) == 5
        assert len(fixed) == 5
        assert trans == fixed  # same functions
        assert "weekend_transient" in TASK_SCORERS
        assert "weekend_fixed" in TASK_SCORERS

    def test_full_scoring_workflow(self, mock_llm):
        from lib.quality_weekend_scorers import _score_weekend_completeness

        ref = json.dumps(
            {
                "age_range": [6, 13],
                "weather": {"sat": "rain"},
                "exclude": [],
                "source_item_names": ["Park"],
            }
        )
        case = make_case(input_text="Park data", reference=ref)
        items = json.dumps(
            [
                {
                    "name": "Park",
                    "location": "Main",
                    "target_ages": "6-12",
                    "price": "free",
                    "weather": "indoor",
                    "day": "sat",
                }
            ]
        )
        s = _score_weekend_completeness(items, case)
        assert s.score == 100.0
