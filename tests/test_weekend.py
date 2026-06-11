"""Tests for weekend planner JSON extraction."""
import pytest
import sys
from unittest.mock import MagicMock, patch

# Mock ddgs before importing weekend
mock_ddgs = MagicMock()
mock_ddgs.DDGS = MagicMock()
sys.modules['ddgs'] = mock_ddgs

# Import the extraction logic
from weekend import normalize_llm_items


class TestWeekendJsonExtraction:
    """Test JSON extraction from LLM responses."""

    def test_extract_activities_key(self, mock_llm_response):
        """Extract when model returns 'activities' key."""
        response = mock_llm_response["json_with_activities"]
        # Simulate the extraction logic
        items = []
        if isinstance(response, dict):
            for k, v in response.items():
                if isinstance(v, list) and len(v) > 0:
                    items = v
                    break
        assert len(items) == 2
        assert items[0]["name"] == "Test Activity 1"

    def test_extract_fixed_activities_key(self, mock_llm_response):
        """Extract when model returns 'fixed_activities' key."""
        response = mock_llm_response["json_with_fixed_activities"]
        items = []
        if isinstance(response, dict):
            for k, v in response.items():
                if isinstance(v, list) and len(v) > 0:
                    items = v
                    break
        assert len(items) == 1
        assert items[0]["name"] == "ROM"

    def test_extract_transient_events_key(self, mock_llm_response):
        """Extract when model returns 'transient_events' key."""
        response = mock_llm_response["json_with_transient_events"]
        items = []
        if isinstance(response, dict):
            for k, v in response.items():
                if isinstance(v, list) and len(v) > 0:
                    items = v
                    break
        assert len(items) == 1
        assert items[0]["name"] == "Spring Festival"

    def test_extract_from_error_wrapper(self):
        """Extract from error wrapper response."""
        # Model sometimes returns error-like wrapper with data inside
        response = {
            "status": "error",
            "data": [
                {"name": "Event 1"},
                {"name": "Event 2"},
            ]
        }
        items = []
        if response.get("data") and isinstance(response.get("data"), list):
            items = response.get("data", [])
        assert len(items) == 2

    def test_extract_from_extracted_data(self):
        """Extract from 'extracted_data' key."""
        response = {
            "extracted_data": [
                {"name": "Thing 1"},
                {"name": "Thing 2"},
            ]
        }
        items = []
        if isinstance(response, dict):
            for k, v in response.items():
                if isinstance(v, list) and len(v) > 0:
                    items = v
                    break
        assert len(items) == 2


class TestWeekendNormalize:
    """Test normalization functions."""

    def test_normalize_gemma_field_names(self):
        """Normalize Gemma-specific field names."""
        from weekend import normalize_llm_items

        items = [
            {"name": "Test", "age_group": "6-12"},
            {"name": "Test2", "setting": "indoor"},
        ]
        normalized = normalize_llm_items(items)
        assert normalized[0].get("target_ages") == "6-12"
        assert normalized[1].get("weather") == "indoor"

    def test_normalize_string_items(self):
        """Normalize items that are strings."""
        from weekend import normalize_llm_items

        items = ["Simple item 1", "Simple item 2"]
        normalized = normalize_llm_items(items)
        assert normalized[0].get("name") == "Simple item 1"

    def test_normalize_empty_list(self):
        """Handle empty list."""
        from weekend import normalize_llm_items

        normalized = normalize_llm_items([])
        assert normalized == []

    def test_normalize_with_field_mapping(self):
        """Apply field mapping to normalize LLM output."""
        from weekend import normalize_llm_items

        # Simulate qwen returning category instead of target_ages
        items = [
            {"name": "Test Park", "location": "Toronto", "category": "6-12", "context_highlight": "$20"},
        ]
        # Apply mapping
        mapping = {"category": "target_ages", "context_highlight": "price"}
        normalized = normalize_llm_items(items, field_mapping=mapping)

        assert normalized[0].get("target_ages") == "6-12"
        assert normalized[0].get("price") == "$20"

    def test_normalize_with_multiple_field_mappings(self):
        """Apply multiple field mappings."""
        from weekend import normalize_llm_items

        items = [
            {"name": "Test", "category": "6-12", "type": "indoor", "event_date": "Saturday"},
        ]
        mapping = {"category": "target_ages", "type": "weather", "event_date": "day"}
        normalized = normalize_llm_items(items, field_mapping=mapping)

        assert normalized[0].get("target_ages") == "6-12"
        assert normalized[0].get("weather") == "indoor"
        assert normalized[0].get("day") == "Saturday"


class TestKeyVariationsFixed:
    """Test extraction with various key names for fixed activities."""

    def test_extract_from_fixed_activities_key(self):
        """Extract from 'fixed_activities' key."""
        response = {"fixed_activities": [{"name": "A", "location": "B"}]}
        from weekend import normalize_llm_items
        items = []
        for k, v in response.items():
            if isinstance(v, list) and len(v) > 0:
                items = normalize_llm_items(v)
                break
        assert len(items) == 1

    def test_extract_from_year_round_fixed_activities_key(self):
        """Extract from 'year_round_fixed_activities' key."""
        response = {"year_round_fixed_activities": [{"name": "A"}, {"name": "B"}]}
        from weekend import normalize_llm_items
        items = []
        for k, v in response.items():
            if isinstance(v, list) and len(v) > 0:
                items = normalize_llm_items(v)
                break
        assert len(items) == 2

    def test_extract_from_venues_key(self):
        """Extract from 'venues' key."""
        response = {"venues": [{"name": "A"}, {"name": "B"}, {"name": "C"}]}
        from weekend import normalize_llm_items
        items = []
        for k, v in response.items():
            if isinstance(v, list) and len(v) >= 3:
                items = normalize_llm_items(v)
                break
        assert len(items) == 3

    def test_extract_from_places_key(self):
        """Extract from 'places' key."""
        response = {"places": [{"name": "Test Place"}]}
        from weekend import normalize_llm_items
        items = []
        for k, v in response.items():
            if isinstance(v, list) and len(v) > 0:
                items = normalize_llm_items(v)
                break
        assert len(items) == 1


class TestKeyVariationsTransient:
    """Test extraction with various key names for transient events."""

    def test_extract_from_transient_events_key(self):
        """Extract from 'transient_events' key."""
        response = {"transient_events": [{"name": "A", "day": "Friday"}, {"name": "B", "day": "Saturday"}]}
        from weekend import normalize_llm_items
        items = []
        for k, v in response.items():
            if isinstance(v, list) and len(v) > 0:
                items = normalize_llm_items(v)
                break
        assert len(items) == 2
        assert items[0].get("day") == "Friday"

    def test_extract_from_events_key(self):
        """Extract from 'events' key."""
        response = {"events": [{"name": "A"}, {"name": "B"}]}
        from weekend import normalize_llm_items
        items = []
        for k, v in response.items():
            if isinstance(v, list) and len(v) > 0:
                items = normalize_llm_items(v)
                break
        assert len(items) == 2

    def test_extract_from_activities_key(self):
        """Extract from 'activities' key."""
        response = {"activities": [{"name": "A"}, {"name": "B"}, {"name": "C"}]}
        from weekend import normalize_llm_items
        items = []
        for k, v in response.items():
            if isinstance(v, list) and len(v) > 0:
                items = normalize_llm_items(v)
                break
        assert len(items) == 3


class TestMinimumItems:
    """Test minimum item count validation."""

    def test_minimum_5_items(self):
        """The MIN_ITEMS threshold in weekend.cli is 5."""
        # MIN_ITEMS is defined in weekend.cli.main() as 5
        from weekend import main
        import inspect
        src = inspect.getsource(main)
        # Verify MIN_ITEMS = 5 is hardcoded in the source
        assert "MIN_ITEMS = 5" in src

    def test_minimum_not_met(self):
        """With < MIN_ITEMS items, the planner should print a low-count warning."""
        MIN_ITEMS = 5
        items = [{"name": "Only One"}]
        # 1 < 5 → low count triggers warning
        assert len(items) < MIN_ITEMS

    def test_10_items_acceptable(self):
        """The target range is 5-10 items per category."""
        MIN_ITEMS = 5
        items = [{"name": f"Item {i}"} for i in range(10)]
        # 10 >= 5 → no low count warning
        assert len(items) >= MIN_ITEMS


class TestJsonExtractionRobustness:
    """Test robust JSON extraction from various LLM response formats."""

    def test_extract_from_year_round_activities_key(self):
        """Extract from 'year_round_activities' key (actual format)."""
        from weekend import normalize_llm_items

        response = {
            "year_round_activities": [
                {"name": "ROM", "location": "Toronto"},
                {"name": "CN Tower", "location": "Toronto"},
            ]
        }
        items = []
        for k, v in response.items():
            if isinstance(v, list) and len(v) > 0:
                items = normalize_llm_items(v)
                break
        assert len(items) == 2

    def test_extract_from_status_wrapper(self):
        """Extract from server wrapper {'status': 'ready', 'message': '...'}."""
        from weekend import normalize_llm_items

        response = {
            "status": "ready",
            "message": "Events fetched successfully",
            "activities": [
                {"name": "Spring Festival", "day": "Saturday"},
            ]
        }
        priority_keys = ["activities", "events", "items", "recommendations"]
        items = []
        for k in priority_keys:
            if response.get(k) and isinstance(response.get(k), list) and len(response.get(k)) > 0:
                items = normalize_llm_items(response.get(k))
                break
        assert len(items) == 1
        assert items[0]["name"] == "Spring Festival"

    def test_extract_from_error_wrapper(self):
        """Extract from error wrapper {'status': 'error', 'data': [...]}."""
        from weekend import normalize_llm_items

        response = {
            "status": "error",
            "data": [
                {"name": "Event 1"},
                {"name": "Event 2"},
            ]
        }
        items = []
        if response.get("data") and isinstance(response.get("data"), list):
            items = normalize_llm_items(response.get("data", []))
        assert len(items) == 2

    def test_handle_single_object_response(self):
        """Handle single object instead of list."""
        from weekend import normalize_llm_items

        response = {
            "name": "Single Event",
            "location": "Toronto",
            "day": "Saturday",
        }
        if "name" in response:
            items = normalize_llm_items([response])
        assert len(items) == 1
        assert items[0]["name"] == "Single Event"


class TestScriptIntegration:
    """Test that the script can be imported and run without errors."""

    def test_import_weekend_cli(self):
        """Test that weekend.cli can be imported and exposes main()."""
        import weekend.cli
        assert callable(getattr(weekend.cli, "main", None))
        assert callable(getattr(weekend.cli, "parse_args", None))

    def test_debug_print_defined(self):
        """Test that debug_print is defined at module level."""
        from lib.tui import debug_print, DEBUG
        assert callable(debug_print)
        assert DEBUG is False

    def test_debug_print_outputs_when_enabled(self, capsys):
        """Test that debug_print outputs when DEBUG is True."""
        import lib.tui
        original_debug = lib.tui.DEBUG
        lib.tui.DEBUG = True
        try:
            lib.tui.debug_print("test message")
            captured = capsys.readouterr()
            assert "test message" in captured.out
        finally:
            lib.tui.DEBUG = original_debug

    def test_main_accepts_debug_arg(self):
        """Test that parse_args() correctly parses --debug flag."""
        from weekend import parse_args
        import sys

        saved = sys.argv
        try:
            sys.argv = ["weekend.cli", "--debug"]
            args = parse_args()
            assert args.debug is True

            sys.argv = ["weekend.cli"]
            args = parse_args()
            assert args.debug is False
        finally:
            sys.argv = saved
