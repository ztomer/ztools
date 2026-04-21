"""Tests for weekend planner JSON extraction."""
import pytest
import sys
from unittest.mock import MagicMock, patch

# Mock ddgs before importing weekend_planner
mock_ddgs = MagicMock()
mock_ddgs.DDGS = MagicMock()
sys.modules['ddgs'] = mock_ddgs


class TestWeekendJsonExtraction:
    """Test JSON extraction from LLM responses."""

    def test_extract_activities_key(self, mock_llm_response):
        """Extract when model returns 'activities' key."""
        response = mock_llm_response["json_with_activities"]
        # Simulate the extraction logic from weekend_planner
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
        from weekend_planner import normalize_llm_items

        items = [
            {"name": "Test", "age_group": "6-12"},
            {"name": "Test2", "setting": "indoor"},
        ]
        normalized = normalize_llm_items(items)
        assert normalized[0].get("target_ages") == "6-12"
        assert normalized[1].get("weather") == "indoor"

    def test_normalize_string_items(self):
        """Normalize items that are strings."""
        from weekend_planner import normalize_llm_items

        items = ["Simple item 1", "Simple item 2"]
        normalized = normalize_llm_items(items)
        assert normalized[0].get("name") == "Simple item 1"

    def test_normalize_empty_list(self):
        """Handle empty list."""
        from weekend_planner import normalize_llm_items

        normalized = normalize_llm_items([])
        assert normalized == []
