"""Shared fixtures for ZTools tests."""
import sys
import os
import pytest
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def mock_llm_response():
    """Sample LLM responses for testing."""
    return {
        "json_with_activities": {
            "activities": [
                {"name": "Test Activity 1", "location": "Toronto", "target_ages": "6-12"},
                {"name": "Test Activity 2", "location": "Vaughan", "target_ages": "8-14"},
            ]
        },
        "json_with_fixed_activities": {
            "fixed_activities": [
                {"name": "ROM", "location": "Toronto", "target_ages": "6-12", "price": "$25", "weather": "indoor"},
            ]
        },
        "json_with_transient_events": {
            "transient_events": [
                {"name": "Spring Festival", "location": "Vaughan", "day": "Saturday"},
            ]
        },
        "qwen_thinking_response": """Here's a thinking process:
1. Analyze the request
2. Formulate response

Output Generation.
{"activities": [{"name": "Test Event"}]}
stats:123""",
        "twitter_response": """Here's a thinking process:
Think about this carefully.

Output: ## Summary
- Main point
- Another point

stats:456""",
    }


@pytest.fixture
def sample_events_data():
    """Sample events data for testing."""
    return """- Event 1 (Toronto): Details here
- Event 2 (Vaughan): More details"""


@pytest.fixture
def sample_venues_data():
    """Sample venues data for testing."""
    return """- Venue 1 (123 Main St): Great place
- Venue 2 (456 Oak Ave): Another great place"""


@pytest.fixture
def sample_tweets():
    """Sample tweets for testing."""
    return [
        {"screen_name": "user1", "text": "Test tweet 1", "created_at": "2026-04-21"},
        {"screen_name": "user2", "text": "Test tweet 2", "created_at": "2026-04-21"},
    ]
