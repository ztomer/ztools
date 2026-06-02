"""
Pytest configuration: shared fixtures for all tests.
"""

import sys
import os
import pytest
from pathlib import Path

from lib.testing import MockLLM

sys.path.insert(0, str(Path(__file__).parent.parent))


# Capture references to lib.mlx_lib functions at conftest load time.
# This MUST happen before any mock patches them.
import lib.mlx_lib  # noqa: E402
_REAL_MLX_FUNCTIONS = {
    "call": lib.mlx_lib.call,
    "call_mlx": lib.mlx_lib.call_mlx,
    "process_mlx_content": lib.mlx_lib.process_mlx_content,
    "find_mlx_model": lib.mlx_lib.find_mlx_model,
    "find_text_mlx_model": lib.mlx_lib.find_text_mlx_model,
}


@pytest.fixture
def real_mlx_functions():
    """Return real (unmocked) lib.mlx_lib functions.

    Captures references at conftest load time, before any mock patches.
    """
    return _REAL_MLX_FUNCTIONS


@pytest.fixture
def mock_llm():
    """Fixture that patches all LLM functions with a MockLLM provider.

    Usage:
        def test_something(mock_llm):
            import eval_run as er
            from unittest.mock import patch
            with patch.object(er, "call", mock_llm.call):
                result = er.run_eval(...)
    """
    mock = MockLLM()
    mock.patch_all()
    yield mock
    mock.unpatch()


@pytest.fixture
def mock_llm_osaurus():
    """Same as mock_llm but only patches osaurus_lib."""
    mock = MockLLM()
    mock.patch_osaurus()
    yield mock
    mock.unpatch()


@pytest.fixture
def mock_llm_osaurus():
    """Same as mock_llm but only patches osaurus_lib."""
    mock = MockLLM()
    mock.patch_osaurus()
    yield mock
    mock.unpatch()


# Legacy fixtures used by existing test files

@pytest.fixture
def mock_llm_response():
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
    return """- Event 1 (Toronto): Details here
- Event 2 (Vaughan): More details"""


@pytest.fixture
def sample_venues_data():
    return """- Venue 1 (123 Main St): Great place
- Venue 2 (456 Oak Ave): Another great place"""


@pytest.fixture
def sample_tweets():
    return [
        {"screen_name": "user1", "text": "Test tweet 1", "created_at": "2026-04-21"},
        {"screen_name": "user2", "text": "Test tweet 2", "created_at": "2026-04-21"},
    ]
