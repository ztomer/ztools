import json

import pytest
from lib.osaurus_lib import _extract_json_only


class TestExtractJsonOnly:
    def test_simple_json(self):
        text = '{"key": "value"}'
        result = _extract_json_only(text)
        assert result == '{"key": "value"}'

    def test_json_with_think_block(self):
        text = """
        <think>
        [Final Check]: Output JSON now.
        {"key": "value"}
        </think>
        {"real": "data"}
        """
        result = _extract_json_only(text)
        # JSON after </think> extracted; first JSON is inside think block (skipped)
        assert result == '{"real": "data"}'
        data = json.loads(result)
        assert "real" in data

    def test_json_after_closing_think(self):
        text = """
        </think>
        {"fixed_activities": [{"name": "The Works Museum", "location": "Vaughan"}]}
        """
        result = _extract_json_only(text)
        # JSON array extracted after think close
        assert result == '[{"name": "The Works Museum", "location": "Vaughan"}]'
        data = json.loads(result)
        assert len(data) == 1

    def test_multiple_json_blocks(self):
        text = """
        {"first": "block"}
        some text
        {"second": "block"}
        """
        result = _extract_json_only(text)
        # First JSON block wins
        assert result == '{"first": "block"}'
        data = json.loads(result)
        assert "first" in data

    def test_no_json(self):
        text = "Just some random text without JSON"
        result = _extract_json_only(text)
        assert result is None

    def test_empty_string(self):
        result = _extract_json_only("")
        assert result is None

    def test_malformed_json(self):
        text = '{"key": broken, "value": here}'
        result = _extract_json_only(text)
        assert result is None

    def test_json_with_code_fence(self):
        text = """
        Here is the result:
        ```json
        {"result": "success", "count": 42}
        ```
        """
        result = _extract_json_only(text)
        # Code fence stripped, JSON extracted
        assert result == '{"result": "success", "count": 42}'
        data = json.loads(result)
        assert data["result"] == "success"
        assert data["count"] == 42

    def test_nested_json_object(self):
        text = '{"level1": {"level2": {"level3": "deep"}}}'
        result = _extract_json_only(text)
        # Nested object passed through unchanged
        assert result == '{"level1": {"level2": {"level3": "deep"}}}'
        data = json.loads(result)
        assert data["level1"]["level2"]["level3"] == "deep"

    def test_json_list_top_level(self):
        text = '[{"name": "item1"}, {"name": "item2"}]'
        result = _extract_json_only(text)
        # List passed through unchanged
        assert result == '[{"name": "item1"}, {"name": "item2"}]'
        data = json.loads(result)
        assert len(data) == 2
        assert data[0]["name"] == "item1"
