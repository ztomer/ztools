
from eval.failures import (
    FAIL_CONTENT,
    FAIL_FORMAT,
    FAIL_INFRA,
    FAIL_NONE,
    FAIL_PARSE,
    FAIL_TIMEOUT,
    _classify_failure,
    _describe_content_failure,
)


class TestClassifyFailure:
    def test_high_score_returns_none(self):
        result = {"content": "good output", "parsed": {"key": "val"}}
        task_cfg = {"parse_json": True}
        diagnosis = _classify_failure(result, task_cfg, 95, "")
        assert diagnosis["category"] is FAIL_NONE

    def test_model_not_found_is_infra(self):
        result = {"content": "", "error": "Model not found: qwen-xxx"}
        task_cfg = {"parse_json": False}
        diagnosis = _classify_failure(result, task_cfg, 0, "")
        assert diagnosis["category"] == FAIL_INFRA
        assert "Model not loaded" in diagnosis["evidence"]

    def test_connection_error_is_infra(self):
        result = {"content": "", "error": "Connection refused"}
        task_cfg = {"parse_json": False}
        diagnosis = _classify_failure(result, task_cfg, 0, "")
        assert diagnosis["category"] == FAIL_INFRA
        assert "unreachable" in diagnosis["evidence"]

    def test_timeout_error(self):
        result = {"content": "", "error": "Timeout after 300s"}
        task_cfg = {"parse_json": False}
        diagnosis = _classify_failure(result, task_cfg, 0, "")
        assert diagnosis["category"] == FAIL_TIMEOUT

    def test_no_json_in_parse_json_task(self):
        result = {"content": "hello world", "parsed": None}
        task_cfg = {"parse_json": True}
        diagnosis = _classify_failure(result, task_cfg, 0, "No JSON")
        assert diagnosis["category"] == FAIL_FORMAT
        assert "no JSON brackets" in diagnosis["evidence"]

    def test_json_chars_but_parsed_fails(self):
        result = {"content": "some text { but not valid json }", "parsed": None}
        task_cfg = {"parse_json": True}
        diagnosis = _classify_failure(result, task_cfg, 0, "extract fail")
        assert diagnosis["category"] == FAIL_PARSE

    def test_parsed_with_prose_before_json(self):
        result = {
            "content": "Let me think about this..." * 30 + '{"key": "val"}',
            "parsed": {"key": "val"},
        }
        task_cfg = {"parse_json": True}
        diagnosis = _classify_failure(result, task_cfg, 30, "prose first")
        assert diagnosis["category"] == FAIL_FORMAT

    def test_parsed_with_some_prose_but_high_score(self):
        result = {
            "content": "Let me think..." * 30 + '{"key": "val"}',
            "parsed": {"key": "val"},
        }
        task_cfg = {"parse_json": True}
        diagnosis = _classify_failure(result, task_cfg, 70, "prose first")
        assert diagnosis["category"] == FAIL_CONTENT

    def test_parsed_json_clean_fallback_to_content(self):
        result = {"content": '{"clean": "json"}', "parsed": {"clean": "json"}}
        task_cfg = {"parse_json": True}
        diagnosis = _classify_failure(result, task_cfg, 60, "not detailed")
        assert diagnosis["category"] == FAIL_CONTENT

    def test_non_parse_json_empty_content(self):
        result = {"content": "", "parsed": None}
        task_cfg = {"parse_json": False}
        diagnosis = _classify_failure(result, task_cfg, 0, "")
        assert diagnosis["category"] == FAIL_FORMAT
        assert "empty" in diagnosis["evidence"].lower()

    def test_reasoning_markers(self):
        result = {
            "content": "Let me think about this problem carefully. " * 10,
            "parsed": None,
        }
        task_cfg = {"parse_json": False}
        diagnosis = _classify_failure(result, task_cfg, 0, "no answer")
        assert diagnosis["category"] == FAIL_FORMAT
        assert "reasoning" in diagnosis["evidence"]

    def test_fallback_content_failure(self):
        result = {"content": "Some short answer", "parsed": None}
        task_cfg = {"parse_json": False}
        diagnosis = _classify_failure(result, task_cfg, 40, "wrong output")
        assert diagnosis["category"] == FAIL_CONTENT
        assert "wrong output" in diagnosis["evidence"]


class TestDescribeContentFailure:
    def test_list_items(self):
        items = [
            {"name": "a", "location": "here"},
            {"name": "b"},
        ]
        result = _describe_content_failure(items, "not detailed enough")
        assert "Parsed 2 items, 1 with details" in result
        assert "not detailed enough" in result

    def test_dict_with_keys(self):
        result = _describe_content_failure({"a": 1, "b": 2}, "missing keys")
        assert "Parsed dict with keys" in result
        assert "missing keys" in result

    def test_unknown_type(self):
        result = _describe_content_failure("plain string", "error msg")
        assert result == "error msg"

    def test_empty_list(self):
        result = _describe_content_failure([], "empty")
        assert "Parsed 0 items" in result
