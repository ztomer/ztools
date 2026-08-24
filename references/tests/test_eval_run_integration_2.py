"""Integration tests for eval_run.run_eval() with mock LLM provider."""

from unittest.mock import patch

import pytest

# Patch the module that OWNS each name, not the `eval.run` shim: rebinding an
# attribute on the shim rebinds a copy nobody reads, and the test then runs
# against the unmocked thing. Each module's docstring says what it owns.
from eval import run_transport, run_validate


@pytest.fixture
def mock_llm():
    from lib.testing import MockLLM

    mock = MockLLM()
    mock.patch_all()
    yield mock
    mock.unpatch()


class TestQualityResultsToEvalFormat:
    def test_conversion_ok(self, mock_llm):
        from eval.run import _quality_results_to_eval_format
        from lib.quality_models import Score, ScoreCard

        sc = ScoreCard("test-model", "json", "test", [Score("format", 95.0, 1.0, [])], "[]", 1.5)
        results = _quality_results_to_eval_format([sc], "test-model")
        assert results[0]["status"] == "ok"
        assert results[0]["quality_score"] == 95.0
        assert results[0]["task"] == "json"

    def test_conversion_fail(self, mock_llm):
        from eval.run import _quality_results_to_eval_format
        from lib.quality_models import Score, ScoreCard

        sc = ScoreCard("test-model", "json", "test", [Score("format", 10.0, 1.0, ["bad"])], "", 0.5)
        results = _quality_results_to_eval_format([sc], "test-model")
        assert results[0]["status"] == "fail"

    def test_json_dict_extracted(self):
        """Lines 57-58: JSON object is wrapped in list."""
        from eval.run import _validate_result

        def fake_validator(items, source_text=None):
            return 95, ""

        task_cfg = {
            "validator": fake_validator,
            "parse_json": True,
        }
        # Content is a dict (not a list)
        result = {"content": '{"single": "item"}', "parsed": None}
        score, failure, diagnosis = _validate_result(result, task_cfg, "json", debug=True)
        assert score == 95

    def test_text_extraction_path(self):
        """Lines 62-67: No JSON match — falls through to _extract_items_from_text (markdown table)."""
        from eval.run import _validate_result

        received = []

        def fake_validator(items, source_text=None):
            received.append(list(items))
            return 60, "fail"

        task_cfg = {
            "validator": fake_validator,
            "parse_json": True,
            "source": "festival park toronto",
        }
        result = {
            "content": "| Name | Location |\n|---|---|\n| Festival Park | Toronto |\n| Beach | Toronto |",
            "parsed": None,
        }
        score, failure, diagnosis = _validate_result(result, task_cfg, "json", debug=True)
        # Validator received 2 extracted items from the markdown table
        assert received == [
            [
                {"name": "Festival Park", "location": "Toronto"},
                {"name": "Beach", "location": "Toronto"},
            ]
        ]
        assert score == 60
        assert failure == "fail"

    def test_prose_only_json_task_scores_zero_as_a_parse_failure(self):
        """A JSON task that emitted no JSON has failed, whatever the prose says.

        This used to route the leftover text to validate_summary — a different
        task's validator — so a refusal earned structure/synthesis points and
        the run was never counted as a parse failure.
        """
        from eval.run import _validate_result

        # Validator is never called in this path — validate_summary runs instead.
        calls = []

        def fake_validator(items, source_text=None):
            calls.append(items)
            return 80, ""

        task_cfg = {
            "validator": fake_validator,
            "parse_json": True,
        }
        # Long content (>50 chars), no JSON, no list - should go through validate_summary
        result = {
            "content": "This is a long descriptive summary of what happened today in toronto with the children and the family event at the park location",
            "parsed": None,
        }
        score, failure, diagnosis = _validate_result(result, task_cfg, "json", debug=True)
        # The task's own validator is not called — there were no items to give it.
        assert calls == []
        assert score == 0
        assert failure == "No JSON in output"

    def test_debug_weekend_with_items(self):
        """Lines 76-83: debug=True, weekend task, with items and source - prints source matching details."""
        from io import StringIO

        from eval.run import _validate_result
        from rich.console import Console

        fake_console = Console(file=StringIO(), force_terminal=False, width=200)

        def fake_validator(items, source_text=None):
            return 100, ""

        task_cfg = {
            "validator": fake_validator,
            "parse_json": True,
            "source": "festival park toronto toronto park festival weekend",
        }
        # parsed must be None to hit the content path (line 51)
        # content has JSON which becomes extracted, items_for_debug gets set
        result = {
            "content": '[{"name": "festival"}, {"name": "toronto"}, {"name": "park"}, {"name": "totally_uniquename_xxxx"}]',
            "parsed": None,
        }
        with patch.object(run_validate, "console", fake_console):
            score, failure, diagnosis = _validate_result(
                result, task_cfg, "weekend_test", debug=True
            )
        assert score == 100
        output = fake_console.file.getvalue()
        assert "Source matching" in output or "Matched" in output

    def test_int_validator_json_match_path(self):
        """Line 88: validator returns int in json_match path."""
        from eval.run import _validate_result

        def int_validator(items, source_text=None):
            return 75

        task_cfg = {
            "validator": int_validator,
            "parse_json": True,
        }
        result = {"content": '[{"a": 1}]', "parsed": None}
        score, failure, diagnosis = _validate_result(result, task_cfg, "json")
        assert score == 75
        assert failure == ""


class TestValidateResultBranches:
    def test_non_tuple_validated_result(self):

        from eval.run import _validate_result

        # A validator that returns a plain int (not a tuple) - line 46
        def int_validator(content, source_text=None):
            return 85

        task_cfg = {
            "validator": int_validator,
            "parse_json": True,
        }
        result = {"content": "x", "parsed": [{"name": "item1"}]}
        score, failure, diagnosis = _validate_result(result, task_cfg, "json")
        assert score == 85
        assert failure == ""

    def test_json_match_invalid_extracted(self):
        from eval.run import _validate_result

        # Validator should NOT be called in this path: json.loads fails and
        # _extract_items_from_text returns nothing (no bullets/tables).
        calls = []

        def fake_validator(items, source_text=None):
            calls.append(items)
            return 50, "fail"

        # Content has "[" "]" (matches JSON regex) but contents aren't valid JSON,
        # and nothing is extractable from the prose either.
        result = {
            "content": "Here is some prose [this is not valid json at all] and more prose after the bracket text to make it long",
            "parsed": None,
        }
        task_cfg = {
            "validator": fake_validator,
            "parse_json": True,
        }
        score, failure, diagnosis = _validate_result(result, task_cfg, "json")
        assert calls == []
        assert score == 0
        assert failure == "No JSON in output"

    def test_short_content_no_extracted(self):
        from eval.run import _validate_result

        def fake_validator(items, source_text=None):
            return 100

        # Very short content, no items - lines 71-74
        task_cfg = {
            "validator": fake_validator,
            "parse_json": True,
        }
        result = {"content": "hi", "parsed": None}
        score, failure, diagnosis = _validate_result(result, task_cfg, "json")
        assert score == 0
        assert failure == "Empty content"

    def test_debug_weekend_task(self, capsys):
        from eval.run import _validate_result

        def fake_validator(items, source_text=None):
            return 100

        task_cfg = {
            "validator": fake_validator,
            "parse_json": True,
            "source": "Source A and Source B both here",
        }
        result = {
            "content": "x",
            "parsed": [{"name": "uniquename_a"}, {"name": "uniquename_b"}],
        }
        score, failure, diagnosis = _validate_result(result, task_cfg, "weekend_thing", debug=True)
        assert score == 100
        assert failure == ""

    def test_text_task_no_content(self):
        from eval.run import _validate_result

        task_cfg = {
            "validator": lambda x: (50, "x"),
            "parse_json": False,
        }
        result = {"content": "", "parsed": None}
        score, failure, diagnosis = _validate_result(result, task_cfg, "filename")
        assert score == 0
        assert failure == "Empty content"

    def test_text_task_int_validator(self):
        from eval.run import _validate_result

        task_cfg = {
            "validator": lambda x, **kw: 75,  # returns int; accepts source_text kwarg
            "parse_json": False,
        }
        result = {"content": "good", "parsed": None}
        score, failure, diagnosis = _validate_result(result, task_cfg, "filename")
        assert score == 75
        assert failure == ""


class TestCallModelBackend:
    def test_default_backend(self, mock_llm):
        import eval.run as er

        with patch.object(run_transport, "call", return_value={"content": "ok", "parse_json": False}):
            result = er._call_model(
                "m", {"messages": [], "parse_json": False}, "task", "h", 99, "osaurus"
            )
        assert result["content"] == "ok"
