"""Integration tests for eval_run.run_eval() with mock LLM provider."""

import json
from unittest.mock import patch
import pytest


@pytest.fixture
def mock_llm():
    from lib.testing import MockLLM
    mock = MockLLM()
    mock.patch_all()
    yield mock
    mock.unpatch()


class TestRunEvalWithMock:
    def test_basic_success(self, mock_llm):
        import eval_run as er
        with patch.object(er, "call", mock_llm.call):
            from eval_tasks_core import TASKS
            tasks = {"json": TASKS["json"]}
            results = er.run_eval("mock-model", tasks=tasks, verbose=False)

        assert len(results) == 1
        assert results[0]["task"] == "json"
        assert results[0]["quality_score"] >= 90

    def test_multiple_tasks(self, mock_llm):
        import eval_run as er
        with patch.object(er, "call", mock_llm.call):
            from eval_tasks_core import TASKS
            tasks = {"json": TASKS["json"], "filename": TASKS["filename"]}
            results = er.run_eval("mock-model", tasks=tasks, verbose=False)

        assert len(results) == 2
        tasks_seen = {r["task"] for r in results}
        assert {"json", "filename"} == tasks_seen

    def test_returns_valid_scores(self, mock_llm):
        import eval_run as er
        with patch.object(er, "call", mock_llm.call):
            from eval_tasks_core import TASKS
            tasks = {"json": TASKS["json"]}
            results = er.run_eval("mock-model", tasks=tasks, verbose=False)

        result = results[0]
        assert 0 <= result["quality_score"] <= 100
        assert result["result"] is not None

    def test_custom_response(self, mock_llm):
        import eval_run as er
        mock_llm.set_response("json", {
            "content": json.dumps([
                {"name": "Custom Venue", "location": "Test City",
                 "target_ages": "All", "price": "Free", "weather": "indoor", "day": "Sat"},
            ]),
            "parsed": [
                {"name": "Custom Venue", "location": "Test City",
                 "target_ages": "All", "price": "Free", "weather": "indoor", "day": "Sat"},
            ],
        })
        with patch.object(er, "call", mock_llm.call):
            from eval_tasks_core import TASKS
            tasks = {"json": TASKS["json"]}
            results = er.run_eval("mock-model", tasks=tasks, verbose=False)

        assert len(results) == 1
        assert "Custom Venue" in results[0]["result"]["content"]

    def test_filename_task(self, mock_llm):
        import eval_run as er
        with patch.object(er, "call", mock_llm.call):
            from eval_tasks_core import TASKS
            tasks = {"filename": TASKS["filename"]}
            results = er.run_eval("mock-model", tasks=tasks, verbose=False)

        assert len(results) == 1
        assert results[0]["task"] == "filename"

    def test_summarize_task(self, mock_llm):
        import eval_run as er
        with patch.object(er, "call", mock_llm.call):
            from eval_tasks_core import TASKS
            tasks = {"summarize": TASKS["summarize"]}
            results = er.run_eval("mock-model", tasks=tasks, verbose=False)

        assert len(results) == 1
        assert results[0]["task"] == "summarize"

    def test_file_summary_task(self, mock_llm):
        import eval_run as er
        with patch.object(er, "call", mock_llm.call):
            from eval_tasks_core import TASKS
            tasks = {"file_summary": TASKS["file_summary"]}
            results = er.run_eval("mock-model", tasks=tasks, verbose=False)
        assert results[0]["quality_score"] >= 50

    def test_server_unreachable(self, mock_llm):
        import eval_run as er
        mock_llm.set_response("json", {"content": "", "error": "Connection refused"})
        with patch.object(er, "call", mock_llm.call):
            from eval_tasks_core import TASKS
            tasks = {"json": TASKS["json"]}
            results = er.run_eval("mock-model", tasks=tasks, verbose=False)
        assert results[0]["quality_score"] == 0

    def test_model_not_found(self, mock_llm):
        import eval_run as er
        mock_llm.set_response("json", {"content": "", "error": "Model not found: xxx"})
        with patch.object(er, "call", mock_llm.call):
            from eval_tasks_core import TASKS
            tasks = {"json": TASKS["json"]}
            results = er.run_eval("mock-model", tasks=tasks, verbose=False)
        assert results[0]["quality_score"] == 0

    def test_empty_content(self, mock_llm):
        import eval_run as er
        mock_llm.set_response("json", {"content": ""})
        with patch.object(er, "call", mock_llm.call):
            from eval_tasks_core import TASKS
            tasks = {"json": TASKS["json"]}
            results = er.run_eval("mock-model", tasks=tasks, verbose=False)
        assert results[0]["quality_score"] == 0

    def test_non_json_content_for_json_task(self, mock_llm):
        import eval_run as er
        mock_llm.set_response("json", {"content": "Just plain text without JSON"})
        with patch.object(er, "call", mock_llm.call):
            from eval_tasks_core import TASKS
            tasks = {"json": TASKS["json"]}
            results = er.run_eval("mock-model", tasks=tasks, verbose=False)
        assert results[0]["quality_score"] < 90


class TestValidateResultDirectly:
    def test_error_result(self, mock_llm):
        from eval_run import _validate_result
        from eval_tasks_core import TASKS

        task_cfg = TASKS["json"]
        result = {"content": "", "error": "Timeout"}
        score, failure, diagnosis = _validate_result(result, task_cfg, "json")
        assert score == 0
        assert failure == "Timeout"

    def test_text_task_result(self, mock_llm):
        from eval_run import _validate_result
        from eval_tasks_core import TASKS

        task_cfg = TASKS["filename"]
        result = {"content": "valid_filename", "parsed": None}
        score, failure, diagnosis = _validate_result(result, task_cfg, "filename")
        assert score >= 50


class TestQualityResultsToEvalFormat:
    def test_conversion_ok(self, mock_llm):
        from eval_run import _quality_results_to_eval_format
        from lib.quality_models import Score
        from lib.quality_entry import ScoreCard

        sc = ScoreCard("test-model", "json", "test",
                       [Score("format", 95.0, 1.0, [])], "[]", 1.5)
        results = _quality_results_to_eval_format([sc], "test-model")
        assert results[0]["status"] == "ok"
        assert results[0]["quality_score"] == 95.0
        assert results[0]["task"] == "json"

    def test_conversion_fail(self, mock_llm):
        from eval_run import _quality_results_to_eval_format
        from lib.quality_models import Score
        from lib.quality_entry import ScoreCard

        sc = ScoreCard("test-model", "json", "test",
                       [Score("format", 10.0, 1.0, ["bad"])], "", 0.5)
        results = _quality_results_to_eval_format([sc], "test-model")
        assert results[0]["status"] == "fail"


    def test_json_dict_extracted(self):
        """Lines 57-58: JSON object is wrapped in list."""
        from eval_run import _validate_result

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
        """Lines 62-67: Content has JSON match but extraction falls through to _extract_items_from_text."""
        from eval_run import _validate_result

        def fake_validator(items, source_text=None):
            return 60, "fail"

        task_cfg = {
            "validator": fake_validator,
            "parse_json": True,
            "source": "festival park toronto",
        }
        result = {"content": "1. festival park\n2. beach toronto", "parsed": None}
        score, failure, diagnosis = _validate_result(result, task_cfg, "json", debug=True)
        # Score is a number (extraction path was taken)
        assert isinstance(score, (int, float))
        assert 0 <= score <= 100
        # Failure reason is a string
        assert isinstance(failure, str)

    def test_summary_validation_path(self):
        """Lines 68-70: long content but no extractable items, falls back to validate_summary."""
        from eval_run import _validate_result

        def fake_validator(items, source_text=None):
            return 80, ""

        task_cfg = {
            "validator": fake_validator,
            "parse_json": True,
        }
        # Long content (>50 chars), no JSON, no list - should go through validate_summary
        result = {"content": "This is a long descriptive summary of what happened today in toronto with the children and the family event at the park location", "parsed": None}
        score, failure, diagnosis = _validate_result(result, task_cfg, "json", debug=True)
        # Should fall through to summary path
        assert score is not None
        assert isinstance(score, (int, float))

    def test_debug_weekend_with_items(self):
        """Lines 76-83: debug=True, weekend task, with items and source - prints source matching details."""
        from eval_run import _validate_result
        from unittest.mock import MagicMock
        from rich.console import Console
        from io import StringIO
        fake_console = Console(file=StringIO(), force_terminal=False, width=200)
        import eval_run as er

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
        with patch.object(er, "console", fake_console):
            score, failure, diagnosis = _validate_result(result, task_cfg, "weekend_test", debug=True)
        assert score == 100
        output = fake_console.file.getvalue()
        assert "Source matching" in output or "Matched" in output

    def test_int_validator_json_match_path(self):
        """Line 88: validator returns int in json_match path."""
        from eval_run import _validate_result

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
        from eval_run import _validate_result
        from unittest.mock import patch

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
        from eval_run import _validate_result

        def fake_validator(items, source_text=None):
            return 50, "fail"

        # Content has "[" "]" but extracted is invalid JSON - lines 55-60
        task_cfg = {
            "validator": fake_validator,
            "parse_json": True,
        }
        result = {"content": "before [not valid json] after", "parsed": None}
        score, failure, diagnosis = _validate_result(result, task_cfg, "json")
        # Falls through to _extract_items_from_text
        assert isinstance(score, (int, float))
        assert 0 <= score <= 100
        assert isinstance(failure, str)

    def test_short_content_no_extracted(self):
        from eval_run import _validate_result

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
        from eval_run import _validate_result

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
        from eval_run import _validate_result

        task_cfg = {
            "validator": lambda x: (50, "x"),
            "parse_json": False,
        }
        result = {"content": "", "parsed": None}
        score, failure, diagnosis = _validate_result(result, task_cfg, "filename")
        assert score == 0
        assert failure == "Empty content"

    def test_text_task_int_validator(self):
        from eval_run import _validate_result

        task_cfg = {
            "validator": lambda x: 75,  # returns int
            "parse_json": False,
        }
        result = {"content": "good", "parsed": None}
        score, failure, diagnosis = _validate_result(result, task_cfg, "filename")
        assert score == 75
        assert failure == ""


class TestCallModelBackend:
    def test_mlx_backend(self, mock_llm):
        import eval_run as er
        with patch.object(er, "mlx_call", return_value={"content": "ok"}):
            result = er._call_model("m", {"messages": []}, "task", "h", 99, "mlx")
        assert result == {"content": "ok"}

    def test_default_backend(self, mock_llm):
        import eval_run as er
        with patch.object(er, "call", return_value={"content": "ok", "parse_json": False}):
            result = er._call_model("m", {"messages": [], "parse_json": False}, "task", "h", 99, "osaurus")
        assert result["content"] == "ok"


class TestRunEvalAllBranches:
    def test_skip_task_no_messages(self, mock_llm, capsys):
        import eval_run as er
        with patch.object(er, "call", mock_llm.call):
            tasks = {"json_no_msgs": {}}  # no messages key
            results = er.run_eval("m", tasks=tasks, verbose=False)
        assert results == []

    def test_model_call_exception(self, mock_llm):
        import eval_run as er
        def bad_call(*args, **kwargs):
            raise RuntimeError("boom")
        with patch.object(er, "call", bad_call):
            from eval_tasks_core import TASKS
            tasks = {"json": TASKS["json"]}
            results = er.run_eval("m", tasks=tasks, verbose=False)
        assert results[0]["quality_score"] == 0
        assert "boom" in (results[0]["error"] or "")

    def test_validation_exception(self, mock_llm):
        import eval_run as er
        from eval_tasks_core import TASKS
        # Set response that makes validation explode
        mock_llm.set_response("json", {"content": "x", "parsed": "not-a-list"})
        with patch.object(er, "call", mock_llm.call), \
             patch.object(er, "_validate_result", side_effect=ValueError("validate boom")):
            tasks = {"json": TASKS["json"]}
            results = er.run_eval("m", tasks=tasks, verbose=False)
        assert results[0]["quality_score"] == 0
        assert "Validation error" in results[0]["failure_reason"]

    def test_verbose_mode(self, mock_llm, capsys):
        import eval_run as er
        with patch.object(er, "call", mock_llm.call):
            from eval_tasks_core import TASKS
            tasks = {"filename": TASKS["filename"]}
            results = er.run_eval("m", tasks=tasks, verbose=True)
        # Filename validation: mock returns "mock_test_filename" which won't pass filename
        # validation since real filename detection may not match. Just check it ran.
        assert results[0]["quality_score"] is not None
        # Verbose mode should print something
        captured = capsys.readouterr()
        assert len(captured.out) > 0 or len(captured.err) > 0

    def test_weekend_quality_summary(self, mock_llm, capsys):
        import eval_run as er
        # Set parsed data that includes items
        from eval_tasks_core import TASKS
        mock_llm.set_response("weekend_things", {
            "content": "[]",
            "parsed": [{"name": "uniquename_a"}, {"name": "uniquename_b"}],
        })
        with patch.object(er, "call", mock_llm.call):
            tasks = {
                "weekend_things": {**TASKS["json"], "source": "uniquename_a uniquename_b uniquename_c"}
            }
            results = er.run_eval("m", tasks=tasks, verbose=False)
        assert len(results) == 1

    def test_weekend_no_source_continue(self, mock_llm):
        """Line 262: weekend task but no source key."""
        import eval_run as er
        with patch.object(er, "call", mock_llm.call):
            from eval_tasks_core import TASKS
            # Weekend task without source - explicitly remove it
            cfg = {k: v for k, v in TASKS["json"].items() if k != "source"}
            tasks = {"weekend_nosource": cfg}
            results = er.run_eval("m", tasks=tasks, verbose=False)
        assert len(results) == 1

    def test_weekend_no_parsed_continue(self, mock_llm):
        """Line 262: weekend task with source but result.parsed is empty."""
        import eval_run as er
        mock_llm.set_response("weekend_noparsed", {"content": "", "parsed": []})
        with patch.object(er, "call", mock_llm.call):
            from eval_tasks_core import TASKS
            tasks = {"weekend_noparsed": {**TASKS["json"], "source": "some source text"}}
            results = er.run_eval("m", tasks=tasks, verbose=False)
        assert len(results) == 1

    def test_run_eval_skip_non_weekend_result(self, mock_llm):
        """Line 258: result has task_name not in weekend_tasks (when the result is built from a non-weekend task that ran)."""
        import eval_run as er
        # Provide BOTH a weekend task AND a non-weekend task
        mock_llm.set_response("json", {"content": "[]", "parsed": []})
        with patch.object(er, "call", mock_llm.call):
            from eval_tasks_core import TASKS
            tasks = {
                "json": TASKS["json"],  # non-weekend
                "weekend_x": {**TASKS["json"], "source": "abc"},  # weekend with source
            }
            results = er.run_eval("m", tasks=tasks, verbose=False)
        # Results has both - the "json" task is not in weekend_tasks so continue
        assert len(results) == 2

    def test_non_weekend_task_continue(self, mock_llm):
        """Line 255: result is in results but task_name is not in weekend_tasks."""
        import eval_run as er
        with patch.object(er, "call", mock_llm.call):
            from eval_tasks_core import TASKS
            # Regular (non-weekend) task
            tasks = {"json": TASKS["json"]}
            results = er.run_eval("m", tasks=tasks, verbose=False)
        assert len(results) == 1
        # Verify the loop ran but no Quality Check Summary printed (since no weekend)

    def test_weekend_debug_summary_path(self, mock_llm):
        """Lines 77-83: weekend task with items and source - exercises debug source matching."""
        import eval_run as er
        from rich.console import Console
        from io import StringIO
        fake_console = Console(file=StringIO(), force_terminal=False, width=200)
        # Set response with parsed=None to ensure we go through content path
        mock_llm.set_response("weekend_debug", {
            "content": '[{"name": "festival"}, {"name": "toronto"}, {"name": "park"}, {"name": "totally_uniquename_xxxx"}]',
            "parsed": None,
        })
        with patch.object(er, "call", mock_llm.call), \
             patch.object(er, "console", fake_console):
            from eval_tasks_core import TASKS
            tasks = {
                "weekend_debug": {**TASKS["json"], "source": "festival toronto park weekend stuff"}
            }
            results = er.run_eval("m", tasks=tasks, verbose=False)
        output = fake_console.file.getvalue()
        assert "Source matching" in output or "Matched" in output
        assert len(results) == 1

    def test_weekend_debug_with_dict_unmatched(self):
        """Line 84: when unmatched item is a dict (not a string) - uses dict path."""
        from eval_run import _validate_result
        from rich.console import Console
        from io import StringIO
        import eval_run as er
        fake_console = Console(file=StringIO(), force_terminal=False, width=200)

        def fake_validator(items, source_text=None):
            return 100, ""

        # Patch get_source_matching_details to return dict-shaped unmatched
        task_cfg = {
            "validator": fake_validator,
            "parse_json": True,
            "source": "festival toronto park",
        }
        result = {
            "content": '[{"name": "festival"}, {"name": "toronto"}, {"name": "park"}, {"name": "unmatched_xxxx"}]',
            "parsed": None,
        }
        with patch.object(er, "console", fake_console), \
             patch("eval_run.get_source_matching_details") as mock_details:
            mock_details.return_value = {
                "matched": [],
                "unmatched": [{"name": "fake_dict_item", "terms": ["a", "b"]}],
                "ratio": 0.0,
                "source_preview": "...",
            }
            score, failure, diagnosis = _validate_result(result, task_cfg, "weekend_test", debug=True)
        output = fake_console.file.getvalue()
        assert "fake_dict_item" in output


class TestQualityResultsFormat:
    def test_partial_status(self):
        from eval_run import _quality_results_to_eval_format
        from lib.quality_models import Score
        from lib.quality_entry import ScoreCard

        sc = ScoreCard("m", "t", "c", [Score("d", 60.0, 1.0, [])], "x", 1.0)
        results = _quality_results_to_eval_format([sc], "m")
        assert results[0]["status"] == "partial"

    def test_failure_list_joins(self):
        from eval_run import _quality_results_to_eval_format
        from lib.quality_models import Score
        from lib.quality_entry import ScoreCard

        sc = ScoreCard("m", "t", "c", [Score("d", 10.0, 1.0, ["e1", "e2"])], "x", 1.0)
        results = _quality_results_to_eval_format([sc], "m")
        assert "e1" in results[0]["failure_reason"]
        assert "e2" in results[0]["failure_reason"]
