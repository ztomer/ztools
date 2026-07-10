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
        import eval.run as er
        with patch.object(er, "call", mock_llm.call):
            from eval.tasks_core import TASKS
            tasks = {"json": TASKS["json"]}
            results = er.run_eval("mock-model", tasks=tasks, verbose=False)

        assert len(results) == 1
        assert results[0]["task"] == "json"
        # Mock returns 2 valid items → score 100 (only fails on <10 items check, not penalized)
        assert results[0]["quality_score"] == 100

    def test_multiple_tasks(self, mock_llm):
        import eval.run as er
        with patch.object(er, "call", mock_llm.call):
            from eval.tasks_core import TASKS
            tasks = {"json": TASKS["json"], "filename": TASKS["filename"]}
            results = er.run_eval("mock-model", tasks=tasks, verbose=False)

        assert len(results) == 2
        tasks_seen = {r["task"] for r in results}
        assert {"json", "filename"} == tasks_seen

    def test_returns_valid_scores(self, mock_llm):
        import eval.run as er
        with patch.object(er, "call", mock_llm.call):
            from eval.tasks_core import TASKS
            tasks = {"json": TASKS["json"]}
            results = er.run_eval("mock-model", tasks=tasks, verbose=False)

        result = results[0]
        # Mock returns valid 2-item JSON → score 100
        assert result["quality_score"] == 100
        assert result["result"] is not None

    def test_custom_response(self, mock_llm):
        import eval.run as er
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
            from eval.tasks_core import TASKS
            tasks = {"json": TASKS["json"]}
            results = er.run_eval("mock-model", tasks=tasks, verbose=False)

        assert len(results) == 1
        assert "Custom Venue" in results[0]["result"]["content"]

    def test_filename_task(self, mock_llm):
        import eval.run as er
        with patch.object(er, "call", mock_llm.call):
            from eval.tasks_core import TASKS
            tasks = {"filename": TASKS["filename"]}
            results = er.run_eval("mock-model", tasks=tasks, verbose=False)

        assert len(results) == 1
        assert results[0]["task"] == "filename"

    def test_summarize_task(self, mock_llm):
        import eval.run as er
        with patch.object(er, "call", mock_llm.call):
            from eval.tasks_core import TASKS
            tasks = {"summarize": TASKS["summarize"]}
            results = er.run_eval("mock-model", tasks=tasks, verbose=False)

        assert len(results) == 1
        assert results[0]["task"] == "summarize"

    def test_file_summary_task(self, mock_llm):
        import eval.run as er
        with patch.object(er, "call", mock_llm.call):
            from eval.tasks_core import TASKS
            tasks = {"file_summary": TASKS["file_summary"]}
            results = er.run_eval("mock-model", tasks=tasks, verbose=False)
        # Mock returns 4 file summaries with detailed descriptions
        # 4 paths matched (40) + 4 detailed (40) + 4 real paths (20) = 100
        assert results[0]["quality_score"] == 100
        assert results[0]["status"] == "ok"

    def test_server_unreachable(self, mock_llm):
        import eval.run as er
        mock_llm.set_response("json", {"content": "", "error": "Connection refused"})
        with patch.object(er, "call", mock_llm.call):
            from eval.tasks_core import TASKS
            tasks = {"json": TASKS["json"]}
            results = er.run_eval("mock-model", tasks=tasks, verbose=False)
        assert results[0]["quality_score"] == 0

    def test_model_not_found(self, mock_llm):
        import eval.run as er
        mock_llm.set_response("json", {"content": "", "error": "Model not found: xxx"})
        with patch.object(er, "call", mock_llm.call):
            from eval.tasks_core import TASKS
            tasks = {"json": TASKS["json"]}
            results = er.run_eval("mock-model", tasks=tasks, verbose=False)
        assert results[0]["quality_score"] == 0

    def test_empty_content(self, mock_llm):
        import eval.run as er
        mock_llm.set_response("json", {"content": ""})
        with patch.object(er, "call", mock_llm.call):
            from eval.tasks_core import TASKS
            tasks = {"json": TASKS["json"]}
            results = er.run_eval("mock-model", tasks=tasks, verbose=False)
        assert results[0]["quality_score"] == 0

    def test_non_json_content_for_json_task(self, mock_llm):
        import eval.run as er
        mock_llm.set_response("json", {"content": "Just plain text without JSON"})
        with patch.object(er, "call", mock_llm.call):
            from eval.tasks_core import TASKS
            tasks = {"json": TASKS["json"]}
            results = er.run_eval("mock-model", tasks=tasks, verbose=False)
        assert results[0]["quality_score"] < 90

    def test_retry_then_success(self, mock_llm, monkeypatch):
        """First attempt scores < 90, retry returns good content — best score wins.

        Verifies the retry loop tracks best_score across attempts and stops
        when score >= 90 on retry.
        """
        import eval.run as er
        # First call returns low-score content, subsequent calls return high-score
        call_count = {"n": 0}
        # 10 items with full details to score >= 90 on the validator
        good_items = [
            {"name": f"Item {i}", "location": f"Place {i}", "target_ages": "All",
             "price": "Free", "weather": "outdoor", "day": "Saturday"}
            for i in range(10)
        ]
        def side_effect(*args, **kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return {"content": "no json here at all", "parsed": None}
            return {
                "content": json.dumps(good_items),
                "parsed": good_items,
            }
        with patch.object(er, "call", side_effect=side_effect), \
             patch.object(er, "MAX_RETRIES", 2):
            from eval.tasks_core import TASKS
            # Drop source so the grounding cap does not apply — this test checks
            # retry logic, not signal-grounding.
            task = dict(TASKS["json"]); task.pop("source", None)
            tasks = {"json": task}
            results = er.run_eval("mock-model", tasks=tasks, verbose=False)
        # Retry happened: exactly 2 calls (first failed, second succeeded with 10 items)
        assert call_count["n"] == 2
        # best_score should be the second (high) score, not the first
        # 10 items + detailed + paths → ~90
        assert results[0]["quality_score"] == 90
        # first_attempt_failed is set when any retry happens
        assert results[0]["first_attempt_failed"] is True

    def test_retry_exhausted_low_score(self, mock_llm, monkeypatch):
        """When all attempts score < 90, the best (lowest) score is reported with FAIL status."""
        import eval.run as er
        # All calls return low-score content
        mock_llm.set_response("json", {"content": "no json", "parsed": None})
        with patch.object(er, "call", mock_llm.call), \
             patch.object(er, "MAX_RETRIES", 2):
            from eval.tasks_core import TASKS
            tasks = {"json": TASKS["json"]}
            results = er.run_eval("mock-model", tasks=tasks, verbose=False)
        # All retries exhausted
        assert results[0]["quality_score"] < 90
        assert results[0]["status"] == "fail"
        assert results[0]["first_attempt_failed"] is True

    def test_fail_content_breaks_retry_early(self, mock_llm, monkeypatch):
        """If diagnosis category is FAIL_CONTENT, retry loop breaks immediately.

        A FAIL_CONTENT happens when the model emits > 200 chars of prose before
        the first JSON bracket. In that case, no point retrying — the model
        burned its context window on reasoning.
        """
        import eval.run as er
        # 200+ chars of prose, then valid JSON (parsed) but only 2 items so
        # score < 90. This triggers has_prose_before_json=True → FAIL_CONTENT.
        prose = "Let me think carefully about this request. " * 10  # > 200 chars
        items = [
            {"name": f"Item {i}", "location": "X", "target_ages": "All",
             "price": "Free", "weather": "outdoor", "day": "Sat"}
            for i in range(2)
        ]
        content = prose + json.dumps(items)
        parsed = items
        call_count = {"n": 0}
        def counting_call(*args, **kwargs):
            call_count["n"] += 1
            return {"content": content, "parsed": parsed}
        with patch.object(er, "call", side_effect=counting_call), \
             patch.object(er, "MAX_RETRIES", 5):
            from eval.tasks_core import TASKS
            # Drop source so the grounding cap does not apply — this test checks
            # the FAIL_CONTENT (prose-before-JSON) early-break, not grounding.
            task = dict(TASKS["json"]); task.pop("source", None)
            tasks = {"json": task}
            results = er.run_eval("mock-model", tasks=tasks, verbose=False)
        # FAIL_CONTENT → break after first attempt (no retry)
        assert call_count["n"] == 1
        # The category is recorded in the result
        assert results[0]["failure_category"] == "CONTENT"


class TestValidateResultDirectly:
    def test_error_result(self, mock_llm):
        from eval.run import _validate_result
        from eval.tasks_core import TASKS

        task_cfg = TASKS["json"]
        result = {"content": "", "error": "Timeout"}
        score, failure, diagnosis = _validate_result(result, task_cfg, "json")
        assert score == 0
        assert failure == "Timeout"

    def test_text_task_result(self, mock_llm):
        from eval.run import _validate_result
        from eval.tasks_core import TASKS

        task_cfg = TASKS["filename"]
        result = {"content": "valid_filename", "parsed": None}
        score, failure, diagnosis = _validate_result(result, task_cfg, "filename")
        # "valid_filename" passes length, chars, format checks
        assert score == 100
        assert failure == ""


class TestQualityResultsToEvalFormat:
    def test_conversion_ok(self, mock_llm):
        from eval.run import _quality_results_to_eval_format
        from lib.quality_models import Score
        from lib.quality_entry import ScoreCard

        sc = ScoreCard("test-model", "json", "test",
                       [Score("format", 95.0, 1.0, [])], "[]", 1.5)
        results = _quality_results_to_eval_format([sc], "test-model")
        assert results[0]["status"] == "ok"
        assert results[0]["quality_score"] == 95.0
        assert results[0]["task"] == "json"

    def test_conversion_fail(self, mock_llm):
        from eval.run import _quality_results_to_eval_format
        from lib.quality_models import Score
        from lib.quality_entry import ScoreCard

        sc = ScoreCard("test-model", "json", "test",
                       [Score("format", 10.0, 1.0, ["bad"])], "", 0.5)
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
        result = {"content": "| Name | Location |\n|---|---|\n| Festival Park | Toronto |\n| Beach | Toronto |", "parsed": None}
        score, failure, diagnosis = _validate_result(result, task_cfg, "json", debug=True)
        # Validator received 2 extracted items from the markdown table
        assert received == [[{"name": "Festival Park", "location": "Toronto"}, {"name": "Beach", "location": "Toronto"}]]
        assert score == 60
        assert failure == "fail"

    def test_summary_validation_path(self):
        """Lines 68-70: long content but no extractable items, falls back to validate_summary."""
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
        result = {"content": "This is a long descriptive summary of what happened today in toronto with the children and the family event at the park location", "parsed": None}
        score, failure, diagnosis = _validate_result(result, task_cfg, "json", debug=True)
        # Validator was bypassed in favor of validate_summary
        assert calls == []
        # validate_summary returns 20 for unstructured prose
        assert score == 20
        assert isinstance(score, (int, float))

    def test_debug_weekend_with_items(self):
        """Lines 76-83: debug=True, weekend task, with items and source - prints source matching details."""
        from eval.run import _validate_result
        from unittest.mock import MagicMock
        from rich.console import Console
        from io import StringIO
        fake_console = Console(file=StringIO(), force_terminal=False, width=200)
        import eval.run as er

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
        from eval.run import _validate_result

        # Validator should NOT be called in this path: json.loads fails,
        # _extract_items_from_text returns nothing (no bullets/tables),
        # content > 50 chars → validate_summary path is taken instead.
        calls = []
        def fake_validator(items, source_text=None):
            calls.append(items)
            return 50, "fail"

        # Content has "[" "]" (matches JSON regex) but contents aren't valid JSON.
        # Long enough (>50 chars) to skip "Empty content" and reach validate_summary.
        result = {"content": "Here is some prose [this is not valid json at all] and more prose after the bracket text to make it long", "parsed": None}
        task_cfg = {
            "validator": fake_validator,
            "parse_json": True,
        }
        score, failure, diagnosis = _validate_result(result, task_cfg, "json")
        assert calls == []
        # validate_summary returns 10 for prose with no structure
        assert score == 10
        assert isinstance(score, (int, float))

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
        with patch.object(er, "call", return_value={"content": "ok", "parse_json": False}):
            result = er._call_model("m", {"messages": [], "parse_json": False}, "task", "h", 99, "osaurus")
        assert result["content"] == "ok"


class TestRunEvalAllBranches:
    def test_skip_task_no_messages(self, mock_llm, capsys):
        import eval.run as er
        with patch.object(er, "call", mock_llm.call):
            tasks = {"json_no_msgs": {}}  # no messages key
            results = er.run_eval("m", tasks=tasks, verbose=False)
        assert results == []

    def test_model_call_exception(self, mock_llm):
        import eval.run as er
        def bad_call(*args, **kwargs):
            raise RuntimeError("boom")
        with patch.object(er, "call", bad_call):
            from eval.tasks_core import TASKS
            tasks = {"json": TASKS["json"]}
            results = er.run_eval("m", tasks=tasks, verbose=False)
        assert results[0]["quality_score"] == 0
        assert "boom" in (results[0]["error"] or "")

    def test_validation_exception(self, mock_llm):
        import eval.run as er
        from eval.tasks_core import TASKS
        # Set response that makes validation explode
        mock_llm.set_response("json", {"content": "x", "parsed": "not-a-list"})
        with patch.object(er, "call", mock_llm.call), \
             patch.object(er, "_validate_result", side_effect=ValueError("validate boom")):
            tasks = {"json": TASKS["json"]}
            results = er.run_eval("m", tasks=tasks, verbose=False)
        assert results[0]["quality_score"] == 0
        assert "Validation error" in results[0]["failure_reason"]

    def test_verbose_mode(self, mock_llm, capsys):
        import eval.run as er
        with patch.object(er, "call", mock_llm.call):
            from eval.tasks_core import TASKS
            tasks = {"filename": TASKS["filename"]}
            results = er.run_eval("m", tasks=tasks, verbose=True)
        # Filename validation: mock returns "mock_test_filename" - all score
        assert results[0]["quality_score"] == 100
        # Verbose mode should print something
        captured = capsys.readouterr()
        assert len(captured.out) > 0 or len(captured.err) > 0

    def test_weekend_quality_summary(self, mock_llm, capsys):
        import eval.run as er
        # Set parsed data that includes items
        from eval.tasks_core import TASKS
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
        import eval.run as er
        with patch.object(er, "call", mock_llm.call):
            from eval.tasks_core import TASKS
            # Weekend task without source - explicitly remove it
            cfg = {k: v for k, v in TASKS["json"].items() if k != "source"}
            tasks = {"weekend_nosource": cfg}
            results = er.run_eval("m", tasks=tasks, verbose=False)
        assert len(results) == 1

    def test_weekend_no_parsed_continue(self, mock_llm):
        """Line 262: weekend task with source but result.parsed is empty."""
        import eval.run as er
        mock_llm.set_response("weekend_noparsed", {"content": "", "parsed": []})
        with patch.object(er, "call", mock_llm.call):
            from eval.tasks_core import TASKS
            tasks = {"weekend_noparsed": {**TASKS["json"], "source": "some source text"}}
            results = er.run_eval("m", tasks=tasks, verbose=False)
        assert len(results) == 1

    def test_run_eval_skip_non_weekend_result(self, mock_llm):
        """Line 258: result has task_name not in weekend_tasks (when the result is built from a non-weekend task that ran)."""
        import eval.run as er
        # Provide BOTH a weekend task AND a non-weekend task
        mock_llm.set_response("json", {"content": "[]", "parsed": []})
        with patch.object(er, "call", mock_llm.call):
            from eval.tasks_core import TASKS
            tasks = {
                "json": TASKS["json"],  # non-weekend
                "weekend_x": {**TASKS["json"], "source": "abc"},  # weekend with source
            }
            results = er.run_eval("m", tasks=tasks, verbose=False)
        # Results has both - the "json" task is not in weekend_tasks so continue
        assert len(results) == 2

    def test_non_weekend_task_continue(self, mock_llm):
        """Line 255: result is in results but task_name is not in weekend_tasks."""
        import eval.run as er
        with patch.object(er, "call", mock_llm.call):
            from eval.tasks_core import TASKS
            # Regular (non-weekend) task
            tasks = {"json": TASKS["json"]}
            results = er.run_eval("m", tasks=tasks, verbose=False)
        assert len(results) == 1
        # Verify the loop ran but no Quality Check Summary printed (since no weekend)

    def test_weekend_debug_summary_path(self, mock_llm):
        """Lines 77-83: weekend task with items and source - exercises debug source matching."""
        import eval.run as er
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
            from eval.tasks_core import TASKS
            tasks = {
                "weekend_debug": {**TASKS["json"], "source": "festival toronto park weekend stuff"}
            }
            results = er.run_eval("m", tasks=tasks, verbose=False)
        output = fake_console.file.getvalue()
        assert "Source matching" in output or "Matched" in output
        assert len(results) == 1

    def test_weekend_debug_with_dict_unmatched(self):
        """Line 84: when unmatched item is a dict (not a string) - uses dict path."""
        from eval.run import _validate_result
        from rich.console import Console
        from io import StringIO
        import eval.run as er
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
             patch("eval.run.get_source_matching_details") as mock_details:
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
        from eval.run import _quality_results_to_eval_format
        from lib.quality_models import Score
        from lib.quality_entry import ScoreCard

        sc = ScoreCard("m", "t", "c", [Score("d", 60.0, 1.0, [])], "x", 1.0)
        results = _quality_results_to_eval_format([sc], "m")
        assert results[0]["status"] == "partial"

    def test_failure_list_joins(self):
        from eval.run import _quality_results_to_eval_format
        from lib.quality_models import Score
        from lib.quality_entry import ScoreCard

        sc = ScoreCard("m", "t", "c", [Score("d", 10.0, 1.0, ["e1", "e2"])], "x", 1.0)
        results = _quality_results_to_eval_format([sc], "m")
        assert "e1" in results[0]["failure_reason"]
        assert "e2" in results[0]["failure_reason"]
