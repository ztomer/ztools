"""Integration tests for eval_run.run_eval() with mock LLM provider."""

import json
from unittest.mock import patch

import pytest

# Patch the module that OWNS each name, not the `eval.run` shim: rebinding an
# attribute on the shim rebinds a copy nobody reads, and the test then runs
# against the unmocked thing. Each module's docstring says what it owns.
from eval import run_attempt, run_transport


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

        with patch.object(run_transport, "call", mock_llm.call):
            from eval.tasks_core import TASKS

            tasks = {"json": TASKS["json"]}
            results = er.run_eval("mock-model", tasks=tasks, verbose=False)

        assert len(results) == 1
        assert results[0]["task"] == "json"
        # The mock returns 2 items. This used to assert 100, with the comment "only
        # fails on <10 items check, not penalized" -- the defect written down and
        # accepted: the count credit is additive, the weights sum to 120 against a
        # ceiling of 100, and the overhang absorbed the penalty whole. Too-few-items
        # is now a cap derived from the credit not earned, so 2 items cannot score
        # what 10 items score.
        from lib.validators.json_validator import DETAILED_COUNT_GOOD, MAX_SCORE

        assert results[0]["quality_score"] == MAX_SCORE - DETAILED_COUNT_GOOD

    def test_multiple_tasks(self, mock_llm):
        import eval.run as er

        with patch.object(run_transport, "call", mock_llm.call):
            from eval.tasks_core import TASKS

            tasks = {"json": TASKS["json"], "filename": TASKS["filename"]}
            results = er.run_eval("mock-model", tasks=tasks, verbose=False)

        assert len(results) == 2
        tasks_seen = {r["task"] for r in results}
        assert {"json", "filename"} == tasks_seen

    def test_returns_valid_scores(self, mock_llm):
        import eval.run as er

        with patch.object(run_transport, "call", mock_llm.call):
            from eval.tasks_core import TASKS

            tasks = {"json": TASKS["json"]}
            results = er.run_eval("mock-model", tasks=tasks, verbose=False)

        result = results[0]
        # 2 items, so the too-few-items cap applies (see test_basic_success). The
        # point of this test is that a valid response produces a real score and a
        # result object, not that the score is any particular number.
        from lib.validators.json_validator import DETAILED_COUNT_GOOD, MAX_SCORE

        assert result["quality_score"] == MAX_SCORE - DETAILED_COUNT_GOOD
        assert result["result"] is not None

    def test_custom_response(self, mock_llm):
        import eval.run as er

        mock_llm.set_response(
            "json",
            {
                "content": json.dumps(
                    [
                        {
                            "name": "Custom Venue",
                            "location": "Test City",
                            "target_ages": "All",
                            "price": "Free",
                            "weather": "indoor",
                            "day": "Sat",
                        },
                    ]
                ),
                "parsed": [
                    {
                        "name": "Custom Venue",
                        "location": "Test City",
                        "target_ages": "All",
                        "price": "Free",
                        "weather": "indoor",
                        "day": "Sat",
                    },
                ],
            },
        )
        with patch.object(run_transport, "call", mock_llm.call):
            from eval.tasks_core import TASKS

            tasks = {"json": TASKS["json"]}
            results = er.run_eval("mock-model", tasks=tasks, verbose=False)

        assert len(results) == 1
        assert "Custom Venue" in results[0]["result"]["content"]

    def test_filename_task(self, mock_llm):
        import eval.run as er

        with patch.object(run_transport, "call", mock_llm.call):
            from eval.tasks_core import TASKS

            tasks = {"filename": TASKS["filename"]}
            results = er.run_eval("mock-model", tasks=tasks, verbose=False)

        assert len(results) == 1
        assert results[0]["task"] == "filename"

    def test_summarize_task(self, mock_llm):
        import eval.run as er

        with patch.object(run_transport, "call", mock_llm.call):
            from eval.tasks_core import TASKS

            tasks = {"summarize": TASKS["summarize"]}
            results = er.run_eval("mock-model", tasks=tasks, verbose=False)

        assert len(results) == 1
        assert results[0]["task"] == "summarize"

    def test_file_summary_task(self, mock_llm):
        import eval.run as er

        with patch.object(run_transport, "call", mock_llm.call):
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
        with patch.object(run_transport, "call", mock_llm.call):
            from eval.tasks_core import TASKS

            tasks = {"json": TASKS["json"]}
            results = er.run_eval("mock-model", tasks=tasks, verbose=False)
        assert results[0]["quality_score"] == 0

    def test_model_not_found(self, mock_llm):
        import eval.run as er

        mock_llm.set_response("json", {"content": "", "error": "Model not found: xxx"})
        with patch.object(run_transport, "call", mock_llm.call):
            from eval.tasks_core import TASKS

            tasks = {"json": TASKS["json"]}
            results = er.run_eval("mock-model", tasks=tasks, verbose=False)
        assert results[0]["quality_score"] == 0

    def test_empty_content(self, mock_llm):
        import eval.run as er

        mock_llm.set_response("json", {"content": ""})
        with patch.object(run_transport, "call", mock_llm.call):
            from eval.tasks_core import TASKS

            tasks = {"json": TASKS["json"]}
            results = er.run_eval("mock-model", tasks=tasks, verbose=False)
        assert results[0]["quality_score"] == 0

    def test_non_json_content_for_json_task(self, mock_llm):
        import eval.run as er

        mock_llm.set_response("json", {"content": "Just plain text without JSON"})
        with patch.object(run_transport, "call", mock_llm.call):
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
        # Columns must actually vary. This fixture used to hold "All"/"Free" in
        # every row -- the mandated-placeholder defect -- and scored 90 only
        # because the scorer could not see it. It now caps at 55, correctly.
        # This test is about the retry loop, so it needs output that is good for
        # real rather than output the scorer failed to fault.
        good_items = [
            {
                "name": f"Item {i}",
                "location": f"{i} Main St, Toronto",
                "target_ages": f"{i}-{i + 6}",
                "price": "Free" if i % 3 == 0 else f"${i * 4}",
                "weather": "outdoor" if i % 2 else "indoor",
                "day": "Saturday" if i % 2 else "Sunday",
            }
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

        with patch.object(run_transport, "call", side_effect=side_effect), patch.object(run_attempt, "MAX_RETRIES", 2):
            from eval.tasks_core import TASKS

            # Drop source so the grounding cap does not apply — this test checks
            # retry logic, not signal-grounding.
            task = dict(TASKS["json"])
            task.pop("source", None)
            tasks = {"json": task}
            # measure_prefill=False: the throughput probe is a real extra
            # transport call, and this test indexes its side effect by call number.
            results = er.run_eval(
                "mock-model", tasks=tasks, verbose=False, measure_prefill=False
            )
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
        with patch.object(run_transport, "call", mock_llm.call), patch.object(run_attempt, "MAX_RETRIES", 2):
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
            {
                "name": f"Item {i}",
                "location": "X",
                "target_ages": "All",
                "price": "Free",
                "weather": "outdoor",
                "day": "Sat",
            }
            for i in range(2)
        ]
        content = prose + json.dumps(items)
        parsed = items
        call_count = {"n": 0}

        def counting_call(*args, **kwargs):
            call_count["n"] += 1
            return {"content": content, "parsed": parsed}

        with (
            patch.object(run_transport, "call", side_effect=counting_call),
            patch.object(run_attempt, "MAX_RETRIES", 5),
        ):
            from eval.tasks_core import TASKS

            # Drop source so the grounding cap does not apply — this test checks
            # the FAIL_CONTENT (prose-before-JSON) early-break, not grounding.
            task = dict(TASKS["json"])
            task.pop("source", None)
            tasks = {"json": task}
            # measure_prefill=False: the throughput probe is a real extra
            # transport call, and this test indexes its side effect by call number.
            results = er.run_eval(
                "mock-model", tasks=tasks, verbose=False, measure_prefill=False
            )
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
        # The name must be RELEVANT to the task's input, not merely well-formed:
        # the filename task now sends the real eval input and scores coverage of
        # it, so a shape-only name like "valid_filename" is capped at 40.
        result = {"content": "login_error_invalid_credentials", "parsed": None}
        score, failure, diagnosis = _validate_result(result, task_cfg, "filename")
        assert score == 100, failure

        shape_only = {"content": "valid_filename", "parsed": None}
        assert _validate_result(shape_only, task_cfg, "filename")[0] <= 40
        assert failure == ""
