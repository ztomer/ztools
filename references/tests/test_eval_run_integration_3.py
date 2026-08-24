"""Integration tests for eval_run.run_eval() with mock LLM provider."""

from unittest.mock import patch

import pytest

# Patch the module that OWNS each name, not the `eval.run` shim: rebinding an
# attribute on the shim rebinds a copy nobody reads, and the test then runs
# against the unmocked thing. Each module's docstring says what it owns.
from eval import run_attempt, run_transport, run_validate


@pytest.fixture
def mock_llm():
    from lib.testing import MockLLM

    mock = MockLLM()
    mock.patch_all()
    yield mock
    mock.unpatch()


class TestRunEvalAllBranches:
    def test_skip_task_no_messages(self, mock_llm, capsys):
        import eval.run as er

        with patch.object(run_transport, "call", mock_llm.call):
            tasks = {"json_no_msgs": {}}  # no messages key
            results = er.run_eval("m", tasks=tasks, verbose=False)
        assert results == []

    def test_model_call_exception(self, mock_llm):
        import eval.run as er

        def bad_call(*args, **kwargs):
            raise RuntimeError("boom")

        with patch.object(run_transport, "call", bad_call):
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
        with (
            patch.object(run_transport, "call", mock_llm.call),
            patch.object(run_attempt, "_validate_result", side_effect=ValueError("validate boom")),
        ):
            tasks = {"json": TASKS["json"]}
            results = er.run_eval("m", tasks=tasks, verbose=False)
        assert results[0]["quality_score"] == 0
        assert "Validation error" in results[0]["failure_reason"]

    def test_verbose_mode(self, mock_llm, capsys):
        import eval.run as er

        with patch.object(run_transport, "call", mock_llm.call):
            from eval.tasks_core import TASKS

            tasks = {"filename": TASKS["filename"]}
            results = er.run_eval("m", tasks=tasks, verbose=True)
        # MockLLM returns "mock_test_filename", which is well-formed but says
        # nothing about the task's input, so the relevance dimension caps it.
        assert results[0]["quality_score"] == 40
        # Verbose mode should print something
        captured = capsys.readouterr()
        assert len(captured.out) > 0 or len(captured.err) > 0

    def test_weekend_quality_summary(self, mock_llm, capsys):
        import eval.run as er

        # Set parsed data that includes items
        from eval.tasks_core import TASKS

        mock_llm.set_response(
            "weekend_things",
            {
                "content": "[]",
                "parsed": [{"name": "uniquename_a"}, {"name": "uniquename_b"}],
            },
        )
        with patch.object(run_transport, "call", mock_llm.call):
            tasks = {
                "weekend_things": {
                    **TASKS["json"],
                    "source": "uniquename_a uniquename_b uniquename_c",
                }
            }
            results = er.run_eval("m", tasks=tasks, verbose=False)
        assert len(results) == 1

    def test_weekend_no_source_continue(self, mock_llm):
        """Line 262: weekend task but no source key."""
        import eval.run as er

        with patch.object(run_transport, "call", mock_llm.call):
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
        with patch.object(run_transport, "call", mock_llm.call):
            from eval.tasks_core import TASKS

            tasks = {"weekend_noparsed": {**TASKS["json"], "source": "some source text"}}
            results = er.run_eval("m", tasks=tasks, verbose=False)
        assert len(results) == 1

    def test_run_eval_skip_non_weekend_result(self, mock_llm):
        """Line 258: result has task_name not in weekend_tasks (when the result is built from a non-weekend task that ran)."""
        import eval.run as er

        # Provide BOTH a weekend task AND a non-weekend task
        mock_llm.set_response("json", {"content": "[]", "parsed": []})
        with patch.object(run_transport, "call", mock_llm.call):
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

        with patch.object(run_transport, "call", mock_llm.call):
            from eval.tasks_core import TASKS

            # Regular (non-weekend) task
            tasks = {"json": TASKS["json"]}
            results = er.run_eval("m", tasks=tasks, verbose=False)
        assert len(results) == 1
        # Verify the loop ran but no Quality Check Summary printed (since no weekend)

    def test_weekend_debug_summary_path(self, mock_llm):
        """Lines 77-83: weekend task with items and source - exercises debug source matching."""
        from io import StringIO

        import eval.run as er
        from rich.console import Console

        fake_console = Console(file=StringIO(), force_terminal=False, width=200)
        # Set response with parsed=None to ensure we go through content path
        mock_llm.set_response(
            "weekend_debug",
            {
                "content": '[{"name": "festival"}, {"name": "toronto"}, {"name": "park"}, {"name": "totally_uniquename_xxxx"}]',
                "parsed": None,
            },
        )
        with patch.object(run_transport, "call", mock_llm.call), patch.object(run_validate, "console", fake_console):
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
        from io import StringIO

        from eval.run import _validate_result
        from rich.console import Console

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
        with (
            patch.object(run_validate, "console", fake_console),
            patch("eval.run_validate.get_source_matching_details") as mock_details,
        ):
            mock_details.return_value = {
                "matched": [],
                "unmatched": [{"name": "fake_dict_item", "terms": ["a", "b"]}],
                "ratio": 0.0,
                "source_preview": "...",
            }
            score, failure, diagnosis = _validate_result(
                result, task_cfg, "weekend_test", debug=True
            )
        output = fake_console.file.getvalue()
        assert "fake_dict_item" in output


class TestQualityResultsFormat:
    def test_partial_status(self):
        from eval.run import _quality_results_to_eval_format
        from lib.quality_models import Score, ScoreCard

        sc = ScoreCard("m", "t", "c", [Score("d", 60.0, 1.0, [])], "x", 1.0)
        results = _quality_results_to_eval_format([sc], "m")
        assert results[0]["status"] == "partial"

    def test_failure_list_joins(self):
        from eval.run import _quality_results_to_eval_format
        from lib.quality_models import Score, ScoreCard

        sc = ScoreCard("m", "t", "c", [Score("d", 10.0, 1.0, ["e1", "e2"])], "x", 1.0)
        results = _quality_results_to_eval_format([sc], "m")
        assert "e1" in results[0]["failure_reason"]
        assert "e2" in results[0]["failure_reason"]


class TestModelIsAbandonedWhenTheServerCannotServe:
    """23 tasks of identical infrastructure failures is not a quality result.

    qwen3.6-35b spent 3h09m returning 23 zeros -- 34 x HTTP 503 "at inference
    capacity" and 12 timeouts -- on a host where its 27b sibling ran fine. The
    503s were leaked server slots: a task exceeds the client timeout, the client
    abandons it, the server keeps the slot, and the next task takes another one.
    After twelve, everything 503s. Stopping early both saves the GPU time and
    keeps a server failure from being recorded as a model's score.
    """

    def _infra_tasks(self, n):
        return {
            f"t{i}": {
                "messages": [{"role": "user", "content": "hi"}],
                "parse_json": False,
                "validator": lambda *a, **k: 100,
            }
            for i in range(n)
        }

    def test_it_stops_after_repeated_infrastructure_failures(self, mock_llm):
        import eval.run as er

        calls = {"n": 0}

        def always_503(*args, **kwargs):
            calls["n"] += 1
            return {"content": None, "error": "HTTP 503: Server is at inference capacity"}

        with patch.object(run_transport, "call", always_503), patch.object(run_attempt, "MAX_RETRIES", 0):
            results = er.run_eval(
                "mock-model", tasks=self._infra_tasks(20), verbose=False, measure_prefill=False
            )

        assert len(results) <= er.MAX_CONSECUTIVE_INFRA_FAILURES, (
            f"ran {len(results)} tasks against a server that cannot serve"
        )
        assert calls["n"] < 20, "kept firing requests at an overloaded server"

    def test_a_recovering_server_is_not_abandoned(self, mock_llm):
        """Intermittent blips must not condemn a model that keeps working.

        Interspersed, not a single failure: with one blip the counter never
        approaches the threshold whether or not it resets, so the test cannot
        tell the two apart. Alternating failures exceed the threshold in TOTAL
        while never being consecutive, which is exactly the distinction.
        """
        import eval.run as er

        state = {"n": 0}

        def alternating(*args, **kwargs):
            state["n"] += 1
            if state["n"] % 2 == 1:
                return {"content": None, "error": "HTTP 503: Server is at inference capacity"}
            return {"content": "[]", "parsed": [], "time": 0.1}

        with patch.object(run_transport, "call", alternating), patch.object(run_attempt, "MAX_RETRIES", 0):
            results = er.run_eval(
                "mock-model", tasks=self._infra_tasks(8), verbose=False, measure_prefill=False
            )

        assert len(results) == 8, (
            "intermittent server blips abandoned a model that kept working -- "
            "the consecutive counter is not resetting on success"
        )
