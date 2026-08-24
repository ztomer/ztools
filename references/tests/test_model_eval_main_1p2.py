"""Tests for model_eval main() flow."""

import subprocess
import sys
from unittest.mock import MagicMock, patch

_REAL_SUBPROCESS_RUN = subprocess.run


def _only_osaurus_commands_fail(cmd, *args, **kwargs):
    """Break the quit/relaunch commands, and nothing else.

    `patch("subprocess.run", side_effect=Exception(...))` reaches EVERY caller in
    the process, not the one under test -- patching an attribute of a shared
    stdlib module always does. That went unnoticed until the GPU lock started
    shelling out to `ps` on the same code path, and this test failed on a
    collaborator it never meant to touch. Failing only the commands whose error
    handling is the subject says what the test means, and cannot be broken by an
    unrelated caller appearing downstream. The real `run` is captured at import,
    before any patch, so the pass-through is genuinely unpatched.
    """
    parts = cmd if isinstance(cmd, (list, tuple)) else [cmd]
    if "osaurus" in " ".join(str(part) for part in parts):
        raise Exception("cmd error")
    return _REAL_SUBPROCESS_RUN(cmd, *args, **kwargs)


class TestMainFlow:
    def test_quality_mode(self, mock_llm, monkeypatch, capsys):
        """--quality mode calls lib.quality.run_suite and prints Quality scores line."""
        import eval.cli as model_eval

        monkeypatch.setattr(sys, "argv", ["eval.cli", "--quality"])
        with (
            patch.object(model_eval, "init_config"),
            patch.object(model_eval, "is_server_running", return_value=True),
            patch.object(model_eval, "is_server_responsive", return_value=True),
            patch.object(model_eval, "get_models", return_value=["m1"]),
            patch.object(model_eval, "build_tasks_from_model", return_value={}),
            patch.object(model_eval, "_print_results"),
            patch.object(model_eval, "is_server_responsive", return_value=True),
            patch.object(model_eval, "estimate_model_memory", return_value=4),
            patch.object(model_eval, "get_memory_percent", return_value=50.0),
        ):
            fake_qm = MagicMock()
            fake_case = MagicMock(task="t1")
            fake_qm.ALL_TEST_CASES = [fake_case]
            fake_qm.run_suite.return_value = [MagicMock()]
            with patch.dict("sys.modules", {"lib.quality": fake_qm}):
                with patch.object(
                    model_eval,
                    "_quality_results_to_eval_format",
                    return_value=[{"task": "t1", "quality_score": 80, "result": {"content": "x"}}],
                ):
                    model_eval.main()
        # lib.quality.run_suite was called with the model
        fake_qm.run_suite.assert_called_once()
        call_args = fake_qm.run_suite.call_args
        assert call_args[0][0] == ["m1"]  # models list
        # _quality_results_to_eval_format was called
        out = capsys.readouterr().out
        # Quality scores summary printed
        assert "Quality scores" in out

    def test_quality_mode_with_task_filter(self, mock_llm, monkeypatch):
        """Quality mode with --task filters cases."""
        import eval.cli as model_eval
        from eval.tasks_core import TASKS

        task_name = next(iter(TASKS.keys()))
        monkeypatch.setattr(sys, "argv", ["eval.cli", "--quality", "--task", task_name])
        with (
            patch.object(model_eval, "init_config"),
            patch.object(model_eval, "is_server_running", return_value=True),
            patch.object(model_eval, "is_server_responsive", return_value=True),
            patch.object(model_eval, "get_models", return_value=["m1"]),
            patch.object(model_eval, "build_tasks_from_model", return_value={}),
            patch.object(model_eval, "_print_results"),
        ):
            fake_case = MagicMock()
            fake_case.task = task_name
            fake_qm = MagicMock()
            fake_qm.ALL_TEST_CASES = [fake_case]
            fake_qm.run_suite.return_value = [MagicMock()]
            with patch.dict("sys.modules", {"lib.quality": fake_qm}):
                with patch.object(
                    model_eval,
                    "_quality_results_to_eval_format",
                    return_value=[
                        {"task": task_name, "quality_score": 80, "result": {"content": "x"}}
                    ],
                ):
                    model_eval.main()
                # Verify run_suite was called with filtered cases
                call_args = fake_qm.run_suite.call_args
                assert len(call_args[0][1]) == 1

    def test_no_scores(self, mock_llm, monkeypatch, capsys):
        """When results have no scores, prints '0 tasks'."""
        import eval.cli as model_eval

        monkeypatch.setattr(sys, "argv", ["eval.cli"])
        with (
            patch.object(model_eval, "init_config"),
            patch.object(model_eval, "is_server_running", return_value=True),
            patch.object(model_eval, "is_server_responsive", return_value=True),
            patch.object(model_eval, "get_models", return_value=["m1"]),
            patch.object(model_eval, "build_tasks_from_model", return_value={}),
            patch.object(model_eval, "run_eval", return_value=[]),
            patch.object(model_eval, "_print_results"),
            patch.object(model_eval, "is_server_responsive", return_value=True),
            patch.object(model_eval, "estimate_model_memory", return_value=4),
            patch.object(model_eval, "get_memory_percent", return_value=50.0),
        ):
            model_eval.main()
        out = capsys.readouterr().out
        assert "0 tasks" in out

    def test_high_avg_score(self, mock_llm, monkeypatch, capsys):
        """All scores >= 90 — STEP status (·)."""
        import eval.cli as model_eval

        monkeypatch.setattr(sys, "argv", ["eval.cli"])
        fake_results = [
            {"task": "t1", "quality_score": 95, "result": {"content": "x"}},
            {"task": "t2", "quality_score": 100, "result": {"content": "y"}},
        ]
        with (
            patch.object(model_eval, "init_config"),
            patch.object(model_eval, "is_server_running", return_value=True),
            patch.object(model_eval, "is_server_responsive", return_value=True),
            patch.object(model_eval, "get_models", return_value=["m1"]),
            patch.object(model_eval, "build_tasks_from_model", return_value={}),
            patch.object(model_eval, "run_eval", return_value=fake_results),
            patch.object(model_eval, "_print_results"),
            patch.object(model_eval, "is_server_responsive", return_value=True),
            patch.object(model_eval, "estimate_model_memory", return_value=4),
            patch.object(model_eval, "get_memory_percent", return_value=50.0),
        ):
            model_eval.main()
        out = capsys.readouterr().out
        # STEP (·) means all scores >= 90
        assert "98%" in out or "97%" in out  # avg of 95+100 = 97.5

    def test_mid_avg_score(self, mock_llm, monkeypatch, capsys):
        """Some scores >= 50 but not all >= 90 — WARN (!)."""
        import eval.cli as model_eval

        monkeypatch.setattr(sys, "argv", ["eval.cli"])
        fake_results = [
            {"task": "t1", "quality_score": 60, "result": {"content": "x"}},
            {"task": "t2", "quality_score": 80, "result": {"content": "y"}},
        ]
        with (
            patch.object(model_eval, "init_config"),
            patch.object(model_eval, "is_server_running", return_value=True),
            patch.object(model_eval, "is_server_responsive", return_value=True),
            patch.object(model_eval, "get_models", return_value=["m1"]),
            patch.object(model_eval, "build_tasks_from_model", return_value={}),
            patch.object(model_eval, "run_eval", return_value=fake_results),
            patch.object(model_eval, "_print_results"),
            patch.object(model_eval, "is_server_responsive", return_value=True),
            patch.object(model_eval, "estimate_model_memory", return_value=4),
            patch.object(model_eval, "get_memory_percent", return_value=50.0),
        ):
            model_eval.main()
        out = capsys.readouterr().out
        # WARN (!) means any score >= 50 but not all >= 90
        assert "70%" in out  # avg of 60+80

    def test_low_avg_score(self, mock_llm, monkeypatch, capsys):
        """All scores < 50 — FAIL (✗)."""
        import eval.cli as model_eval

        monkeypatch.setattr(sys, "argv", ["eval.cli"])
        fake_results = [
            {"task": "t1", "quality_score": 30, "result": {"content": "x"}},
            {"task": "t2", "quality_score": 20, "result": {"content": "y"}},
        ]
        with (
            patch.object(model_eval, "init_config"),
            patch.object(model_eval, "is_server_running", return_value=True),
            patch.object(model_eval, "is_server_responsive", return_value=True),
            patch.object(model_eval, "get_models", return_value=["m1"]),
            patch.object(model_eval, "build_tasks_from_model", return_value={}),
            patch.object(model_eval, "run_eval", return_value=fake_results),
            patch.object(model_eval, "_print_results"),
            patch.object(model_eval, "is_server_responsive", return_value=True),
            patch.object(model_eval, "estimate_model_memory", return_value=4),
            patch.object(model_eval, "get_memory_percent", return_value=50.0),
        ):
            model_eval.main()
        out = capsys.readouterr().out
        # FAIL (✗) means all scores < 50
        assert "25%" in out  # avg of 30+20

    def test_multiple_models_with_flush(self, mock_llm, monkeypatch, capsys):
        """Multiple models trigger flush_between_models (which calls call)."""
        import eval.cli as model_eval

        monkeypatch.setattr(sys, "argv", ["eval.cli"])
        fake_results = [{"task": "t1", "quality_score": 80, "result": {"content": "x"}}]
        with (
            patch.object(model_eval, "init_config"),
            patch.object(model_eval, "is_server_running", return_value=True),
            patch.object(model_eval, "is_server_responsive", return_value=True),
            patch.object(model_eval, "get_models", return_value=["m1", "m2"]),
            patch.object(model_eval, "build_tasks_from_model", return_value={}),
            patch.object(model_eval, "run_eval", return_value=fake_results),
            patch("eval.cli_runtime.call", return_value={}) as mock_call,
            patch.object(model_eval, "_print_results"),
            patch.object(model_eval, "is_server_responsive", return_value=True),
            patch.object(model_eval, "estimate_model_memory", return_value=4),
            patch.object(model_eval, "get_memory_percent", return_value=50.0),
            patch("time.sleep"),
        ):
            model_eval.main()
        # Flush calls call() once for the next model
        assert mock_call.call_count == 1
        out = capsys.readouterr().out
        assert "Flushing" in out or "m1" in out

    def test_low_memory_warning(self, mock_llm, monkeypatch, capsys):
        """Memory > threshold triggers warning."""
        import eval.cli as model_eval

        monkeypatch.setattr(sys, "argv", ["eval.cli"])
        with (
            patch.object(model_eval, "init_config"),
            patch.object(model_eval, "is_server_running", return_value=True),
            patch.object(model_eval, "is_server_responsive", return_value=True),
            patch.object(model_eval, "get_models", return_value=["m1"]),
            patch.object(model_eval, "build_tasks_from_model", return_value={}),
            patch.object(model_eval, "run_eval", return_value=[]),
            patch.object(model_eval, "_print_results"),
            patch.object(model_eval, "is_server_responsive", return_value=True),
            patch.object(model_eval, "estimate_model_memory", return_value=4),
            patch.object(model_eval, "get_memory_percent", return_value=95.0),
        ):
            model_eval.main()
        out = capsys.readouterr().out
        assert "95.0%" in out

    def test_model_too_big_for_memory(self, mock_llm, monkeypatch, capsys):
        """Model too big for available memory — REFUSED, not warned.

        This assertion used to be `"70GB" in out`, which was satisfied by the
        warn-and-continue line it was written for AND by the refusal that
        replaced it: the test could not see the change in consequence, only the
        number. Warn-and-continue is what produced qwen3.8-27b-mxfp8's 0.1158
        tok/s reading and the ~138,000s derived timeout that followed from it.
        """
        import eval.cli as model_eval

        monkeypatch.setattr(sys, "argv", ["eval.cli"])
        with (
            patch.object(model_eval, "init_config"),
            patch.object(model_eval, "is_server_running", return_value=True),
            patch.object(model_eval, "is_server_responsive", return_value=True),
            patch.object(model_eval, "get_models", return_value=["m1-70b"]),
            patch.object(model_eval, "build_tasks_from_model", return_value={}),
            patch.object(model_eval, "run_eval", return_value=[]) as ran,
            patch.object(model_eval, "_print_results"),
            patch.object(model_eval, "is_server_responsive", return_value=True),
            patch.object(model_eval, "estimate_model_memory", return_value=70),
            patch.object(model_eval, "get_memory_percent", return_value=50.0),
        ):
            model_eval.main()
        out = capsys.readouterr().out
        # 70GB of weights against 32GB available → refused, and the model is
        # SKIPPED rather than measured under swap.
        assert "Skipping m1-70b" in out
        assert "70GB" in out and "64GB reclaimable" in out
        assert "EVAL_ALLOW_OVERSIZE" in out, "the refusal must name its escape hatch"
        # The REFUSAL, not just the message. Printing "Skipping" and then
        # measuring anyway is the warn-and-continue bug wearing a new word, and
        # a mutation that deleted the `continue` passed every assertion above.
        ran.assert_not_called()

    def test_server_not_responsive(self, mock_llm, monkeypatch, capsys):
        """Server not responsive — print FAIL."""
        import eval.cli as model_eval

        monkeypatch.setattr(sys, "argv", ["eval.cli"])
        with (
            patch.object(model_eval, "init_config"),
            patch.object(model_eval, "is_server_running", return_value=True),
            patch.object(model_eval, "is_server_responsive", return_value=True),
            patch.object(model_eval, "get_models", return_value=["m1"]),
            patch.object(model_eval, "build_tasks_from_model", return_value={}),
            patch.object(model_eval, "run_eval", return_value=[]),
            patch.object(model_eval, "_print_results"),
            patch.object(model_eval, "is_server_responsive", return_value=False),
            patch.object(model_eval, "estimate_model_memory", return_value=4),
            patch.object(model_eval, "get_memory_percent", return_value=50.0),
        ):
            model_eval.main()
        out = capsys.readouterr().out
        assert "Server not responsive" in out or "restart" in out.lower()

    def test_task_arg_in_hardcoded_tasks(self, mock_llm, monkeypatch):
        """--task filters to single task from hardcoded TASKS."""
        import eval.cli as model_eval

        # Get a real task from eval_tasks_core
        from eval.tasks_core import TASKS

        first_task = next(iter(TASKS.keys()))
        monkeypatch.setattr(sys, "argv", ["eval.cli", "--task", first_task])
        with (
            patch.object(model_eval, "init_config"),
            patch.object(model_eval, "is_server_running", return_value=True),
            patch.object(model_eval, "is_server_responsive", return_value=True),
            patch.object(model_eval, "get_models", return_value=["m1"]),
            patch.object(model_eval, "build_tasks_from_model", return_value={}),
            patch.object(model_eval, "run_eval", return_value=[]) as mock_run,
            patch.object(model_eval, "_print_results"),
        ):
            model_eval.main()
        # Only the specified task is in tasks
        called_tasks = mock_run.call_args.kwargs["tasks"]
        assert first_task in called_tasks
        assert len(called_tasks) == 1
