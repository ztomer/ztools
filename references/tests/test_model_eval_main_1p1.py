"""Tests for model_eval main() flow."""

import subprocess
import sys
from unittest.mock import patch

import pytest

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
    def test_no_server_no_models(self, mock_llm, monkeypatch):
        import eval.cli as model_eval

        monkeypatch.setattr(sys, "argv", ["eval.cli"])
        with (
            patch.object(model_eval, "init_config"),
            patch.object(model_eval, "is_server_running", return_value=False),
            patch.object(model_eval, "get_models", return_value=[]),
            patch.object(model_eval, "build_tasks_from_model", return_value={}),
        ):
            with pytest.raises(SystemExit) as e:
                model_eval.main()
        assert e.value.code == 1

    def test_server_with_models(self, mock_llm, monkeypatch, capsys):
        import eval.cli as model_eval

        monkeypatch.setattr(sys, "argv", ["eval.cli"])
        fake_results = [{"task": "t1", "quality_score": 80, "result": {"content": "x"}}]
        with (
            patch.object(model_eval, "init_config"),
            patch.object(model_eval, "is_server_running", return_value=True),
            patch.object(model_eval, "is_server_responsive", return_value=True),
            patch.object(model_eval, "get_models", return_value=["m1"]),
            patch.object(model_eval, "build_tasks_from_model", return_value={}),
            patch.object(model_eval, "run_eval", return_value=fake_results),
            patch.object(model_eval, "_print_results"),
        ):
            model_eval.main()
        # Should have logged "Found 1 models"
        captured = capsys.readouterr()
        assert "1 models" in captured.out or "Found" in captured.out

    def test_specific_model_filter(self, mock_llm, monkeypatch):
        import eval.cli as model_eval

        monkeypatch.setattr(sys, "argv", ["eval.cli", "--model", "m1"])
        fake_results = [{"task": "t1", "quality_score": 80, "result": {"content": "x"}}]
        with (
            patch.object(model_eval, "init_config"),
            patch.object(model_eval, "is_server_running", return_value=True),
            patch.object(model_eval, "is_server_responsive", return_value=True),
            patch.object(model_eval, "get_models", return_value=["m1", "m2"]),
            patch.object(model_eval, "build_tasks_from_model", return_value={}),
            patch.object(model_eval, "run_eval", return_value=fake_results),
            patch.object(model_eval, "_print_results") as mock_print,
        ):
            model_eval.main()
        # Both models in list, then filtered to just m1
        mock_print.assert_called_once()
        results = mock_print.call_args[0][0]
        assert len(results) == 1
        assert results[0]["model"] == "m1"

    def test_specific_model_not_found(self, mock_llm, monkeypatch):
        import eval.cli as model_eval

        monkeypatch.setattr(sys, "argv", ["eval.cli", "--model", "missing"])
        with (
            patch.object(model_eval, "init_config"),
            patch.object(model_eval, "is_server_running", return_value=True),
            patch.object(model_eval, "is_server_responsive", return_value=True),
            patch.object(model_eval, "get_models", return_value=["m1"]),
            patch.object(model_eval, "build_tasks_from_model", return_value={}),
        ):
            with pytest.raises(SystemExit) as e:
                model_eval.main()
        assert e.value.code == 1

    def test_specific_task(self, mock_llm, monkeypatch, capsys):
        import eval.cli as model_eval

        monkeypatch.setattr(sys, "argv", ["eval.cli", "--task", "filename"])
        fake_results = [{"task": "filename", "quality_score": 80, "result": {"content": "x"}}]
        with (
            patch.object(model_eval, "init_config"),
            patch.object(model_eval, "is_server_running", return_value=True),
            patch.object(model_eval, "is_server_responsive", return_value=True),
            patch.object(model_eval, "get_models", return_value=["m1"]),
            patch.object(model_eval, "build_tasks_from_model", return_value={}),
            patch.object(model_eval, "run_eval", return_value=fake_results) as mock_run,
            patch.object(model_eval, "_print_results"),
            patch.object(model_eval, "is_server_responsive", return_value=True),
            patch.object(model_eval, "estimate_model_memory", return_value=4),
            patch.object(model_eval, "get_memory_percent", return_value=50.0),
        ):
            model_eval.main()
        # run_eval was called with the specific task
        called_tasks = mock_run.call_args.kwargs.get("tasks", {})
        assert "filename" in called_tasks
        assert len(called_tasks) == 1
        # Score rendered
        out = capsys.readouterr().out
        assert "80%" in out

    def test_unknown_task_exits(self, mock_llm, monkeypatch):
        import eval.cli as model_eval

        monkeypatch.setattr(sys, "argv", ["eval.cli", "--task", "nonexistent"])
        with (
            patch.object(model_eval, "init_config"),
            patch.object(model_eval, "is_server_running", return_value=True),
            patch.object(model_eval, "is_server_responsive", return_value=True),
            patch.object(model_eval, "get_models", return_value=["m1"]),
            patch.object(model_eval, "build_tasks_from_model", return_value={}),
        ):
            with pytest.raises(SystemExit) as e:
                model_eval.main()
        assert e.value.code == 1

    def test_config_tasks_loaded(self, mock_llm, monkeypatch, capsys):
        """Without --config-tasks, hardcoded TASKS are used."""
        import eval.cli as model_eval
        from eval.tasks_core import TASKS

        monkeypatch.setattr(sys, "argv", ["eval.cli"])
        config_tasks = {"t1": {"prompt": "P"}, "t2": {"prompt": "P"}}
        with (
            patch.object(model_eval, "init_config"),
            patch.object(model_eval, "is_server_running", return_value=True),
            patch.object(model_eval, "is_server_responsive", return_value=True),
            patch.object(model_eval, "get_models", return_value=["m1"]),
            patch.object(model_eval, "build_tasks_from_model", return_value=config_tasks),
            patch.object(model_eval, "run_eval", return_value=[]) as mock_run,
            patch.object(model_eval, "_print_results"),
            patch.object(model_eval, "is_server_responsive", return_value=True),
            patch.object(model_eval, "estimate_model_memory", return_value=4),
            patch.object(model_eval, "get_memory_percent", return_value=50.0),
        ):
            model_eval.main()
        # Without --config-tasks, hardcoded TASKS are used, not config_tasks
        called_tasks = mock_run.call_args.kwargs.get("tasks", {})
        assert called_tasks == TASKS
        out = capsys.readouterr().out
        assert "hardcoded TASKS" in out

    def test_config_tasks_flag_uses_config_tasks(self, mock_llm, monkeypatch):
        """With --config-tasks, config tasks are used instead of hardcoded TASKS."""
        import eval.cli as model_eval

        monkeypatch.setattr(sys, "argv", ["eval.cli", "--config-tasks"])
        config_tasks = {"t1": {"prompt": "P"}, "t2": {"prompt": "P"}}
        with (
            patch.object(model_eval, "init_config"),
            patch.object(model_eval, "is_server_running", return_value=True),
            patch.object(model_eval, "is_server_responsive", return_value=True),
            patch.object(model_eval, "get_models", return_value=["m1"]),
            patch.object(model_eval, "build_tasks_from_model", return_value=config_tasks),
            patch.object(model_eval, "run_eval", return_value=[]) as mock_run,
            patch.object(model_eval, "_print_results"),
            patch.object(model_eval, "is_server_responsive", return_value=True),
            patch.object(model_eval, "estimate_model_memory", return_value=4),
            patch.object(model_eval, "get_memory_percent", return_value=50.0),
        ):
            model_eval.main()
        called_tasks = mock_run.call_args.kwargs.get("tasks", {})
        assert called_tasks == config_tasks

    def test_config_tasks_with_specific_task_match(self, mock_llm, monkeypatch):
        """--task matching config task uses that task only."""
        import eval.cli as model_eval

        # Use a real task name that exists in hardcoded TASKS
        from eval.tasks_core import TASKS

        task_name = next(iter(TASKS.keys()))
        monkeypatch.setattr(sys, "argv", ["eval.cli", "--task", task_name])
        config_tasks = {task_name: {"prompt": "P"}, "t2": {"prompt": "P"}}
        with (
            patch.object(model_eval, "init_config"),
            patch.object(model_eval, "is_server_running", return_value=True),
            patch.object(model_eval, "is_server_responsive", return_value=True),
            patch.object(model_eval, "get_models", return_value=["m1"]),
            patch.object(model_eval, "build_tasks_from_model", return_value=config_tasks),
            patch.object(model_eval, "run_eval", return_value=[]) as mock_run,
            patch.object(model_eval, "_print_results"),
        ):
            model_eval.main()
        # run_eval called with only the task
        called_tasks = mock_run.call_args.kwargs["tasks"]
        assert task_name in called_tasks
        assert len(called_tasks) == 1

    def test_config_tasks_with_specific_task_no_match_config(self, mock_llm, monkeypatch, capsys):
        """--task + --config-tasks with no overlap prints FAIL and exits."""
        import eval.cli as model_eval

        monkeypatch.setattr(sys, "argv", ["eval.cli", "--config-tasks", "--task", "nonexistent"])
        config_tasks = {"other1": {"prompt": "P"}, "other2": {"prompt": "P"}}
        with (
            patch.object(model_eval, "init_config"),
            patch.object(model_eval, "is_server_running", return_value=True),
            patch.object(model_eval, "is_server_responsive", return_value=True),
            patch.object(model_eval, "get_models", return_value=["m1"]),
            patch.object(model_eval, "build_tasks_from_model", return_value=config_tasks),
            patch.object(model_eval, "run_eval", return_value=[]),
            patch.object(model_eval, "_print_results"),
            patch.object(model_eval, "is_server_responsive", return_value=True),
            patch.object(model_eval, "estimate_model_memory", return_value=4),
            patch.object(model_eval, "get_memory_percent", return_value=50.0),
        ):
            with pytest.raises(SystemExit):
                model_eval.main()
        out = capsys.readouterr().out
        assert "not in config" in out

    def test_quick_mode(self, mock_llm, monkeypatch, capsys):
        """--quick flag runs with no retries (MAX_RETRIES=0)."""
        import eval.cli as model_eval

        monkeypatch.setattr(sys, "argv", ["eval.cli", "--quick"])
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
            model_eval.main()
        out = capsys.readouterr().out
        # Quick mode banner
        assert "Quick mode" in out
