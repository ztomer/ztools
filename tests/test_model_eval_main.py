"""Tests for model_eval main() flow."""
import pytest
import sys
from unittest.mock import patch, MagicMock


class TestMainFlow:
    def test_no_server_no_models(self, mock_llm, monkeypatch):
        import eval.cli as model_eval
        monkeypatch.setattr(sys, "argv", ["eval.cli"])
        with patch.object(model_eval, "init_config"), \
             patch.object(model_eval, "is_server_running", return_value=False), \
             patch.object(model_eval, "get_models", return_value=[]), \
             patch.object(model_eval, "build_tasks_from_model", return_value={}):
            with pytest.raises(SystemExit) as e:
                model_eval.main()
        assert e.value.code == 1

    def test_server_with_models(self, mock_llm, monkeypatch, capsys):
        import eval.cli as model_eval
        monkeypatch.setattr(sys, "argv", ["eval.cli"])
        fake_results = [{"task": "t1", "quality_score": 80, "result": {"content": "x"}}]
        with patch.object(model_eval, "init_config"), \
             patch.object(model_eval, "is_server_running", return_value=True), \
             patch.object(model_eval, "get_models", return_value=["m1"]), \
             patch.object(model_eval, "build_tasks_from_model", return_value={}), \
             patch.object(model_eval, "run_eval", return_value=fake_results), \
             patch.object(model_eval, "_print_results"):
            model_eval.main()
        # Should have logged "Found 1 models"
        captured = capsys.readouterr()
        assert "1 models" in captured.out or "Found" in captured.out

    def test_specific_model_filter(self, mock_llm, monkeypatch):
        import eval.cli as model_eval
        monkeypatch.setattr(sys, "argv", ["eval.cli", "--model", "m1"])
        fake_results = [{"task": "t1", "quality_score": 80, "result": {"content": "x"}}]
        with patch.object(model_eval, "init_config"), \
             patch.object(model_eval, "is_server_running", return_value=True), \
             patch.object(model_eval, "get_models", return_value=["m1", "m2"]), \
             patch.object(model_eval, "build_tasks_from_model", return_value={}), \
             patch.object(model_eval, "run_eval", return_value=fake_results), \
             patch.object(model_eval, "_print_results") as mock_print:
            model_eval.main()
        # Both models in list, then filtered to just m1
        mock_print.assert_called_once()
        results = mock_print.call_args[0][0]
        assert len(results) == 1
        assert results[0]["model"] == "m1"

    def test_specific_model_not_found(self, mock_llm, monkeypatch):
        import eval.cli as model_eval
        monkeypatch.setattr(sys, "argv", ["eval.cli", "--model", "missing"])
        with patch.object(model_eval, "init_config"), \
             patch.object(model_eval, "is_server_running", return_value=True), \
             patch.object(model_eval, "get_models", return_value=["m1"]), \
             patch.object(model_eval, "build_tasks_from_model", return_value={}):
            with pytest.raises(SystemExit) as e:
                model_eval.main()
        assert e.value.code == 1

    def test_specific_task(self, mock_llm, monkeypatch, capsys):
        import eval.cli as model_eval
        monkeypatch.setattr(sys, "argv", ["eval.cli", "--task", "filename"])
        fake_results = [{"task": "filename", "quality_score": 80, "result": {"content": "x"}}]
        with patch.object(model_eval, "init_config"), \
             patch.object(model_eval, "is_server_running", return_value=True), \
             patch.object(model_eval, "get_models", return_value=["m1"]), \
             patch.object(model_eval, "build_tasks_from_model", return_value={}), \
             patch.object(model_eval, "run_eval", return_value=fake_results) as mock_run, \
             patch.object(model_eval, "_print_results"), \
             patch.object(model_eval, "is_server_responsive", return_value=True), \
             patch.object(model_eval, "estimate_model_memory", return_value=4), \
             patch.object(model_eval, "get_memory_percent", return_value=50.0):
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
        with patch.object(model_eval, "init_config"), \
             patch.object(model_eval, "is_server_running", return_value=True), \
             patch.object(model_eval, "get_models", return_value=["m1"]), \
             patch.object(model_eval, "build_tasks_from_model", return_value={}):
            with pytest.raises(SystemExit) as e:
                model_eval.main()
        assert e.value.code == 1

    def test_config_tasks_loaded(self, mock_llm, monkeypatch, capsys):
        """Without --config-tasks, hardcoded TASKS are used."""
        import eval.cli as model_eval
        from eval.tasks_core import TASKS
        monkeypatch.setattr(sys, "argv", ["eval.cli"])
        config_tasks = {"t1": {"prompt": "P"}, "t2": {"prompt": "P"}}
        with patch.object(model_eval, "init_config"), \
             patch.object(model_eval, "is_server_running", return_value=True), \
             patch.object(model_eval, "get_models", return_value=["m1"]), \
             patch.object(model_eval, "build_tasks_from_model", return_value=config_tasks), \
             patch.object(model_eval, "run_eval", return_value=[]) as mock_run, \
             patch.object(model_eval, "_print_results"), \
             patch.object(model_eval, "is_server_responsive", return_value=True), \
             patch.object(model_eval, "estimate_model_memory", return_value=4), \
             patch.object(model_eval, "get_memory_percent", return_value=50.0):
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
        with patch.object(model_eval, "init_config"), \
             patch.object(model_eval, "is_server_running", return_value=True), \
             patch.object(model_eval, "get_models", return_value=["m1"]), \
             patch.object(model_eval, "build_tasks_from_model", return_value=config_tasks), \
             patch.object(model_eval, "run_eval", return_value=[]) as mock_run, \
             patch.object(model_eval, "_print_results"), \
             patch.object(model_eval, "is_server_responsive", return_value=True), \
             patch.object(model_eval, "estimate_model_memory", return_value=4), \
             patch.object(model_eval, "get_memory_percent", return_value=50.0):
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
        with patch.object(model_eval, "init_config"), \
             patch.object(model_eval, "is_server_running", return_value=True), \
             patch.object(model_eval, "get_models", return_value=["m1"]), \
             patch.object(model_eval, "build_tasks_from_model", return_value=config_tasks), \
             patch.object(model_eval, "run_eval", return_value=[]) as mock_run, \
             patch.object(model_eval, "_print_results"):
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
        with patch.object(model_eval, "init_config"), \
             patch.object(model_eval, "is_server_running", return_value=True), \
             patch.object(model_eval, "get_models", return_value=["m1"]), \
             patch.object(model_eval, "build_tasks_from_model", return_value=config_tasks), \
             patch.object(model_eval, "run_eval", return_value=[]) as mock_run, \
             patch.object(model_eval, "_print_results"), \
             patch.object(model_eval, "is_server_responsive", return_value=True), \
             patch.object(model_eval, "estimate_model_memory", return_value=4), \
             patch.object(model_eval, "get_memory_percent", return_value=50.0):
            with pytest.raises(SystemExit):
                model_eval.main()
        out = capsys.readouterr().out
        assert "not in config" in out

    def test_quick_mode(self, mock_llm, monkeypatch, capsys):
        """--quick flag runs with no retries (MAX_RETRIES=0)."""
        import eval.cli as model_eval
        monkeypatch.setattr(sys, "argv", ["eval.cli", "--quick"])
        with patch.object(model_eval, "init_config"), \
             patch.object(model_eval, "is_server_running", return_value=True), \
             patch.object(model_eval, "get_models", return_value=["m1"]), \
             patch.object(model_eval, "build_tasks_from_model", return_value={}), \
             patch.object(model_eval, "_print_results"), \
             patch.object(model_eval, "is_server_responsive", return_value=True), \
             patch.object(model_eval, "estimate_model_memory", return_value=4), \
             patch.object(model_eval, "get_memory_percent", return_value=50.0):
            model_eval.main()
        out = capsys.readouterr().out
        # Quick mode banner
        assert "Quick mode" in out

    def test_quality_mode(self, mock_llm, monkeypatch, capsys):
        """--quality mode calls lib.quality.run_suite and prints Quality scores line."""
        import eval.cli as model_eval
        monkeypatch.setattr(sys, "argv", ["eval.cli", "--quality"])
        with patch.object(model_eval, "init_config"), \
             patch.object(model_eval, "is_server_running", return_value=True), \
             patch.object(model_eval, "get_models", return_value=["m1"]), \
             patch.object(model_eval, "build_tasks_from_model", return_value={}), \
             patch.object(model_eval, "_print_results"), \
             patch.object(model_eval, "is_server_responsive", return_value=True), \
             patch.object(model_eval, "estimate_model_memory", return_value=4), \
             patch.object(model_eval, "get_memory_percent", return_value=50.0):
            fake_qm = MagicMock()
            fake_case = MagicMock(task="t1")
            fake_qm.ALL_TEST_CASES = [fake_case]
            fake_qm.run_suite.return_value = [MagicMock()]
            with patch.dict("sys.modules", {"lib.quality": fake_qm}):
                with patch.object(model_eval, "_quality_results_to_eval_format",
                                 return_value=[{"task": "t1", "quality_score": 80, "result": {"content": "x"}}]):
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
        with patch.object(model_eval, "init_config"), \
             patch.object(model_eval, "is_server_running", return_value=True), \
             patch.object(model_eval, "get_models", return_value=["m1"]), \
             patch.object(model_eval, "build_tasks_from_model", return_value={}), \
             patch.object(model_eval, "_print_results"):
            fake_case = MagicMock()
            fake_case.task = task_name
            fake_qm = MagicMock()
            fake_qm.ALL_TEST_CASES = [fake_case]
            fake_qm.run_suite.return_value = [MagicMock()]
            with patch.dict("sys.modules", {"lib.quality": fake_qm}):
                with patch.object(model_eval, "_quality_results_to_eval_format",
                                 return_value=[{"task": task_name, "quality_score": 80, "result": {"content": "x"}}]):
                    model_eval.main()
                # Verify run_suite was called with filtered cases
                call_args = fake_qm.run_suite.call_args
                assert len(call_args[0][1]) == 1

    def test_no_scores(self, mock_llm, monkeypatch, capsys):
        """When results have no scores, prints '0 tasks'."""
        import eval.cli as model_eval
        monkeypatch.setattr(sys, "argv", ["eval.cli"])
        with patch.object(model_eval, "init_config"), \
             patch.object(model_eval, "is_server_running", return_value=True), \
             patch.object(model_eval, "get_models", return_value=["m1"]), \
             patch.object(model_eval, "build_tasks_from_model", return_value={}), \
             patch.object(model_eval, "run_eval", return_value=[]), \
             patch.object(model_eval, "_print_results"), \
             patch.object(model_eval, "is_server_responsive", return_value=True), \
             patch.object(model_eval, "estimate_model_memory", return_value=4), \
             patch.object(model_eval, "get_memory_percent", return_value=50.0):
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
        with patch.object(model_eval, "init_config"), \
             patch.object(model_eval, "is_server_running", return_value=True), \
             patch.object(model_eval, "get_models", return_value=["m1"]), \
             patch.object(model_eval, "build_tasks_from_model", return_value={}), \
             patch.object(model_eval, "run_eval", return_value=fake_results), \
             patch.object(model_eval, "_print_results"), \
             patch.object(model_eval, "is_server_responsive", return_value=True), \
             patch.object(model_eval, "estimate_model_memory", return_value=4), \
             patch.object(model_eval, "get_memory_percent", return_value=50.0):
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
        with patch.object(model_eval, "init_config"), \
             patch.object(model_eval, "is_server_running", return_value=True), \
             patch.object(model_eval, "get_models", return_value=["m1"]), \
             patch.object(model_eval, "build_tasks_from_model", return_value={}), \
             patch.object(model_eval, "run_eval", return_value=fake_results), \
             patch.object(model_eval, "_print_results"), \
             patch.object(model_eval, "is_server_responsive", return_value=True), \
             patch.object(model_eval, "estimate_model_memory", return_value=4), \
             patch.object(model_eval, "get_memory_percent", return_value=50.0):
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
        with patch.object(model_eval, "init_config"), \
             patch.object(model_eval, "is_server_running", return_value=True), \
             patch.object(model_eval, "get_models", return_value=["m1"]), \
             patch.object(model_eval, "build_tasks_from_model", return_value={}), \
             patch.object(model_eval, "run_eval", return_value=fake_results), \
             patch.object(model_eval, "_print_results"), \
             patch.object(model_eval, "is_server_responsive", return_value=True), \
             patch.object(model_eval, "estimate_model_memory", return_value=4), \
             patch.object(model_eval, "get_memory_percent", return_value=50.0):
            model_eval.main()
        out = capsys.readouterr().out
        # FAIL (✗) means all scores < 50
        assert "25%" in out  # avg of 30+20

    def test_multiple_models_with_flush(self, mock_llm, monkeypatch, capsys):
        """Multiple models trigger flush_between_models (which calls call)."""
        import eval.cli as model_eval
        monkeypatch.setattr(sys, "argv", ["eval.cli"])
        fake_results = [{"task": "t1", "quality_score": 80, "result": {"content": "x"}}]
        with patch.object(model_eval, "init_config"), \
             patch.object(model_eval, "is_server_running", return_value=True), \
             patch.object(model_eval, "get_models", return_value=["m1", "m2"]), \
             patch.object(model_eval, "build_tasks_from_model", return_value={}), \
             patch.object(model_eval, "run_eval", return_value=fake_results), \
             patch.object(model_eval, "call", return_value={}) as mock_call, \
             patch.object(model_eval, "_print_results"), \
             patch.object(model_eval, "is_server_responsive", return_value=True), \
             patch.object(model_eval, "estimate_model_memory", return_value=4), \
             patch.object(model_eval, "get_memory_percent", return_value=50.0), \
             patch("time.sleep"):
            model_eval.main()
        # Flush calls call() once for the next model
        assert mock_call.call_count == 1
        out = capsys.readouterr().out
        assert "Flushing" in out or "m1" in out

    def test_low_memory_warning(self, mock_llm, monkeypatch, capsys):
        """Memory > threshold triggers warning."""
        import eval.cli as model_eval
        monkeypatch.setattr(sys, "argv", ["eval.cli"])
        with patch.object(model_eval, "init_config"), \
             patch.object(model_eval, "is_server_running", return_value=True), \
             patch.object(model_eval, "get_models", return_value=["m1"]), \
             patch.object(model_eval, "build_tasks_from_model", return_value={}), \
             patch.object(model_eval, "run_eval", return_value=[]), \
             patch.object(model_eval, "_print_results"), \
             patch.object(model_eval, "is_server_responsive", return_value=True), \
             patch.object(model_eval, "estimate_model_memory", return_value=4), \
             patch.object(model_eval, "get_memory_percent", return_value=95.0):
            model_eval.main()
        out = capsys.readouterr().out
        assert "95.0%" in out

    def test_model_too_big_for_memory(self, mock_llm, monkeypatch, capsys):
        """Model too big for available memory — warning."""
        import eval.cli as model_eval
        monkeypatch.setattr(sys, "argv", ["eval.cli"])
        with patch.object(model_eval, "init_config"), \
             patch.object(model_eval, "is_server_running", return_value=True), \
             patch.object(model_eval, "get_models", return_value=["m1-70b"]), \
             patch.object(model_eval, "build_tasks_from_model", return_value={}), \
             patch.object(model_eval, "run_eval", return_value=[]), \
             patch.object(model_eval, "_print_results"), \
             patch.object(model_eval, "is_server_responsive", return_value=True), \
             patch.object(model_eval, "estimate_model_memory", return_value=70), \
             patch.object(model_eval, "get_memory_percent", return_value=50.0):
            model_eval.main()
        out = capsys.readouterr().out
        # 70b model with 50% memory free (32GB) → too big
        assert "70GB" in out or "70b" in out

    def test_server_not_responsive(self, mock_llm, monkeypatch, capsys):
        """Server not responsive — print FAIL."""
        import eval.cli as model_eval
        monkeypatch.setattr(sys, "argv", ["eval.cli"])
        with patch.object(model_eval, "init_config"), \
             patch.object(model_eval, "is_server_running", return_value=True), \
             patch.object(model_eval, "get_models", return_value=["m1"]), \
             patch.object(model_eval, "build_tasks_from_model", return_value={}), \
             patch.object(model_eval, "run_eval", return_value=[]), \
             patch.object(model_eval, "_print_results"), \
             patch.object(model_eval, "is_server_responsive", return_value=False), \
             patch.object(model_eval, "estimate_model_memory", return_value=4), \
             patch.object(model_eval, "get_memory_percent", return_value=50.0):
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
        with patch.object(model_eval, "init_config"), \
             patch.object(model_eval, "is_server_running", return_value=True), \
             patch.object(model_eval, "get_models", return_value=["m1"]), \
             patch.object(model_eval, "build_tasks_from_model", return_value={}), \
             patch.object(model_eval, "run_eval", return_value=[]) as mock_run, \
             patch.object(model_eval, "_print_results"):
            model_eval.main()
        # Only the specified task is in tasks
        called_tasks = mock_run.call_args.kwargs["tasks"]
        assert first_task in called_tasks
        assert len(called_tasks) == 1


class TestFlushBetweenModels:
    """flush_between_models is a nested function inside main(). Use main() to exercise it."""

    def test_flush_success_path(self, mock_llm, monkeypatch, capsys):
        """Test the success path of flush via main() with multiple models."""
        import eval.cli as model_eval
        monkeypatch.setattr(sys, "argv", ["eval.cli"])
        with patch.object(model_eval, "init_config"), \
             patch.object(model_eval, "is_server_running", return_value=True), \
             patch.object(model_eval, "get_models", return_value=["m1", "m2"]), \
             patch.object(model_eval, "build_tasks_from_model", return_value={}), \
             patch.object(model_eval, "run_eval", return_value=[]), \
             patch.object(model_eval, "call", return_value={}) as mock_call, \
             patch.object(model_eval, "_print_results"), \
             patch.object(model_eval, "is_server_responsive", return_value=True), \
             patch.object(model_eval, "estimate_model_memory", return_value=4), \
             patch.object(model_eval, "get_memory_percent", return_value=50.0), \
             patch("time.sleep"):
            model_eval.main()
        # call() was invoked for the flush
        assert mock_call.call_count == 1
        out = capsys.readouterr().out
        assert "Flushing" in out

    def test_flush_with_error_triggers_restart(self, mock_llm, monkeypatch, capsys):
        import eval.cli as model_eval
        monkeypatch.setattr(sys, "argv", ["eval.cli"])
        with patch.object(model_eval, "init_config"), \
             patch.object(model_eval, "is_server_running", return_value=True), \
             patch.object(model_eval, "get_models", return_value=["m1", "m2"]), \
             patch.object(model_eval, "build_tasks_from_model", return_value={}), \
             patch.object(model_eval, "run_eval", return_value=[]), \
             patch.object(model_eval, "call", return_value={"error": "fail"}), \
             patch.object(model_eval, "_print_results"), \
             patch.object(model_eval, "is_server_responsive", return_value=True), \
             patch.object(model_eval, "estimate_model_memory", return_value=4), \
             patch.object(model_eval, "get_memory_percent", return_value=50.0), \
             patch("subprocess.run") as mock_subprocess, \
             patch("time.sleep"), \
             patch("requests.Session") as mock_session:
            s = mock_session.return_value.__enter__.return_value
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            s.get.return_value = mock_resp
            model_eval.main()
        # subprocess was called for both quit and open. Verify the right commands.
        sub_calls = [str(c) for c in mock_subprocess.call_args_list]
        # Quit command (osascript)
        assert any("osascript" in s and "quit" in s for s in sub_calls)
        # Open command
        assert any("open" in s and "osaurus" in s for s in sub_calls)
        out = capsys.readouterr().out
        assert "Flush failed" in out or "restart" in out.lower()

    def test_flush_subprocess_error_caught(self, mock_llm, monkeypatch, capsys):
        import eval.cli as model_eval
        monkeypatch.setattr(sys, "argv", ["eval.cli"])
        with patch.object(model_eval, "init_config"), \
             patch.object(model_eval, "is_server_running", return_value=True), \
             patch.object(model_eval, "get_models", return_value=["m1", "m2"]), \
             patch.object(model_eval, "build_tasks_from_model", return_value={}), \
             patch.object(model_eval, "run_eval", return_value=[]), \
             patch.object(model_eval, "call", return_value={"error": "fail"}), \
             patch.object(model_eval, "_print_results"), \
             patch.object(model_eval, "is_server_responsive", return_value=True), \
             patch.object(model_eval, "estimate_model_memory", return_value=4), \
             patch.object(model_eval, "get_memory_percent", return_value=50.0), \
             patch("subprocess.run", side_effect=Exception("cmd error")), \
             patch("time.sleep"), \
             patch("requests.Session") as mock_session:
            s = mock_session.return_value.__enter__.return_value
            s.get.return_value = MagicMock(status_code=200)
            # Should not raise — exception caught
            model_eval.main()
        out = capsys.readouterr().out
        # We got to the flush step
        assert "Flushing" in out
        # And the error was caught (no traceback printed to stdout)
        assert "Traceback" not in out

    def test_flush_request_error_caught(self, mock_llm, monkeypatch, capsys):
        import eval.cli as model_eval
        monkeypatch.setattr(sys, "argv", ["eval.cli"])
        with patch.object(model_eval, "init_config"), \
             patch.object(model_eval, "is_server_running", return_value=True), \
             patch.object(model_eval, "get_models", return_value=["m1", "m2"]), \
             patch.object(model_eval, "build_tasks_from_model", return_value={}), \
             patch.object(model_eval, "run_eval", return_value=[]), \
             patch.object(model_eval, "call", return_value={"error": "fail"}), \
             patch.object(model_eval, "_print_results"), \
             patch.object(model_eval, "is_server_responsive", return_value=True), \
             patch.object(model_eval, "estimate_model_memory", return_value=4), \
             patch.object(model_eval, "get_memory_percent", return_value=50.0), \
             patch("subprocess.run"), \
             patch("time.sleep"), \
             patch("requests.Session") as mock_session:
            s = mock_session.return_value.__enter__.return_value
            s.get.side_effect = Exception("conn err")
            # Should not raise
            model_eval.main()
        out = capsys.readouterr().out
        assert "Flushing" in out
        # requests exception is caught by inner try/except → no traceback
        assert "Traceback" not in out

    def test_flush_call_exception_caught(self, mock_llm, monkeypatch, capsys):
        import eval.cli as model_eval
        monkeypatch.setattr(sys, "argv", ["eval.cli"])
        with patch.object(model_eval, "init_config"), \
             patch.object(model_eval, "is_server_running", return_value=True), \
             patch.object(model_eval, "get_models", return_value=["m1", "m2"]), \
             patch.object(model_eval, "build_tasks_from_model", return_value={}), \
             patch.object(model_eval, "run_eval", return_value=[]), \
             patch.object(model_eval, "call", side_effect=Exception("api boom")), \
             patch.object(model_eval, "_print_results"), \
             patch.object(model_eval, "is_server_responsive", return_value=True), \
             patch.object(model_eval, "estimate_model_memory", return_value=4), \
             patch.object(model_eval, "get_memory_percent", return_value=50.0), \
             patch("time.sleep"):
            model_eval.main()
        out = capsys.readouterr().out
        # Outer try/except caught the exception
        assert "Flush error" in out and "api boom" in out


class TestMainBlock:
    def test_main_block(self, monkeypatch):
        """Exec the real `if __name__ == "__main__":` block from eval/__main__.py and verify it calls main()."""
        import re
        import sys
        import textwrap
        from unittest.mock import MagicMock
        monkeypatch.setattr(sys, "argv", ["eval.cli"])
        from pathlib import Path
        import eval.__main__ as eval_main
        import eval.cli as model_eval
        mock_main = MagicMock()
        saved = model_eval.main
        model_eval.main = mock_main
        try:
            source = Path(eval_main.__file__).read_text()
            match = re.search(
                r'if __name__ == "__main__":\n(?:    .*\n)+',
                source,
            )
            assert match is not None, "eval/__main__.py must have an if __name__ == __main__ block"
            block = textwrap.dedent(match.group())
            exec(block, {"__name__": "__main__", "main": mock_main})
        finally:
            model_eval.main = saved
        mock_main.assert_called_once()
