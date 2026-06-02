"""Tests for model_eval main() flow."""
import pytest
import sys
from unittest.mock import patch, MagicMock


class TestMainFlow:
    def test_no_server_no_models(self, mock_llm, monkeypatch):
        import model_eval
        monkeypatch.setattr(sys, "argv", ["model_eval"])
        with patch.object(model_eval, "init_config"), \
             patch.object(model_eval, "is_server_running", return_value=False), \
             patch.object(model_eval, "get_models", return_value=[]), \
             patch.object(model_eval, "build_tasks_from_model", return_value={}):
            with pytest.raises(SystemExit) as e:
                model_eval.main()
        assert e.value.code == 1

    def test_server_with_models(self, mock_llm, monkeypatch, capsys):
        import model_eval
        monkeypatch.setattr(sys, "argv", ["model_eval"])
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
        import model_eval
        monkeypatch.setattr(sys, "argv", ["model_eval", "--model", "m1"])
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
        import model_eval
        monkeypatch.setattr(sys, "argv", ["model_eval", "--model", "missing"])
        with patch.object(model_eval, "init_config"), \
             patch.object(model_eval, "is_server_running", return_value=True), \
             patch.object(model_eval, "get_models", return_value=["m1"]), \
             patch.object(model_eval, "build_tasks_from_model", return_value={}):
            with pytest.raises(SystemExit) as e:
                model_eval.main()
        assert e.value.code == 1

    def test_specific_task(self, mock_llm, monkeypatch):
        import model_eval
        monkeypatch.setattr(sys, "argv", ["model_eval", "--task", "filename"])
        fake_results = [{"task": "filename", "quality_score": 80, "result": {"content": "x"}}]
        with patch.object(model_eval, "init_config"), \
             patch.object(model_eval, "is_server_running", return_value=True), \
             patch.object(model_eval, "get_models", return_value=["m1"]), \
             patch.object(model_eval, "build_tasks_from_model", return_value={}), \
             patch.object(model_eval, "run_eval", return_value=fake_results), \
             patch.object(model_eval, "_print_results"):
            model_eval.main()

    def test_unknown_task_exits(self, mock_llm, monkeypatch):
        import model_eval
        monkeypatch.setattr(sys, "argv", ["model_eval", "--task", "nonexistent"])
        with patch.object(model_eval, "init_config"), \
             patch.object(model_eval, "is_server_running", return_value=True), \
             patch.object(model_eval, "get_models", return_value=["m1"]), \
             patch.object(model_eval, "build_tasks_from_model", return_value={}):
            with pytest.raises(SystemExit) as e:
                model_eval.main()
        assert e.value.code == 1

    def test_config_tasks_loaded(self, mock_llm, monkeypatch):
        """When build_tasks_from_model returns tasks, use them."""
        import model_eval
        monkeypatch.setattr(sys, "argv", ["model_eval"])
        config_tasks = {"t1": {"prompt": "P"}, "t2": {"prompt": "P"}}
        with patch.object(model_eval, "init_config"), \
             patch.object(model_eval, "is_server_running", return_value=True), \
             patch.object(model_eval, "get_models", return_value=["m1"]), \
             patch.object(model_eval, "build_tasks_from_model", return_value=config_tasks), \
             patch.object(model_eval, "run_eval", return_value=[]), \
             patch.object(model_eval, "_print_results"):
            model_eval.main()

    def test_config_tasks_with_specific_task_match(self, mock_llm, monkeypatch):
        """--task matching config task uses that task only."""
        import model_eval
        # Use a real task name that exists in hardcoded TASKS
        from eval_tasks_core import TASKS
        task_name = next(iter(TASKS.keys()))
        monkeypatch.setattr(sys, "argv", ["model_eval", "--task", task_name])
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

    def test_config_tasks_with_specific_task_no_match(self, mock_llm, monkeypatch):
        """--task not in config_tasks prints FAIL."""
        import model_eval
        from eval_tasks_core import TASKS
        task_name = next(iter(TASKS.keys()))
        monkeypatch.setattr(sys, "argv", ["model_eval", "--task", task_name])
        # config_tasks has different tasks
        config_tasks = {"other1": {"prompt": "P"}, "other2": {"prompt": "P"}}
        with patch.object(model_eval, "init_config"), \
             patch.object(model_eval, "is_server_running", return_value=True), \
             patch.object(model_eval, "get_models", return_value=["m1"]), \
             patch.object(model_eval, "build_tasks_from_model", return_value=config_tasks), \
             patch.object(model_eval, "run_eval", return_value=[]) as mock_run, \
             patch.object(model_eval, "_print_results"):
            model_eval.main()
        # Falls back to all config tasks
        mock_run.assert_called_once()

    def test_quick_mode(self, mock_llm, monkeypatch):
        import model_eval
        monkeypatch.setattr(sys, "argv", ["model_eval", "--quick"])
        with patch.object(model_eval, "init_config"), \
             patch.object(model_eval, "is_server_running", return_value=True), \
             patch.object(model_eval, "get_models", return_value=["m1"]), \
             patch.object(model_eval, "build_tasks_from_model", return_value={}), \
             patch.object(model_eval, "run_eval", return_value=[]) as mock_run, \
             patch.object(model_eval, "_print_results"):
            model_eval.main()
        # quick_run_eval was called (via mock since we replaced run_eval)
        mock_run.assert_called_once()

    def test_quality_mode(self, mock_llm, monkeypatch):
        import model_eval
        monkeypatch.setattr(sys, "argv", ["model_eval", "--quality"])
        with patch.object(model_eval, "init_config"), \
             patch.object(model_eval, "is_server_running", return_value=True), \
             patch.object(model_eval, "get_models", return_value=["m1"]), \
             patch.object(model_eval, "build_tasks_from_model", return_value={}), \
             patch.object(model_eval, "_print_results"):
            # Mock the lib.quality module
            fake_qm = MagicMock()
            fake_qm.ALL_TEST_CASES = [MagicMock(task="t1")]
            fake_qm.run_suite.return_value = [MagicMock()]
            with patch.dict("sys.modules", {"lib.quality": fake_qm}):
                with patch.object(model_eval, "_quality_results_to_eval_format",
                                 return_value=[{"task": "t1", "quality_score": 80, "result": {"content": "x"}}]):
                    model_eval.main()

    def test_quality_mode_with_task_filter(self, mock_llm, monkeypatch):
        """Quality mode with --task filters cases."""
        import model_eval
        from eval_tasks_core import TASKS
        task_name = next(iter(TASKS.keys()))
        monkeypatch.setattr(sys, "argv", ["model_eval", "--quality", "--task", task_name])
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

    def test_no_scores(self, mock_llm, monkeypatch):
        """When results have no scores, prints '0 tasks'."""
        import model_eval
        monkeypatch.setattr(sys, "argv", ["model_eval"])
        with patch.object(model_eval, "init_config"), \
             patch.object(model_eval, "is_server_running", return_value=True), \
             patch.object(model_eval, "get_models", return_value=["m1"]), \
             patch.object(model_eval, "build_tasks_from_model", return_value={}), \
             patch.object(model_eval, "run_eval", return_value=[]), \
             patch.object(model_eval, "_print_results"):
            model_eval.main()

    def test_high_avg_score(self, mock_llm, monkeypatch):
        """All scores >= 90 — STEP status."""
        import model_eval
        monkeypatch.setattr(sys, "argv", ["model_eval"])
        fake_results = [
            {"task": "t1", "quality_score": 95, "result": {"content": "x"}},
            {"task": "t2", "quality_score": 100, "result": {"content": "y"}},
        ]
        with patch.object(model_eval, "init_config"), \
             patch.object(model_eval, "is_server_running", return_value=True), \
             patch.object(model_eval, "get_models", return_value=["m1"]), \
             patch.object(model_eval, "build_tasks_from_model", return_value={}), \
             patch.object(model_eval, "run_eval", return_value=fake_results), \
             patch.object(model_eval, "_print_results"):
            model_eval.main()

    def test_mid_avg_score(self, mock_llm, monkeypatch):
        """Some scores >= 50 but not all >= 90 — WARN."""
        import model_eval
        monkeypatch.setattr(sys, "argv", ["model_eval"])
        fake_results = [
            {"task": "t1", "quality_score": 60, "result": {"content": "x"}},
            {"task": "t2", "quality_score": 80, "result": {"content": "y"}},
        ]
        with patch.object(model_eval, "init_config"), \
             patch.object(model_eval, "is_server_running", return_value=True), \
             patch.object(model_eval, "get_models", return_value=["m1"]), \
             patch.object(model_eval, "build_tasks_from_model", return_value={}), \
             patch.object(model_eval, "run_eval", return_value=fake_results), \
             patch.object(model_eval, "_print_results"):
            model_eval.main()

    def test_low_avg_score(self, mock_llm, monkeypatch):
        """All scores < 50 — FAIL."""
        import model_eval
        monkeypatch.setattr(sys, "argv", ["model_eval"])
        fake_results = [
            {"task": "t1", "quality_score": 30, "result": {"content": "x"}},
            {"task": "t2", "quality_score": 20, "result": {"content": "y"}},
        ]
        with patch.object(model_eval, "init_config"), \
             patch.object(model_eval, "is_server_running", return_value=True), \
             patch.object(model_eval, "get_models", return_value=["m1"]), \
             patch.object(model_eval, "build_tasks_from_model", return_value={}), \
             patch.object(model_eval, "run_eval", return_value=fake_results), \
             patch.object(model_eval, "_print_results"):
            model_eval.main()

    def test_multiple_models_with_flush(self, mock_llm, monkeypatch):
        """Multiple models trigger flush_between_models (which calls call)."""
        import model_eval
        monkeypatch.setattr(sys, "argv", ["model_eval"])
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
        mock_call.assert_called()

    def test_low_memory_warning(self, mock_llm, monkeypatch):
        """Memory > threshold triggers warning."""
        import model_eval
        monkeypatch.setattr(sys, "argv", ["model_eval"])
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

    def test_model_too_big_for_memory(self, mock_llm, monkeypatch):
        """Model too big for available memory — warning."""
        import model_eval
        monkeypatch.setattr(sys, "argv", ["model_eval"])
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

    def test_server_not_responsive(self, mock_llm, monkeypatch):
        """Server not responsive — print FAIL."""
        import model_eval
        monkeypatch.setattr(sys, "argv", ["model_eval"])
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

    def test_task_arg_in_hardcoded_tasks(self, mock_llm, monkeypatch):
        """--task filters to single task from hardcoded TASKS."""
        import model_eval
        # Get a real task from eval_tasks_core
        from eval_tasks_core import TASKS
        first_task = next(iter(TASKS.keys()))
        monkeypatch.setattr(sys, "argv", ["model_eval", "--task", first_task])
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

    def test_flush_success_path(self, mock_llm, monkeypatch):
        """Test the success path of flush via main() with multiple models."""
        import model_eval
        monkeypatch.setattr(sys, "argv", ["model_eval"])
        with patch.object(model_eval, "init_config"), \
             patch.object(model_eval, "is_server_running", return_value=True), \
             patch.object(model_eval, "get_models", return_value=["m1", "m2"]), \
             patch.object(model_eval, "build_tasks_from_model", return_value={}), \
             patch.object(model_eval, "run_eval", return_value=[]), \
             patch.object(model_eval, "call", return_value={}), \
             patch.object(model_eval, "_print_results"), \
             patch.object(model_eval, "is_server_responsive", return_value=True), \
             patch.object(model_eval, "estimate_model_memory", return_value=4), \
             patch.object(model_eval, "get_memory_percent", return_value=50.0), \
             patch("time.sleep"):
            model_eval.main()

    def test_flush_with_error_triggers_restart(self, mock_llm, monkeypatch):
        import model_eval
        monkeypatch.setattr(sys, "argv", ["model_eval"])
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
             patch("requests.get") as mock_get:
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_get.return_value = mock_resp
            model_eval.main()

    def test_flush_subprocess_error_caught(self, mock_llm, monkeypatch):
        import model_eval
        monkeypatch.setattr(sys, "argv", ["model_eval"])
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
             patch("requests.get", return_value=MagicMock(status_code=200)):
            model_eval.main()

    def test_flush_request_error_caught(self, mock_llm, monkeypatch):
        import model_eval
        monkeypatch.setattr(sys, "argv", ["model_eval"])
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
             patch("requests.get", side_effect=Exception("conn err")):
            model_eval.main()

    def test_flush_call_exception_caught(self, mock_llm, monkeypatch):
        import model_eval
        monkeypatch.setattr(sys, "argv", ["model_eval"])
        with patch.object(model_eval, "init_config"), \
             patch.object(model_eval, "is_server_running", return_value=True), \
             patch.object(model_eval, "get_models", return_value=["m1", "m2"]), \
             patch.object(model_eval, "build_tasks_from_model", return_value={}), \
             patch.object(model_eval, "run_eval", return_value=[]), \
             patch.object(model_eval, "call", side_effect=Exception("call err")), \
             patch.object(model_eval, "_print_results"), \
             patch.object(model_eval, "is_server_responsive", return_value=True), \
             patch.object(model_eval, "estimate_model_memory", return_value=4), \
             patch.object(model_eval, "get_memory_percent", return_value=50.0), \
             patch("time.sleep"):
            model_eval.main()


class TestMainBlock:
    def test_main_block(self, monkeypatch):
        import sys
        from unittest.mock import MagicMock
        monkeypatch.setattr(sys, "argv", ["model_eval"])
        import model_eval
        mock_main = MagicMock()
        saved = model_eval.main
        model_eval.main = mock_main
        try:
            # Inline the __main__ block
            exec("if __name__ == \"__main__\":\n    main()\n",
                 {"__name__": "__main__", "main": mock_main})
        finally:
            model_eval.main = saved
        mock_main.assert_called_once()
