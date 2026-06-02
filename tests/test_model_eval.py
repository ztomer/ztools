"""Tests for model_eval: memory, tasks, main flow."""
import pytest
import os
import json
from unittest.mock import patch, MagicMock
from io import StringIO
from rich.console import Console


def _capture_console():
    """Replace model_eval.console with a buffered one."""
    import model_eval
    buf = StringIO()
    new_console = Console(file=buf, force_terminal=True, force_interactive=True, width=120)
    return model_eval.console, new_console, buf


class TestGetMemoryPercent:
    def test_with_psutil(self, mock_llm):
        import model_eval
        fake_mem = MagicMock()
        fake_mem.virtual_memory.return_value = MagicMock(percent=50.0)
        with patch.dict("sys.modules", {"psutil": fake_mem}):
            result = model_eval.get_memory_percent()
        assert result == 50.0

    def test_without_psutil(self, mock_llm):
        import model_eval
        # Force ImportError
        import builtins
        real_import = builtins.__import__
        def mock_import(name, *args, **kwargs):
            if name == "psutil":
                raise ImportError("no psutil")
            return real_import(name, *args, **kwargs)
        with patch.object(builtins, "__import__", side_effect=mock_import):
            result = model_eval.get_memory_percent()
        assert result == 0.0


class TestCheckMemorySafe:
    def test_safe_memory(self, mock_llm):
        import model_eval
        old, new, buf = _capture_console()
        try:
            model_eval.console = new
            with patch.object(model_eval, "get_memory_percent", return_value=50.0):
                assert model_eval.check_memory_safe() is True
        finally:
            model_eval.console = old

    def test_unsafe_memory(self, mock_llm):
        import model_eval
        old, new, buf = _capture_console()
        try:
            model_eval.console = new
            with patch.object(model_eval, "get_memory_percent", return_value=95.0):
                assert model_eval.check_memory_safe() is False
            assert "Memory" in buf.getvalue()
        finally:
            model_eval.console = old


class TestIsServerResponsive:
    def test_responsive(self, mock_llm):
        import model_eval
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        with patch("requests.get", return_value=mock_resp):
            assert model_eval.is_server_responsive() is True

    def test_not_responsive(self, mock_llm):
        import model_eval
        mock_resp = MagicMock()
        mock_resp.status_code = 500
        with patch("requests.get", return_value=mock_resp):
            assert model_eval.is_server_responsive() is False

    def test_exception(self, mock_llm):
        import model_eval
        with patch("requests.get", side_effect=Exception("conn error")):
            assert model_eval.is_server_responsive() is False


class TestMonitorMemoryLoop:
    def test_starts_thread(self, mock_llm):
        import model_eval
        with patch.object(model_eval, "get_memory_percent", return_value=50.0):
            t = model_eval.monitor_memory_loop(interval=1)
        # Real thread started
        assert t is not None
        assert t.daemon is True  # daemon=True set in code
        # Stop it so the test doesn't hang
        t.running = False

    def test_monitor_thread_logs_high_memory(self, mock_llm):
        """Line 153: when mem > threshold, log warning."""
        import model_eval
        from io import StringIO
        from rich.console import Console
        buf = StringIO()
        old_console = model_eval.console
        new_console = Console(file=buf, force_terminal=True, force_interactive=True, width=120)
        model_eval.console = new_console
        try:
            with patch.object(model_eval, "get_memory_percent", return_value=95.0):
                t = model_eval.monitor_memory_loop(interval=0)  # no sleep
            t.join(timeout=2)  # wait for thread to finish one iteration
            t.running = False
            assert "Memory" in buf.getvalue()
        finally:
            model_eval.console = old_console


class TestEstimateModelMemory:
    def test_7b_model(self, mock_llm):
        import model_eval
        assert model_eval.estimate_model_memory("qwen2.5-7b") == 7

    def test_27b_model(self, mock_llm):
        import model_eval
        assert model_eval.estimate_model_memory("some-27b-model") == 27

    def test_no_size_in_name(self, mock_llm):
        import model_eval
        assert model_eval.estimate_model_memory("unknown-model") == 4

    def test_case_insensitive(self, mock_llm):
        import model_eval
        assert model_eval.estimate_model_memory("Qwen2-72B") == 72


class TestLoadTasksFromConfig:
    def test_no_prompts(self, mock_llm):
        import model_eval
        with patch.object(model_eval, "get_model_prompts_all", return_value=None):
            assert model_eval.load_tasks_from_config("m1") is None

    def test_empty_prompts(self, mock_llm):
        import model_eval
        with patch.object(model_eval, "get_model_prompts_all", return_value={}):
            # Empty dict is falsy, so `if not prompts: return None`
            assert model_eval.load_tasks_from_config("m1") is None

    def test_full_prompts(self, mock_llm):
        import model_eval
        prompts = {
            "weekend_fixed": "WF",
            "weekend_transient": "WT",
            "summarize": "S",
            "filename": "F",
            "file_summary": "FS",
        }
        with patch.object(model_eval, "get_model_prompts_all", return_value=prompts):
            result = model_eval.load_tasks_from_config("m1")
        assert result["detailed_json"] == "WF"
        assert result["json"] == "WT"
        assert result["summarize"] == "S"
        assert result["filename"] == "F"
        assert result["file_summary"] == "FS"

    def test_partial_prompts(self, mock_llm):
        import model_eval
        prompts = {"filename": "F", "summarize": "S"}
        with patch.object(model_eval, "get_model_prompts_all", return_value=prompts):
            result = model_eval.load_tasks_from_config("m1")
        assert "filename" in result
        assert "summarize" in result
        assert "json" not in result


class TestUpdateConfig:
    def test_no_config_file(self, mock_llm, tmp_path, monkeypatch):
        import model_eval
        old, new, buf = _capture_console()
        try:
            model_eval.console = new
            # Patch __file__ to point to tmp_path
            monkeypatch.setattr(model_eval, "__file__", str(tmp_path / "model_eval.py"))
            model_eval.update_config({"task1": "model-x"})
            assert "Config file not found" in buf.getvalue()
        finally:
            model_eval.console = old

    def test_updates_config(self, mock_llm, tmp_path, monkeypatch):
        import model_eval
        # Create conf/config.yaml
        conf_dir = tmp_path / "conf"
        conf_dir.mkdir()
        config_file = conf_dir / "config.yaml"
        config_file.write_text("existing_key: value\n")
        old, new, buf = _capture_console()
        try:
            model_eval.console = new
            monkeypatch.setattr(model_eval, "__file__", str(tmp_path / "model_eval.py"))
            model_eval.update_config({"task1": "model-x", "task2": "model-y"})
            content = config_file.read_text()
            assert "best_models" in content
        finally:
            model_eval.console = old

    def test_existing_best_models(self, mock_llm, tmp_path, monkeypatch):
        import model_eval
        conf_dir = tmp_path / "conf"
        conf_dir.mkdir()
        config_file = conf_dir / "config.yaml"
        config_file.write_text("best_models:\n  task1: old_model\n")
        old, new, buf = _capture_console()
        try:
            model_eval.console = new
            monkeypatch.setattr(model_eval, "__file__", str(tmp_path / "model_eval.py"))
            model_eval.update_config({"task1": "new_model"})
            content = config_file.read_text()
            assert "new_model" in content
        finally:
            model_eval.console = old

    def test_empty_model_value(self, mock_llm, tmp_path, monkeypatch):
        """If model value is falsy, don't add to best_models."""
        import model_eval
        conf_dir = tmp_path / "conf"
        conf_dir.mkdir()
        config_file = conf_dir / "config.yaml"
        config_file.write_text("original: value\n")
        old, new, buf = _capture_console()
        try:
            model_eval.console = new
            monkeypatch.setattr(model_eval, "__file__", str(tmp_path / "model_eval.py"))
            model_eval.update_config({"task1": None})
            content = config_file.read_text()
            assert "task1" not in content
        finally:
            model_eval.console = old


class TestPrintResults:
    def test_basic(self, mock_llm, tmp_path, monkeypatch):
        """Test _print_results saves to file and prints."""
        import model_eval
        monkeypatch.setattr("os.path.expanduser", lambda p: str(tmp_path) if p.startswith("~") else p)
        old, new, buf = _capture_console()
        try:
            model_eval.console = new
            all_results = [
                {"model": "m1", "results": [
                    {"task": "t1", "quality_score": 80, "result": {"content": "x" * 100}},
                ]},
            ]
            best_scores = {"t1": 80}
            best_models = {"t1": "m1"}
            model_eval._print_results(all_results, best_scores, best_models)
            # Saved file
            assert (tmp_path / "eval_results.json").exists()
        finally:
            model_eval.console = old
