"""Tests for lib.osaurus_server - server lifecycle and connection tests."""

from unittest.mock import patch

import pytest


@pytest.fixture
def mock_llm():
    from lib.testing import MockLLM

    mock = MockLLM()
    mock.patch_all()
    yield mock
    mock.unpatch()


class TestRestartServer:
    def test_restart_server_success(self, mock_llm):
        from lib.osaurus_server import restart_server

        with (
            patch("lib.osaurus_server.subprocess.run"),
            patch("lib.osaurus_server.subprocess.Popen"),
            patch("lib.osaurus_server.is_server_running", return_value=True),
        ):
            assert restart_server() is True

    def test_restart_server_timeout(self, mock_llm):
        from lib.osaurus_server import restart_server

        with (
            patch("lib.osaurus_server.subprocess.run"),
            patch("lib.osaurus_server.subprocess.Popen"),
            patch("lib.osaurus_server.is_server_running", return_value=False),
            patch("lib.osaurus_server.time.sleep"),
        ):  # Skip sleep
            assert restart_server(wait=2) is False

    def test_restart_pkill_exception(self, mock_llm):
        """When pkill fails, should still try to restart."""
        from lib.osaurus_server import restart_server

        with (
            patch("lib.osaurus_server.subprocess.run", side_effect=Exception("pkill fail")),
            patch("lib.osaurus_server.subprocess.Popen"),
            patch("lib.osaurus_server.is_server_running", return_value=True),
            patch("lib.osaurus_server.time.sleep"),
        ):
            assert restart_server() is True

    def test_restart_popen_exception(self, mock_llm):
        """When Popen fails, return False."""
        from lib.osaurus_server import restart_server

        with (
            patch("lib.osaurus_server.subprocess.run"),
            patch("lib.osaurus_server.subprocess.Popen", side_effect=Exception("popen fail")),
            patch("lib.osaurus_server.time.sleep"),
        ):
            assert restart_server() is False


class TestEnsureServer:
    def test_ensure_server_already_running(self, mock_llm):
        from lib.osaurus_server import ensure_server

        with patch("lib.osaurus_server.is_server_running", return_value=True):
            assert ensure_server() is True

    def test_ensure_server_restart_succeeds(self, mock_llm):
        from lib.osaurus_server import ensure_server

        with (
            patch("lib.osaurus_server.is_server_running", side_effect=[False, True]),
            patch("lib.osaurus_server.restart_server", return_value=True),
            patch("lib.osaurus_server.print"),
        ):
            assert ensure_server() is True

    def test_ensure_server_restart_fails(self, mock_llm):
        from lib.osaurus_server import ensure_server

        with (
            patch("lib.osaurus_server.is_server_running", return_value=False),
            patch("lib.osaurus_server.restart_server", return_value=False),
            patch("lib.osaurus_server.print"),
        ):
            assert ensure_server() is False

    def test_ensure_server_max_retries(self, mock_llm):
        from lib.osaurus_server import ensure_server

        # All attempts fail, last check still false
        with (
            patch("lib.osaurus_server.is_server_running", return_value=False),
            patch("lib.osaurus_server.restart_server", return_value=False),
            patch("lib.osaurus_server.print"),
        ):
            assert ensure_server(max_retries=2) is False

    def test_ensure_server_loop_completes(self, mock_llm):
        """When all retries are exhausted but restart_server succeeded, falls through to final check."""
        from lib.osaurus_server import ensure_server

        # Two loop iterations (2 False), then final is_server_running() check (True)
        with (
            patch("lib.osaurus_server.is_server_running", side_effect=[False, False, True]),
            patch("lib.osaurus_server.restart_server", return_value=True),
            patch("lib.osaurus_server.print"),
        ):
            assert ensure_server(max_retries=2) is True


class TestTestConnection:
    def test_connection_server_not_running(self, mock_llm):
        from lib.osaurus_server import test_connection

        with patch("lib.osaurus_server.is_server_running", return_value=False):
            result = test_connection()
        assert result["status"] == "error"
        assert "not running" in result["message"]

    def test_connection_no_models(self, mock_llm):
        from lib.osaurus_server import test_connection

        with (
            patch("lib.osaurus_server.is_server_running", return_value=True),
            patch("lib.osaurus_server.get_models", return_value=[]),
        ):
            result = test_connection()
        assert result["status"] == "error"
        assert "No models" in result["message"]

    def test_connection_success(self, mock_llm):
        import lib.osaurus_lib
        from lib.osaurus_server import test_connection

        mock_result = {"content": "Hello response", "error": None}
        with (
            patch("lib.osaurus_server.is_server_running", return_value=True),
            patch("lib.osaurus_server.get_models", return_value=["model-a"]),
            patch.object(lib.osaurus_lib, "call", return_value=mock_result),
        ):
            result = test_connection()
        assert result["status"] == "ok"
        assert "Hello response" in result["response_preview"]
        assert result["test_model"] == "model-a"

    def test_connection_error(self, mock_llm):
        import lib.osaurus_lib
        from lib.osaurus_server import test_connection

        mock_result = {"content": "", "error": "API failed"}
        with (
            patch("lib.osaurus_server.is_server_running", return_value=True),
            patch("lib.osaurus_server.get_models", return_value=["model-a"]),
            patch.object(lib.osaurus_lib, "call", return_value=mock_result),
        ):
            result = test_connection()
        assert result["status"] == "error"
        assert result["message"] == "API failed"

    def test_connection_exception(self, mock_llm):
        from lib.osaurus_server import test_connection

        with (
            patch("lib.osaurus_server.is_server_running", return_value=True),
            patch("lib.osaurus_server.get_models", side_effect=Exception("boom")),
        ):
            result = test_connection()
        assert result["status"] == "error"
        assert "boom" in result["message"]

    def test_connection_with_model(self, mock_llm):
        import lib.osaurus_lib
        from lib.osaurus_server import test_connection

        mock_result = {"content": "Hi back", "error": None}
        with (
            patch("lib.osaurus_server.is_server_running", return_value=True),
            patch("lib.osaurus_server.get_models", return_value=["model-a", "model-b"]),
            patch.object(lib.osaurus_lib, "call", return_value=mock_result) as call,
        ):
            result = test_connection(model="specific-model")
        assert result["status"] == "ok"
        assert result["test_model"] == "specific-model"
        assert call.call_args.args[0] == "specific-model"


class TestPanicDump:
    def test_panic_dump(self, mock_llm, tmp_path, monkeypatch):
        from lib.osaurus_server import panic_dump

        monkeypatch.setattr(__import__("pathlib").Path, "home", lambda: tmp_path)
        panic_dump("test content")
        files = list((tmp_path / "llm_dumps").iterdir())
        assert len(files) == 1
        assert "test content" in files[0].read_text()

    def test_panic_dump_empty(self, mock_llm, tmp_path, monkeypatch):
        from lib.osaurus_server import panic_dump

        monkeypatch.setattr(__import__("pathlib").Path, "home", lambda: tmp_path)
        panic_dump("")
        files = list((tmp_path / "llm_dumps").iterdir())
        assert len(files) == 1
        assert "(empty)" in files[0].read_text()

    def test_panic_dump_creates_dir(self, mock_llm, tmp_path, monkeypatch):
        from lib.osaurus_server import panic_dump

        monkeypatch.setattr(__import__("pathlib").Path, "home", lambda: tmp_path)
        # llm_dumps should not exist yet
        assert not (tmp_path / "llm_dumps").exists()
        panic_dump("content")
        assert (tmp_path / "llm_dumps").exists()
