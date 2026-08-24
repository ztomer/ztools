"""Tests for model_eval main() flow."""

import pathlib
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


class TestFlushBetweenModels:
    """flush_between_models is a nested function inside main(). Use main() to exercise it."""

    def test_flush_success_path(self, mock_llm, monkeypatch, capsys):
        """Test the success path of flush via main() with multiple models."""
        import eval.cli as model_eval

        monkeypatch.setattr(sys, "argv", ["eval.cli"])
        with (
            patch.object(model_eval, "init_config"),
            patch.object(model_eval, "is_server_running", return_value=True),
            patch.object(model_eval, "is_server_responsive", return_value=True),
            patch.object(model_eval, "get_models", return_value=["m1", "m2"]),
            patch.object(model_eval, "build_tasks_from_model", return_value={}),
            patch.object(model_eval, "run_eval", return_value=[]),
            patch("eval.cli_runtime.call", return_value={}) as mock_call,
            patch.object(model_eval, "_print_results"),
            patch.object(model_eval, "is_server_responsive", return_value=True),
            patch.object(model_eval, "estimate_model_memory", return_value=4),
            patch.object(model_eval, "get_memory_percent", return_value=50.0),
            patch("time.sleep"),
        ):
            model_eval.main()
        # call() was invoked for the flush
        assert mock_call.call_count == 1
        out = capsys.readouterr().out
        assert "Flushing" in out

    def test_flush_with_error_restarts_through_osaurus_one_not_open_n(
        self, mock_llm, monkeypatch, capsys
    ):
        """The restart must go through tools/osaurus_one.sh, never `open -n`.

        This test previously asserted the OPPOSITE -- that flush ran `osascript
        quit` and then `open`. That pinned a real defect: `open -n` forces a NEW
        osaurus instance, which is the second server osaurus_one.sh exists to
        prevent, and two servers each load their own copy of the model. A sweep
        lost three models to it before anyone read the flag.
        """
        import eval.cli as model_eval
        import eval.cli_runtime as cli_runtime

        monkeypatch.setattr(sys, "argv", ["eval.cli"])
        monkeypatch.setattr(cli_runtime, "RESTART_READY_BUDGET", 0)
        monkeypatch.setattr(
            cli_runtime, "osaurus_one_script", lambda: pathlib.Path("/fake/osaurus_one.sh")
        )
        with (
            patch.object(model_eval, "init_config"),
            patch.object(model_eval, "is_server_running", return_value=True),
            patch.object(model_eval, "is_server_responsive", return_value=True),
            patch.object(model_eval, "get_models", return_value=["m1", "m2"]),
            patch.object(model_eval, "build_tasks_from_model", return_value={}),
            patch.object(model_eval, "run_eval", return_value=[]),
            patch("eval.cli_runtime.call", return_value={"error": "fail"}),
            patch.object(model_eval, "_print_results"),
            patch.object(model_eval, "estimate_model_memory", return_value=4),
            patch.object(model_eval, "get_memory_percent", return_value=50.0),
            patch("subprocess.run") as mock_subprocess,
            patch("time.sleep"),
        ):
            mock_subprocess.return_value = MagicMock(returncode=0, stdout="", stderr="")
            model_eval.main()

        sub_calls = [str(c) for c in mock_subprocess.call_args_list]
        assert any("osaurus_one.sh" in s and "--restart" in s for s in sub_calls), sub_calls
        assert not any("-n" in s and "open" in s for s in sub_calls), (
            f"`open -n` starts a SECOND server: {sub_calls}"
        )
        assert not any("osascript" in s for s in sub_calls), (
            f"the quit must be osaurus_one.sh's job, not a raw osascript: {sub_calls}"
        )

    def test_an_unrecoverable_server_is_announced_not_silent(
        self, mock_llm, monkeypatch, capsys
    ):
        """The old code exhausted its retries and printed NOTHING, then ran the
        model anyway -- which is how a column of INFRA zeros reached the results
        table with no warning above it."""
        import eval.cli as model_eval
        import eval.cli_runtime as cli_runtime

        monkeypatch.setattr(sys, "argv", ["eval.cli"])
        monkeypatch.setattr(cli_runtime, "osaurus_one_script", lambda: None)
        with (
            patch.object(model_eval, "init_config"),
            patch.object(model_eval, "is_server_running", return_value=True),
            patch.object(model_eval, "is_server_responsive", return_value=True),
            patch.object(model_eval, "get_models", return_value=["m1", "m2"]),
            patch.object(model_eval, "build_tasks_from_model", return_value={}),
            patch.object(model_eval, "run_eval", return_value=[]),
            patch("eval.cli_runtime.call", return_value={"error": "fail"}),
            patch.object(model_eval, "_print_results"),
            patch.object(model_eval, "estimate_model_memory", return_value=4),
            patch.object(model_eval, "get_memory_percent", return_value=50.0),
            patch("time.sleep"),
        ):
            model_eval.main()
        out = capsys.readouterr().out
        assert "NOT quality results" in out, out

    def test_flush_subprocess_error_caught(self, mock_llm, monkeypatch, capsys):
        import eval.cli as model_eval

        monkeypatch.setattr(sys, "argv", ["eval.cli"])
        with (
            patch.object(model_eval, "init_config"),
            patch.object(model_eval, "is_server_running", return_value=True),
            patch.object(model_eval, "is_server_responsive", return_value=True),
            patch.object(model_eval, "get_models", return_value=["m1", "m2"]),
            patch.object(model_eval, "build_tasks_from_model", return_value={}),
            patch.object(model_eval, "run_eval", return_value=[]),
            patch("eval.cli_runtime.call", return_value={"error": "fail"}),
            patch.object(model_eval, "_print_results"),
            patch.object(model_eval, "is_server_responsive", return_value=True),
            patch.object(model_eval, "estimate_model_memory", return_value=4),
            patch.object(model_eval, "get_memory_percent", return_value=50.0),
            patch("subprocess.run", side_effect=_only_osaurus_commands_fail),
            patch("time.sleep"),
        ):
            # A restart that raises must degrade ONE model, not abort a sweep
            # that has already spent hours. Previously this escaped and killed main().
            model_eval.main()
        out = capsys.readouterr().out
        assert "Flushing" in out
        assert "Traceback" not in out
        assert "NOT quality results" in out, out

    def test_flush_request_error_caught(self, mock_llm, monkeypatch, capsys):
        import eval.cli as model_eval

        monkeypatch.setattr(sys, "argv", ["eval.cli"])
        with (
            patch.object(model_eval, "init_config"),
            patch.object(model_eval, "is_server_running", return_value=True),
            patch.object(model_eval, "is_server_responsive", return_value=True),
            patch.object(model_eval, "get_models", return_value=["m1", "m2"]),
            patch.object(model_eval, "build_tasks_from_model", return_value={}),
            patch.object(model_eval, "run_eval", return_value=[]),
            patch("eval.cli_runtime.call", return_value={"error": "fail"}),
            patch.object(model_eval, "_print_results"),
            patch.object(model_eval, "is_server_responsive", return_value=True),
            patch.object(model_eval, "estimate_model_memory", return_value=4),
            patch.object(model_eval, "get_memory_percent", return_value=50.0),
            patch("subprocess.run"),
            patch("time.sleep"),
            patch("requests.Session") as mock_session,
        ):
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
        with (
            patch.object(model_eval, "init_config"),
            patch.object(model_eval, "is_server_running", return_value=True),
            patch.object(model_eval, "is_server_responsive", return_value=True),
            patch.object(model_eval, "get_models", return_value=["m1", "m2"]),
            patch.object(model_eval, "build_tasks_from_model", return_value={}),
            patch.object(model_eval, "run_eval", return_value=[]),
            patch("eval.cli_runtime.call", side_effect=Exception("api boom")),
            patch.object(model_eval, "_print_results"),
            patch.object(model_eval, "estimate_model_memory", return_value=4),
            patch.object(model_eval, "get_memory_percent", return_value=50.0),
            patch("time.sleep"),
            # Without this the restart path executed the REAL osaurus_one.sh and
            # this unit test killed the developer's server. conftest now blocks
            # that outright; patching here is what the gate asks for.
            patch("subprocess.run") as mock_subprocess,
        ):
            mock_subprocess.return_value = MagicMock(returncode=1, stdout="", stderr="no")
            model_eval.main()
        out = capsys.readouterr().out
        # One message for both the error-dict and the exception path. They used to
        # diverge, and the exception route skipped the GPU-lock guard the other
        # honoured -- so the strings were unified along with the control flow.
        assert "Flush failed" in out and "api boom" in out


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
