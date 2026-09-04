"""The eval's server-restart path.

This code exists because `flush_between_models` used to hand-roll its own restart:
`osascript` quit, a fixed `sleep 3`, then `open -n -a osaurus`. `-n` starts a NEW
instance, which is the second server `tools/osaurus_one.sh` exists to prevent, and
two servers each load their own copy of the model rather than queueing. A sweep lost
three of its first four models to it and reported one of them as "62% avg" over a
partial task set.

The readiness check was the other half: it polled the model-LIST endpoint and treated
200 as servable. Listing models proves the HTTP layer is up and says nothing about
whether a 13-35GB model can be loaded, which is why the log printed "Server restarted"
and "Server: OK" immediately before failing every completion.
"""

from unittest.mock import MagicMock, patch

import eval.cli_runtime as cli_runtime


class TestScriptDiscovery:
    def test_no_script_when_running_from_an_install(self):
        """The wheel does not ship tools/, so None is a legitimate answer and the
        caller must degrade with a stated reason rather than invent a path."""
        with patch("lib.paths.repo_root", return_value=None):
            assert cli_runtime.osaurus_one_script() is None

    def test_no_script_when_the_file_is_absent_from_a_real_checkout(self, tmp_path):
        with patch("lib.paths.repo_root", return_value=tmp_path):
            assert cli_runtime.osaurus_one_script() is None

    def test_the_script_is_found_in_a_normal_checkout(self, tmp_path):
        (tmp_path / "tools").mkdir()
        script = tmp_path / "tools" / "osaurus_one.sh"
        script.write_text("#!/bin/sh\n")
        with patch("lib.paths.repo_root", return_value=tmp_path):
            assert cli_runtime.osaurus_one_script() == script


class TestWaitUntilModelServes:
    def test_a_serving_model_returns_immediately(self):
        with patch.object(cli_runtime, "call", return_value={}) as c:
            assert cli_runtime.wait_until_model_serves("m", out=MagicMock(), budget=30) is True
        assert c.call_count == 1, "polled again after the model already answered"

    def test_it_keeps_trying_while_the_model_is_still_loading(self):
        """The whole point of the budget: a big model is not servable the instant
        the port opens, and the previous ~21s allowance gave up too early."""
        with (
            patch.object(cli_runtime, "call", side_effect=[{"error": "loading"}, {}]) as c,
            patch.object(cli_runtime.time, "sleep"),
        ):
            assert cli_runtime.wait_until_model_serves("m", out=MagicMock(), budget=30) is True
        assert c.call_count == 2

    def test_it_gives_up_after_the_budget_and_names_the_last_error(self):
        out = MagicMock()
        with (
            patch.object(cli_runtime, "call", return_value={"error": "still loading"}),
            patch.object(cli_runtime.time, "sleep"),
            patch.object(cli_runtime.time, "monotonic", side_effect=[0, 0, 999]),
        ):
            assert cli_runtime.wait_until_model_serves("m", out=out, budget=30) is False
        printed = " ".join(str(c) for c in out.print.call_args_list)
        assert "still not serving" in printed and "still loading" in printed

    def test_a_transport_that_raises_is_not_fatal(self):
        """Connection refused arrives as an exception, not an error dict — that is
        the shape the real failure took."""
        out = MagicMock()
        with (
            patch.object(cli_runtime, "call", side_effect=OSError("connection refused")),
            patch.object(cli_runtime.time, "sleep"),
            patch.object(cli_runtime.time, "monotonic", side_effect=[0, 0, 999]),
        ):
            assert cli_runtime.wait_until_model_serves("m", out=out, budget=30) is False
        assert "connection refused" in " ".join(str(c) for c in out.print.call_args_list)

    def test_the_budget_is_read_at_call_time_not_bound_at_import(self, monkeypatch):
        """As a default argument this bound once at import, so patching the constant
        could not shorten it and the suite sat through the full wait."""
        monkeypatch.setattr(cli_runtime, "RESTART_READY_BUDGET", 0)
        out = MagicMock()
        with patch.object(cli_runtime, "call", return_value={"error": "no"}) as c:
            assert cli_runtime.wait_until_model_serves("m", out=out) is False
        assert c.call_count == 0, "a zero budget still polled; the constant was ignored"


class TestRestartServer:
    def test_a_missing_script_is_reported_not_papered_over(self):
        out = MagicMock()
        with patch.object(cli_runtime, "osaurus_one_script", return_value=None):
            assert cli_runtime.restart_server(out=out) is False
        assert "single-server invariant" in " ".join(str(c) for c in out.print.call_args_list)

    def test_a_nonzero_exit_is_reported_with_its_last_line(self, tmp_path):
        out = MagicMock()
        with (
            patch.object(cli_runtime, "osaurus_one_script", return_value=tmp_path / "s.sh"),
            patch("subprocess.run", return_value=MagicMock(returncode=2, stdout="", stderr="port busy")),
        ):
            assert cli_runtime.restart_server(out=out) is False
        assert "port busy" in " ".join(str(c) for c in out.print.call_args_list)

    def test_a_raising_subprocess_returns_false_rather_than_aborting_the_sweep(self, tmp_path):
        """Contract is "return a bool, never raise". A restart failure must cost one
        model, not a sweep that has already spent hours."""
        out = MagicMock()
        with (
            patch.object(cli_runtime, "osaurus_one_script", return_value=tmp_path / "s.sh"),
            patch("subprocess.run", side_effect=Exception("boom")),
        ):
            assert cli_runtime.restart_server(out=out) is False

    def test_success_uses_the_script_with_restart_and_never_open_n(self, tmp_path):
        out = MagicMock()
        with (
            patch.object(cli_runtime, "osaurus_one_script", return_value=tmp_path / "s.sh"),
            patch("subprocess.run", return_value=MagicMock(returncode=0, stdout="", stderr="")) as run,
        ):
            assert cli_runtime.restart_server(out=out) is True
        argv = run.call_args[0][0]
        assert argv[1] == "--restart"
        assert "-n" not in argv, "`open -n` is what started a second server"


class TestFlushReportsAnUnservableModel:
    def test_restart_succeeds_but_model_never_serves(self, tmp_path):
        """The gap that produced silent INFRA zeros: the server came back, the model
        did not, and nothing said so before the scores were printed."""
        out = MagicMock()
        with (
            patch.object(cli_runtime, "call", return_value={"error": "nope"}),
            patch.object(cli_runtime, "restart_server", return_value=True),
            patch.object(cli_runtime, "wait_until_model_serves", return_value=False),
            patch.object(cli_runtime.time, "sleep"),
        ):
            cli_runtime.flush_between_models("prev", "next", out=out)
        printed = " ".join(str(c) for c in out.print.call_args_list)
        assert "NOT quality results" in printed and "next" in printed

    def test_a_successful_recovery_says_nothing_alarming(self):
        """The happy path after a restart: no 'NOT quality results' warning, because
        the model really is servable and its scores really are comparable."""
        out = MagicMock()
        with (
            patch.object(cli_runtime, "call", return_value={"error": "nope"}),
            patch.object(cli_runtime, "restart_server", return_value=True),
            patch.object(cli_runtime, "wait_until_model_serves", return_value=True),
            patch.object(cli_runtime.time, "sleep"),
        ):
            cli_runtime.flush_between_models("prev", "next", out=out)
        printed = " ".join(str(c) for c in out.print.call_args_list)
        assert "NOT quality results" not in printed, printed

    def test_a_healthy_server_never_restarts_at_all(self):
        """Guards the common case: if the flush call succeeds there must be no
        restart, or every model transition would bounce the server."""
        out = MagicMock()
        with (
            patch.object(cli_runtime, "call", return_value={}),
            patch.object(cli_runtime, "restart_server") as restart,
            patch.object(cli_runtime.time, "sleep"),
        ):
            cli_runtime.flush_between_models("prev", "next", out=out)
        restart.assert_not_called()
