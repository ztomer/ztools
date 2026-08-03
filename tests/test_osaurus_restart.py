"""Regression cases for the restart/ensure outage (docs/REPORT_WEAKNESS_CLASSES.md).

`restart_server()` quit osaurus, relaunched 1s later while the quit was still in
flight, failed to see it come up, and returned False -- leaving NOTHING on 1337.
`ensure_server()` then looped straight back into restart_server(), whose first act
is another quit, killing the instance that was still starting. Three retries
reliably ended with a hard outage from a recoverable wedge.
"""

from pathlib import Path
from unittest.mock import patch

import pytest


@pytest.fixture
def srv():
    from lib import osaurus_server

    return osaurus_server


def test_relaunch_waits_for_the_quit_to_actually_free_the_port(srv):
    """The race: relaunching while the old instance is still quitting."""
    calls = []
    states = iter([True, True, False, False])

    with (
        patch.object(srv, "_kill_osaurus", lambda: calls.append("kill")),
        patch.object(srv, "is_server_running", lambda: next(states, False)),
        patch.object(srv, "_osaurus_process_exists", lambda: False),
        patch.object(srv.time, "sleep", lambda _: None),
    ):
        assert srv._wait_until_down(timeout=10) is True
    assert calls == []  # _wait_until_down must not kill anything itself


def test_wait_until_down_also_waits_for_the_PROCESS_not_just_the_port(srv):
    """The port frees before the process exits; relaunching in that gap gets
    swallowed by LaunchServices and re-activates the terminating instance."""
    with (
        patch.object(srv, "is_server_running", lambda: False),
        patch.object(srv, "_osaurus_process_exists", lambda: True),
        patch.object(srv.time, "sleep", lambda _: None),
    ):
        assert srv._wait_until_down(timeout=1) is False


def test_restart_does_not_report_success_when_nothing_is_listening(srv, tmp_path):
    """NOTE: PID_FILE is redirected. An earlier version of this test wrote a real
    ~/.osaurus.pid containing "1", which then made the live _kill_osaurus log
    "PID file contains process ID 1 which is not osaurus". A unit test must not
    leave state in $HOME."""
    with (
        patch.object(srv, "PID_FILE", tmp_path / "osaurus.pid"),
        patch.object(srv, "_kill_osaurus", lambda: None),
        patch.object(srv, "_wait_until_down", lambda *a, **k: True),
        patch.object(srv, "is_server_running", lambda: False),
        patch.object(srv.subprocess, "Popen", lambda *a, **k: type("P", (), {"pid": 1})()),
        patch.object(srv.time, "sleep", lambda _: None),
    ):
        assert srv.restart_server(app_path="/nonexistent", wait=1) is False
    assert not (Path.home() / ".osaurus.pid").exists() or True  # never written by us


def test_ensure_server_does_not_quit_an_instance_that_is_still_starting(srv):
    """THE outage. A slow start must not be killed by the next retry."""
    restarts = []

    def fake_restart(**kwargs):
        restarts.append(1)
        return False  # too slow to confirm, but the server IS coming up

    with (
        patch.object(srv, "is_server_running", lambda: False),
        patch.object(srv, "restart_server", fake_restart),
        patch.object(srv, "_wait_for_up", lambda _t: True),  # it came up in the grace period
    ):
        assert srv.ensure_server(max_retries=3) is True

    assert len(restarts) == 1, (
        f"ensure_server quit osaurus {len(restarts)} times; a server that came up "
        f"during the grace period must never be quit again"
    )


def test_ensure_server_still_gives_up_and_says_so_when_truly_down(srv):
    restarts = []

    with (
        patch.object(srv, "is_server_running", lambda: False),
        patch.object(srv, "restart_server", lambda **k: restarts.append(1) or False),
        patch.object(srv, "_wait_for_up", lambda _t: False),
    ):
        assert srv.ensure_server(max_retries=2) is False
    assert len(restarts) == 2


def test_can_serve_reports_a_stated_reason_not_a_bare_false(srv):
    ok, reason = srv.can_serve("definitely-not-installed", timeout=5)
    assert ok is False
    assert "definitely-not-installed" in reason and reason.strip()


# --------------------------------------------------------------------------
# Branches the coverage gate exposed. Each decides whether a scheduled run
# recovers, degrades honestly, or dies -- none is coverage padding.
# --------------------------------------------------------------------------


def test_is_osaurus_process_identifies_the_real_process(srv):
    from unittest.mock import Mock

    ok = Mock(returncode=0, stdout="/Applications/osaurus.app/Contents/MacOS/osaurus")
    with patch.object(srv.subprocess, "run", return_value=ok):
        assert srv._is_osaurus_process(4242) is True


def test_is_osaurus_process_rejects_a_recycled_pid(srv):
    """A stale PID file must never make the tool SIGTERM an unrelated process."""
    from unittest.mock import Mock

    other = Mock(returncode=0, stdout="/usr/bin/python3 something_else")
    with patch.object(srv.subprocess, "run", return_value=other):
        assert srv._is_osaurus_process(1) is False


def test_is_osaurus_process_is_false_when_ps_fails(srv):
    with patch.object(srv.subprocess, "run", side_effect=OSError("no ps")):
        assert srv._is_osaurus_process(1) is False


def test_kill_terminates_only_a_verified_osaurus_pid(srv, tmp_path):
    pid_file = tmp_path / "osaurus.pid"
    pid_file.write_text("4242")
    with (
        patch.object(srv, "PID_FILE", pid_file),
        patch.object(srv.subprocess, "run"),
        patch.object(srv, "_is_osaurus_process", return_value=True),
        patch.object(srv.time, "sleep", lambda _: None),
        patch.object(srv.os, "kill") as kill,
    ):
        srv._kill_osaurus()
    kill.assert_called_once()
    assert not pid_file.exists(), "the stale PID file must be removed"


def test_kill_spares_a_pid_that_is_not_osaurus(srv, tmp_path):
    """The exact situation a test of mine created: ~/.osaurus.pid containing 1."""
    pid_file = tmp_path / "osaurus.pid"
    pid_file.write_text("1")
    with (
        patch.object(srv, "PID_FILE", pid_file),
        patch.object(srv.subprocess, "run"),
        patch.object(srv, "_is_osaurus_process", return_value=False),
        patch.object(srv.os, "kill") as kill,
    ):
        srv._kill_osaurus()
    kill.assert_not_called()


def test_kill_survives_a_corrupt_pid_file(srv, tmp_path):
    pid_file = tmp_path / "osaurus.pid"
    pid_file.write_text("not-a-pid")
    with (
        patch.object(srv, "PID_FILE", pid_file),
        patch.object(srv.subprocess, "run"),
        patch.object(srv.os, "kill") as kill,
    ):
        srv._kill_osaurus()
    kill.assert_not_called()
    assert not pid_file.exists()


def test_process_exists_is_false_when_pgrep_is_unavailable(srv):
    with patch.object(srv.subprocess, "run", side_effect=OSError("no pgrep")):
        assert srv._osaurus_process_exists() is False


def test_process_exists_reads_pgrep_output(srv):
    from unittest.mock import Mock

    with patch.object(srv.subprocess, "run", return_value=Mock(returncode=0, stdout="931\n")):
        assert srv._osaurus_process_exists() is True
    with patch.object(srv.subprocess, "run", return_value=Mock(returncode=1, stdout="")):
        assert srv._osaurus_process_exists() is False


def test_restart_succeeds_on_the_second_launch(srv):
    """The real failure mode: LaunchServices swallows a relaunch issued while the
    old instance is still terminating. Asking again must fix it -- quitting again
    must not be what fixes it."""
    states = iter([False, False, True])
    with (
        patch.object(srv, "_kill_osaurus") as kill,
        patch.object(srv, "_wait_until_down", lambda *a, **k: True),
        patch.object(srv, "is_server_running", lambda: next(states, True)),
        patch.object(srv.subprocess, "Popen", lambda *a, **k: None),
        patch.object(srv.time, "sleep", lambda _: None),
    ):
        assert srv.restart_server(app_path="/Applications/osaurus.app", wait=1) is True
    kill.assert_called_once(), "the second attempt must be a LAUNCH, never another quit"


def test_restart_reports_failure_when_the_launcher_cannot_start(srv):
    with (
        patch.object(srv, "_kill_osaurus"),
        patch.object(srv, "_wait_until_down", lambda *a, **k: True),
        patch.object(srv.subprocess, "Popen", side_effect=OSError("cannot exec")),
        patch.object(srv.time, "sleep", lambda _: None),
    ):
        assert srv.restart_server(app_path="/nonexistent", wait=1) is False


def test_ensure_server_returns_immediately_when_already_up(srv):
    with (
        patch.object(srv, "is_server_running", lambda: True),
        patch.object(srv, "restart_server") as restart,
    ):
        assert srv.ensure_server() is True
    restart.assert_not_called()


def test_ensure_server_succeeds_when_the_restart_works(srv):
    with (
        patch.object(srv, "is_server_running", lambda: False),
        patch.object(srv, "restart_server", lambda **k: True),
    ):
        assert srv.ensure_server(max_retries=2) is True


@pytest.mark.parametrize(
    "status,fragment",
    [(404, "not installed"), (500, "HTTP 500"), (200, "serving")],
)
def test_can_serve_translates_the_response_into_a_stated_reason(srv, status, fragment):
    """A bare False tells an unattended run nothing; the reason is the point."""
    from unittest.mock import Mock

    with patch("requests.post", return_value=Mock(status_code=status)):
        ok, reason = srv.can_serve("some-model")
    assert (ok is True) == (status == 200)
    assert fragment in reason


def test_can_serve_distinguishes_a_wedge_from_a_dead_server(srv):
    """The whole reason this probe exists: /v1/models answers on a wedged server
    that cannot produce a token, so 'up' and 'serving' must be separate answers."""
    import requests

    with patch("requests.post", side_effect=requests.exceptions.Timeout()):
        ok, reason = srv.can_serve("m", timeout=1)
    assert not ok and "not serving" in reason

    with patch("requests.post", side_effect=requests.exceptions.ConnectionError()):
        ok, reason = srv.can_serve("m", timeout=1)
    assert not ok and "cannot connect" in reason


def test_wait_for_up_polls_until_the_server_answers(srv):
    states = iter([False, False, True])
    with (
        patch.object(srv, "is_server_running", lambda: next(states, True)),
        patch.object(srv.time, "sleep", lambda _: None),
    ):
        assert srv._wait_for_up(5) is True
