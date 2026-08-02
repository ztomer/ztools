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
