from unittest.mock import MagicMock, patch

import pytest

import tui.app as app_mod


@pytest.mark.asyncio
async def test_scheduler_add_remove():
    app = app_mod.ZToolsApp()
    app.active_schedules = []

    mock_status = MagicMock()
    mock_type = MagicMock()
    mock_type.value = "twitter"
    mock_interval = MagicMock()
    mock_interval.value = "60"

    def mock_query(selector):
        if selector == "#sched-form-status":
            return mock_status
        elif selector == "#sched-task-type":
            return mock_type
        elif selector == "#sched-interval":
            return mock_interval
        elif selector == "#sched-list":
            return MagicMock()
        raise ValueError(f"Unexpected selector: {selector}")

    with patch.object(app, "query_one", side_effect=mock_query):
        app.add_scheduler_task()
        assert len(app.active_schedules) == 1
        assert app.active_schedules[0]["task_type"] == "twitter"
        assert app.active_schedules[0]["interval_seconds"] == 60
        mock_status.update.assert_any_call("[green]Task scheduled successfully.[/green]")

        mock_status.reset_mock()
        app.add_scheduler_task()
        assert len(app.active_schedules) == 1
        mock_status.update.assert_any_call("[red]Error: twitter is already scheduled.[/red]")

        app.remove_scheduler_task("1")
        assert len(app.active_schedules) == 0


class _FakeProc:
    def __init__(self, returncode, stderr=b""):
        self.returncode = returncode
        self._stderr = stderr

    async def communicate(self):
        return b"", self._stderr


def _sched(task_type="twitter"):
    from datetime import datetime

    return {
        "id": "1",
        "task_type": task_type,
        "interval_seconds": 60,
        "next_run": datetime.now(),
        "last_run_status": "Idle",
        "is_running": True,
    }


def test_scheduled_task_command_targets_the_real_tools():
    app = app_mod.ZToolsApp()
    for task_type, module in (("twitter", "twitter"), ("weekend", "weekend")):
        cmd = app.scheduled_task_command(task_type)
        assert cmd[1:] == ["-m", module]
    # rename needs a directory; without a valid one there is no runnable command
    with patch.object(app, "query_one", side_effect=ValueError):
        assert app.scheduled_task_command("rename") is None
    assert app.scheduled_task_command("nonsense") is None


@pytest.mark.asyncio
async def test_run_scheduled_task_reports_the_real_exit_status():
    """The status must come from a process, not from a sleep.

    run_scheduled_task used to `await asyncio.sleep(2)` and set "Success"
    without dispatching anything.
    """
    app = app_mod.ZToolsApp()
    sched = _sched()
    spawned = []

    async def fake_exec(*command, **kwargs):
        spawned.append(command)
        return _FakeProc(0)

    with (
        patch.object(app, "refresh_scheduler_display"),
        patch("asyncio.create_subprocess_exec", fake_exec),
    ):
        await app.run_scheduled_task(sched)

    assert spawned and spawned[0][1:] == ("-m", "twitter")
    assert sched["last_run_status"] == "Success"
    assert sched["is_running"] is False


@pytest.mark.asyncio
async def test_run_scheduled_task_surfaces_failure():
    app = app_mod.ZToolsApp()
    sched = _sched()

    async def fake_exec(*command, **kwargs):
        return _FakeProc(2, b"Traceback...\nRuntimeError: server offline\n")

    with (
        patch.object(app, "refresh_scheduler_display"),
        patch("asyncio.create_subprocess_exec", fake_exec),
    ):
        await app.run_scheduled_task(sched)

    assert sched["last_run_status"].startswith("Failed:")
    assert "server offline" in sched["last_run_status"]


def test_scheduler_cards_are_built_with_their_children():
    """Mounting into an unmounted Horizontal raised MountError on every add."""
    from textual.containers import Horizontal

    app = app_mod.ZToolsApp()
    app.active_schedules = [_sched()]
    container = MagicMock()
    container.children = []

    with patch.object(app, "query_one", return_value=container):
        app.refresh_scheduler_display()

    assert container.mount.call_count == 1
    card = container.mount.call_args[0][0]
    assert isinstance(card, Horizontal)
    # Children came from the constructor (Textual holds them as _pending_children
    # until the card itself is mounted); mounting into the unmounted card is the
    # MountError this guards against.
    assert len(card._pending_children) == 2
