import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime, timedelta
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
        mock_status.update.assert_any_call("[green]🟢 Task scheduled successfully.[/green]")
        
        mock_status.reset_mock()
        app.add_scheduler_task()
        assert len(app.active_schedules) == 1
        mock_status.update.assert_any_call("[red]Error: twitter is already scheduled.[/red]")
        
        app.remove_scheduler_task("1")
        assert len(app.active_schedules) == 0
