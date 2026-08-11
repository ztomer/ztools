"""Scheduler tab actions for the ztools dashboard.

Split out of app_actions.py to keep both modules under the repo's 500-line
limit. Bound onto the app class in tui/app.py exactly as the other actions are.
"""

import asyncio
import sys
from datetime import datetime, timedelta
from pathlib import Path

from textual.containers import Horizontal
from textual.widgets import Button, Label


async def scheduler_loop(self) -> None:
    while True:
        await asyncio.sleep(1)
        for sched in list(self.active_schedules):
            if sched["next_run"] <= datetime.now() and not sched["is_running"]:
                sched["is_running"] = True
                self.run_worker(self.run_scheduled_task(sched))

def scheduled_task_command(self, task_type: str) -> list[str] | None:
    """The command a scheduled task runs, or None if it cannot be built."""
    if task_type == "twitter":
        return [sys.executable, "-m", "twitter"]
    if task_type == "weekend":
        return [sys.executable, "-m", "weekend"]
    if task_type == "rename":
        try:
            directory = self.query_one("#rn-dir").value
        except Exception:
            directory = ""
        if not directory or not Path(directory).is_dir():
            return None
        return [sys.executable, "-m", "rename", directory]
    return None

async def run_scheduled_task(self, sched: dict) -> None:
    """Run the scheduled tool for real and report its actual exit status.

    This used to sleep two seconds and report "Success" without dispatching
    anything — a fabricated result presented as completion.
    """
    try:
        sched["last_run_status"] = "Running"
        self.refresh_scheduler_display()

        command = self.scheduled_task_command(sched["task_type"])
        if command is None:
            sched["last_run_status"] = f"Failed: no runnable command for {sched['task_type']}"
            return

        proc = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await proc.communicate()
        if proc.returncode == 0:
            sched["last_run_status"] = "Success"
        else:
            detail = (stderr or b"").decode(errors="replace").strip().splitlines()
            reason = detail[-1] if detail else f"exit {proc.returncode}"
            sched["last_run_status"] = f"Failed: {reason[:80]}"
    except Exception as e:
        sched["last_run_status"] = f"Failed: {e}"
    finally:
        sched["is_running"] = False
        sched["next_run"] = datetime.now() + timedelta(seconds=sched["interval_seconds"])
        self.refresh_scheduler_display()

def refresh_scheduler_display(self) -> None:
    try:
        container = self.query_one("#sched-list")
    except Exception:
        return
    for child in list(container.children):
        child.remove()
    if not self.active_schedules:
        container.mount(Label("No scheduled tasks configured."))
        return
    for sched in self.active_schedules:
        task_type_label = {
            "twitter": "Twitter Summarizer",
            "rename": "Screenshot Renamer",
            "weekend": "Weekend Planner"
        }.get(sched["task_type"], sched["task_type"])
        interval_sec = sched["interval_seconds"]
        if interval_sec == 10:
            interval_str = "Every 10 seconds"
        elif interval_sec == 60:
            interval_str = "Every 1 minute"
        else:
            hours = interval_sec // 3600
            interval_str = f"Every {hours}h"
        status_text = sched["last_run_status"]
        next_run_str = sched["next_run"].strftime("%H:%M:%S")
        card_content = (
            f"[bold white]{task_type_label}[/bold white]\n"
            f"Interval: {interval_str} | Status: {status_text} | Next Run: {next_run_str}"
        )
        btn_id = f"btn-sched-del-{sched['id']}"
        # Children are passed to the constructor: mounting into a widget that is
        # not itself mounted raises MountError, which the blanket except turned
        # into a red "Error" on every add even though the schedule was stored.
        card_row = Horizontal(
            Label(card_content, classes="sched-card-info"),
            Button("Delete", variant="error", id=btn_id, classes="sched-card-btn"),
            classes="sched-card",
        )
        container.mount(card_row)

def add_scheduler_task(self) -> None:
    status = self.query_one("#sched-form-status")
    try:
        task_type = self.query_one("#sched-task-type").value
        interval_str = self.query_one("#sched-interval").value

        if not task_type or not interval_str:
            status.update("[red]Error: Please select both task type and interval.[/red]")
            return

        interval = int(interval_str)
        task_id = str(len(self.active_schedules) + 1)
        for s in self.active_schedules:
            if s["task_type"] == task_type:
                status.update(f"[red]Error: {task_type} is already scheduled.[/red]")
                return
        self.active_schedules.append({
            "id": task_id,
            "task_type": task_type,
            "interval_seconds": interval,
            "next_run": datetime.now() + timedelta(seconds=interval),
            "last_run_status": "Idle",
            "is_running": False
        })
        status.update("[green]Task scheduled successfully.[/green]")
        self.refresh_scheduler_display()
    except Exception as e:
        status.update(f"[red]Error: {e}[/red]")

def remove_scheduler_task(self, task_id: str) -> None:
    self.active_schedules = [s for s in self.active_schedules if s["id"] != task_id]
    self.refresh_scheduler_display()

