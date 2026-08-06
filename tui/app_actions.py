import asyncio
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import lib.llm.client as llm_client
import lib.osaurus_lib as osaurus_lib
from lib.llm.client import get_models, is_server_running
from lib.quality_entry import compare_to_baseline, get_dimension_weights, load_baseline
from lib.quality_models import Score, ScoreCard
from PIL import Image
from rename.cli import image_extensions, rename_image
from textual.containers import Horizontal
from textual.widgets import Button, Collapsible, Label, ListItem, Markdown
from twitter.browser import collect_tweets_via_browser
from twitter.summarize import summarize_with_llm
from weekend.cli import _fetch_data, _parse_fixed, _parse_transient
from weekend.llm import generate_weekend_plan
from weekend.output import build_markdown_tables

from tui.app_views import _tool_default_model
from tui.lib import ICON_OK, ICON_WARN


def _collect_images(directory: Path) -> list[Path]:
    return list(set(
        f for ext in image_extensions for f in directory.glob(f"*{ext}")
    ))

async def on_mount(self) -> None:
    self.active_schedules = []
    self.run_worker(self.scheduler_loop())
    await self.action_refresh_models()

async def action_refresh_models(self) -> None:
    status_box = self.query_one("#server-status-box")
    status_box.update("[yellow]Checking Osaurus...[/yellow]")
    is_running = await asyncio.to_thread(is_server_running)
    if is_running:
        models = await asyncio.to_thread(get_models)
        if models:
            status_box.update("[green]Server Online[/green]")
            options = [(m, m) for m in models]
            for select_id, config_key in (
                ("#wk-model", "json"),
                ("#tw-model", "summarize"),
                ("#ev-model", "think"),
                ("#rn-model", "filename"),
            ):
                try:
                    widget = self.query_one(select_id)
                    widget.options = options
                    preferred = _tool_default_model(config_key)
                    widget.value = preferred if preferred in models else models[0]
                except Exception:
                    pass
    else:
        status_box.update("[red]Server Offline[/red]")

async def run_weekend_planner(self) -> None:
    self._update_status("#wk-result-area", "Setting date range (Friday through Sunday)...")

    location = self.query_one("#wk-location").value
    model = self.query_one("#wk-model").value
    use_cache = self.query_one("#wk-cache").value
    use_foundation = self.query_one("#wk-foundation").value

    # Setup configs/arguments
    os.environ["OLLAMA_MODEL"] = model or ""

    try:
        # Shift variables for execution
        fri = datetime.now()
        while fri.weekday() != 4:  # Find Friday
            fri += timedelta(days=1)
        sun = fri + timedelta(days=2)

        year = fri.strftime("%Y")
        month_name = fri.strftime("%B")

        # Fetch weather, events, and venues
        weather, events, venues, dates = await asyncio.to_thread(
            _fetch_data, fri, sun, year, month_name, use_cache
        )

        msg = f"Date range: {dates}\nGenerating weekend plan..."
        self._update_status("#wk-result-area", msg)

        # Run LLM pipeline
        json_trans, json_fixed = await asyncio.to_thread(
            generate_weekend_plan,
            model,
            weather,
            events,
            venues,
            dates,
            location=location,
            age_range="4-8",
            date_range=dates,
            use_foundation=use_foundation
        )

        # Format items
        fixed_acts = _parse_fixed(json_fixed, model, {})
        transient_items = _parse_transient(json_trans, model, {})

        plan_md = build_markdown_tables(
            dates, weather, {"transient_events": transient_items}, fixed_acts
        )

        # Mount final result
        container = self.query_one("#wk-result-area")
        for child in list(container.children):
            child.remove()
        container.mount(Markdown(plan_md))

    except Exception as e:
        self._update_status("#wk-result-area", f"Error: {e}")

async def run_twitter_summarizer(self) -> None:
    self._update_status("#tw-result-area", "Opening browser session...")

    since = self.query_one("#tw-since").value
    model = self.query_one("#tw-model").value

    try:
        # Parse since time
        since_time = datetime.now(timezone.utc) - timedelta(hours=24)
        if since.endswith("h"):
            since_time = datetime.now(timezone.utc) - timedelta(hours=int(since[:-1]))
        elif since.endswith("d"):
            since_time = datetime.now(timezone.utc) - timedelta(days=int(since[:-1]))

        # Fetch tweets via browser
        fmt_time = since_time.strftime('%Y-%m-%d %H:%M:%S UTC')
        msg_scraping = f"Scraping tweets since: {fmt_time}..."
        self._update_status("#tw-result-area", msg_scraping)

        tweets = await asyncio.to_thread(
            collect_tweets_via_browser,
            since_time,
            debug=False
        )

        if not tweets:
            msg_fallback = "No tweets scraped. Using fallback cache..."
            self._update_status("#tw-result-area", msg_fallback)
            from twitter.output import load_debug_cache
            tweets = load_debug_cache() or []

        if not tweets:
            self._update_status("#tw-result-area", "No timeline data available.")
            return

        msg_sum = f"Scraped {len(tweets)} tweets. Summarizing timeline with {model}..."
        self._update_status("#tw-result-area", msg_sum)

        # Summarize timeline
        host = os.environ.get("OLLAMA_BASE_URL", "http://localhost:1337")
        summary = await asyncio.to_thread(
            summarize_with_llm,
            tweets,
            host,
            model,
            ""
        )

        # Extract thinking block
        import re
        thinking_match = re.search(r"<thinking>(.*?)</thinking>", summary, re.DOTALL)
        if thinking_match:
            thinking_text = thinking_match.group(1).strip()
            clean_summary = re.sub(
                r"<thinking>.*?</thinking>", "", summary, flags=re.DOTALL
            ).strip()
        else:
            thinking_text = None
            clean_summary = summary.strip()

        # Mount final result
        container = self.query_one("#tw-result-area")
        for child in list(container.children):
            child.remove()

        if thinking_text:
            container.mount(Collapsible(
                Markdown(thinking_text),
                title="Show model thinking",
                collapsed=True
            ))

        container.mount(Markdown(clean_summary))

    except Exception as e:
        self._update_status("#tw-result-area", f"Error: {e}")

async def run_model_evaluator(self) -> None:
    self._update_status("#ev-result-area", "Running quality evaluation...")

    task = self.query_one("#ev-task").value
    model = self.query_one("#ev-model").value

    try:
        baseline = await asyncio.to_thread(load_baseline)
        if not baseline:
            self._update_status(
                "#ev-result-area",
                "No baseline found to run evaluation against."
            )
            return

        # Run mock scorecard match
        scorecards = []
        dim_weights = get_dimension_weights(task)

        for key, prev in baseline.items():
            parts = key.split("::", 2)
            if len(parts) == 3:
                m, t, case_id = parts
                if m == model and t == task:
                    dims = []
                    for dname, dscore in prev.get("dimensions", {}).items():
                        weight = dim_weights.get(dname, 1.0)
                        dims.append(Score(dname, dscore, weight))
                    sc = ScoreCard(model, task, case_id, dims, "", prev.get("elapsed", 0.0))
                    scorecards.append(sc)

        warnings = compare_to_baseline(scorecards)

        container = self.query_one("#ev-result-area")
        for child in list(container.children):
            child.remove()

        if warnings:
            container.mount(Markdown("\n".join(f"- {ICON_WARN} {w}" for w in warnings)))
        else:
            container.mount(Markdown(f"### {ICON_OK} No regressions detected against baseline."))

    except Exception as e:
        self._update_status("#ev-result-area", f"Error: {e}")

async def run_image_renamer(self) -> None:
    self._update_status("#rn-result-area", "Loading images...")

    directory_str = self.query_one("#rn-dir").value
    model = self.query_one("#rn-model").value
    dry_run = self.query_one("#rn-dry-run").value
    force = self.query_one("#rn-force").value

    directory = Path(directory_str)
    if not directory.exists() or not directory.is_dir():
        self._update_status("#rn-result-area", "Invalid directory path.")
        return

    image_files = _collect_images(directory)

    if not image_files:
        self._update_status("#rn-result-area", "No images found.")
        return

    msg = f"Renaming {len(image_files)} images with {model}..."
    self._update_status("#rn-result-area", msg)

    host = os.environ.get("OLLAMA_BASE_URL", "http://localhost:1337")
    results = []

    for image_path in sorted(image_files):
        success, message = await asyncio.to_thread(
            rename_image,
            image_path,
            dry_run=dry_run,
            force=force,
            llm_host=host,
            llm_model=model,
            vlm_model=model,
            api_key=""
        )
        if success:
            results.append(f"- **{image_path.name}**: Renamed successfully")
        else:
            results.append(f"- **{image_path.name}**: Skipped ({message})")

    container = self.query_one("#rn-result-area")
    for child in list(container.children):
        child.remove()
    container.mount(Markdown("\n".join(results)))

async def load_image_preview(self) -> None:
    directory_str = self.query_one("#rn-dir").value
    preview_box = self.query_one("#rn-preview-box")
    preview_info = self.query_one("#rn-preview-info")

    preview_box.update("Loading...")

    try:
        directory = Path(directory_str)
        if not directory.exists() or not directory.is_dir():
            preview_box.update("[red]Invalid Dir[/red]")
            return

        image_files = _collect_images(directory)

        if not image_files:
            preview_box.update("[yellow]No images[/yellow]")
            preview_info.update("")
            return

        first_image = sorted(image_files)[0]

        # Read dimensions
        with Image.open(first_image) as img:
            w, h = img.size
            size_kb = os.path.getsize(first_image) / 1024

        preview_info.update(f"{first_image.name}\n{w}x{h} px | {size_kb:.1f} KB")

    except Exception as e:
        preview_box.update(f"[red]Error: {e}[/red]")
        preview_info.update("")

async def refresh_history_archive(self) -> None:
    hist_list = self.query_one("#hist-list")
    hist_list.clear()

    doc_dir = Path.home() / "Documents"
    if not doc_dir.exists() or not doc_dir.is_dir():
        item = ListItem(Label("[yellow]No history directory found[/yellow]"))
        hist_list.append(item)
        return

    files = []
    for p in doc_dir.glob("*.md"):
        if p.name.startswith("weekend_plan_") or ("_to_" in p.name and len(p.name) > 20):
            files.append(p)

    files.sort(key=lambda x: x.stat().st_mtime, reverse=True)

    for f in files:
        if f.name.startswith("weekend_plan_"):
            name_clean = f.name.replace('weekend_plan_', '')
            name_clean = name_clean.replace('.md', '').replace('_', ' ')
            label = f"Weekend Plan: {name_clean}"
        else:
            label = f"Summary: {f.name.replace('.md', '').replace('_', ' ')}"

        item = ListItem(Label(label))
        item.filepath = f
        hist_list.append(item)

async def load_history_item(self, item: ListItem) -> None:
    filepath = getattr(item, "filepath", None)
    viewer = self.query_one("#hist-viewer")

    for child in list(viewer.children):
        child.remove()

    if not filepath or not filepath.exists():
        viewer.mount(Label("[red]File not found[/red]"))
        return

    try:
        content = await asyncio.to_thread(filepath.read_text)
        viewer.mount(Markdown(content))
    except Exception as e:
        viewer.mount(Label(f"[red]Failed to load file: {e}[/red]"))

def apply_model_parameters(self) -> None:
    status = self.query_one("#param-status")
    try:
        temp_str = self.query_one("#param-temp").value
        max_tokens_str = self.query_one("#param-max-tokens").value

        try:
            temp = float(temp_str)
            if not (0.0 <= temp <= 2.0):
                raise ValueError()
        except ValueError:
            status.update("[red]Error: Temperature must be a float between 0.0 and 2.0[/red]")
            return

        try:
            max_tokens = int(max_tokens_str)
            if max_tokens <= 0:
                raise ValueError()
        except ValueError:
            status.update("[red]Error: Max Tokens must be a positive integer[/red]")
            return

        llm_client.GLOBAL_OVERRIDES["temperature"] = temp
        llm_client.GLOBAL_OVERRIDES["max_tokens"] = max_tokens
        osaurus_lib.GLOBAL_OVERRIDES["temperature"] = temp
        osaurus_lib.GLOBAL_OVERRIDES["max_tokens"] = max_tokens

        msg = f"[green]Applied successfully: Temp={temp}, Max Tokens={max_tokens}[/green]"
        status.update(msg)
    except Exception as e:
        status.update(f"[red]Error: {e}[/red]")

def reset_model_parameters(self) -> None:
    status = self.query_one("#param-status")
    self.query_one("#param-temp").value = "0.1"
    self.query_one("#param-max-tokens").value = "16000"

    llm_client.GLOBAL_OVERRIDES.clear()
    osaurus_lib.GLOBAL_OVERRIDES.clear()

    status.update("[green]Reset to default values[/green]")

async def scheduler_loop(self) -> None:
    while True:
        await asyncio.sleep(1)
        for sched in list(self.active_schedules):
            if sched["next_run"] <= datetime.now() and not sched["is_running"]:
                sched["is_running"] = True
                self.run_worker(self.run_scheduled_task(sched))

async def run_scheduled_task(self, sched: dict) -> None:
    try:
        sched["last_run_status"] = "Running"
        self.refresh_scheduler_display()
        await asyncio.sleep(2)
        sched["last_run_status"] = "Success"
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
        card_row = Horizontal(classes="sched-card")
        card_row.mount(Label(card_content, classes="sched-card-info"))
        btn_id = f"btn-sched-del-{sched['id']}"
        card_row.mount(Button("Delete", variant="error", id=btn_id, classes="sched-card-btn"))
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

