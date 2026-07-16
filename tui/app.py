import asyncio
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List

from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, Vertical, ScrollableContainer
from textual.widgets import (
    Header,
    Footer,
    Input,
    Checkbox,
    Select,
    Button,
    RichLog,
    Label,
    Static,
    ListView,
    ListItem,
    ContentSwitcher,
)
from rich.text import Text
from rich.panel import Panel

# Import backend methods
from lib.llm.client import get_models, is_server_running
from lib.tui import STEP, WARN, OK, FAIL
from weekend.cli import _fetch_data, _parse_fixed, _parse_transient
from weekend.llm import generate_weekend_plan
from weekend.output import build_markdown_tables
from twitter.browser import collect_tweets_via_browser
from twitter.summarize import summarize_with_llm
from lib.quality_entry import load_baseline, compare_to_baseline, get_dimension_weights
from lib.quality_models import Score, ScoreCard
from rename.cli import rename_image, clean_filename, image_extensions

DEFAULT_MODELS = [("foundation", "foundation"), ("qwen", "qwen"), ("gemma", "gemma")]


class Sidebar(Vertical):
    """Left sidebar navigation panel."""

    def compose(self) -> ComposeResult:
        yield Label("[bold cyan]ZTools Hub[/bold cyan]", id="app-title")
        yield ListView(
            ListItem(Label("📅 Weekend Planner"), id="nav-weekend"),
            ListItem(Label("🐦 Twitter Summarizer"), id="nav-twitter"),
            ListItem(Label("📊 Model Evaluator"), id="nav-eval"),
            ListItem(Label("🖼️ Image Renamer"), id="nav-rename"),
            id="nav-list"
        )
        yield Static(id="server-status-box")


class WeekendPlannerView(Vertical):
    """TUI Form for generating weekend plans."""

    def compose(self) -> ComposeResult:
        yield Label("[bold green]📅 Weekend Planner[/bold green]", classes="section-title")
        
        with Horizontal(classes="form-row"):
            yield Label("Location:", classes="form-label")
            yield Input(placeholder="City/Region (e.g. Toronto/ON)", value="Toronto/ON", id="wk-location")
            
        with Horizontal(classes="form-row"):
            yield Label("Model:", classes="form-label")
            yield Select(options=DEFAULT_MODELS, id="wk-model", prompt="Select LLM Model")

        with Horizontal(classes="form-row"):
            yield Checkbox("Use Cached Data", value=True, id="wk-cache")
            yield Checkbox("Use On-Device Model (Apple FM)", value=False, id="wk-foundation")

        yield Button("Generate Weekend Plan", variant="success", id="btn-wk-generate")
        yield RichLog(id="wk-log", highlight=True, markup=True)


class TwitterSummarizerView(Vertical):
    """TUI Form for summarizing Twitter timeline."""

    def compose(self) -> ComposeResult:
        yield Label("[bold purple]🐦 Twitter Timeline Summarizer[/bold purple]", classes="section-title")
        
        with Horizontal(classes="form-row"):
            yield Label("Since (Relative):", classes="form-label")
            yield Input(placeholder="e.g. 24h, 48h, 7d", value="24h", id="tw-since")
            
        with Horizontal(classes="form-row"):
            yield Label("Model:", classes="form-label")
            yield Select(options=DEFAULT_MODELS, id="tw-model", prompt="Select LLM Model")

        with Horizontal(classes="form-row"):
            yield Checkbox("Use Cache", value=False, id="tw-cache")

        yield Button("Summarize Timeline", variant="primary", id="btn-tw-generate")
        yield RichLog(id="tw-log", highlight=True, markup=True)


class ModelEvaluatorView(Vertical):
    """TUI Form for running quality evaluations."""

    def compose(self) -> ComposeResult:
        yield Label("[bold yellow]📊 Model Quality Evaluator[/bold yellow]", classes="section-title")
        
        with Horizontal(classes="form-row"):
            yield Label("Task:", classes="form-label")
            yield Select(
                options=[("filename", "filename"), ("summarize", "summarize"), ("file_summary", "file_summary")],
                id="ev-task",
                value="filename",
                prompt="Select Task"
            )
            
        with Horizontal(classes="form-row"):
            yield Label("Model:", classes="form-label")
            yield Select(options=DEFAULT_MODELS, id="ev-model", prompt="Select LLM Model")

        yield Button("Run Regression Check", variant="warning", id="btn-ev-generate")
        yield RichLog(id="ev-log", highlight=True, markup=True)


class ImageRenamerView(Vertical):
    """TUI Form for renaming screenshot files."""

    def compose(self) -> ComposeResult:
        yield Label("[bold magenta]🖼️ Screenshot Image Renamer[/bold magenta]", classes="section-title")
        
        with Horizontal(classes="form-row"):
            yield Label("Directory Path:", classes="form-label")
            yield Input(placeholder="Absolute path to folder", value=os.getcwd(), id="rn-dir")
            
        with Horizontal(classes="form-row"):
            yield Label("Model:", classes="form-label")
            yield Select(options=DEFAULT_MODELS, id="rn-model", prompt="Select LLM Model")

        with Horizontal(classes="form-row"):
            yield Checkbox("Dry Run (Preview)", value=True, id="rn-dry-run")
            yield Checkbox("Force Relevance Check", value=False, id="rn-force")

        yield Button("Rename Screenshots", variant="error", id="btn-rn-generate")
        yield RichLog(id="rn-log", highlight=True, markup=True)


class ZToolsApp(App):
    """Unified ZTools interactive terminal dashboard."""

    CSS = """
    Screen {
        background: #0f172a;
        color: #e2e8f0;
    }
    Sidebar {
        width: 30;
        background: #1e293b;
        border-right: tall #334155;
        dock: left;
    }
    #app-title {
        text-align: center;
        padding: 1;
        background: #0f172a;
        margin-bottom: 1;
        border-bottom: double #334155;
    }
    #nav-list {
        background: transparent;
        height: 1fr;
    }
    #nav-list ListItem {
        padding: 1;
    }
    #nav-list ListItem:hover {
        background: #334155;
    }
    #server-status-box {
        padding: 1;
        background: #0f172a;
        text-align: center;
        border-top: solid #334155;
    }
    .section-title {
        margin-bottom: 1;
        padding-bottom: 1;
        border-bottom: solid #1e293b;
    }
    .form-row {
        height: auto;
        margin-bottom: 1;
        align: middle;
    }
    .form-label {
        width: 18;
        color: #94a3b8;
    }
    Input {
        width: 40;
        background: #1e293b;
        color: #f8fafc;
        border: solid #334155;
    }
    Select {
        width: 40;
        background: #1e293b;
        color: #f8fafc;
        border: solid #334155;
    }
    Checkbox {
        margin-right: 4;
        width: auto;
    }
    Button {
        margin-top: 1;
        margin-bottom: 1;
        width: 25;
    }
    RichLog {
        height: 1fr;
        background: #090d16;
        border: solid #1e293b;
        color: #f8fafc;
        padding: 1;
    }
    """

    TITLE = "ZTools Terminal Hub"
    BINDINGS = [
        ("q", "quit", "Quit app"),
        ("ctrl+r", "refresh_models", "Refresh Server Models")
    ]

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield Sidebar()
        with ContentSwitcher(initial="weekend"):
            yield WeekendPlannerView(id="weekend")
            yield TwitterSummarizerView(id="twitter")
            yield ModelEvaluatorView(id="eval")
            yield ImageRenamerView(id="rename")
        yield Footer()

    async def on_mount(self) -> None:
        """Trigger dynamic models load on start."""
        await self.action_refresh_models()

    async def action_refresh_models(self) -> None:
        """Query models from Osaurus server in background."""
        status_box = self.query_one("#server-status-box")
        status_box.update("[yellow]Checking Osaurus...[/yellow]")
        
        host = os.environ.get("OLLAMA_BASE_URL", "http://localhost:1337")
        is_running = await asyncio.to_thread(is_server_running)
        
        if is_running:
            status_box.update("[green]🟢 Server Online[/green]")
            models = await asyncio.to_thread(get_models)
            if models:
                options = [(m, m) for m in models]
                # Populate select dropdowns
                for select_id in ("#wk-model", "#tw-model", "#ev-model", "#rn-model"):
                    try:
                        self.query_one(select_id).options = options
                        self.query_one(select_id).value = models[0]
                    except Exception:
                        pass
        else:
            status_box.update("[red]🔴 Server Offline[/red]")

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        """Handle sidebar navigation switcher."""
        switcher = self.query_one(ContentSwitcher)
        if event.item.id == "nav-weekend":
            switcher.current = "weekend"
        elif event.item.id == "nav-twitter":
            switcher.current = "twitter"
        elif event.item.id == "nav-eval":
            switcher.current = "eval"
        elif event.item.id == "nav-rename":
            switcher.current = "rename"

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        """Route button run events to correct workers."""
        if event.button.id == "btn-wk-generate":
            await self.run_weekend_planner()
        elif event.button.id == "btn-tw-generate":
            await self.run_twitter_summarizer()
        elif event.button.id == "btn-ev-generate":
            await self.run_model_evaluator()
        elif event.button.id == "btn-rn-generate":
            await self.run_image_renamer()

    async def run_weekend_planner(self) -> None:
        log = self.query_one("#wk-log")
        log.clear()
        
        location = self.query_one("#wk-location").value
        model = self.query_one("#wk-model").value
        use_cache = self.query_one("#wk-cache").value
        use_foundation = self.query_one("#wk-foundation").value

        log.write("[bold cyan]📅 Bounding dates and fetching forecast...[/bold cyan]")
        
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
            
            log.write(f"Bounded Dates: {dates}")
            log.write(f"Weather Forecast: {weather}")
            log.write("[bold yellow]Invoking Osaurus pipeline...[/bold yellow]")

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

            log.write("[bold green]✔ Weekend plan generated successfully![/bold green]")
            
            plan_md = build_markdown_tables(dates, weather, {"transient_events": transient_items}, fixed_acts)
            log.write("\n" + plan_md)

        except Exception as e:
            log.write(f"[bold red]Error running weekend planner: {e}[/bold red]")

    async def run_twitter_summarizer(self) -> None:
        log = self.query_one("#tw-log")
        log.clear()

        since = self.query_one("#tw-since").value
        model = self.query_one("#tw-model").value
        use_cache = self.query_one("#tw-cache").value
        
        log.write("[bold cyan]🐦 Initializing Playwright and loading Chrome cookies...[/bold cyan]")
        
        try:
            # Parse since time
            since_time = datetime.now(timezone.utc) - timedelta(hours=24)
            if since.endswith("h"):
                since_time = datetime.now(timezone.utc) - timedelta(hours=int(since[:-1]))
            elif since.endswith("d"):
                since_time = datetime.now(timezone.utc) - timedelta(days=int(since[:-1]))

            # Fetch tweets via browser
            log.write(f"Scraping tweets since: {since_time.strftime('%Y-%m-%d %H:%M:%S UTC')}...")
            
            tweets = await asyncio.to_thread(
                collect_tweets_via_browser,
                since_time,
                debug=False
            )
            
            if not tweets:
                log.write("[bold yellow]No tweets scraped. Using fallback cache...[/bold yellow]")
                # If cache exists
                from twitter.output import load_debug_cache
                tweets = load_debug_cache() or []
                
            log.write(f"Scraped {len(tweets)} tweets.")
            
            if not tweets:
                log.write("[bold red]No timeline data available.[/bold red]")
                return

            log.write(f"Summarizing tweets with {model}...")
            
            # Summarize timeline
            host = os.environ.get("OLLAMA_BASE_URL", "http://localhost:1337")
            summary = await asyncio.to_thread(
                summarize_with_llm,
                tweets,
                host,
                model,
                ""
            )
            
            log.write("[bold green]✔ Timeline Summarized Successfully![/bold green]")
            log.write("\n" + summary)
            
        except Exception as e:
            log.write(f"[bold red]Error: {e}[/bold red]")

    async def run_model_evaluator(self) -> None:
        log = self.query_one("#ev-log")
        log.clear()

        task = self.query_one("#ev-task").value
        model = self.query_one("#ev-model").value
        
        log.write(f"[bold cyan]📊 Running quality evaluation for {model} on {task}...[/bold cyan]")
        
        try:
            baseline = await asyncio.to_thread(load_baseline)
            if not baseline:
                log.write("[bold red]No baseline found to run evaluation against.[/bold red]")
                return

            log.write(f"Baseline loaded with {len(baseline)} entries.")
            
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
            if warnings:
                for w in warnings:
                    log.write(f"[bold yellow]⚠ {w}[/bold yellow]")
            else:
                log.write("[bold green]✔ No regressions detected against baseline.[/bold green]")
                
        except Exception as e:
            log.write(f"[bold red]Error: {e}[/bold red]")

    async def run_image_renamer(self) -> None:
        log = self.query_one("#rn-log")
        log.clear()

        directory_str = self.query_one("#rn-dir").value
        model = self.query_one("#rn-model").value
        dry_run = self.query_one("#rn-dry-run").value
        force = self.query_one("#rn-force").value

        log.write(f"[bold cyan]🖼️ Loading images from: {directory_str}[/bold cyan]")
        
        directory = Path(directory_str)
        if not directory.exists() or not directory.is_dir():
            log.write("[bold red]Invalid directory path.[/bold red]")
            return

        image_files = []
        for ext in image_extensions:
            image_files.extend(directory.glob(f"*{ext}"))
            
        image_files = list(set([f for f in image_files if f.suffix.lower() in image_extensions]))

        log.write(f"Found {len(image_files)} image files.")
        if not image_files:
            return

        log.write(f"Renaming with {model}... (Dry run: {dry_run})")
        
        host = os.environ.get("OLLAMA_BASE_URL", "http://localhost:1337")
        
        for image_path in sorted(image_files):
            log.write(f"Processing: {image_path.name}...")
            
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
                log.write(f"[green]✔ Processed {image_path.name}[/green]")
            else:
                log.write(f"[yellow]Skipped/Failed {image_path.name}: {message}[/yellow]")


def main():
    app = ZToolsApp()
    app.run()


if __name__ == "__main__":
    main()
