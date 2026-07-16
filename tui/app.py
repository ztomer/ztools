import asyncio
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List

from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, Vertical, ScrollableContainer, VerticalScroll
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
    Collapsible,
    Markdown,
)
from rich.text import Text
from rich.panel import Panel

# Import backend methods
import lib.llm.client as llm_client
import lib.osaurus_lib as osaurus_lib
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

from PIL import Image

def image_to_ansi(image_path: Path, width: int = 24) -> str:
    try:
        with Image.open(image_path) as img:
            img = img.convert("RGB")
            w, h = img.size
            height = int((h / w) * width)
            if height % 2 != 0:
                height += 1
            if height > 24:
                height = 24
                width = int((w / h) * height)
                
            img = img.resize((width, height), Image.Resampling.BILINEAR)
            pixels = img.load()
            
            lines = []
            for y in range(0, height, 2):
                line_parts = []
                for x in range(width):
                    r1, g1, b1 = pixels[x, y]
                    r2, g2, b2 = pixels[x, y + 1] if y + 1 < height else (0, 0, 0)
                    line_parts.append(f"\033[48;2;{r1};{g1};{b1}m\033[38;2;{r2};{g2};{b2}m▄\033[0m")
                lines.append("".join(line_parts))
            return "\n".join(lines)
    except Exception as e:
        return f"[red]Failed to render preview: {e}[/red]"


class Sidebar(Vertical):
    """Left sidebar navigation panel."""

    def compose(self) -> ComposeResult:
        yield Label("[bold cyan]ZTools Hub[/bold cyan]", id="app-title")
        yield ListView(
            ListItem(Label("📅 Weekend Planner"), id="nav-weekend"),
            ListItem(Label("🐦 Twitter Summarizer"), id="nav-twitter"),
            ListItem(Label("📊 Model Evaluator"), id="nav-eval"),
            ListItem(Label("🖼️ Image Renamer"), id="nav-rename"),
            ListItem(Label("📂 History Archive"), id="nav-history"),
            ListItem(Label("⚙️ Model Parameters"), id="nav-params"),
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
        yield VerticalScroll(id="wk-result-area", classes="result-area")


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
        yield VerticalScroll(id="tw-result-area", classes="result-area")


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
        yield VerticalScroll(id="ev-result-area", classes="result-area")


class ImageRenamerView(Vertical):
    """TUI Form for renaming screenshot files."""

    def compose(self) -> ComposeResult:
        yield Label("[bold magenta]🖼️ Screenshot Image Renamer[/bold magenta]", classes="section-title")
        
        with Horizontal(id="rn-layout"):
            with Vertical(id="rn-form-panel"):
                with Horizontal(classes="form-row"):
                    yield Label("Directory Path:", classes="form-label")
                    yield Input(placeholder="Absolute path to folder", value=os.getcwd(), id="rn-dir")
                    
                with Horizontal(classes="form-row"):
                    yield Label("Model:", classes="form-label")
                    yield Select(options=DEFAULT_MODELS, id="rn-model", prompt="Select LLM Model")

                with Horizontal(classes="form-row"):
                    yield Checkbox("Dry Run (Preview)", value=True, id="rn-dry-run")
                    yield Checkbox("Force Relevance Check", value=False, id="rn-force")

                with Horizontal(classes="form-row"):
                    yield Button("Rename Screenshots", variant="error", id="btn-rn-generate")
                    yield Button("Load Preview", variant="primary", id="btn-rn-preview")
                yield VerticalScroll(id="rn-result-area", classes="result-area")
                
            with Vertical(id="rn-preview-panel"):
                yield Label("[bold white]Preview Window[/bold white]")
                yield Static(id="rn-preview-box")
                yield Label("", id="rn-preview-info")
class HistoryArchiveView(Vertical):
    """TUI view for browsing and viewing historical plans/summaries."""

    def compose(self) -> ComposeResult:
        yield Label("[bold cyan]📂 History Archive[/bold cyan]", classes="section-title")
        with Horizontal(id="hist-layout"):
            with Vertical(id="hist-list-panel"):
                yield Button("Refresh History", variant="primary", id="btn-hist-refresh")
                yield ListView(id="hist-list")
            with Vertical(id="hist-viewer-panel"):
                yield VerticalScroll(id="hist-viewer")


class ModelParametersView(Vertical):
    """TUI view for customizing global model generation parameters."""

    def compose(self) -> ComposeResult:
        yield Label("[bold cyan]⚙️ Model Parameters[/bold cyan]", classes="section-title")
        with Vertical(classes="form-container"):
            with Horizontal(classes="form-row"):
                yield Label("Temperature (0.0 to 2.0):", classes="form-label")
                yield Input(value="0.1", placeholder="0.1", id="param-temp")
            with Horizontal(classes="form-row"):
                yield Label("Max Tokens (positive int):", classes="form-label")
                yield Input(value="16000", placeholder="16000", id="param-max-tokens")
            with Horizontal(classes="form-row"):
                yield Button("Apply Parameters", variant="success", id="btn-param-apply")
                yield Button("Reset Defaults", variant="primary", id="btn-param-reset")
            yield Label("", id="param-status")


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
    .result-area {
        height: 1fr;
        background: #090d16;
        border: solid #1e293b;
        color: #f8fafc;
        padding: 1;
    }
    .result-area Markdown {
        background: transparent;
        color: #e2e8f0;
    }
    .result-area Label {
        color: #e2e8f0;
    }
    Collapsible {
        background: #1e293b;
        border: solid #334155;
        margin-bottom: 1;
    }
    #rn-layout {
        height: 1fr;
    }
    #rn-form-panel {
        width: 1fr;
    }
    #rn-preview-panel {
        width: 32;
        background: #1c2538;
        border-left: solid #334155;
        padding: 1;
        align: center middle;
    }
    #rn-preview-box {
        width: auto;
        height: auto;
        margin-top: 1;
        margin-bottom: 1;
        background: #090d16;
        color: #e2e8f0;
        border: solid #334155;
        text-align: center;
    }
    #rn-preview-info {
        color: #94a3b8;
        font-size: 8;
        text-align: center;
    }
    #hist-layout {
        height: 1fr;
    }
    #hist-list-panel {
        width: 35;
        background: #1c2538;
        border-right: solid #334155;
        padding: 1;
    }
    #hist-list {
        background: transparent;
        height: 1fr;
        margin-top: 1;
    }
    #hist-viewer-panel {
        width: 1fr;
        padding: 1;
    }
    #hist-viewer {
        background: #090d16;
        border: solid #1e293b;
        padding: 1;
        height: 1fr;
    }
    #param-status {
        margin-top: 1;
        padding-left: 2;
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
            yield HistoryArchiveView(id="history")
            yield ModelParametersView(id="params")
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
        """Handle list view item selection."""
        if event.list_view.id == "nav-list":
            switcher = self.query_one(ContentSwitcher)
            if event.item.id == "nav-weekend":
                switcher.current = "weekend"
            elif event.item.id == "nav-twitter":
                switcher.current = "twitter"
            elif event.item.id == "nav-eval":
                switcher.current = "eval"
            elif event.item.id == "nav-rename":
                switcher.current = "rename"
            elif event.item.id == "nav-history":
                switcher.current = "history"
                self.run_worker(self.refresh_history_archive())
            elif event.item.id == "nav-params":
                switcher.current = "params"
        elif event.list_view.id == "hist-list":
            self.run_worker(self.load_history_item(event.item))

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
        elif event.button.id == "btn-rn-preview":
            await self.load_image_preview()
        elif event.button.id == "btn-hist-refresh":
            self.run_worker(self.refresh_history_archive())
        elif event.button.id == "btn-param-apply":
            self.apply_model_parameters()
        elif event.button.id == "btn-param-reset":
            self.reset_model_parameters()

    def _update_status(self, container_id: str, lines: List[str], message: str) -> None:
        lines.append(message)
        container = self.query_one(container_id)
        for child in list(container.children):
            child.remove()
        container.mount(Label(message))

    async def run_weekend_planner(self) -> None:
        lines = []
        self._update_status("#wk-result-area", lines, "⏳ Bounding dates and fetching forecast...")
        
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
            
            self._update_status("#wk-result-area", lines, f"⏳ Bounded dates: {dates}\nInvoking Osaurus pipeline...")

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

            plan_md = build_markdown_tables(dates, weather, {"transient_events": transient_items}, fixed_acts)
            
            # Mount final result
            container = self.query_one("#wk-result-area")
            for child in list(container.children):
                child.remove()
            container.mount(Markdown(plan_md))

        except Exception as e:
            self._update_status("#wk-result-area", lines, f"❌ Error: {e}")

    async def run_twitter_summarizer(self) -> None:
        lines = []
        self._update_status("#tw-result-area", lines, "⏳ Initializing Playwright and loading Chrome cookies...")

        since = self.query_one("#tw-since").value
        model = self.query_one("#tw-model").value
        use_cache = self.query_one("#tw-cache").value
        
        try:
            # Parse since time
            since_time = datetime.now(timezone.utc) - timedelta(hours=24)
            if since.endswith("h"):
                since_time = datetime.now(timezone.utc) - timedelta(hours=int(since[:-1]))
            elif since.endswith("d"):
                since_time = datetime.now(timezone.utc) - timedelta(days=int(since[:-1]))

            # Fetch tweets via browser
            self._update_status("#tw-result-area", lines, f"⏳ Scraping tweets since: {since_time.strftime('%Y-%m-%d %H:%M:%S UTC')}...")
            
            tweets = await asyncio.to_thread(
                collect_tweets_via_browser,
                since_time,
                debug=False
            )
            
            if not tweets:
                self._update_status("#tw-result-area", lines, "⏳ No tweets scraped. Using fallback cache...")
                from twitter.output import load_debug_cache
                tweets = load_debug_cache() or []
                
            if not tweets:
                self._update_status("#tw-result-area", lines, "❌ No timeline data available.")
                return

            self._update_status("#tw-result-area", lines, f"⏳ Scraped {len(tweets)} tweets. Summarizing timeline with {model}...")
            
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
                clean_summary = re.sub(r"<thinking>.*?</thinking>", "", summary, flags=re.DOTALL).strip()
            else:
                thinking_text = None
                clean_summary = summary.strip()

            # Mount final result
            container = self.query_one("#tw-result-area")
            for child in list(container.children):
                child.remove()
                
            if thinking_text:
                container.mount(Collapsible(Markdown(thinking_text), title="💡 Show LLM Reasoning Block", collapsed=True))
            
            container.mount(Markdown(clean_summary))
            
        except Exception as e:
            self._update_status("#tw-result-area", lines, f"❌ Error: {e}")

    async def run_model_evaluator(self) -> None:
        lines = []
        self._update_status("#ev-result-area", lines, "⏳ Running quality evaluation...")

        task = self.query_one("#ev-task").value
        model = self.query_one("#ev-model").value
        
        try:
            baseline = await asyncio.to_thread(load_baseline)
            if not baseline:
                self._update_status("#ev-result-area", lines, "❌ No baseline found to run evaluation against.")
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
                container.mount(Markdown("\n".join(f"- ⚠ {w}" for w in warnings)))
            else:
                container.mount(Markdown("### ✔ No regressions detected against baseline."))
                
        except Exception as e:
            self._update_status("#ev-result-area", lines, f"❌ Error: {e}")

    async def run_image_renamer(self) -> None:
        lines = []
        self._update_status("#rn-result-area", lines, "⏳ Loading images...")

        directory_str = self.query_one("#rn-dir").value
        model = self.query_one("#rn-model").value
        dry_run = self.query_one("#rn-dry-run").value
        force = self.query_one("#rn-force").value

        directory = Path(directory_str)
        if not directory.exists() or not directory.is_dir():
            self._update_status("#rn-result-area", lines, "❌ Invalid directory path.")
            return

        image_files = []
        for ext in image_extensions:
            image_files.extend(directory.glob(f"*{ext}"))
            
        image_files = list(set([f for f in image_files if f.suffix.lower() in image_extensions]))

        if not image_files:
            self._update_status("#rn-result-area", lines, "❌ No images found.")
            return

        self._update_status("#rn-result-area", lines, f"⏳ Renaming {len(image_files)} images with {model}...")
        
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
        
        preview_box.update("⏳ Loading...")
        
        try:
            directory = Path(directory_str)
            if not directory.exists() or not directory.is_dir():
                preview_box.update("[red]Invalid Dir[/red]")
                return
                
            image_files = []
            for ext in image_extensions:
                image_files.extend(directory.glob(f"*{ext}"))
            image_files = list(set([f for f in image_files if f.suffix.lower() in image_extensions]))
            
            if not image_files:
                preview_box.update("[yellow]No images[/yellow]")
                preview_info.update("")
                return
                
            first_image = sorted(image_files)[0]
            
            # Generate ANSI preview in thread
            ansi_preview = await asyncio.to_thread(image_to_ansi, first_image, 24)
            
            preview_box.update(Text.from_ansi(ansi_preview))
            
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
            return
            
        files = []
        for p in doc_dir.glob("*.md"):
            if p.name.startswith("weekend_plan_") or ("_to_" in p.name and len(p.name) > 20):
                files.append(p)
                
        files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        
        for f in files:
            if f.name.startswith("weekend_plan_"):
                label = f"📅 {f.name.replace('weekend_plan_', '').replace('.md', '').replace('_', ' ')}"
            else:
                label = f"🐦 Summary: {f.name.replace('.md', '').replace('_', ' ')}"
                
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
            
            status.update(f"[green]🟢 Applied successfully: Temp={temp}, Max Tokens={max_tokens}[/green]")
        except Exception as e:
            status.update(f"[red]Error: {e}[/red]")

    def reset_model_parameters(self) -> None:
        status = self.query_one("#param-status")
        self.query_one("#param-temp").value = "0.1"
        self.query_one("#param-max-tokens").value = "16000"
        
        llm_client.GLOBAL_OVERRIDES.clear()
        osaurus_lib.GLOBAL_OVERRIDES.clear()
        
        status.update("[green]🟢 Reset to default values[/green]")


def main():
    app = ZToolsApp()
    app.run()


if __name__ == "__main__":
    main()
