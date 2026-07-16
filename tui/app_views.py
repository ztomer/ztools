import os
from pathlib import Path

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import (
    Button,
    Checkbox,
    Input,
    Label,
    ListItem,
    ListView,
    Select,
    Static,
)

from rename.cli import image_extensions

_DEFAULT_MODEL = "foundation"
try:
    from lib.config import get_config
    _cfg = get_config()
    _DEFAULT_MODEL = _cfg.get("default_model", _DEFAULT_MODEL)
except Exception:
    pass

_KNOWN_MODELS = ["foundation", "qwen", "gemma"]
if _DEFAULT_MODEL not in _KNOWN_MODELS:
    _KNOWN_MODELS.insert(0, _DEFAULT_MODEL)


def _tool_default_model(config_key: str) -> str:
    try:
        from lib.config import get_config
        _best = get_config().get("best_models", {})
        return _best.get(config_key, _DEFAULT_MODEL)
    except Exception:
        return _DEFAULT_MODEL


for _k in ("json", "summarize", "think", "filename"):
    _m = _tool_default_model(_k)
    if _m not in _KNOWN_MODELS:
        _KNOWN_MODELS.insert(0, _m)
DEFAULT_MODELS = [(m, m) for m in _KNOWN_MODELS]


def _collect_images(directory: Path) -> list[Path]:
    image_files = []
    for ext in image_extensions:
        image_files.extend(directory.glob(f"*{ext}"))
class Sidebar(Vertical):
    """Left sidebar navigation panel."""

    def compose(self) -> ComposeResult:
        yield Label("[bold cyan]ZTools Hub[/bold cyan]", id="app-title")
        yield ListView(
            ListItem(Label("Weekend Planner"), id="nav-weekend"),
            ListItem(Label("Twitter Summarizer"), id="nav-twitter"),
            ListItem(Label("Model Evaluator"), id="nav-eval"),
            ListItem(Label("Image Renamer"), id="nav-rename"),
            ListItem(Label("History Archive"), id="nav-history"),
            ListItem(Label("Model Parameters"), id="nav-params"),
            ListItem(Label("Task Scheduler"), id="nav-scheduler"),
            id="nav-list"
        )
        yield Static(id="server-status-box")


class WeekendPlannerView(Vertical):
    """TUI Form for generating weekend plans."""
    CONFIG_KEY = "json"

    def compose(self) -> ComposeResult:
        yield Label("[bold]Weekend Planner[/bold]", classes="section-title")

        with Horizontal(classes="form-row"):
            yield Label("Location:", classes="form-label")
            yield Input(
                placeholder="City/Region (e.g. Toronto/ON)",
                value="Toronto/ON",
                id="wk-location"
            )

        with Horizontal(classes="form-row"):
            yield Label("Model:", classes="form-label")
            yield Select(options=DEFAULT_MODELS, value=_tool_default_model("json"),
                         id="wk-model", prompt="Select LLM Model")

        with Horizontal(classes="form-row"):
            yield Checkbox("Use Cached Data", value=True, id="wk-cache")
            yield Checkbox("Use On-Device Model (Apple FM)", value=False, id="wk-foundation")

        yield Button("Generate Weekend Plan", variant="success", id="btn-wk-generate")
        yield VerticalScroll(id="wk-result-area", classes="result-area")


class TwitterSummarizerView(Vertical):
    """TUI Form for summarizing Twitter timeline."""
    CONFIG_KEY = "summarize"

    def compose(self) -> ComposeResult:
        msg = "[bold]Twitter Timeline Summarizer[/bold]"
        yield Label(msg, classes="section-title")

        with Horizontal(classes="form-row"):
            yield Label("Since (Relative):", classes="form-label")
            yield Input(placeholder="e.g. 24h, 48h, 7d", value="24h", id="tw-since")

        with Horizontal(classes="form-row"):
            yield Label("Model:", classes="form-label")
            yield Select(options=DEFAULT_MODELS, value=_tool_default_model("summarize"),
                         id="tw-model", prompt="Select LLM Model")

        with Horizontal(classes="form-row"):
            yield Checkbox("Use Cache", value=False, id="tw-cache")

        yield Button("Summarize Timeline", variant="primary", id="btn-tw-generate")
        yield VerticalScroll(id="tw-result-area", classes="result-area")


class ModelEvaluatorView(Vertical):
    """TUI Form for running quality evaluations."""
    CONFIG_KEY = "think"

    def compose(self) -> ComposeResult:
        msg = "[bold]Model Quality Evaluator[/bold]"
        yield Label(msg, classes="section-title")

        with Horizontal(classes="form-row"):
            yield Label("Task:", classes="form-label")
            yield Select(
                options=[
                    ("filename", "filename"),
                    ("summarize", "summarize"),
                    ("file_summary", "file_summary")
                ],
                id="ev-task",
                value="filename",
                prompt="Select Task"
            )

        with Horizontal(classes="form-row"):
            yield Label("Model:", classes="form-label")
            yield Select(options=DEFAULT_MODELS, value=_tool_default_model("think"),
                         id="ev-model", prompt="Select LLM Model")

        yield Button("Run Evaluation", variant="warning", id="btn-ev-generate")
        yield VerticalScroll(id="ev-result-area", classes="result-area")


class ImageRenamerView(Vertical):
    """TUI Form for renaming screenshot files."""
    CONFIG_KEY = "filename"

    def compose(self) -> ComposeResult:
        msg = "[bold]Screenshot Image Renamer[/bold]"
        yield Label(msg, classes="section-title")

        with Horizontal(id="rn-layout"):
            with Vertical(id="rn-form-panel"):
                with Horizontal(classes="form-row"):
                    yield Label("Directory Path:", classes="form-label")
                    yield Input(
                        placeholder="Absolute path to folder",
                        value=os.getcwd(),
                        id="rn-dir"
                    )

                with Horizontal(classes="form-row"):
                    yield Label("Model:", classes="form-label")
                    yield Select(options=DEFAULT_MODELS, value=_tool_default_model("filename"),
                                 id="rn-model", prompt="Select LLM Model")

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
        yield Label("[bold]History Archive[/bold]", classes="section-title")
        with Horizontal(id="hist-layout"):
            with Vertical(id="hist-list-panel"):
                yield Button("Refresh History", variant="primary", id="btn-hist-refresh")
                yield ListView(id="hist-list")
            with Vertical(id="hist-viewer-panel"):
                yield VerticalScroll(id="hist-viewer")


class ModelParametersView(Vertical):
    """TUI view for customizing global model generation parameters."""

    def compose(self) -> ComposeResult:
        yield Label("[bold]Model Parameters[/bold]", classes="section-title")
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
            yield Label("[yellow]Session only — lost on quit[/yellow]", id="param-hint")


class TaskSchedulerView(Vertical):
    """TUI view for defining and monitoring background automation tasks."""

    def compose(self) -> ComposeResult:
        yield Label("[bold]Task Scheduler[/bold]", classes="section-title")
        with Horizontal(id="sched-layout"):
            with Vertical(id="sched-form-panel"):
                yield Label("[bold white]Create New Task Schedule[/bold white]")
                with Horizontal(classes="form-row"):
                    yield Label("Task Type:", classes="form-label")
                    yield Select(
                        options=[
                            ("Twitter Summarizer", "twitter"),
                            ("Screenshot Renamer", "rename"),
                            ("Weekend Planner", "weekend")
                        ],
                        id="sched-task-type",
                        prompt="Select Task"
                    )
                with Horizontal(classes="form-row"):
                    yield Label("Interval:", classes="form-label")
                    yield Select(
                        options=[
                            ("Every 10 seconds", "10"),
                            ("Every 1 minute", "60"),
                            ("Every 1 hour", "3600"),
                            ("Every 6 hours", "21600"),
                            ("Every 12 hours", "43200"),
                            ("Every 24 hours", "86400")
                        ],
                        id="sched-interval",
                        prompt="Select Interval"
                    )
                yield Button("Schedule Task", variant="success", id="btn-sched-add")
                yield Label("", id="sched-form-status")
            with Vertical(id="sched-list-panel"):
                yield Label("[bold white]Active Background Tasks[/bold white]")
                yield VerticalScroll(id="sched-list")


