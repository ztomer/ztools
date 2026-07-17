import asyncio

from textual.app import App, ComposeResult
from textual.widgets import (
    Button,
    ContentSwitcher,
    Footer,
    Header,
    Label,
    ListView,
)

from tui.app_actions import (
    action_refresh_models,
    add_scheduler_task,
    apply_model_parameters,
    load_history_item,
    load_image_preview,
    on_mount,
    refresh_history_archive,
    refresh_scheduler_display,
    remove_scheduler_task,
    reset_model_parameters,
    run_image_renamer,
    run_model_evaluator,
    run_scheduled_task,
    run_twitter_summarizer,
    run_weekend_planner,
    scheduler_loop,
)
from tui.app_views import (
    HistoryArchiveView,
    ImageRenamerView,
    ModelEvaluatorView,
    ModelParametersView,
    Sidebar,
    TaskSchedulerView,
    TwitterSummarizerView,
    WeekendPlannerView,
)

APP_CSS = """
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
    align: center middle;
}
.form-label {
    width: 18;
    color: #94a3b8;
}
Input {
    width: 40;
    background: #1e293b;
    color: #e2e8f0;
    border: solid #334155;
}
Select {
    width: 40;
    background: #1e293b;
    color: #e2e8f0;
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
    color: #e2e8f0;
    padding: 1;
}
.result-area Markdown {
    background: transparent;
    color: #e2e8f0;
    overflow-x: auto;
    width: 100%;
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
    background: #1e293b;
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
    text-align: center;
}
#hist-layout {
    height: 1fr;
}
#hist-list-panel {
    width: 35;
    background: #1e293b;
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
#param-hint {
    margin-top: 1;
    padding-left: 2;
}
#sched-layout {
    height: 1fr;
}
#sched-form-panel {
    width: 1fr;
    padding: 1;
}
#sched-form-status {
    margin-top: 1;
    padding-left: 2;
}
#sched-list-panel {
    width: 50;
    background: #1e293b;
    border-left: solid #334155;
    padding: 1;
}
#sched-list {
    background: transparent;
    height: 1fr;
    margin-top: 1;
}
.sched-card {
    height: auto;
    background: #090d16;
    border: solid #334155;
    margin-bottom: 1;
    padding: 1;
    align: center middle;
}
.sched-card-info {
    width: 1fr;
}
.sched-card-btn {
    width: 12;
}
"""


class ZToolsApp(App):
    """Unified ZTools interactive terminal dashboard."""

    CSS = APP_CSS

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
            yield TaskSchedulerView(id="scheduler")
        yield Footer()

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        switcher = self.query_one(ContentSwitcher)
        if event.item.id == "nav-history":
            switcher.current = "history"
            asyncio.create_task(self.refresh_history_archive())
        elif event.item.id:
            view_id = event.item.id.replace("nav-", "")
            switcher.current = view_id

    def on_button_pressed(self, event: Button.Pressed) -> None:
        button_id = event.button.id
        handlers = {
            "btn-wk-generate": self.run_weekend_planner,
            "btn-tw-generate": self.run_twitter_summarizer,
            "btn-ev-generate": self.run_model_evaluator,
            "btn-rn-generate": self.run_image_renamer,
            "btn-rn-preview": self.load_image_preview,
            "btn-hist-refresh": self.refresh_history_archive,
            "btn-param-apply": self.apply_model_parameters,
            "btn-param-reset": self.reset_model_parameters,
            "btn-sched-add": self.add_scheduler_task,
        }
        if button_id in handlers:
            handler = handlers[button_id]
            if asyncio.iscoroutinefunction(handler):
                asyncio.create_task(handler())
            else:
                handler()
        elif button_id and button_id.startswith("btn-sched-del-"):
            task_id = button_id.replace("btn-sched-del-", "")
            self.remove_scheduler_task(task_id)

    def _update_status(self, container_id: str, message: str) -> None:
        container = self.query_one(container_id)
        for child in list(container.children):
            child.remove()
        container.mount(Label(message))


for cls in [ZToolsApp]:
    cls.on_mount = on_mount
    cls.run_weekend_planner = run_weekend_planner
    cls.run_twitter_summarizer = run_twitter_summarizer
    cls.run_model_evaluator = run_model_evaluator
    cls.run_image_renamer = run_image_renamer
    cls.load_image_preview = load_image_preview
    cls.refresh_history_archive = refresh_history_archive
    cls.load_history_item = load_history_item
    cls.apply_model_parameters = apply_model_parameters
    cls.reset_model_parameters = reset_model_parameters
    cls.scheduler_loop = scheduler_loop
    cls.run_scheduled_task = run_scheduled_task
    cls.refresh_scheduler_display = refresh_scheduler_display
    cls.add_scheduler_task = add_scheduler_task
    cls.remove_scheduler_task = remove_scheduler_task
    cls.action_refresh_models = action_refresh_models


def main():
    app = ZToolsApp()
    app.run()


if __name__ == "__main__":
    main()
