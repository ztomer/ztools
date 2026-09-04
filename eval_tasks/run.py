"""Backward-compat shim — routes to eval.run."""

from eval.run import (
    run_eval,
)
from eval.tasks_core import TASKS
from lib.osaurus_lib import is_server_running
from lib.tui import console


def run_model_eval(
    model: str,
    tasks: dict = None,
    host: str = "localhost",
    port: int = 1337,
    backend: str = "osaurus",
) -> list:
    """Run evaluation on a model. Returns list of result dicts."""
    if tasks is None:
        tasks = TASKS
    if not is_server_running():
        console.print("[yellow]Warning: Osaurus server not running[/yellow]")
        return []
    return run_eval(model, tasks=tasks, host=host, port=port, backend=backend)


def main():
    """Legacy CLI entry point — delegates to eval.cli."""
    from eval.cli import main as eval_main

    eval_main()


if __name__ == "__main__":
    main()
