"""Backward-compat shim — routes to eval package.

Kill criterion: this file + README.md + data/ is all that remains
of the old eval_tasks/ package.
"""

from eval.tasks_core import TASKS as _TASKS

# Copy, not alias (fix shared mutable reference aliasing)
TASKS = dict(_TASKS)
TASKS["json"] = dict(_TASKS["weekend_transient"])
TASKS["detailed_json"] = dict(_TASKS["weekend_fixed"])


def load_tasks_from_config(model: str):
    """Backward compat — delegates to eval.cli."""
    from eval.cli import load_tasks_from_config as _load

    return _load(model)


def get_tasks(model: str = None):
    """Backward compat — delegates to eval.cli."""
    return load_tasks_from_config(model or "default")
