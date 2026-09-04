"""Model evaluator — public API surface."""

from eval.cli import main
from eval.run import run_eval
from eval.tasks_core import TASKS

__all__ = ["main", "run_eval", "TASKS"]
