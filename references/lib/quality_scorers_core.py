"""Scorer registry and dimension weights.

Split out of quality_scorers.py, which had grown past the repo's 500-line
limit. The per-task scorer modules register themselves against this registry;
quality_scorers.py imports them and exposes score_output.
"""

from typing import Callable, Dict, List

from lib.quality_models import TestCase

_scorer_registry: Dict[str, List[Callable]] = {}


def register_scorer(*tasks: str):
    """Decorator that registers a scorer function for one or more task types."""

    def decorator(func):
        for task in tasks:
            _scorer_registry.setdefault(task, []).append(func)
        return func

    return decorator


def get_scorers(task: str) -> List[Callable]:
    return list(_scorer_registry.get(task, []))


def get_dimension_weights(task: str) -> dict[str, float]:
    """Return {dimension_name: weight} for a task from the registered scorers."""
    weights: dict[str, float] = {}
    for scorer in get_scorers(task):
        dummy = scorer("", TestCase(task, "", "", ""))
        weights[dummy.name] = dummy.weight
    return weights


