"""Quality scoring entry point.

Shim: the registry lives in quality_scorers_core, the per-task scorers in
scorers_filename / scorers_summarize / scorers_file. Importing this module
registers all of them, so `from lib.quality_scorers import score_output`
behaves exactly as before the split.
"""

from typing import Callable, Dict, List

from lib.quality_models import GENERIC_FILENAMES, Score, ScoreCard, TestCase, _lower
from lib.quality_scorers_core import (
    _scorer_registry,
    get_dimension_weights,
    get_scorers,
    register_scorer,
)
from lib.scorers_file import (  # noqa: F401  (re-export; import registers the scorers)
    _score_file_accuracy,
    _score_file_completeness,
    _score_file_format,
)

# Imported for their registration side effects, and re-exported so the private
# scorers stay importable from this module as they were before the split.
from lib.scorers_filename import (  # noqa: F401  (re-export; import registers the scorers)
    _score_filename_conciseness,
    _score_filename_format,
    _score_filename_relevance,
)
from lib.scorers_summarize import (  # noqa: F401  (re-export; import registers the scorers)
    _score_summarize_completeness,
    _score_summarize_specificity,
    _score_summarize_structure,
    _score_summarize_synthesis,
)

# Registry snapshot — every scorer module above has registered by this point.
TASK_SCORERS: Dict[str, List[Callable]] = _scorer_registry

__all__ = [
    "TASK_SCORERS",
    "register_scorer",
    "get_scorers",
    "get_dimension_weights",
    "score_output",
]


def score_output(output: str, task: str, case: TestCase) -> ScoreCard:
    out = output.strip()

    if not out:
        return ScoreCard(
            model="",
            task=task,
            case_id=case.description,
            dimensions=[],
            output=output,
        )

    if task == "filename":
        if _lower(out) in GENERIC_FILENAMES:
            return ScoreCard(
                model="",
                task=task,
                case_id=case.description,
                dimensions=[
                    Score("Relevance", 0, 0.40, failures=["generic"]),
                    Score("Format", 0, 0.35, failures=["generic"]),
                    Score("Conciseness", 0, 0.25, failures=["generic"]),
                ],
                output=output,
            )

    scorers = get_scorers(task)
    dimensions = [scorer(output, case) for scorer in scorers]
    return ScoreCard(
        model="",
        task=task,
        case_id=case.description,
        dimensions=dimensions,
        output=output,
    )
