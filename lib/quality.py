"""
Dimension-based quality scoring for model evaluation.

Each task has 3-5 named quality dimensions, each scored 0-100 independently.
Composite score = weighted average of dimensions.
Scorers are calibrated against human-judged reference outputs.
"""

from lib.quality_models import Score, ScoreCard, TestCase, _str, _lower  # noqa: F401
from lib.quality_scorers import (  # noqa: F401
    TASK_SCORERS,
    score_output,
    _score_filename_relevance,
    _score_filename_format,
    _score_filename_conciseness,
    _score_summarize_completeness,
    _score_summarize_synthesis,
    _score_summarize_structure,
    _score_summarize_specificity,
    _score_file_completeness,
    _score_file_accuracy,
    _score_file_format,
)
from lib.quality_runner import (  # noqa: F401
    FILENAME_CASES,
    SUMMARIZE_CASES,
    FILE_SUMMARY_CASES,
    ALL_TEST_CASES,
    query_model,
    run_suite,
)
from lib.quality_report import (  # noqa: F401
    BASELINE_PATH,
    generate_report,
    save_baseline,
    load_baseline,
    compare_to_baseline,
)
from lib.quality_entry import main  # noqa: F401

__all__ = [
    "Score", "ScoreCard", "TestCase",
    "_str", "_lower",
    "TASK_SCORERS", "score_output",
    "_score_filename_relevance", "_score_filename_format", "_score_filename_conciseness",
    "_score_summarize_completeness", "_score_summarize_synthesis",
    "_score_summarize_structure", "_score_summarize_specificity",
    "_score_file_completeness", "_score_file_accuracy", "_score_file_format",
    "FILENAME_CASES", "SUMMARIZE_CASES", "FILE_SUMMARY_CASES", "ALL_TEST_CASES",
    "query_model", "run_suite",
    "BASELINE_PATH", "generate_report", "save_baseline", "load_baseline",
    "compare_to_baseline",
    "main",
]
