#!/usr/bin/env python3
"""Reporting and analysis for model evaluation.

Shim: split across report_core (tables, comparison, stats), report_history
(persistence and trends) and report_metrics (tokens, verbosity, errors,
winners, diffs, CSV) to stay under the 500-line limit. Every name they define
is re-exported here, so existing imports are unaffected.
"""

from eval.report_core import (  # noqa: F401
    _get_eval_dir,
    _make_table,
    categorize_failures,
    compute_score_stats,
    print_cross_model_comparison,
    print_failure_summary,
    print_score_stats,
)
from eval.report_history import (  # noqa: F401
    check_model_history,
    load_historical_stats,
    print_historical_trends,
    save_historical_results,
)
from eval.report_metrics import (  # noqa: F401
    compute_error_rates,
    compute_task_winners,
    compute_token_estimates,
    compute_verbosity,
    diff_from_last_run,
    export_to_csv,
    print_diff,
    print_error_rates,
    print_verbosity,
)

__all__ = [
    "print_cross_model_comparison",
    "compute_score_stats",
    "print_score_stats",
    "categorize_failures",
    "print_failure_summary",
    "save_historical_results",
    "load_historical_stats",
    "check_model_history",
    "print_historical_trends",
    "compute_token_estimates",
    "compute_verbosity",
    "print_verbosity",
    "compute_error_rates",
    "print_error_rates",
    "compute_task_winners",
    "diff_from_last_run",
    "print_diff",
    "export_to_csv",
]
