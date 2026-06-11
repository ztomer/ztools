"""Backward-compat shim — routes to eval.report for all analysis functions."""

from eval.report import (
    print_cross_model_comparison,
    compute_score_stats,
    print_score_stats,
    categorize_failures,
    print_failure_summary,
    save_historical_results,
    load_historical_stats,
    check_model_history,
    print_historical_trends,
    compute_token_estimates,
    compute_verbosity,
    print_verbosity,
    compute_error_rates,
    print_error_rates,
    compute_task_winners,
    diff_from_last_run,
    print_diff,
    export_to_csv,
)

__all__ = [
    "print_cross_model_comparison", "compute_score_stats", "print_score_stats",
    "categorize_failures", "print_failure_summary",
    "save_historical_results", "load_historical_stats", "check_model_history",
    "print_historical_trends", "compute_token_estimates",
    "compute_verbosity", "print_verbosity",
    "compute_error_rates", "print_error_rates",
    "compute_task_winners", "diff_from_last_run", "print_diff", "export_to_csv",
]
