"""Model evaluator."""

from eval.cli import main

from eval.tasks_core import (
    TASKS, _extract_items_from_text,
    WEEKEND_SYS_TRANSIENT, WEEKEND_SYS_FIXED,
    WEEKEND_USR_TRANSIENT, WEEKEND_USR_FIXED,
    RENAME_PROMPT, FILE_SUMMARY_PROMPT, TWITTER_PROMPT,
)
from eval.validate import safe_content, validate_file_summary
from eval.failures import (
    FAIL_INFRA, FAIL_TIMEOUT, FAIL_PARSE, FAIL_FORMAT,
    FAIL_CONTENT, FAIL_NONE,
    _classify_failure, _describe_content_failure,
)
from eval.run import (
    MAX_RETRIES, DEFAULT_EVAL_TIMEOUT, MEMORY_WARNING_THRESHOLD,
    _validate_result, _call_model,
    _quality_results_to_eval_format, run_eval,
)
from eval.report import (
    print_cross_model_comparison, compute_score_stats,
    print_score_stats, categorize_failures,
    print_failure_summary, save_historical_results,
    load_historical_stats, check_model_history,
    print_historical_trends, compute_token_estimates,
    compute_verbosity, print_verbosity,
    compute_error_rates, print_error_rates,
    compute_task_winners, diff_from_last_run, print_diff,
    export_to_csv,
)
