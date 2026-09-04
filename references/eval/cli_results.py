#!/usr/bin/env python3
"""Final results rendering for the evaluator CLI.

Split out of cli.py for the repo's 500-line limit.
"""

import json

from lib.tui import STEP, WARN

from eval import cli_runtime
from eval.discrimination import disagreements
from eval.report import (
    categorize_failures,
    compute_error_rates,
    compute_score_stats,
    compute_task_winners,
    compute_verbosity,
    diff_from_last_run,
    export_to_csv,
    print_cross_model_comparison,
    print_diff,
    print_error_rates,
    print_failure_summary,
    print_historical_trends,
    print_score_stats,
    print_verbosity,
    save_historical_results,
)
from eval.report_core import default_eval_dir


def _print_results(all_results, best_scores, best_models_dict, eval_dir=None):
    eval_dir = eval_dir or default_eval_dir()
    cli_runtime.console.print("\nBest Models per Task:")
    for task, model in best_models_dict.items():
        cli_runtime.console.print(f"  {task}: {model} ({best_scores[task]}%)")

    print_cross_model_comparison(all_results)

    stats = compute_score_stats(all_results)
    print_score_stats(stats)

    # The gate list is a record of a measurement, so every run re-checks it
    # against its own data. A hand-typed list nothing verifies is how
    # `conf/config.toml` came to name models that had been deleted from disk.
    for note in disagreements(all_results):
        cli_runtime.console.print(f"{WARN} task classification: {note}")

    categories = categorize_failures(all_results)
    print_failure_summary(categories)

    verbosity = compute_verbosity(all_results)
    print_verbosity(verbosity)

    error_rates = compute_error_rates(all_results)
    print_error_rates(error_rates)

    diffs = diff_from_last_run(all_results)
    print_diff(diffs)

    winners = compute_task_winners(all_results)
    cli_runtime.console.print("")
    cli_runtime.console.print("Task Winners:")
    for task, (model, score) in sorted(winners.items()):
        cli_runtime.console.print(f"  {task}: {model} ({score}%)")

    save_historical_results(all_results, stats, categories)
    print_historical_trends()

    export_to_csv(all_results)

    # Resolved through the module so eval.report_core._get_eval_dir stays the
    # single patch point for the whole report surface.

    eval_dir = eval_dir
    eval_dir.mkdir(parents=True, exist_ok=True)
    results_file = eval_dir / "eval_results.json"

    with open(results_file, "w") as f:
        json.dump(
            {
                "models": all_results,
                "best_scores": best_scores,
                "best_models": best_models_dict,
            },
            f,
            indent=2,
        )
    cli_runtime.console.print(f"{STEP} Saved to {results_file}")
