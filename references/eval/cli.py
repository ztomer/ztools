#!/usr/bin/env python3
"""
Model Evaluator - Test models on REAL-WORLD tasks from ZTools.

Evaluates local models against the actual prompts used in the tools.

Split into modules:
  eval_tasks.py     - task definitions and prompts
  eval_validate.py  - validation functions
  eval_failures.py  - failure diagnosis
  eval_run.py       - main eval loop
  eval_report.py    - reporting and analysis
"""

import argparse
import importlib
import os
import sys

from lib import init_config
from lib.config import build_tasks_from_model
from lib.llm.constants import DEFAULT_HOST, DEFAULT_PORT
from lib.osaurus_lib import (
    get_models,
    is_server_running,
)
from lib.signal_handling import setup_signals
from lib.tui import FAIL, STEP, WARN

from eval.failures import (
    FAIL_CONTENT,
    FAIL_FORMAT,
    FAIL_INFRA,
    FAIL_NONE,
    FAIL_PARSE,
    FAIL_TIMEOUT,
    _classify_failure,
    _describe_content_failure,
)
from eval.report import (
    categorize_failures,
    check_model_history,
    compute_error_rates,
    compute_score_stats,
    compute_task_winners,
    compute_token_estimates,
    compute_verbosity,
    diff_from_last_run,
    export_to_csv,
    load_historical_stats,
    print_cross_model_comparison,
    print_diff,
    print_error_rates,
    print_failure_summary,
    print_historical_trends,
    print_score_stats,
    print_verbosity,
    save_historical_results,
)
from eval.run import (
    DEFAULT_EVAL_TIMEOUT,
    MAX_RETRIES,
    MEMORY_WARNING_THRESHOLD,
    _call_model,
    _quality_results_to_eval_format,
    _validate_result,
    run_eval,
    run_eval_quick,
)
from eval.tasks_core import (
    FILE_SUMMARY_PROMPT,
    RENAME_PROMPT,
    TASKS,
    TWITTER_PROMPT,
    WEEKEND_SYS_FIXED,
    WEEKEND_SYS_TRANSIENT,
    WEEKEND_USR_FIXED,
    WEEKEND_USR_TRANSIENT,
    _extract_items_from_text,
)
from eval.validate import safe_content, validate_file_summary

__all__ = [
    "TASKS",
    "MAX_RETRIES",
    "DEFAULT_EVAL_TIMEOUT",
    "MEMORY_WARNING_THRESHOLD",
    "safe_content",
    "validate_file_summary",
    "FAIL_INFRA",
    "FAIL_TIMEOUT",
    "FAIL_PARSE",
    "FAIL_FORMAT",
    "FAIL_CONTENT",
    "FAIL_NONE",
    "run_eval",
    "_validate_result",
    "_call_model",
    "_quality_results_to_eval_format",
    "_extract_items_from_text",
    "_classify_failure",
    "_describe_content_failure",
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
    "get_memory_percent",
    "check_memory_safe",
    "is_server_responsive",
    "print_memory_usage",
    "estimate_model_memory",
    "load_tasks_from_config",
    "update_config",
    "main",
    "WEEKEND_SYS_TRANSIENT",
    "WEEKEND_SYS_FIXED",
    "WEEKEND_USR_TRANSIENT",
    "WEEKEND_USR_FIXED",
    "RENAME_PROMPT",
    "FILE_SUMMARY_PROMPT",
    "TWITTER_PROMPT",
]


# Re-exported so `from eval.cli import ...` is unchanged after the split.
from eval import cli_runtime
from eval.cli_results import _print_results  # noqa: E402,F401
from eval.cli_runtime import (  # noqa: E402,F401
    check_memory_safe,
    estimate_model_memory,
    flush_between_models,
    get_memory_percent,
    hold_gpu_for_eval,
    is_server_responsive,
    load_tasks_from_config,
    print_memory_usage,
    update_config,
)


def _print_capabilities(host=None):
    """Print the probed capability table for every model the server lists.

    Budget and timeout come from conf/config.toml's `think` slot, addressed as a
    plain string because `think` is a config key with no member in the Task enum.
    It is the heaviest configured budget (16000 tokens in the tightest timeout), so
    a model that clears it clears every other task.
    """
    from lib.config import get_timeout
    from lib.model_resolve import audit_configured_models, fetch_roster, format_audit

    from eval.capabilities import capability_report, format_capability_table
    from eval.signals import _load_eval_signals

    roster = fetch_roster(host) if host else fetch_roster()
    if not roster:
        cli_runtime.console.print(f"{FAIL} Osaurus server not reachable — nothing to probe")
        return
    models = [entry["model"] for entry in roster]
    rows = capability_report(models, roster, _load_eval_signals())
    from eval.capabilities import expected_output_tokens

    table = format_capability_table(rows, expected_output_tokens(), get_timeout("think"))
    for line in table:
        cli_runtime.console.print(line, highlight=False)

    for line in format_audit(audit_configured_models(installed=models)):
        cli_runtime.console.print(f"{WARN} {line}" if not line.startswith(" ") else line)


def main():
    """Main entry point for model evaluation."""
    setup_signals()
    init_config()
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="Run evaluation for a specific model")
    parser.add_argument(
        "--task",
        help="Run a specific task (weekend_transient, weekend_fixed, "
        "filename, summarize, file_summary)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode: run single task with one retry (faster iteration)",
    )
    parser.add_argument(
        "--config-tasks",
        action="store_true",
        help="Load tasks from YAML config instead of hardcoded prompts",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging to console")
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Show raw model output for debugging quality"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_EVAL_TIMEOUT,
        help=f"Per-task timeout in seconds (default {DEFAULT_EVAL_TIMEOUT})",
    )
    parser.add_argument(
        "--quality",
        action="store_true",
        help="Use quality.py dimension-based scoring instead of old validators",
    )
    parser.add_argument(
        "--host",
        default=None,
        help=f"Osaurus/Ollama server URL (default: $OLLAMA_BASE_URL or http://{DEFAULT_HOST}:{DEFAULT_PORT})",
    )
    parser.add_argument("--api-key", default=None, help="Bearer token for the LLM API")
    parser.add_argument(
        "--capabilities",
        action="store_true",
        help="Probe what each installed model IS (family, vision, size, viability) "
        "and print it, without running any task. Replaces the hand-maintained "
        "roster table in docs/MODEL_QUIRKS.md.",
    )
    args = parser.parse_args()

    if args.capabilities:
        _print_capabilities(args.host)
        return
    timeout = args.timeout

    if args.host:
        os.environ["OLLAMA_BASE_URL"] = args.host
    if args.api_key:
        os.environ["OLLAMA_API_KEY"] = args.api_key

    # Claim the GPU before anything reaches the server, and hold it for the whole
    # run. Below this line every number produced is only worth recording if no
    # other session restarts osaurus underneath it -- and on this machine several
    # agent sessions run at once. Taken AFTER --capabilities returns: that path
    # only reads recorded facts and a model list, so making a read-only report
    # queue behind a peer's measurement would be pure obstruction.
    #
    # Raises GpuBusy rather than degrading. A run that cannot get the GPU has
    # nothing useful to do: proceeding would produce a full set of results that
    # look ordinary and are quietly wrong, which is worse than not running.
    hold_gpu_for_eval(f"eval {args.model or 'all models'} (pid {os.getpid()})")

    models_to_test = []

    from lib.model_caps import is_generative_model
    from lib.model_resolve import audit_configured_models, format_audit

    from eval.capabilities import record_static_capabilities

    if is_server_running():
        osaurus_models = get_models()

        # Surface a stale conf/config.toml here rather than waiting for a tool to trip
        # over it: the roster changes on disk, the config does not follow, and until
        # this ran the only symptom was an HTTP 404 from whichever tool hit it first.
        # Audited against the list just fetched — a second request would be both
        # wasteful and a live connection in a path the test suite forbids one in.
        for line in format_audit(audit_configured_models(installed=osaurus_models)):
            cli_runtime.console.print(f"{WARN} {line}" if not line.startswith(" ") else line)

        # Probe once per run and persist, so every offline consumer -- family
        # routing, VLM selection, the memory estimate -- reads a recorded fact
        # instead of guessing from the model name. No roster is passed: every field
        # is derivable from disk, and a second HTTP request here is exactly what the
        # no-live-server gate exists to catch.
        for m in osaurus_models:
            record_static_capabilities(m)
            # Judged by the model's own config.json rather than by keywords in its
            # name. The name list ("model2vec", "potion", "embedding", ...) works
            # until the next embedding model arrives called something else.
            if not is_generative_model(m):
                cli_runtime.console.print(f"{WARN} Skipping {m} (not a generative model)")
                continue
            models_to_test.append((m, "osaurus"))
    else:
        cli_runtime.console.print(
            f"{WARN} Osaurus server not running — install/start it to evaluate local models:"
        )
        cli_runtime.console.print("  brew install --cask osaurus")
        # Not `osaurus serve &`, which this line used to advise. A hand-started
        # server takes no GPU lock and checks for no existing one, so following
        # that advice on a machine already running an eval produces the two-server
        # contention the whole lock exists to prevent -- and the sample guard,
        # which reads swap and compressor, records the result as CLEAN.
        cli_runtime.console.print("  ./tools/osaurus_one.sh   (never start one by hand)")

    if args.model:
        models_to_test = [(m, b) for m, b in models_to_test if m == args.model]

    if not models_to_test:
        cli_runtime.console.print(f"{FAIL} No models found")
        sys.exit(1)

    cli_runtime.console.print(f"{STEP} Found {len(models_to_test)} models to test")

    tasks_to_run = TASKS
    from lib.config import get_config

    _default_eval_model = get_config().get("default_model", "foundation")
    config_model = args.model if args.model else _default_eval_model

    if args.config_tasks:
        config_tasks = build_tasks_from_model(config_model)
        if not config_tasks:
            cli_runtime.console.print(
                f"{WARN} Config loading failed, falling back to hardcoded tasks"
            )
        else:
            if args.task:
                if args.task in config_tasks:
                    tasks_to_run = {args.task: config_tasks[args.task]}
                    cli_runtime.console.print(f"{STEP} Using config task: {args.task}")
                else:
                    cli_runtime.console.print(
                        f"{FAIL} Task '{args.task}' not in config. "
                        f"Available: {list(config_tasks.keys())}"
                    )
                    sys.exit(1)
            else:
                tasks_to_run = config_tasks
            cli_runtime.console.print(f"{STEP} Loaded {len(tasks_to_run)} tasks from config")
    else:
        if args.task:
            if args.task not in TASKS:
                cli_runtime.console.print(
                    f"{FAIL} Unknown task: {args.task}. "
                    f"Available: {list(TASKS.keys())}"
                )
                sys.exit(1)
            tasks_to_run = {args.task: TASKS[args.task]}
            cli_runtime.console.print(f"{WARN} Running only task: {args.task}")
        cli_runtime.console.print(f"{STEP} Loaded {len(tasks_to_run)} tasks from hardcoded TASKS")

    if args.quick:
        cli_runtime.console.print(f"{STEP} Quick mode: single run, no retries")

        def quick_run_eval(model, backend="osaurus", **kwargs):
            return run_eval_quick(
                model,
                tasks=kwargs.get("tasks"),
                backend=backend,
                verbose=kwargs.get("verbose", False),
                timeout=timeout,
                on_complete=print_memory_usage,
            )

        _run_eval = quick_run_eval
    else:
        def _run_eval(model, backend="osaurus", **kwargs):
            return run_eval(
                model,
                tasks=kwargs.get("tasks"),
                backend=backend,
                verbose=kwargs.get("verbose", False),
                timeout=timeout,
                on_complete=print_memory_usage,
            )

    all_results = []
    best_scores = {task: -1 for task in tasks_to_run.keys()}
    best_models_dict = {task: None for task in tasks_to_run.keys()}

    prev_model = None
    for i, (model, backend) in enumerate(models_to_test):
        # Visual separator between models
        if i > 0:
            cli_runtime.console.rule(f"[bold]{model}[/bold]", style="dim")
        else:
            cli_runtime.console.rule(f"[bold]{model}[/bold]", style="dim")

        if prev_model and model != prev_model:
            flush_between_models(prev_model, model)
        prev_model = model

        mem_pct = get_memory_percent()
        model_mem_gb = estimate_model_memory(model)
        _total_gb = (
            os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") / (1024**3)
            if hasattr(os, "sysconf")
            else 64
        )
        avail_mem_gb = (100 - mem_pct) / 100 * _total_gb

        if mem_pct > MEMORY_WARNING_THRESHOLD:
            cli_runtime.console.print(f"{WARN} Memory at {mem_pct}% - model may be slow")

        if model_mem_gb > avail_mem_gb * 0.8:
            cli_runtime.console.print(
                f"{WARN} Model needs ~{model_mem_gb}GB, low memory - will be slower"
            )

        if not is_server_responsive():
            cli_runtime.console.print(f"{FAIL} Server not responsive - attempting restart...")

        cli_runtime.console.print(f"{STEP} Memory: {mem_pct}%, Server: OK")

        if args.quality:
            qm = importlib.import_module("lib.quality")

            cases = qm.ALL_TEST_CASES
            if args.task:
                cases = [c for c in cases if c.task == args.task]
            scorecards = qm.run_suite([model], cases, verbose=True)
            results = _quality_results_to_eval_format(scorecards, model)
            task_avgs = {}
            for r in results:
                lst = task_avgs.setdefault(r["task"], [])
                lst.append(r["quality_score"])
            summary = "  ".join(f"{t}={int(round(sum(s) / len(s)))}%" for t, s in task_avgs.items())
            cli_runtime.console.print(f"{STEP} Quality scores: {summary}")
        else:
            results = _run_eval(model, tasks=tasks_to_run, backend=backend, verbose=args.verbose)

        scores = [r["quality_score"] for r in results]
        if not scores:
            cli_runtime.console.print(f"{STEP} {model} ({backend}): 0 tasks")
        else:
            avg = sum(scores) / len(scores)
            status = (
                STEP
                if all(s >= 90 for s in scores)
                else (WARN if any(s >= 50 for s in scores) else FAIL)
            )
            cli_runtime.console.print(f"{status} {model} ({backend}): {avg:.0f}% avg")

        all_results.append({"model": model, "backend": backend, "results": results})
        for r in results:
            task = r["task"]
            score = r["quality_score"]
            if task not in best_scores:
                best_scores[task] = -1
            if score > best_scores[task]:
                best_scores[task] = score
                best_models_dict[task] = model

    _print_results(all_results, best_scores, best_models_dict)
