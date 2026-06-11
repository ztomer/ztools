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

import sys
import re
import json
import time
import argparse
import os
from pathlib import Path
from rich.console import Console

from lib import init_config
from lib.osaurus_lib import (
    call,
    get_models,
    is_server_running,
)

from lib.tui import STEP, WARN, FAIL
from lib.config import get_model_prompts_all, build_tasks_from_model

from eval.tasks_core import (
    TASKS,
    _extract_items_from_text,
    WEEKEND_SYS_TRANSIENT,
    WEEKEND_SYS_FIXED,
    WEEKEND_USR_TRANSIENT,
    WEEKEND_USR_FIXED,
    RENAME_PROMPT,
    FILE_SUMMARY_PROMPT,
    TWITTER_PROMPT,
)
from eval.validate import safe_content, validate_file_summary
from eval.failures import (
    FAIL_INFRA,
    FAIL_TIMEOUT,
    FAIL_PARSE,
    FAIL_FORMAT,
    FAIL_CONTENT,
    FAIL_NONE,
    _classify_failure,
    _describe_content_failure,
)
import eval.run as eval_run
from eval.run import (
    MAX_RETRIES,
    EVAL_TIMEOUT,
    MEMORY_WARNING_THRESHOLD,
    _validate_result,
    _call_model,
    _quality_results_to_eval_format,
    run_eval,
)
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
    "TASKS", "MAX_RETRIES", "EVAL_TIMEOUT", "MEMORY_WARNING_THRESHOLD",
    "safe_content", "validate_file_summary",
    "FAIL_INFRA", "FAIL_TIMEOUT", "FAIL_PARSE", "FAIL_FORMAT", "FAIL_CONTENT", "FAIL_NONE",
    "run_eval", "_validate_result", "_call_model", "_quality_results_to_eval_format",
    "_extract_items_from_text", "_classify_failure", "_describe_content_failure",
    "print_cross_model_comparison", "compute_score_stats", "print_score_stats",
    "categorize_failures", "print_failure_summary",
    "save_historical_results", "load_historical_stats", "check_model_history",
    "print_historical_trends", "compute_token_estimates",
    "compute_verbosity", "print_verbosity",
    "compute_error_rates", "print_error_rates",
    "compute_task_winners", "diff_from_last_run", "print_diff", "export_to_csv",
    "get_memory_percent", "check_memory_safe", "is_server_responsive",
    "monitor_memory_loop", "estimate_model_memory",
    "load_tasks_from_config", "update_config", "main",
    "WEEKEND_SYS_TRANSIENT", "WEEKEND_SYS_FIXED",
    "WEEKEND_USR_TRANSIENT", "WEEKEND_USR_FIXED",
    "RENAME_PROMPT", "FILE_SUMMARY_PROMPT", "TWITTER_PROMPT",
]

console = Console()


# ==========================================================
# Memory monitoring
# ==========================================================

def get_memory_percent() -> float:
    """Get current memory usage percent."""
    try:
        import psutil
        return psutil.virtual_memory().percent
    except ImportError:
        return 0.0


def check_memory_safe() -> bool:
    """Check if memory is safe to run eval."""
    mem_pct = get_memory_percent()
    if mem_pct > MEMORY_WARNING_THRESHOLD:
        console.print(f"{WARN} Memory at {mem_pct}% - may cause OOM")
        return False
    return True


def is_server_responsive(host: str = "localhost", port: int = 1337, timeout: int = 5) -> bool:
    """Check if osaurus server is responsive."""
    import requests
    try:
        resp = requests.get(f"http://{host}:{port}/api/tags", timeout=timeout)
        return resp.status_code == 200
    except Exception:
        return False


def monitor_memory_loop(interval: int = 30):
    """Background thread to monitor memory during eval."""
    import threading
    import time

    def _monitor():
        while getattr(threading.current_thread(), "running", True):
            mem = get_memory_percent()
            if mem > MEMORY_WARNING_THRESHOLD:
                console.print(f"{WARN} Memory at {mem}%")
            time.sleep(interval)

    t = threading.Thread(target=_monitor, daemon=True)
    t.running = True
    t.start()
    return t


def estimate_model_memory(model: str) -> int:
    """Estimate memory needed for a model (in GB). Extract size from model name."""
    import re
    match = re.search(r'(\d+)b', model.lower())
    if match:
        return int(match.group(1))
    return 4


# ==========================================================
# Helper to build tasks from model configs
# ==========================================================

def load_tasks_from_config(model: str):
    """Build task prompts from model config YAML."""
    prompts = get_model_prompts_all(model)
    if not prompts:
        return None

    built = {}

    if "weekend_fixed" in prompts:
        built["detailed_json"] = prompts["weekend_fixed"]
    if "weekend_transient" in prompts:
        built["json"] = prompts["weekend_transient"]
    if "summarize" in prompts:
        built["summarize"] = prompts["summarize"]
    if "filename" in prompts:
        built["filename"] = prompts["filename"]
    if "file_summary" in prompts:
        built["file_summary"] = prompts["file_summary"]

    return built


def update_config(best_models: dict):
    """Update config.yaml with best models per task."""
    import yaml
    config_path = Path(__file__).parent.parent / "conf" / "config.yaml"
    if not config_path.exists():
        console.print(f"{WARN} Config file not found, skipping update.")
        return

    with open(config_path, "r") as f:
        config = yaml.safe_load(f) or {}

    if "best_models" not in config:
        config["best_models"] = {}

    for task, model in best_models.items():
        if model:
            config["best_models"][task] = model

    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    console.print(f"{STEP} Updated conf/config.yaml with best models.")


def main():
    """Main entry point for model evaluation."""
    init_config()
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="Run evaluation for a specific model")
    parser.add_argument("--task", help="Run a specific task (weekend_transient, weekend_fixed, filename, summarize, file_summary)")
    parser.add_argument("--quick", action="store_true", help="Quick mode: run single task with one retry (faster iteration)")
    parser.add_argument("--config-tasks", action="store_true", help="Load tasks from YAML config instead of hardcoded prompts")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging to console")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show raw model output for debugging quality")
    parser.add_argument("--timeout", type=int, default=EVAL_TIMEOUT, help=f"Per-task timeout in seconds (default {EVAL_TIMEOUT})")
    parser.add_argument("--quality", action="store_true", help="Use quality.py dimension-based scoring instead of old validators")
    args = parser.parse_args()

    eval_run.EVAL_TIMEOUT = args.timeout

    models_to_test = []

    if is_server_running():
        osaurus_models = get_models()
        for m in osaurus_models:
            models_to_test.append((m, "osaurus"))
    else:
        console.print(f"{WARN} Osaurus server not running")

    if args.model:
        models_to_test = [(m, b) for m, b in models_to_test if m == args.model]

    if not models_to_test:
        console.print(f"{FAIL} No models found")
        sys.exit(1)

    console.print(f"{STEP} Found {len(models_to_test)} models to test")

    tasks_to_run = TASKS
    if args.task:
        if args.task not in TASKS:
            console.print(f"{FAIL} Unknown task: {args.task}. Available: {list(TASKS.keys())}")
            sys.exit(1)
        tasks_to_run = {args.task: TASKS[args.task]}
        console.print(f"{WARN} Running only task: {args.task}")

    config_model = args.model if args.model else "qwen"
    config_tasks = build_tasks_from_model(config_model)
    if config_tasks:
        if args.task:
            if args.task in config_tasks:
                tasks_to_run = {args.task: config_tasks[args.task]}
                console.print(f"{STEP} Using config task: {args.task}")
            else:
                console.print(f"{FAIL} Task '{args.task}' not in config")
        else:
            tasks_to_run = config_tasks
        console.print(f"{STEP} Loaded {len(tasks_to_run)} tasks from config")
    else:
        console.print(f"{WARN} Config loading failed, using minimal tasks")

    if args.quick:
        console.print(f"{STEP} Quick mode: single run, no retries")
        orig_run_eval = run_eval

        def quick_run_eval(model, backend="osaurus", **kwargs):
            orig_retries = MAX_RETRIES
            eval_run.MAX_RETRIES = 0
            try:
                return orig_run_eval(
                    model,
                    tasks=kwargs.get("tasks"),
                    backend=backend,
                    verbose=kwargs.get("verbose", False),
                )
            finally:
                eval_run.MAX_RETRIES = orig_retries

        import eval.cli as me
        me.run_eval = quick_run_eval

    all_results = []
    best_scores = {task: -1 for task in tasks_to_run.keys()}
    best_models_dict = {task: None for task in tasks_to_run.keys()}

    def flush_between_models(prev_model: str, next_model: str) -> None:
        import time
        import subprocess
        console.print(f"{STEP} Flushing {prev_model} -> {next_model}...")

        try:
            r = call(next_model, [{"role": "user", "content": "ok"}], timeout=30)
            if r.get("error"):
                console.print(f"{WARN} Flush failed, attempting restart...")
                try:
                    subprocess.run(["osascript", "-e", 'quit app "osaurus"'], capture_output=True)
                except Exception:
                    pass
                time.sleep(3)
                try:
                    subprocess.run(["open", "-n", "-a", "osaurus"], capture_output=True)
                except Exception:
                    pass
                time.sleep(8)
                for _ in range(5):
                    try:
                        import requests
                        resp = requests.get("http://localhost:1337/api/tags", timeout=2)
                        if resp.status_code == 200:
                            console.print(f"{STEP} Server restarted")
                            break
                    except Exception:
                        time.sleep(2)
        except Exception as e:
            console.print(f"{FAIL} Flush error: {e}")
        time.sleep(2)

    prev_model = None
    for model, backend in models_to_test:
        if prev_model and model != prev_model:
            flush_between_models(prev_model, model)
        prev_model = model

        mem_pct = get_memory_percent()
        model_mem_gb = estimate_model_memory(model)
        avail_mem_gb = (100 - mem_pct) / 100 * 64

        if mem_pct > MEMORY_WARNING_THRESHOLD:
            console.print(f"{WARN} Memory at {mem_pct}% - model may be slow")

        if model_mem_gb > avail_mem_gb * 0.8:
            console.print(f"{WARN} Model needs ~{model_mem_gb}GB, low memory - will be slower")

        if not is_server_responsive():
            console.print(f"{FAIL} Server not responsive - attempting restart...")

        console.print(f"{STEP} Memory: {mem_pct}%, Server: OK")

        if args.quality:
            from lib import quality as qm
            cases = qm.ALL_TEST_CASES
            if args.task:
                cases = [c for c in cases if c.task == args.task]
            scorecards = qm.run_suite([model], cases, verbose=True)
            results = _quality_results_to_eval_format(scorecards, model)
            task_avgs = {}
            for r in results:
                lst = task_avgs.setdefault(r["task"], [])
                lst.append(r["quality_score"])
            summary = "  ".join(f"{t}={int(round(sum(s)/len(s)))}%" for t, s in task_avgs.items())
            console.print(f"{STEP} Quality scores: {summary}")
        else:
            results = run_eval(model, tasks=tasks_to_run, backend=backend, verbose=args.verbose)

        scores = [r["quality_score"] for r in results]
        if not scores:
            console.print(f"{STEP} {model} ({backend}): 0 tasks")
        else:
            avg = sum(scores) / len(scores)
            status = STEP if all(s >= 90 for s in scores) else (WARN if any(s >= 50 for s in scores) else FAIL)
            console.print(f"{status} {model} ({backend}): {avg:.0f}% avg")

        all_results.append({'model': model, 'backend': backend, 'results': results})
        for r in results:
            task = r["task"]
            score = r["quality_score"]
            if task not in best_scores:
                best_scores[task] = -1
            if score > best_scores[task]:
                best_scores[task] = score
                best_models_dict[task] = model

    _print_results(all_results, best_scores, best_models_dict)


def _print_results(all_results, best_scores, best_models_dict):
    console.print("\nBest Models per Task:")
    for task, model in best_models_dict.items():
        console.print(f"  {task}: {model} ({best_scores[task]}%)")

    print_cross_model_comparison(all_results)

    stats = compute_score_stats(all_results)
    print_score_stats(stats)

    categories = categorize_failures(all_results)
    print_failure_summary(categories)

    verbosity = compute_verbosity(all_results)
    print_verbosity(verbosity)

    error_rates = compute_error_rates(all_results)
    print_error_rates(error_rates)

    diffs = diff_from_last_run(all_results)
    print_diff(diffs)

    winners = compute_task_winners(all_results)
    console.print("")
    console.print("Task Winners:")
    for task, (model, score) in sorted(winners.items()):
        console.print(f"  {task}: {model} ({score}%)")

    save_historical_results(all_results, stats, categories)
    print_historical_trends()

    export_to_csv(all_results)

    eval_dir = Path(os.path.expanduser("~/.config/ztools"))
    eval_dir.mkdir(parents=True, exist_ok=True)
    results_file = eval_dir / "eval_results.json"

    with open(results_file, "w") as f:
        json.dump({
            "models": all_results,
            "best_scores": best_scores,
            "best_models": best_models_dict,
        }, f, indent=2)
    console.print(f"{STEP} Saved to {results_file}")
