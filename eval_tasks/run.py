#!/usr/bin/env python3
"""
Quick evaluation runner.

Usage:
    python3 -m eval_tasks.run                  # Run all models
    python3 -m eval_tasks.run --model qwen    # Run specific model
    python3 -m eval_tasks.run --task weekend  # Run specific task
    python3 -m eval_tasks.run --quick        # Single run, no retries
"""

import sys
sys.path.insert(0, "/Users/ztomer/Projects/ztools")

from model_eval import run_eval, TASKS, is_server_running, get_models
from lib import init_config
from rich.console import Console

console = Console()


def run_model_eval(model: str, tasks: dict = None, host: str = "localhost", port: int = 1337, backend: str = "osaurus") -> list:
    """Run evaluation on a model. Returns list of result dicts."""
    if tasks is None:
        from model_eval import TASKS
        tasks = TASKS
    
    if not is_server_running():
        console.print("[yellow]Warning: Osaurus server not running[/yellow]")
        return []
    
    return run_eval(model, tasks=tasks, host=host, port=port, backend=backend)


def main():
    init_config()
    import argparse
    
    parser = argparse.ArgumentParser(description="Run model evaluation")
    parser.add_argument("--model", help="Model name to test")
    parser.add_argument("--task", help="Run specific task only")
    parser.add_argument("--quick", action="store_true", help="Quick mode")
    args = parser.parse_args()
    
    if args.model:
        models = [args.model]
    else:
        models = get_models()
    
    if not models:
        console.print("[red]No models found[/red]")
        return
    
    console.print(f"[green]Testing {len(models)} model(s)[/green]")
    
    for model in models:
        results = run_model_eval(model)
        
        console.print("")
        for r in results:
            status = "[PASS]" if r["quality_score"] >= 90 else "[WARN]" if r["quality_score"] >= 50 else "[FAIL]"
            console.print(f"  {status} {r['task']}: {r['quality_score']}%")


if __name__ == "__main__":
    main()