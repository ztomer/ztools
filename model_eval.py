#!/usr/bin/env python3
"""
Model Evaluator - Test models on standard tasks.
Uses osaurus_lib for generic utilities.
"""

import os
import sys
import json
import time
import requests
from pathlib import Path
from rich.console import Console

from osaurus_lib import (
    call,
    get_api_url,
    get_models,
    is_server_running,
    extract_json,
    validate_json,
    validate_detailed_json,
    validate_summary,
    validate_filename,
    BEST_MODELS,
    TIMEOUTS,
)

console = Console()

# TASKS - Task prompts for model evaluation
# Each task has a prompt that asks the model to generate the expected output format
TASKS = {
    # JSON tasks - model generates JSON
    "json_simple": {
        "prompt": {
            "role": "user",
            "content": "List 3 activities: Reading, Swimming, Cooking. Output JSON list.",
        }
    },
    "json_medium": {
        "prompt": {
            "role": "user",
            "content": "List 5 indoor kid activities in Toronto. Output JSON with name, location, weather.",
        }
    },
    "json_complex": {
        "prompt": {
            "role": "user",
            "content": "List 10 activities for kids in Toronto. Output JSON with name, location, price, ages.",
        }
    },
    "summarize": {
        "prompt": {
            "role": "user",
            "content": "Tweets: [1] Breaking news [2] More details. Summarize with ## Summary and ## Key Facts headers.",
        }
    },
    "filename": {
        "prompt": {
            "role": "user",
            "content": "Text: Quarterly Financial Report Q4 2024. Output a short filename under 50 chars.",
        }
    },
}

JSON_TASKS = {"json_simple", "json_medium", "json_complex"}

VALIDATORS = {
    "json_simple": validate_json,
    "json_medium": validate_detailed_json,
    "json_complex": validate_detailed_json,
    "summarize": validate_summary,
    "filename": validate_filename,
}


def run_eval(
    model: str, tasks: dict = None, host: str = "localhost", port: int = 1337
) -> dict:
    """Run evaluation on model."""
    tasks = tasks or TASKS
    results = []

    console.print(f"[cyan]Testing {model}...[/cyan]")

    for task_name, task in tasks.items():
        # Task format: either {prompt, user} or single {prompt} with all content in prompt
        if "user" in task:
            messages = [
                {"role": task["prompt"]["role"], "content": task["prompt"]["content"]},
                {"role": "user", "content": task["user"]},
            ]
        else:
            # Single prompt contains full instruction
            messages = [
                {"role": task["prompt"]["role"], "content": task["prompt"]["content"]}
            ]

        parse_json = task_name in JSON_TASKS

        result = call(
            model,
            messages,
            host,
            port,
            task=task_name,
            parse_json=parse_json,
            validator=VALIDATORS.get(task_name),
        )

        status = (
            "ok"
            if result.get("quality_score", 0) >= 90
            else ("partial" if result.get("quality_score", 0) >= 50 else "fail")
        )

        results.append(
            {
                "task": task_name,
                "status": status,
                "quality_score": result.get("quality_score", 0),
                "time": result.get("time"),
                "error": result.get("error"),
            }
        )

        status_symbol = (
            "✅" if status == "ok" else ("⚠️" if status == "partial" else "❌")
        )
        console.print(
            f"  {status_symbol} {task_name}: {result.get('quality_score', 0)}% ({result.get('time', 0)}s)"
        )

    return results


def main():
    if not is_server_running():
        console.print("[red]Error: Osaurus server not running[/red]")
        sys.exit(1)

    models = get_models()
    if not models:
        console.print("[red]No models found[/red]")
        sys.exit(1)

    console.print(f"[green]Found {len(models)} models[/green]")

    for model in models:
        results = run_eval(model)
        scores = [r["quality_score"] for r in results]
        avg = sum(scores) / len(scores) if scores else 0
        console.print(f"[bold]{model}: {avg:.0f}%[/bold]")


if __name__ == "__main__":
    main()
