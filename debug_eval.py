#!/usr/bin/env python3
"""Quick debug script to test a single model+task via the eval pipeline."""

from lib.osaurus_lib import call as osaurus_call
from lib.validators_lib import validate_detailed_json
from model_eval import TASKS, _validate_result

task_name = "json"
task = TASKS[task_name]
model = "foundation"

print(f"Calling {model} for task '{task_name}'...")
result = osaurus_call(
    model=model,
    messages=task["messages"],
    task=task_name,
    parse_json=task["parse_json"],
)

print("\n--- RAW CONTENT ---")
print(repr(result.get("content", "")[:500]))
print("\n--- PARSED ---")
print(repr(result.get("parsed", "")))

score, reason = _validate_result(result, task, task_name)
print(f"\n--- QUALITY SCORE ---\n{score}/100")
print(f"\n--- FAILURE REASON ---\n{reason}")
