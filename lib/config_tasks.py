"""Task builder - creates eval tasks from model config."""

from pathlib import Path
from typing import Any, Dict

from .config_core import Task
from .config_getters import get_model_prompts_all
from .config_toml import load_config

_eval_inputs_cache: Dict[str, str] = {}


def clear_eval_inputs_cache():
    _eval_inputs_cache.clear()


def _load_eval_inputs() -> Dict[str, str]:
    global _eval_inputs_cache
    if _eval_inputs_cache:
        return _eval_inputs_cache
    inputs_path = Path(__file__).parent.parent / "conf" / "eval_inputs.toml"
    if not inputs_path.exists():
        raise FileNotFoundError(f"Missing eval inputs: {inputs_path}")
    data = load_config(inputs_path) or {}
    _eval_inputs_cache = data.get("test_inputs", {})
    if not _eval_inputs_cache:
        raise ValueError(f"Empty test_inputs in {inputs_path}")
    return _eval_inputs_cache


def get_eval_input(task: str) -> str:
    inputs = _load_eval_inputs()
    if task not in inputs:
        raise KeyError(f"Unknown task: {task}. Available: {list(inputs.keys())}")
    return inputs[task]


def _safe_format_prompt(prompt_template: str, test_input: str) -> str:
    result = prompt_template

    import json

    location = ""
    target_ages = ""
    try:
        data = json.loads(test_input)
        if data:
            first = data[0] if isinstance(data, list) else data
            location = first.get("location", "")
            target_ages = first.get("target_ages", "")
    except Exception:
        pass

    if "{location}" in result and location:
        result = result.replace("{location}", location)
    if "{age_range}" in result and target_ages:
        result = result.replace("{age_range}", target_ages)
    if "{date_range}" in result:
        result = result.replace("{date_range}", "this weekend")
    if "{text}" in result:
        result = result.replace("{text}", test_input)
    if "{}" in result:
        result = result.replace("{}", test_input)

    return result


def build_tasks_from_model(model: str) -> Dict[str, Any]:
    prompts = get_model_prompts_all(model)
    if not prompts:
        return {}
    tasks = {}
    from eval.validate import validate_file_summary
    from lib.validators_lib import validate_detailed_json, validate_filename, validate_summary

    if Task.WEEKEND_FIXED.value in prompts:
        test_input = get_eval_input("weekend_fixed")
        prompt = _safe_format_prompt(prompts[Task.WEEKEND_FIXED.value], test_input)
        tasks["detailed_json"] = {
            "messages": [{"role": "user", "content": prompt}],
            "validator": validate_detailed_json,
            "parse_json": True,
            "source": test_input,
        }
    if Task.WEEKEND_TRANSIENT.value in prompts:
        test_input = get_eval_input("weekend_transient")
        prompt = _safe_format_prompt(prompts[Task.WEEKEND_TRANSIENT.value], test_input)
        tasks["json"] = {
            "messages": [{"role": "user", "content": prompt}],
            "validator": validate_detailed_json,
            "parse_json": True,
        }
    if Task.FILENAME.value in prompts:
        test_input = get_eval_input("filename")
        prompt = _safe_format_prompt(prompts[Task.FILENAME.value], test_input)
        tasks["filename"] = {
            "messages": [{"role": "user", "content": prompt}],
            "validator": validate_filename,
            "parse_json": False,
        }
    if Task.SUMMARIZE.value in prompts:
        test_input = get_eval_input("summarize")
        prompt = _safe_format_prompt(prompts[Task.SUMMARIZE.value], test_input)
        tasks["summarize"] = {
            "messages": [{"role": "user", "content": prompt}],
            "validator": validate_summary,
            "parse_json": False,
        }
    if Task.FILE_SUMMARY.value in prompts:
        test_input = get_eval_input("file_summary")
        prompt = _safe_format_prompt(prompts[Task.FILE_SUMMARY.value], test_input)
        tasks["file_summary"] = {
            "messages": [{"role": "user", "content": prompt}],
            "validator": validate_file_summary,
            "parse_json": True,
            "source": test_input,
        }
    return tasks
