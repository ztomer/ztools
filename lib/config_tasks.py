"""Task builder - creates eval tasks from model config."""

from pathlib import Path
from typing import Dict, Any
import yaml

from .config_core import Task
from .config_getters import get_model_prompts_all


_eval_inputs_cache: Dict[str, str] = {}


def _load_eval_inputs() -> Dict[str, str]:
    global _eval_inputs_cache
    if _eval_inputs_cache:
        return _eval_inputs_cache
    inputs_path = Path(__file__).parent.parent / "conf" / "eval_inputs.yaml"
    if not inputs_path.exists():
        raise FileNotFoundError(f"Missing eval inputs: {inputs_path}")
    with open(inputs_path) as f:
        data = yaml.safe_load(f) or {}
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
    if "{}" in prompt_template:
        try:
            return prompt_template.format(test_input)
        except (KeyError, ValueError):
            return prompt_template.replace("{}", test_input)
    if test_input and ("{" in prompt_template or "}" in prompt_template):
        import json
        try:
            data = json.loads(test_input)
            if data and len(data) > 0:
                first_item = data[0]
                location = first_item.get("location", "")
                target_ages = first_item.get("target_ages", "")
                result = prompt_template
                if location:
                    result = result.replace("{location}", location)
                if target_ages:
                    result = result.replace("{age_range}", target_ages)
                    result = result.replace("{age_range}", target_ages)
                return result
        except Exception:
            pass
    return prompt_template


def build_tasks_from_model(model: str) -> Dict[str, Any]:
    prompts = get_model_prompts_all(model)
    if not prompts:
        return {}
    tasks = {}
    from lib.validators_lib import validate_detailed_json, validate_summary, validate_filename
    try:
        from eval.validate import validate_file_summary
    except ImportError:
        def validate_file_summary(data, source_text=""):
            from lib.validators_lib import validate_summary
            return validate_summary(data)

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
            "source": test_input,
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
