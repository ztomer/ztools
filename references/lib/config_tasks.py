"""Task builder - creates eval tasks from model config."""

from typing import Any, Dict

from .config_core import Task
from .config_getters import get_model_prompts_all
from .config_toml import load_config
from .paths import conf_path
from .prompt_render import POSITIONAL_SLOT, render_prompt, unrendered_placeholders

_eval_inputs_cache: Dict[str, str] = {}


def clear_eval_inputs_cache():
    _eval_inputs_cache.clear()


def _load_eval_inputs() -> Dict[str, str]:
    global _eval_inputs_cache
    if _eval_inputs_cache:
        return _eval_inputs_cache
    inputs_path = conf_path("eval_inputs.toml")
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
    """Render an eval prompt through the SAME renderer production uses.

    Class C12. This used to be a private `str.replace()` helper that silently
    succeeded on templates `str.format()` could not render, so `ev` scored green
    on prompts production shipped corrupted. Both paths now go through
    `lib.prompt_render.render_prompt`, so the divergence cannot silently return.

    The eval's own placeholder values are derived from `test_input`; only the
    substitution mechanism is shared, not the data.
    """
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

    fields = {
        "location": location or "the eval location",
        "age_range": target_ages or "6-13",
        "date_range": "this weekend",
        "text": test_input,
        "exclusions": "",
    }
    # Only offer fields the template actually asks for, so an unexpected
    # placeholder still raises rather than being quietly absorbed.
    wanted = set(unrendered_placeholders(prompt_template))
    return render_prompt(
        prompt_template,
        template_id="eval",
        positional=test_input if POSITIONAL_SLOT in prompt_template else None,
        **{k: v for k, v in fields.items() if k in wanted},
    )


def _embeds_source(template: str) -> bool:
    """Whether the template itself carries the source text into the prompt."""
    return POSITIONAL_SLOT in template or "{text}" in template


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
        template = prompts[Task.WEEKEND_TRANSIENT.value]
        prompt = _safe_format_prompt(template, test_input)
        messages = [{"role": "user", "content": prompt}]
        # The shipped transient templates use named placeholders only, so the
        # source events never reached the model: the prompt ordered "Copy every
        # value from the source text. NEVER invent one" with no source text, and
        # without a "source" key validate_detailed_json skipped both the
        # grounding score and the no-source cap — schema-shaped hallucination
        # could score 100. Production's fallback delivers the events as a second
        # message; do the same here.
        if not _embeds_source(template):
            messages.append({"role": "user", "content": f"Source events:\n{test_input}"})
        tasks["json"] = {
            "messages": messages,
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
