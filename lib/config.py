#!/usr/bin/env python3
"""
Config Management - Single source of truth from conf/config.yaml.
Auto-loads configuration on first access.
"""

import functools
import os
import sys
import yaml
from pathlib import Path
from typing import Dict, Any, Optional

# ==========================================================
# CONSTANTS - Task names used throughout the system
# ==========================================================

class TaskKeys:
    """Known task keys for configuration."""
    WEEKEND_FIXED = "weekend_fixed"
    WEEKEND_TRANSIENT = "weekend_transient"
    SUMMARIZE = "summarize"
    FILENAME = "filename"
    JSON = "json"
    DETAILED_JSON = "detailed_json"

# Minimal hardcoded fallbacks - only used if config.yaml is completely missing
_FALLBACK_TIMEOUT = 600
_FALLBACK_MAX_TOKENS = 16000
_FALLBACK_MODEL = "foundation"

# Global config state
_config_loaded = False
_config: Dict[str, Any] = {}


def _auto_load():
    """Auto-load config from conf/config.yaml if not yet loaded."""
    global _config_loaded, _config

    if _config_loaded:
        return

    config_path = Path(__file__).parent.parent / "conf" / "config.yaml"
    if not config_path.exists():
        print(f"[ Wrn ] Config file not found: {config_path}, using fallback defaults")
        _config_loaded = True
        return

    with open(config_path, 'r') as f:
        loaded = yaml.safe_load(f)
    _config = loaded if isinstance(loaded, dict) else {}
    _config_loaded = True


def init_config(config_path: Optional[str] = None) -> bool:
    """
    Explicitly initialize configuration from a YAML file.
    Use this to override the default conf/config.yaml path.

    Args:
        config_path: Path to config.yaml file. If None, uses conf/config.yaml

    Returns:
        bool: True if config loaded successfully

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If config is invalid
    """
    global _config_loaded, _config

    if config_path is None:
        config_path = Path(__file__).parent.parent / "conf" / "config.yaml"

    config_file = Path(config_path)

    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")

    with open(config_file, 'r') as f:
        loaded = yaml.safe_load(f)

    if loaded is None:
        loaded = {}

    if not isinstance(loaded, dict):
        raise ValueError("Config must be a dictionary")

    _config = loaded
    _config_loaded = True
    return True


def get_timeouts() -> Dict[str, int]:
    """Get timeout configuration from config.yaml."""
    _auto_load()
    return _config.get("timeouts", {})


def get_max_tokens() -> Dict[str, int]:
    """Get max tokens configuration from config.yaml."""
    _auto_load()
    return _config.get("max_tokens", {})


def get_best_models() -> Dict[str, str]:
    """Get best models configuration from config.yaml."""
    _auto_load()
    return _config.get("best_models", {})


def get_best_model(task: str) -> str:
    """Get the best model for a specific task."""
    models = get_best_models()
    return models.get(task, _config.get("default_model", _FALLBACK_MODEL))


def get_timeout(task: str) -> int:
    """Get timeout for a specific task."""
    timeouts = get_timeouts()
    return timeouts.get(task, _FALLBACK_TIMEOUT)


def get_max_tokens_for_task(task: str) -> int:
    """Get max tokens for a specific task."""
    tokens = get_max_tokens()
    return tokens.get(task, _FALLBACK_MAX_TOKENS)


def get_config() -> Dict[str, Any]:
    """Get the full raw config dict."""
    _auto_load()
    return _config.copy()


def is_config_loaded() -> bool:
    """Check if configuration has been loaded."""
    _auto_load()
    return _config_loaded


def reset_config():
    """Reset configuration state (for testing)."""
    global _config_loaded, _config
    _config_loaded = False
    _config = {}
    _model_configs_cache.clear()


# ==========================================================
# MODEL-SPECIFIC CONFIG
# ==========================================================

@functools.lru_cache(maxsize=16)
def get_model_family(model: str) -> str:
    """Extract model family from full model name.

    Examples:
      qwen3.6-35b-a3b-mxfp4 -> qwen
      gemma-4-26b-a4b-it-4bit -> gemma
      foundation -> foundation
    """
    if not model:
        return "default"

    model_lower = model.lower()

    if "qwen" in model_lower:
        return "qwen"
    elif "gemma" in model_lower:
        return "gemma"
    elif "foundation" in model_lower:
        return "foundation"
    else:
        return "default"


# Use a plain dict with explicit clear method instead of lru_cache
# since we need to store complex dicts, not just strings
_model_configs_cache: Dict[str, Dict] = {}


def clear_model_config_cache():
    """Clear the model config cache. Call after modifying YAML files."""
    global _model_configs_cache
    _model_configs_cache.clear()


def get_model_config(model: str) -> Dict:
    """Load model-specific configuration from conf/models/{family}.yaml"""
    family = get_model_family(model)

    if family in _model_configs_cache:
        return _model_configs_cache[family]

    # Load from YAML
    config_path = Path(__file__).parent.parent / "conf" / "models" / f"{family}.yaml"

    if config_path.exists():
        with open(config_path) as f:
            _model_configs_cache[family] = yaml.safe_load(f) or {}
    else:
        # Return empty config
        _model_configs_cache[family] = {"name": family, "prompts": {}, "key_mappings": {}, "quirks": []}

    return _model_configs_cache[family]


def get_model_prompt(model: str, task: str) -> str:
    """Get model-specific prompt for a task."""
    config = get_model_config(model)
    prompts = config.get("prompts", {})
    return prompts.get(task, "")


def get_model_prompts_all(model: str) -> Dict[str, str]:
    """Get all prompts for a model."""
    config = get_model_config(model)
    return config.get("prompts", {})


# ==========================================================
# TASK BUILDER - Creates eval tasks from model config
# ==========================================================

# Test inputs for eval tasks - should be minimal but realistic
_EVAL_TEST_INPUTS = {
    TaskKeys.WEEKEND_FIXED: "Vaughan Sports Arena: indoor.",
    TaskKeys.WEEKEND_TRANSIENT: "Spring Festival April 20.",
    TaskKeys.FILENAME: "Screenshot of login page with error message.",
    TaskKeys.SUMMARIZE: "[@user1 | 10:00]: Test tweet",
}


def _safe_format_prompt(prompt_template: str, test_input: str) -> str:
    """Safely format a prompt template with test input.

    Only formats if the template contains {} placeholder.
    Returns the template as-is if no placeholder found.
    """
    if "{}" in prompt_template:
        return prompt_template.format(test_input)
    return prompt_template


def build_tasks_from_model(model: str) -> Dict:
    """Build eval TASKS dict from model config.

    Maps config prompts to task format used in model_eval.py.
    """
    prompts = get_model_prompts_all(model)
    if not prompts:
        return {}

    tasks = {}

    # Import validators lazily to avoid circular imports
    from lib.validators_lib import validate_detailed_json, validate_summary, validate_filename

    # Map prompt keys to task definitions using constants
    if TaskKeys.WEEKEND_FIXED in prompts:
        test_input = _EVAL_TEST_INPUTS[TaskKeys.WEEKEND_FIXED]
        tasks["detailed_json"] = {
            "messages": [
                {"role": "system", "content": prompts[TaskKeys.WEEKEND_FIXED]},
                {"role": "user", "content": f"Extract venues from this context: {test_input}"},
            ],
            "validator": validate_detailed_json,
            "parse_json": True,
            "source": test_input,
        }

    if TaskKeys.WEEKEND_TRANSIENT in prompts:
        test_input = _EVAL_TEST_INPUTS[TaskKeys.WEEKEND_TRANSIENT]
        tasks["json"] = {
            "messages": [
                {"role": "system", "content": prompts[TaskKeys.WEEKEND_TRANSIENT]},
                {"role": "user", "content": f"Extract events from this context: {test_input}"},
            ],
            "validator": validate_detailed_json,
            "parse_json": True,
            "source": test_input,
        }

    if TaskKeys.FILENAME in prompts:
        test_input = _EVAL_TEST_INPUTS[TaskKeys.FILENAME]
        prompt = _safe_format_prompt(prompts[TaskKeys.FILENAME], test_input)
        tasks["filename"] = {
            "messages": [
                {"role": "user", "content": prompt},
            ],
            "validator": validate_filename,
            "parse_json": False,
        }

    if TaskKeys.SUMMARIZE in prompts:
        test_input = _EVAL_TEST_INPUTS[TaskKeys.SUMMARIZE]
        prompt = _safe_format_prompt(prompts[TaskKeys.SUMMARIZE], test_input)
        tasks["summarize"] = {
            "messages": [
                {"role": "user", "content": prompt},
            ],
            "validator": validate_summary,
            "parse_json": False,
        }

    return tasks
