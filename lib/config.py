#!/usr/bin/env python3
"""
Config Management - Single source of truth from conf/config.yaml.
Auto-loads configuration on first access.
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List

# ==========================================================
# CONSTANTS - Task names used throughout the system
# ==========================================================

import enum


class Task(enum.Enum):
    """Task types for configuration - use these instead of strings for type safety."""
    WEEKEND_FIXED = "weekend_fixed"
    WEEKEND_TRANSIENT = "weekend_transient"
    SUMMARIZE = "summarize"
    FILENAME = "filename"
    JSON = "json"
    DETAILED_JSON = "detailed_json"


# Backward compatibility - use Task enum instead
TaskKeys = Task

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


def get_best_model(task: Task) -> str:
    """Get the best model for a specific task.
    
    Args:
        task: Task enum value (e.g., Task.SUMMARIZE)
        
    Returns:
        Model name string
    """
    models = get_best_models()
    task_key = task.value if isinstance(task, Task) else task
    return models.get(task_key, _config.get("default_model", _FALLBACK_MODEL))


def get_timeout(task: Task) -> int:
    """Get timeout for a specific task."""
    timeouts = get_timeouts()
    task_key = task.value if isinstance(task, Task) else task
    return timeouts.get(task_key, _FALLBACK_TIMEOUT)


def get_max_tokens_for_task(task: Task) -> int:
    """Get max tokens for a specific task."""
    tokens = get_max_tokens()
    task_key = task.value if isinstance(task, Task) else task
    return tokens.get(task_key, _FALLBACK_MAX_TOKENS)


def get_config() -> Dict[str, Any]:
    """Get the full raw config dict."""
    _auto_load()
    return _config.copy()


def is_config_loaded() -> bool:
    """Check if configuration has been loaded."""
    _auto_load()
    return _config_loaded


# ==========================================================
# MODEL-SPECIFIC CONFIG
# ==========================================================

# Cache for model configs - defined before functions that use it
_model_configs_cache: Dict[str, Dict] = {}


def reset_config():
    """Reset configuration state (for testing)."""
    global _config_loaded, _config
    _config_loaded = False
    _config = {}
    _model_configs_cache.clear()


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


def clear_model_config_cache():
    """Clear the model config cache. Call after modifying YAML files."""
    global _model_configs_cache
    _model_configs_cache.clear()


def get_model_config(model: str) -> Dict:
    """Load model-specific configuration from conf/models/{family}.yaml or version configs"""
    family = get_model_family(model)
    version = model.replace(family + "-", "") if family in model else ""

    # Check cache first
    if family in _model_configs_cache:
        family_config = _model_configs_cache[family]
        # Check if we have version-specific override
        if version and "models" in family_config:
            version_config = family_config["models"].get(model, {})
            if version_config:
                # Merge family config with version-specific overrides
                merged = {k: v for k, v in family_config.items() if k != "models"}
                merged.update(version_config)
                merged["version"] = version
                return merged
        return family_config

    # Load from YAML - check for version configs first
    version_config_path = Path(__file__).parent.parent / "conf" / "models" / f"{family}_versions.yaml"
    config_path = Path(__file__).parent.parent / "conf" / "models" / f"{family}.yaml"

    if version_config_path.exists():
        with open(version_config_path) as f:
            loaded = yaml.safe_load(f) or {}
            _model_configs_cache[family] = loaded
            if "models" in loaded:
                version_specific = loaded["models"].get(model, {})
                if version_specific:
                    merged = {k: v for k, v in loaded.items() if k != "models"}
                    merged.update(version_specific)
                    merged["version"] = version
                    return merged
            return loaded
    elif config_path.exists():
        with open(config_path) as f:
            _model_configs_cache[family] = yaml.safe_load(f) or {}
    else:
        _model_configs_cache[family] = {"name": family, "prompts": {}, "key_mappings": {}, "quirks": []}

    return _model_configs_cache.get(family, {"name": family, "prompts": {}, "key_mappings": {}, "quirks": []})


def get_model_field_mapping(model: str) -> Dict[str, str]:
    """Get model-specific field mapping (e.g., category -> target_ages).

    Args:
        model: Model name (e.g., "qwen3.6-35b-a3b-mxfp4")

    Returns:
        Dict mapping model field names to standard field names
    """
    config = get_model_config(model)
    return config.get("field_mapping", {})


def get_model_top_keys(model: str) -> Dict[str, List[str]]:
    """Get model-specific top-level JSON keys for extraction.

    Args:
        model: Model name (e.g., "qwen3.6-35b-a3b-mxfp4")

    Returns:
        Dict with 'fixed' and 'transient' keys lists, in priority order
    """
    config = get_model_config(model)
    return config.get("top_keys", {
        "fixed": ["fixed_activities", "year_round_fixed_activities", "venues", "places", "activities", "items"],
        "transient": ["transient_events", "events", "activities", "recommendations"],
    })


def get_model_quirks(model: str) -> List[Dict]:
    """Get model-specific quirks for processing.

    Args:
        model: Model name (e.g., "qwen3.6-35b-a3b-mxfp4")

    Returns:
        List of quirk dicts with type, name, description, etc.
    """
    config = get_model_config(model)
    return config.get("quirks", [])


def get_model_prompt(model: str, task: Task) -> str:
    """Get model-specific prompt for a task.

    Args:
        model: Model name (e.g., "qwen", "gemma", "foundation")
        task: Task enum value (e.g., Task.SUMMARIZE)

    Returns:
        Prompt string or empty if not found
    """
    config = get_model_config(model)
    prompts = config.get("prompts", {})
    task_key = task.value if isinstance(task, Task) else task
    return prompts.get(task_key, "")


def get_model_prompts_all(model: str) -> Dict[str, str]:
    """Get all prompts for a model."""
    config = get_model_config(model)
    return config.get("prompts", {})


# ==========================================================
# TASK BUILDER - Creates eval tasks from model config
# ==========================================================

# Test inputs for eval tasks - should be minimal but realistic
_EVAL_TEST_INPUTS = {
    Task.WEEKEND_FIXED: "Vaughan Sports Arena: indoor.",
    Task.WEEKEND_TRANSIENT: "Spring Festival April 20.",
    Task.FILENAME: "Screenshot of login page with error message.",
    Task.SUMMARIZE: "[@user1 | 10:00]: Test tweet",
}


def _safe_format_prompt(prompt_template: str, test_input: str) -> str:
    """Safely format a prompt template with test input.

    Only formats if the template contains {} placeholder.
    Returns the template as-is if no placeholder found.
    """
    if "{}" in prompt_template:
        return prompt_template.format(test_input)
    return prompt_template


def build_tasks_from_model(model: str) -> Dict[str, Any]:
    """Build eval TASKS dict from model config.

    Maps config prompts to task format used in model_eval.py.
    """
    prompts = get_model_prompts_all(model)
    if not prompts:
        return {}

    tasks = {}

    # Import validators lazily to avoid circular imports
    from lib.validators_lib import validate_detailed_json, validate_summary, validate_filename

    # Map prompt keys to task definitions using Task enum
    if Task.WEEKEND_FIXED.value in prompts:
        test_input = _EVAL_TEST_INPUTS[Task.WEEKEND_FIXED]
        tasks["detailed_json"] = {
            "messages": [
                {"role": "system", "content": prompts[Task.WEEKEND_FIXED.value]},
                {"role": "user", "content": f"Extract venues from this context: {test_input}"},
            ],
            "validator": validate_detailed_json,
            "parse_json": True,
            "source": test_input,
        }

    if Task.WEEKEND_TRANSIENT.value in prompts:
        test_input = _EVAL_TEST_INPUTS[Task.WEEKEND_TRANSIENT]
        tasks["json"] = {
            "messages": [
                {"role": "system", "content": prompts[Task.WEEKEND_TRANSIENT.value]},
                {"role": "user", "content": f"Extract events from this context: {test_input}"},
            ],
            "validator": validate_detailed_json,
            "parse_json": True,
            "source": test_input,
        }

    if Task.FILENAME.value in prompts:
        test_input = _EVAL_TEST_INPUTS[Task.FILENAME]
        prompt = _safe_format_prompt(prompts[Task.FILENAME.value], test_input)
        tasks["filename"] = {
            "messages": [
                {"role": "user", "content": prompt},
            ],
            "validator": validate_filename,
            "parse_json": False,
        }

    if Task.SUMMARIZE.value in prompts:
        test_input = _EVAL_TEST_INPUTS[Task.SUMMARIZE]
        prompt = _safe_format_prompt(prompts[Task.SUMMARIZE.value], test_input)
        tasks["summarize"] = {
            "messages": [
                {"role": "user", "content": prompt},
            ],
            "validator": validate_summary,
            "parse_json": False,
        }

    return tasks
