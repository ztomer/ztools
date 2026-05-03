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
    FILE_SUMMARY = "file_summary"
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
      nemotron-3-nano-omni-30b-a3b-mxfp4 -> nemotron
      laguna-xs.2-mxfp4 -> laguna
    """
    if not model:
        return "default"

    model_lower = model.lower()

    if "qwen" in model_lower:
        return "qwen"
    elif "gemma" in model_lower:
        return "gemma"
    elif "nemotron" in model_lower:
        return "nemotron"
    elif "laguna" in model_lower:
        return "laguna"
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
        # Fallback config for unknown models - use foundation-style prompts as default
        _model_configs_cache[family] = {
            "name": family,
            "timeout": 300,
            "prompts": {
                "json": "Output JSON now. Use EXACT schema.",
                "weekend_fixed": "Output JSON now. Use EXACT schema: {\"fixed_activities\": [{\"name\": \"str\", \"location\": \"str\", \"target_ages\": \"str\", \"price\": \"str\", \"weather\": \"str\"}]}\n\nExtract venues. Use exact fields. Output ONLY JSON.",
                "weekend_transient": "Output JSON now. Schema: {\"transient_events\": [{\"name\": \"str\", \"location\": \"str\", \"target_ages\": \"str\", \"price\": \"str\", \"duration\": \"str\", \"weather\": \"str\", \"day\": \"str\"]}\n\nFind events. Use exact fields. Output ONLY JSON.",
                "summarize": "Output a detailed summary with ## headers and bullet points.\n\n{}\n\nSummarize thoroughly.",
                "filename": "Output ONLY the filename (lowercase, underscores).",
                "file_summary": "Output JSON array with path and desc fields.",
            },
            "key_mappings": {
                "event": "name", "title": "name", "activity": "name",
                "venue": "location", "address": "location", "place": "location",
                "age_group": "target_ages", "ages": "target_ages", "age_range": "target_ages",
                "cost": "price", "pricing": "price", "fee": "price",
                "type": "weather", "category": "weather",
            },
            "quirks": [
                {"type": "prefix", "pattern": "Output JSON now.", "reason": "Ensures clean JSON"}
            ],
            "top_keys": {
                "fixed": ["fixed_activities", "venues", "places", "activities", "items"],
                "transient": ["transient_events", "events", "activities", "recommendations"],
            },
            "field_mapping": {},
        }

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
# ==========================================================
# TASK BUILDER - Creates eval tasks from model config
# ==========================================================

# Load test inputs from YAML config
_eval_inputs_cache: Dict[str, str] = {}

def _load_eval_inputs() -> Dict[str, str]:
    """Load eval test inputs from conf/eval_inputs.yaml.
    
    Fail-fast if file missing - configs should always exist.
    """
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
    """Get test input for a task from config."""
    inputs = _load_eval_inputs()
    if task not in inputs:
        raise KeyError(f"Unknown task: {task}. Available: {list(inputs.keys())}")
    return inputs[task]


def _safe_format_prompt(prompt_template: str, test_input: str) -> str:
    """Safely format a prompt template with test input.

    Supports both:
    - {} placeholder (eval mode - test_input gets injected)
    - {location}, {age_range} style (production - uses runtime values)
    
    Priority: Try {} first, then try keyword placeholders from Task placeholders.
    """
    # First try {}
    if "{}" in prompt_template:
        try:
            return prompt_template.format(test_input)
        except (KeyError, ValueError):
            return prompt_template.replace("{}", test_input)
    
    # Check for Task-based placeholders and extract from test_input JSON
    # This allows prompts like "Extract {location} venues" to work in eval
    if test_input and ("{" in prompt_template or "}" in prompt_template):
        # Try to infer values from test_input JSON
        import json
        try:
            data = json.loads(test_input)
            if data and len(data) > 0:
                first_item = data[0]
                # Extract common values
                location = first_item.get("location", "")
                target_ages = first_item.get("target_ages", "")
                
                # Replace known placeholders
                result = prompt_template
                if location:
                    result = result.replace("{location}", location)
                if target_ages:
                    result = result.replace("{age_range}", target_ages)
                    result = result.replace("{age_range}", target_ages)
                return result
        except:
            pass
    
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
    try:
        from model_eval import validate_file_summary
    except ImportError:
        def validate_file_summary(data, source_text=""):
            from lib.validators_lib import validate_summary
            return validate_summary(data)

    # Map prompt keys to task definitions using Task enum
    if Task.WEEKEND_FIXED.value in prompts:
        test_input = get_eval_input("weekend_fixed")
        prompt = _safe_format_prompt(prompts[Task.WEEKEND_FIXED.value], test_input)
        tasks["detailed_json"] = {
            "messages": [
                {"role": "user", "content": prompt},
            ],
            "validator": validate_detailed_json,
            "parse_json": True,
            "source": test_input,
        }

    if Task.WEEKEND_TRANSIENT.value in prompts:
        test_input = get_eval_input("weekend_transient")
        prompt = _safe_format_prompt(prompts[Task.WEEKEND_TRANSIENT.value], test_input)
        tasks["json"] = {
            "messages": [
                {"role": "user", "content": prompt},
            ],
            "validator": validate_detailed_json,
            "parse_json": True,
            "source": test_input,
        }

    if Task.FILENAME.value in prompts:
        test_input = get_eval_input("filename")
        prompt = _safe_format_prompt(prompts[Task.FILENAME.value], test_input)
        tasks["filename"] = {
            "messages": [
                {"role": "user", "content": prompt},
            ],
            "validator": validate_filename,
            "parse_json": False,
        }

    if Task.SUMMARIZE.value in prompts:
        test_input = get_eval_input("summarize")
        prompt = _safe_format_prompt(prompts[Task.SUMMARIZE.value], test_input)
        tasks["summarize"] = {
            "messages": [
                {"role": "user", "content": prompt},
            ],
            "validator": validate_summary,
            "parse_json": False,
        }

    if Task.FILE_SUMMARY.value in prompts:
        test_input = get_eval_input("file_summary")
        prompt = _safe_format_prompt(prompts[Task.FILE_SUMMARY.value], test_input)
        tasks["file_summary"] = {
            "messages": [
                {"role": "user", "content": prompt},
            ],
            "validator": validate_file_summary,
            "parse_json": True,
            "source": test_input,
        }

    return tasks


# ==========================================================
# FILENAME HELPERS - Get config for image renaming
# ==========================================================


def get_filename_models() -> List[str]:
    """Get model list for filename generation, with fallback."""
    _auto_load()
    models = _config.get("filename_models", [])
    return models if models else ["foundation"]


def get_filename_prompt() -> str:
    """Get prompt template for filename generation."""
    _auto_load()
    prompts = _config.get("prompts", {})
    return prompts.get("filename", "Give a short summary of: {text}")
