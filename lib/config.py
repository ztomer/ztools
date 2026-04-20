#!/usr/bin/env python3
"""
Config Management - Single source of truth from conf/config.yaml.
Auto-loads configuration on first access.
"""

import os
import sys
import yaml
from pathlib import Path
from typing import Dict, Any, Optional

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

    try:
        import yaml
        with open(config_path, 'r') as f:
            loaded = yaml.safe_load(f)
        _config = loaded if isinstance(loaded, dict) else {}
        _config_loaded = True
    except Exception as e:
        print(f"[ Err ] Failed to load config: {e}")
        _config_loaded = True


def init_config(config_path: Optional[str] = None) -> bool:
    """
    Explicitly initialize configuration from a YAML file.
    Use this to override the default conf/config.yaml path.

    Args:
        config_path: Path to config.yaml file. If None, uses conf/config.yaml

    Returns:
        bool: True if config loaded successfully
    """
    global _config_loaded, _config

    if config_path is None:
        config_path = Path(__file__).parent.parent / "conf" / "config.yaml"

    config_file = Path(config_path)

    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")

    try:
        import yaml
        with open(config_file, 'r') as f:
            loaded = yaml.safe_load(f)

        if loaded is None:
            loaded = {}

        if not isinstance(loaded, dict):
            raise ValueError("Config must be a dictionary")

        _config = loaded
        _config_loaded = True
        return True

    except Exception as e:
        raise ValueError(f"Error loading config: {e}")


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


# ==========================================================
# TASK-SPECIFIC PROMPTS
# ==========================================================

# ==========================================================
# MODEL-SPECIFIC CONFIG
# ==========================================================

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


_model_configs: Dict[str, Dict] = {}


def get_model_config(model: str) -> Dict:
    """Load model-specific configuration from conf/models/{family}.yaml"""
    global _model_configs

    family = get_model_family(model)

    if family in _model_configs:
        return _model_configs[family]

    # Load from YAML
    config_path = Path(__file__).parent.parent / "conf" / "models" / f"{family}.yaml"

    if config_path.exists():
        import yaml
        with open(config_path) as f:
            _model_configs[family] = yaml.safe_load(f)
    else:
        # Return empty config
        _model_configs[family] = {"name": family, "prompts": {}, "key_mappings": {}, "quirks": []}

    return _model_configs[family]


def get_model_prompt(model: str, task: str) -> str:
    """Get model-specific prompt for a task."""
    config = get_model_config(model)
    prompts = config.get("prompts", {})
    return prompts.get(task, "")


def get_model_prompts_all(model: str) -> Dict[str, str]:
    """Get all prompts for a model."""
    config = get_model_config(model)
    return config.get("prompts", {})


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

    # Map prompt keys to task definitions
    if "weekend_fixed" in prompts:
        tasks["detailed_json"] = {
            "messages": [
                {"role": "system", "content": prompts["weekend_fixed"]},
                {"role": "user", "content": "Extract venues from this context: Vaughan Sports Arena: indoor."},
            ],
            "validator": validate_detailed_json,
            "parse_json": True,
            "source": "Vaughan Sports Arena: indoor.",
        }

    if "weekend_transient" in prompts:
        tasks["json"] = {
            "messages": [
                {"role": "system", "content": prompts["weekend_transient"]},
                {"role": "user", "content": "Extract events from this context: Spring Festival April 20."},
            ],
            "validator": validate_detailed_json,
            "parse_json": True,
            "source": "Spring Festival April 20.",
        }

    if "filename" in prompts:
        tasks["filename"] = {
            "messages": [
                {"role": "user", "content": prompts["filename"] + "\n\nScreenshot of login page with error message."},
            ],
            "validator": validate_filename,
            "parse_json": False,
        }

    if "summarize" in prompts:
        tasks["summarize"] = {
            "messages": [
                {"role": "user", "content": prompts["summarize"].format("[@user1 | 10:00]: Test tweet")},
            ],
            "validator": validate_summary,
            "parse_json": False,
        }

    return tasks
