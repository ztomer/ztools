#!/usr/bin/env python3
"""
Config Management - Single source of truth from conf/config.yaml.
Auto-loads configuration on first access.
"""

import os
import sys
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
