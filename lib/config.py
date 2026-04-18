#!/usr/bin/env python3
"""
Config Management - Explicit configuration loading.
Replaces silent config loading at module import time.
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional

# Default configuration values
DEFAULT_TIMEOUTS = {
    "think": 30,
    "json": 60,
    "summarize": 30,
    "filename": 15,
    "vlm": 45,
}

DEFAULT_MAX_TOKENS = {
    "think": 2000,
    "json": 2000,
    "summarize": 2000,
    "filename": 500,
    "vlm": 3000,
}

DEFAULT_BEST_MODELS = {
    "think": "gemma-4-26b-a4b-it-4bit",
    "json": "gemma-4-26b-a4b-it-4bit",
    "summarize": "gemma-4-26b-a4b-it-4bit",
    "filename": "gemma-4-26b-a4b-it-4bit",
    "vlm": "gemma-4-26b-a4b-it-4bit",
}

# Global config state - must be initialized explicitly
_config_loaded = False
_timeouts = DEFAULT_TIMEOUTS.copy()
_max_tokens = DEFAULT_MAX_TOKENS.copy()
_best_models = DEFAULT_BEST_MODELS.copy()


def init_config(config_path: Optional[str] = None) -> bool:
    """
    Initialize configuration from YAML file.

    Args:
        config_path: Path to config.yaml file. If None, looks in conf/config.yaml

    Returns:
        bool: True if config loaded successfully, False otherwise

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If config file is invalid YAML
    """
    global _config_loaded, _timeouts, _max_tokens, _best_models

    if config_path is None:
        config_path = Path(__file__).parent.parent / "conf" / "config.yaml"

    config_file = Path(config_path)

    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")

    try:
        import yaml
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)

        if config is None:
            config = {}

        # Validate config structure
        if not isinstance(config, dict):
            raise ValueError("Config must be a dictionary")

        # Apply overrides
        if "best_models" in config:
            if not isinstance(config["best_models"], dict):
                raise ValueError("best_models must be a dictionary")
            _best_models.update(config["best_models"])

        if "timeouts" in config:
            if not isinstance(config["timeouts"], dict):
                raise ValueError("timeouts must be a dictionary")
            _timeouts.update(config["timeouts"])

        if "max_tokens" in config:
            if not isinstance(config["max_tokens"], dict):
                raise ValueError("max_tokens must be a dictionary")
            _max_tokens.update(config["max_tokens"])

        _config_loaded = True
        return True

    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML in config file: {e}")
    except Exception as e:
        raise ValueError(f"Error loading config: {e}")


def get_timeouts() -> Dict[str, int]:
    """Get current timeout configuration."""
    return _timeouts.copy()


def get_max_tokens() -> Dict[str, int]:
    """Get current max tokens configuration."""
    return _max_tokens.copy()


def get_best_models() -> Dict[str, str]:
    """Get current best models configuration."""
    return _best_models.copy()


def is_config_loaded() -> bool:
    """Check if configuration has been loaded."""
    return _config_loaded


def reset_config():
    """Reset configuration to defaults (for testing)."""
    global _config_loaded, _timeouts, _max_tokens, _best_models
    _config_loaded = False
    _timeouts = DEFAULT_TIMEOUTS.copy()
    _max_tokens = DEFAULT_MAX_TOKENS.copy()
    _best_models = DEFAULT_BEST_MODELS.copy()
