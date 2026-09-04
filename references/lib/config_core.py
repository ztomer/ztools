"""Config core - shared state, Task enum, init/reset."""

import enum
import threading
from pathlib import Path
from typing import Any, Dict, Optional

from lib.llm.constants import (
    DEFAULT_MAX_TOKENS as _FALLBACK_MAX_TOKENS,
)
from lib.llm.constants import (
    DEFAULT_MODEL as _FALLBACK_MODEL,
)
from lib.llm.constants import (
    DEFAULT_TIMEOUT as _FALLBACK_TIMEOUT,
)

from .config_toml import load_config
from .paths import conf_path
from .tui import WARN


class ConfigurationError(Exception):
    """Exception raised when configuration loading or parsing fails."""

    pass


class Task(enum.Enum):
    WEEKEND_FIXED = "weekend_fixed"
    WEEKEND_TRANSIENT = "weekend_transient"
    SUMMARIZE = "summarize"
    FILENAME = "filename"
    FILE_SUMMARY = "file_summary"
    JSON = "json"
    DETAILED_JSON = "detailed_json"


TaskKeys = Task


_config_loaded = False
_config: Dict[str, Any] = {}
_model_configs_cache: Dict[str, Dict] = {}
_config_lock = threading.Lock()


USER_CONFIG_PATH = Path.home() / ".config" / "ztools" / "config.toml"


def _auto_load():
    global _config_loaded, _config
    with _config_lock:
        if _config_loaded:
            return
        config_path = conf_path("config.toml")
        if not config_path.exists():
            print(f"{WARN} Config file not found, using fallback defaults")
            _config_loaded = True
            return
        try:
            loaded = load_config(config_path)
        except Exception as e:
            raise ConfigurationError(
                f"Failed to read configuration file '{config_path}': {e}"
            ) from e
        _config.clear()
        _config.update(loaded if isinstance(loaded, dict) else {})
        _config_loaded = True
        # User config overlay — ~/.config/ztools/config.toml overrides defaults
        if USER_CONFIG_PATH.exists():
            try:
                user_loaded = load_config(USER_CONFIG_PATH)
                if isinstance(user_loaded, dict):
                    _config.update(user_loaded)
            except Exception as e:
                print(f"{WARN} Failed to read user config '{USER_CONFIG_PATH}': {e}")


def init_config(config_path: Optional[str] = None) -> bool:
    global _config_loaded, _config
    if config_path is None:
        config_path = conf_path("config.toml")
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")
    try:
        loaded = load_config(config_file)
    except Exception as e:
        raise ConfigurationError(f"Failed to read configuration file '{config_file}': {e}") from e
    if loaded is None:
        loaded = {}
    if not isinstance(loaded, dict):
        raise ValueError("Config must be a dictionary")
    _config.clear()
    _config.update(loaded)
    _config_loaded = True
    return True


def reset_config():
    global _config_loaded
    _config_loaded = False
    _config.clear()
    _model_configs_cache.clear()


def get_config() -> Dict[str, Any]:
    _auto_load()
    return _config.copy()


def is_config_loaded() -> bool:
    _auto_load()
    return _config_loaded


__all__ = [
    "_FALLBACK_MAX_TOKENS",
    "_FALLBACK_MODEL",
    "_FALLBACK_TIMEOUT",
    "ConfigurationError",
    "Task",
    "TaskKeys",
    "_auto_load",
    "_config",
    "_config_loaded",
    "_model_configs_cache",
    "get_config",
    "init_config",
    "is_config_loaded",
    "reset_config",
]
