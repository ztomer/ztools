"""Config core - shared state, Task enum, init/reset."""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import enum


class Task(enum.Enum):
    WEEKEND_FIXED = "weekend_fixed"
    WEEKEND_TRANSIENT = "weekend_transient"
    SUMMARIZE = "summarize"
    FILENAME = "filename"
    FILE_SUMMARY = "file_summary"
    JSON = "json"
    DETAILED_JSON = "detailed_json"


TaskKeys = Task

_FALLBACK_TIMEOUT = 600
_FALLBACK_MAX_TOKENS = 16000
_FALLBACK_MODEL = "foundation"

_config_loaded = False
_config: Dict[str, Any] = {}
_model_configs_cache: Dict[str, Dict] = {}


def _auto_load():
    global _config_loaded, _config
    if _config_loaded:
        return
    config_path = Path(__file__).parent.parent / "conf" / "config.yaml"
    if not config_path.exists():
        print(f"Config file not found: {config_path}, using fallback defaults")
        _config_loaded = True
        return
    with open(config_path, 'r') as f:
        loaded = yaml.safe_load(f)
    _config.clear()
    _config.update(loaded if isinstance(loaded, dict) else {})
    _config_loaded = True


def init_config(config_path: Optional[str] = None) -> bool:
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
