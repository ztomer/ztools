import os
import yaml
from pathlib import Path

from lib.osaurus_lib import (
    restart_server,
    get_best_model,
    is_server_running,
    ensure_server as _osaurus_ensure_server,
)

DEBUG_EVENTS_FILE = Path.home() / ".weekend_events_debug_cache.json"
DEBUG_VENUES_FILE = Path.home() / ".weekend_venues_debug_cache.json"


def load_events_cache():
    if DEBUG_EVENTS_FILE.exists():
        return DEBUG_EVENTS_FILE.read_text()
    return None


def save_events_cache(events_str):
    DEBUG_EVENTS_FILE.write_text(events_str)


def load_venues_cache():
    if DEBUG_VENUES_FILE.exists():
        return DEBUG_VENUES_FILE.read_text()
    return None


def save_venues_cache(venues_str):
    DEBUG_VENUES_FILE.write_text(venues_str)


def load_weekend_config():
    config_path = Path(__file__).parent.parent / "conf" / "weekend.yaml"
    try:
        with open(config_path) as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"Failed to load weekend.yaml: {e}")
        return {}


try:
    WEEKEND_CONFIG = load_weekend_config()
except Exception:
    WEEKEND_CONFIG = {}
EXCLUDE_PLACES = WEEKEND_CONFIG.get("exclude_places", [])
CHILDREN = WEEKEND_CONFIG.get("children", [])
CHILDREN_STR = ", ".join([f"{c['age']}yo {c['gender']}" for c in CHILDREN]) if CHILDREN else "{CHILDREN_STR}"
CITY = WEEKEND_CONFIG.get("location", {}).get("city", "Vaughan")
REGION = WEEKEND_CONFIG.get("location", {}).get("region", "Toronto")
AGE_RANGE = f"{min(c['age'] for c in CHILDREN)}-{max(c['age'] for c in CHILDREN)}" if CHILDREN else "4-12"
DATES_STR = "April 24 to April 26"

MODEL_CONFIG = os.path.expanduser("~/.config/model_eval.json")

from lib.config import Task
MODEL_NAME = os.environ.get(
    "OLLAMA_MODEL", get_best_model(Task.JSON) or "gemma-4-26b-a4b-it-4bit"
)
OSAURUS_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:1337")
OSAURUS_APP = "/Applications/osaurus.app"


def is_server_running_ours():
    return is_server_running()


def restart_osaurus(wait=20):
    return restart_server(app_path=OSAURUS_APP, wait=wait)


def ensure_server(max_retries=3, wait=20):
    return _osaurus_ensure_server(max_retries=max_retries, wait=wait)
