import datetime
import os
from pathlib import Path

from lib.config import Task
from lib.config_toml import load_config
from lib.osaurus_lib import (
    ensure_server as _osaurus_ensure_server,
)
from lib.osaurus_lib import (
    get_best_model,
    is_server_running,
    restart_server,
)
from lib.osaurus_server import ENSURE_MAX_RETRIES
from lib.osaurus_server import SERVER_WAIT as RESTART_WAIT
from lib.paths import conf_path

_cache_dir = Path(os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache"))) / "weekend"
DEBUG_EVENTS_FILE = _cache_dir / "events_debug_cache.json"
DEBUG_VENUES_FILE = _cache_dir / "venues_debug_cache.json"


def load_events_cache():
    if DEBUG_EVENTS_FILE.exists():
        return DEBUG_EVENTS_FILE.read_text()
    return None


def save_events_cache(events_str):
    DEBUG_EVENTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    DEBUG_EVENTS_FILE.write_text(events_str)


def load_venues_cache():
    if DEBUG_VENUES_FILE.exists():
        return DEBUG_VENUES_FILE.read_text()
    return None


def save_venues_cache(venues_str):
    DEBUG_VENUES_FILE.parent.mkdir(parents=True, exist_ok=True)
    DEBUG_VENUES_FILE.write_text(venues_str)


def load_weekend_config():
    config_path = conf_path("weekend.toml")
    return load_config(config_path)


try:
    WEEKEND_CONFIG = load_weekend_config()
except Exception:
    WEEKEND_CONFIG = {}
EXCLUDE_PLACES = WEEKEND_CONFIG.get("exclude_places", [])
_raw_children = WEEKEND_CONFIG.get("children", [])
CHILDREN = []
for c in _raw_children:
    if "birthday" in c:
        bday = (
            c["birthday"]
            if isinstance(c["birthday"], datetime.date)
            else datetime.datetime.strptime(c["birthday"], "%Y-%m-%d").date()
        )
        _today = datetime.date.today()
        c["age"] = _today.year - bday.year - ((_today.month, _today.day) < (bday.month, bday.day))
    CHILDREN.append(c)
CHILDREN_STR = (
    ", ".join([f"{c['age']}yo {c['gender']}" for c in CHILDREN]) if CHILDREN else "{CHILDREN_STR}"
)
LATITUDE = WEEKEND_CONFIG.get("location", {}).get("latitude", 43.8361)
LONGITUDE = WEEKEND_CONFIG.get("location", {}).get("longitude", -79.5083)
TIMEZONE = WEEKEND_CONFIG.get("location", {}).get("timezone", "America/Toronto")
CITY = WEEKEND_CONFIG.get("location", {}).get("city", "Vaughan")
REGION = WEEKEND_CONFIG.get("location", {}).get("region", "Toronto")
AGE_RANGE = (
    f"{min(c['age'] for c in CHILDREN)}-{max(c['age'] for c in CHILDREN)}" if CHILDREN else "4-12"
)

# Dynamic date fallback for LLM prompts — computed from this week's Friday
_today = datetime.date.today()
_friday = _today + datetime.timedelta((4 - _today.weekday()) % 7)
_sunday = _friday + datetime.timedelta(days=2)
DATES_STR = f"{_friday.strftime('%B %d')} to {_sunday.strftime('%B %d, %Y')}"

MODEL_CONFIG = str(Path.home() / ".config" / "model_eval.json")

_fallback_model = os.environ.get("WEEKEND_FALLBACK_MODEL", "gemma-4-26b-a4b-it-4bit")
MODEL_NAME = os.environ.get("OLLAMA_MODEL", get_best_model(Task.JSON) or _fallback_model)
OSAURUS_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:1337")
OSAURUS_APP = os.environ.get("OSAURUS_APP", "/Applications/osaurus.app")


def is_server_running_ours():
    return is_server_running()


def restart_osaurus(wait=RESTART_WAIT):
    return restart_server(app_path=OSAURUS_APP, wait=wait)


def ensure_server(max_retries=ENSURE_MAX_RETRIES, wait=RESTART_WAIT):
    return _osaurus_ensure_server(max_retries=max_retries, wait=wait)
