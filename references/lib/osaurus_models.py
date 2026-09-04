#!/usr/bin/env python3
import os
from typing import List, Optional

import requests

from lib.llm.constants import DEFAULT_HOST, DEFAULT_PORT

# Timeouts for HTTP requests (seconds)
HTTP_READ_TIMEOUT = 10
SERVER_CHECK_TIMEOUT = 3

# API endpoints
API_CHAT_PATH = "/v1/chat/completions"
API_MODELS_PATH = "/v1/models"

# HTTP protocols & status codes
HTTP_PREFIX = "http"
HTTP_STATUS_OK = 200
HTTP_STATUS_NOT_FOUND = 404

# Model defaults
DEFAULT_ENV_VAR = "OLLAMA_MODEL"
FALLBACK_MODEL = "foundation"
DEFAULT_PREFERRED_MODELS = os.environ.get(
    "OSAURUS_PREFERRED_MODELS", "foundation,qwen,gemma"
).split(",")
DEFAULT_VLM_KEYWORDS = os.environ.get("OSAURUS_VLM_KEYWORDS", "vl,vision,qwen,llamavl").split(",")


def get_api_url(host: str = DEFAULT_HOST, port: int = DEFAULT_PORT) -> str:
    if "://" in host:
        return f"{host.rstrip('/')}{API_CHAT_PATH}"
    return f"http://{host}:{port}{API_CHAT_PATH}"


def get_base_url(host: str = DEFAULT_HOST, port: int = DEFAULT_PORT) -> str:
    if "://" in host:
        return host.rstrip("/")
    return f"http://{host}:{port}"


def get_models(host: str = DEFAULT_HOST, port: int = DEFAULT_PORT, api_key: str = "") -> List[str]:
    try:
        if host.startswith(HTTP_PREFIX):
            url = f"{host}{API_MODELS_PATH}"
        else:
            url = f"http://{host}:{port}{API_MODELS_PATH}"
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        with requests.Session() as s:
            resp = s.get(url, timeout=HTTP_READ_TIMEOUT, headers=headers)
        if resp.status_code == HTTP_STATUS_OK:
            return [m["id"] for m in resp.json().get("data", [])]
    except requests.exceptions.Timeout:
        pass
    except requests.exceptions.ConnectionError:
        pass
    except Exception as e:
        print(f"Warning: get_models failed: {e}")
    return []


def is_server_running(host: str = DEFAULT_HOST, port: int = DEFAULT_PORT) -> bool:
    try:
        if host.startswith(HTTP_PREFIX):
            url = f"{host}{API_MODELS_PATH}"
        else:
            url = f"http://{host}:{port}{API_MODELS_PATH}"
        with requests.Session() as s:
            resp = s.get(url, timeout=SERVER_CHECK_TIMEOUT)
        return resp.status_code in (HTTP_STATUS_OK, HTTP_STATUS_NOT_FOUND)
    except requests.exceptions.Timeout:
        return False
    except requests.exceptions.ConnectionError:
        return False
    except Exception as e:
        print(f"Warning: server check failed: {e}")
        return False


def check_llm_availability(
    host: str = DEFAULT_HOST, port: int = DEFAULT_PORT, api_key: str = ""
) -> bool:
    return is_server_running(host, port)


get_available_models = get_models


def get_best_model(task: str = None, env_var: str = DEFAULT_ENV_VAR) -> str:
    from .config import get_best_model as _get_best

    if task:
        return os.environ.get(env_var, _get_best(task))
    return os.environ.get(env_var, FALLBACK_MODEL)


def select_best_vlm_model(available_models: List[str]) -> Optional[str]:
    """Pick a model that can actually take an image.

    Asks each model's own config.json for a vision tower, and only falls back to
    DEFAULT_VLM_KEYWORDS for models it cannot find on disk.

    The keyword list is wrong in both directions, which is why it is now the last
    resort rather than the first: it accepts any model whose NAME contains a matching
    substring whether or not it has an image tower, and rejects every vision-capable
    model whose name happens not to. Run `ev --capabilities` for the current roster's
    actual answers -- a name is not evidence of a capability, and a list of today's
    exceptions written here would be stale by the next model install.

    Order is preserved, so the caller's preference still decides among models that
    genuinely have vision.
    """
    from lib.model_caps import probe_vision

    unknown = []
    for model in available_models:
        vision = probe_vision(model)
        if vision is True:
            return model
        if vision is None:
            unknown.append(model)

    # Nothing on disk said yes. For models we could not inspect, the keyword guess is
    # better than refusing outright -- foundation lands here.
    for keyword in DEFAULT_VLM_KEYWORDS:
        for model in unknown:
            if keyword.lower() in model.lower():
                return model
    return None


def select_best_model(models: list, preferred: list = None) -> str:
    if not models:
        return None
    if preferred is None:
        preferred = DEFAULT_PREFERRED_MODELS
    for pref in preferred:
        for model in models:
            if pref.lower() in model.lower():
                return model
    return models[0] if models else None
